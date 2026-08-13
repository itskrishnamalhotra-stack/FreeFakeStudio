# ============================================================
#  FreeFakeStudio — Gemini-like AI Image Studio
#  Conversational image generation & editing interface
#  Models: Z-Image Turbo · FLUX.2-klein 4B · ERNIE-Image-Turbo
#  Built for Google Colab T4 (15GB VRAM)
# ============================================================

import os, random, time, sys, gc, re, uuid, json, base64, traceback, html
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
from datetime import datetime

# ── Environment detection ──────────────────────────────────
def _running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IS_COLAB = _running_in_colab()
DEV_MODE = os.environ.get("FREEFAKESTUDIO_DEV", "").lower() in ("1", "true", "yes") or not IS_COLAB
DEBUG_MODE = os.environ.get("FFS_DEBUG", "1").lower() not in ("0", "false", "no", "off")

# ── Import model manager ──────────────────────────────────
import model_manager

# ── Conditional imports ────────────────────────────────────
if not DEV_MODE:
    import torch
    import cv2

import gradio as gr

# ── Save directory ─────────────────────────────────────────
import workspace as ws
SAVE_DIR = ws.get_save_dir()

def get_save_path(prefix="img"):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)[:20]
    uid = uuid.uuid4().hex[:6]
    return os.path.join(SAVE_DIR, f"{safe}_{uid}.png")


def _write_runtime_error(context, exc):
    """Write full runtime traceback to the persistent results/debug folder."""
    if not DEBUG_MODE:
        return None
    debug_dir = os.path.join(SAVE_DIR, "_debug")
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Context: {context}\n")
        f.write(f"Model loaded: {model_manager.get_current_model()}\n")
        f.write(f"DEV_MODE: {DEV_MODE}\n")
        f.write(f"IS_COLAB: {IS_COLAB}\n")
        f.write(f"COMFYUI_ROOT: {os.environ.get('COMFYUI_ROOT')}\n")
        f.write(f"WORKSPACE: {os.environ.get('FREEFAKESTUDIO_WORKSPACE')}\n\n")
        f.write("Traceback:\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return path

def make_seed(seed):
    if seed == 0 or seed == -1:
        random.seed(int(time.time()))
        seed = random.randint(0, 2**63)
    return int(seed)


# ═══════════════════════════════════════════════════════════
#  AUTO-MASK HELPERS (preserved from original)
# ═══════════════════════════════════════════════════════════
_rembg_session = None

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session

def auto_mask_background(image_pil):
    if DEV_MODE:
        # Simple mock mask for dev testing
        w, h = image_pil.size
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[h//4:3*h//4, w//4:3*w//4] = 0
        return mask
    from rembg import remove
    session = get_rembg_session()
    result = remove(image_pil, session=session, only_mask=True)
    mask_np = np.array(result)
    return 255 - mask_np

def auto_mask_except_face(image_pil):
    if DEV_MODE:
        w, h = image_pil.size
        mask = np.ones((h, w), dtype=np.uint8) * 255
        cx, cy = w // 2, h // 3
        rr = min(w, h) // 6
        Y, X = np.ogrid[:h, :w]
        circle = ((X - cx)**2 + (Y - cy)**2) <= rr**2
        mask[circle] = 0
        return mask
    img_np = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    h, w = img_np.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            pad_w, pad_h = int(fw * 0.3), int(fh * 0.3)
            x1, y1 = max(0, fx - pad_w), max(0, fy - pad_h)
            x2, y2 = min(w, fx + fw + pad_w), min(h, fy + fh + pad_h)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = (x2 - x1) // 2, (y2 - y1) // 2
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 0, -1)
    else:
        mask = auto_mask_background(image_pil)
    return mask

def generate_auto_mask_preview(image_pil, mask_mode):
    if image_pil is None:
        return None
    if mask_mode == "Background Only":
        mask_np = auto_mask_background(image_pil)
    elif mask_mode == "Everything Except Face":
        mask_np = auto_mask_except_face(image_pil)
    else:
        return None
    img_np = np.array(image_pil.convert("RGB")).copy()
    img_np[mask_np > 127] = [255, 255, 255]
    return Image.fromarray(img_np)


# ═══════════════════════════════════════════════════════════
#  SMART MASK SELECTION (preserved from original)
# ═══════════════════════════════════════════════════════════
def _select_mask_for_prompt(prompt, image_pil):
    """Select mask and denoise based on prompt keywords (FLUX models)."""
    p = prompt.lower().strip()
    bg_keywords = r'\b(background|backdrop|bg|behind|surroundings|scenery|scene)\b'
    if re.search(bg_keywords, p):
        mask = auto_mask_background(image_pil)
        pattern = r'^(?:change|make|turn|set|replace|convert)\s+(?:the\s+)?(?:background|bg|backdrop)\s+(?:to|into|with)\s+'
        cleaned = re.sub(pattern, '', prompt.strip(), flags=re.IGNORECASE).strip()
        if cleaned and cleaned != prompt.strip():
            cleaned = f"{cleaned} background"
        else:
            cleaned = prompt
        return mask, cleaned, 1.0
    # Non-background edit: clothing/body mask
    except_face = auto_mask_except_face(image_pil)
    background = auto_mask_background(image_pil)
    clothing_mask = np.where((except_face > 127) & (background < 127), 255, 0).astype(np.uint8)
    return clothing_mask, prompt, 0.75


# ═══════════════════════════════════════════════════════════
#  GENERATION CORE — yields status updates for streaming
# ═══════════════════════════════════════════════════════════
def do_generate(model_name, prompt, negative, aspect_ratio,
                seed, cfg, denoise, num_images, steps,
                input_image=None, mask_mode=None, editor_data=None):
    """
    Unified generation function. Handles:
    - text→image (no input_image)
    - img2img (input_image, no mask)
    - inpaint (input_image + mask)

    Yields (status_html, result_images, result_paths, seed_str) tuples for streaming.
    """
    seed = make_seed(seed)
    w, h = _parse_aspect(aspect_ratio)
    mode = "generate"

    # Determine mode
    if input_image is not None:
        if mask_mode and mask_mode != "None":
            mode = "inpaint"
        else:
            mode = "img2img"

    # Check capabilities
    caps = model_manager.get_capabilities(model_name)
    if mode == "img2img" and "img2img" not in caps:
        yield (_status_html("error", f"{model_name} doesn't support image editing. Switch to FLUX.2-klein 4B."),
               [], [], str(seed))
        return
    if mode == "inpaint" and "inpaint" not in caps:
        yield (_status_html("error", f"{model_name} doesn't support inpainting. Switch to FLUX.2-klein 4B."),
               [], [], str(seed))
        return

    # Status: preparing
    yield (_status_html("active", "Preparing request"), [], [], str(seed))

    # Ensure model is loaded. The actual load is blocking, so emit the planned
    # operational stage before entering the locked model switch.
    active_model = model_manager.get_current_model()
    if active_model == model_name:
        yield (_status_html("active", f"Using loaded {model_name}"), [], [], str(seed))
    elif active_model:
        yield (_status_html("active", f"Unloading {active_model} and loading {model_name}"), [], [], str(seed))
    else:
        yield (_status_html("active", f"Loading {model_name}"), [], [], str(seed))

    status_msgs = []
    def _on_status(msg):
        status_msgs.append(msg)

    try:
        engine = model_manager.ensure_model(model_name, status_callback=_on_status)
    except RuntimeError as e:
        debug_path = _write_runtime_error("model load", e)
        if debug_path:
            print(f"Runtime debug log written to: {debug_path}")
        yield (_status_html("error", f"{e} Check results/_debug for the full traceback."), [], [], str(seed))
        return

    if status_msgs:
        clean_msgs = [m.replace("[..]", "").replace("[ok]", "").strip() for m in status_msgs[-3:]]
        yield (_status_html("active", " / ".join(clean_msgs)), [], [], str(seed))

    # Status: generating
    if mode == "generate":
        yield (_status_html("active", f"Generating with {model_name}"), [], [], str(seed))
    elif mode == "img2img":
        yield (_status_html("active", f"Editing image with {model_name}"), [], [], str(seed))
    else:
        yield (_status_html("active", f"Inpainting with {model_name}"), [], [], str(seed))

    try:
        paths = []
        images = []

        for i in range(int(num_images)):
            if mode == "generate":
                img = engine.generate(prompt, negative, w, h,
                                      seed + i, cfg, denoise, int(steps))
            elif mode == "img2img":
                # For FLUX models, use smart mask selection
                if model_name == "FLUX.2-klein 4B":
                    mask, img_prompt, effective_denoise = _select_mask_for_prompt(prompt, input_image)
                    img = engine.img2img(input_image, img_prompt, negative,
                                         seed + i, cfg, effective_denoise, int(steps), mask=mask)
                else:
                    img = engine.img2img(input_image, prompt, negative,
                                         seed + i, cfg, denoise, int(steps))
            elif mode == "inpaint":
                mask_combined = _resolve_mask(input_image, mask_mode, editor_data)
                if mask_combined is None:
                    yield (_status_html("error", "No mask detected. Paint or select a mask mode."),
                           [], [], str(seed))
                    return
                img = engine.inpaint(input_image, mask_combined, prompt, negative,
                                     seed + i, cfg, denoise, int(steps))

            path = get_save_path("gen" if mode == "generate" else mode)
            img.save(path)
            paths.append(path)
            images.append(img)

        # Status: complete
        yield (_status_html("done", "Complete"), images, paths, str(seed))

    except Exception as e:
        debug_path = _write_runtime_error("generation", e)
        if debug_path:
            print(f"Runtime debug log written to: {debug_path}")
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            model_manager.unload_current()
            error_msg = f"GPU out of memory. The model has been unloaded. Try again or use fewer images."
        elif debug_path:
            error_msg = f"{error_msg} Check results/_debug for the full traceback."
        yield (_status_html("error", error_msg), [], [], str(seed))


def _resolve_mask(image_pil, mask_mode, editor_data):
    """Resolve mask from mode/editor data."""
    if mask_mode == "Manual Paint":
        return _extract_manual_mask(editor_data, image_pil)
    elif mask_mode == "Background Only":
        return auto_mask_background(image_pil)
    elif mask_mode == "Everything Except Face":
        return auto_mask_except_face(image_pil)
    return None


def _extract_manual_mask(editor_data, fallback_image):
    """Extract painted mask from Gradio ImageEditor data."""
    if editor_data is None:
        return None
    if isinstance(editor_data, dict):
        bg = editor_data.get("background")
        layers = editor_data.get("layers", [])
        if bg is None:
            return None
        if not isinstance(bg, Image.Image):
            bg = Image.fromarray(bg)
        manual_mask = np.zeros((bg.size[1], bg.size[0]), dtype=np.uint8)
        for layer in layers:
            if not isinstance(layer, Image.Image):
                layer = Image.fromarray(layer)
            layer = layer.resize(bg.size)
            arr = np.array(layer)
            if arr.ndim == 3 and arr.shape[2] == 4:
                manual_mask = np.maximum(manual_mask, arr[:, :, 3])
            elif arr.ndim == 3:
                manual_mask = np.maximum(manual_mask, np.mean(arr[:, :, :3], axis=2).astype(np.uint8))
            elif arr.ndim == 2:
                manual_mask = np.maximum(manual_mask, arr)
        if np.sum(manual_mask > 0) > 0:
            return manual_mask
    return None


def _parse_aspect(aspect_str):
    """Parse aspect ratio string like '1024x1024 (1:1)' to (w, h)."""
    try:
        dims = aspect_str.split("(")[0].strip().split("x")
        return int(dims[0]), int(dims[1])
    except Exception:
        return 1024, 1024


# ═══════════════════════════════════════════════════════════
#  STATUS HTML HELPERS
# ═══════════════════════════════════════════════════════════
def _status_html(state, message):
    """Generate status indicator HTML."""
    if state == "active":
        icon = '<span class="ffs-status-dot ffs-pulse"></span>'
        cls = "ffs-status-active"
    elif state == "done":
        icon = '<span class="ffs-status-check">✓</span>'
        cls = "ffs-status-done"
    elif state == "error":
        icon = '<span class="ffs-status-error">✗</span>'
        cls = "ffs-status-error-text"
    else:
        icon = '<span class="ffs-status-dot"></span>'
        cls = ""
    return f'<div class="ffs-status-line {cls}">{icon} {message}</div>'


def _request_html(prompt, model_name, has_image, mask_mode):
    prompt = html.escape(prompt or "(image edit)")
    mode = "image edit" if has_image else "text to image"
    mask_note = f" / {html.escape(mask_mode)}" if mask_mode and mask_mode != "None" else ""
    return (
        '<div class="ffs-turn ffs-turn-user">'
        '<div class="ffs-role">You</div>'
        f'<div class="ffs-bubble">{prompt}</div>'
        f'<div class="ffs-meta">{html.escape(model_name)} / {mode}{mask_note}</div>'
        '</div>'
    )


def _assistant_html(paths, seed_str):
    count = len(paths or [])
    noun = "image" if count == 1 else "images"
    return (
        '<div class="ffs-turn ffs-turn-assistant">'
        '<div class="ffs-role">FreeFakeStudio</div>'
        f'<div class="ffs-bubble">Generated {count} {noun}. Seed: {html.escape(str(seed_str))}.</div>'
        '</div>'
    )


def _render_history(history):
    return "".join(history or [])


# ═══════════════════════════════════════════════════════════
#  ASPECT RATIOS
# ═══════════════════════════════════════════════════════════
ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
]
DEFAULT_NEG = "low quality, blurry, pixelated, noise, watermark, text, logo"


# ═══════════════════════════════════════════════════════════
#  CSS — Gemini-like design with dark/light support
# ═══════════════════════════════════════════════════════════
CSS = """
/* ── Root Variables ────────────────────────────────────── */
:root {
    --ffs-bg: #f8f9fa;
    --ffs-surface: #ffffff;
    --ffs-surface-2: #f1f3f5;
    --ffs-text: #1a1a2e;
    --ffs-text-2: #6b7280;
    --ffs-border: #e5e7eb;
    --ffs-accent: #4f6df5;
    --ffs-accent-2: #7c5bf0;
    --ffs-accent-bg: rgba(79, 109, 245, 0.08);
    --ffs-user-bg: #e8edf5;
    --ffs-assistant-bg: #ffffff;
    --ffs-danger: #ef4444;
    --ffs-success: #10b981;
    --ffs-radius: 16px;
    --ffs-radius-sm: 10px;
    --ffs-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --ffs-shadow-lg: 0 4px 16px rgba(0,0,0,0.08);
    --ffs-max-width: 860px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --ffs-bg: #0f0f14;
        --ffs-surface: #1a1a24;
        --ffs-surface-2: #22222e;
        --ffs-text: #e8e8f0;
        --ffs-text-2: #9ca3af;
        --ffs-border: #2d2d3a;
        --ffs-accent: #6b8aff;
        --ffs-accent-2: #9b7dff;
        --ffs-accent-bg: rgba(107, 138, 255, 0.1);
        --ffs-user-bg: #1e1e2e;
        --ffs-assistant-bg: #16161e;
        --ffs-shadow: 0 1px 3px rgba(0,0,0,0.3);
        --ffs-shadow-lg: 0 4px 16px rgba(0,0,0,0.4);
    }
}

/* Force dark mode class */
.dark {
    --ffs-bg: #0f0f14;
    --ffs-surface: #1a1a24;
    --ffs-surface-2: #22222e;
    --ffs-text: #e8e8f0;
    --ffs-text-2: #9ca3af;
    --ffs-border: #2d2d3a;
    --ffs-accent: #6b8aff;
    --ffs-accent-2: #9b7dff;
    --ffs-accent-bg: rgba(107, 138, 255, 0.1);
    --ffs-user-bg: #1e1e2e;
    --ffs-assistant-bg: #16161e;
    --ffs-shadow: 0 1px 3px rgba(0,0,0,0.3);
    --ffs-shadow-lg: 0 4px 16px rgba(0,0,0,0.4);
}

/* ── Base ──────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--ffs-bg) !important;
    max-width: 100% !important;
    padding: 0 !important;
}

footer { display: none !important; }

/* ── Header Bar ────────────────────────────────────────── */
.ffs-header {
    background: var(--ffs-surface);
    border-bottom: 1px solid var(--ffs-border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
}
.ffs-logo {
    font-size: 1.25em;
    font-weight: 700;
    color: var(--ffs-text);
    display: flex;
    align-items: center;
    gap: 8px;
}
.ffs-logo-icon {
    font-size: 1.3em;
}
.ffs-logo-gradient {
    background: linear-gradient(135deg, var(--ffs-accent), var(--ffs-accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Model Status Badge ────────────────────────────────── */
.ffs-model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 500;
    background: var(--ffs-accent-bg);
    color: var(--ffs-accent);
    border: 1px solid transparent;
}
.ffs-model-badge.ready {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
}
.ffs-model-badge.loading {
    background: rgba(245, 158, 11, 0.1);
    color: #f59e0b;
}

/* ── Chat Area ─────────────────────────────────────────── */
.ffs-chat-area {
    max-width: var(--ffs-max-width);
    margin: 0 auto;
    padding: 24px 16px;
    min-height: 60vh;
}

/* ── Status Indicators ─────────────────────────────────── */
.ffs-status-line {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 16px;
    font-size: 0.9em;
    color: var(--ffs-text-2);
    font-weight: 500;
}
.ffs-status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--ffs-accent);
}
.ffs-pulse {
    animation: ffs-pulse 1.5s ease-in-out infinite;
}
@keyframes ffs-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}
.ffs-status-check {
    color: var(--ffs-success);
    font-weight: 700;
    font-size: 1.1em;
}
.ffs-status-error {
    color: var(--ffs-danger);
    font-weight: 700;
}
.ffs-status-done { color: var(--ffs-success); }
.ffs-status-error-text { color: var(--ffs-danger); }
.ffs-status-active { color: var(--ffs-text); }

/* Chat turns */
.ffs-history {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.ffs-turn {
    max-width: 100%;
}
.ffs-role {
    font-size: 0.78em;
    font-weight: 700;
    color: var(--ffs-text-2);
    margin-bottom: 5px;
}
.ffs-bubble {
    background: var(--ffs-surface);
    color: var(--ffs-text);
    border: 1px solid var(--ffs-border);
    border-radius: 8px;
    padding: 11px 13px;
    line-height: 1.45;
    box-shadow: var(--ffs-shadow);
    overflow-wrap: anywhere;
}
.ffs-turn-user .ffs-bubble {
    background: var(--ffs-user-bg);
}
.ffs-meta {
    margin-top: 5px;
    color: var(--ffs-text-2);
    font-size: 0.76em;
}

/* ── Result Card ───────────────────────────────────────── */
.ffs-result-card {
    background: var(--ffs-surface);
    border: 1px solid var(--ffs-border);
    border-radius: var(--ffs-radius);
    padding: 16px;
    margin-top: 12px;
    box-shadow: var(--ffs-shadow);
}
.ffs-result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
}
.ffs-result-grid img {
    width: 100%;
    border-radius: var(--ffs-radius-sm);
    cursor: pointer;
    transition: transform 0.2s;
}
.ffs-result-grid img:hover {
    transform: scale(1.02);
}
.ffs-result-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    padding-top: 8px;
    border-top: 1px solid var(--ffs-border);
}

/* ── Composer (bottom bar) ─────────────────────────────── */
.ffs-composer-wrap {
    position: sticky;
    bottom: 0;
    background: var(--ffs-bg);
    border-top: 1px solid var(--ffs-border);
    padding: 12px 16px 16px;
    z-index: 50;
}
.ffs-composer {
    max-width: var(--ffs-max-width);
    margin: 0 auto;
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: var(--ffs-surface);
    border: 1px solid var(--ffs-border);
    border-radius: 24px;
    padding: 8px 12px;
    box-shadow: var(--ffs-shadow-lg);
    transition: border-color 0.2s;
}
.ffs-composer:focus-within {
    border-color: var(--ffs-accent);
}

/* ── Attachment Thumbnail ──────────────────────────────── */
.ffs-attachment {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: var(--ffs-accent-bg);
    border-radius: var(--ffs-radius-sm);
    margin-bottom: 8px;
    max-width: var(--ffs-max-width);
    margin-left: auto;
    margin-right: auto;
}
.ffs-attachment img {
    width: 48px;
    height: 48px;
    border-radius: 8px;
    object-fit: cover;
}
.ffs-attachment-info {
    flex: 1;
    font-size: 0.85em;
    color: var(--ffs-text-2);
}

/* ── Buttons ───────────────────────────────────────────── */
.ffs-btn {
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.82em;
    font-weight: 500;
    border: 1px solid var(--ffs-border);
    background: var(--ffs-surface);
    color: var(--ffs-text);
    cursor: pointer;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.ffs-btn:hover {
    background: var(--ffs-surface-2);
    border-color: var(--ffs-accent);
}
.ffs-btn-primary {
    background: linear-gradient(135deg, var(--ffs-accent), var(--ffs-accent-2)) !important;
    color: white !important;
    border: none !important;
}
.ffs-btn-primary:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Settings Panel ────────────────────────────────────── */
.ffs-settings-panel {
    max-width: var(--ffs-max-width);
    margin: 0 auto;
}

/* ── Empty State ───────────────────────────────────────── */
.ffs-empty {
    text-align: center;
    padding: 80px 20px;
    color: var(--ffs-text-2);
}
.ffs-empty-icon {
    font-size: 3em;
    margin-bottom: 16px;
    opacity: 0.5;
}
.ffs-empty-title {
    font-size: 1.3em;
    font-weight: 600;
    color: var(--ffs-text);
    margin-bottom: 8px;
}
.ffs-empty-sub {
    font-size: 0.95em;
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── Shimmer loading ───────────────────────────────────── */
@keyframes ffs-shimmer {
    0% { background-position: -200px 0; }
    100% { background-position: calc(200px + 100%) 0; }
}
.ffs-shimmer {
    background: linear-gradient(90deg,
        var(--ffs-surface-2) 0px,
        var(--ffs-surface) 40px,
        var(--ffs-surface-2) 80px);
    background-size: 200px 100%;
    animation: ffs-shimmer 1.5s ease-in-out infinite;
    border-radius: var(--ffs-radius-sm);
    height: 200px;
}

/* ── Mobile Responsive ─────────────────────────────────── */
@media (max-width: 640px) {
    .ffs-header { padding: 10px 12px; }
    .ffs-chat-area { padding: 12px 8px; }
    .ffs-result-grid {
        grid-template-columns: 1fr;
    }
    .ffs-result-actions {
        justify-content: center;
    }
    .ffs-composer {
        border-radius: 16px;
    }
    .ffs-empty { padding: 40px 16px; }
}

/* ── Gradio overrides ──────────────────────────────────── */
.contain { max-width: 100% !important; }
#component-0 { max-width: 100% !important; }

/* Hide default Gradio submit buttons we override */
.ffs-hidden { display: none !important; }

/* Image editor fixes */
.image-editor-container, .image-editor-container .wrapper {
    overflow: hidden !important;
}
button[aria-label="Pan"], button[aria-label="Move"] {
    display: none !important;
}

/* Gallery in results */
.ffs-result-gallery .gallery-item {
    border-radius: var(--ffs-radius-sm) !important;
}

/* Accordion styling */
.ffs-settings .label-wrap {
    background: var(--ffs-surface) !important;
    border-radius: var(--ffs-radius-sm) !important;
}
"""

# ═══════════════════════════════════════════════════════════
#  JAVASCRIPT — brush size boost, keyboard shortcuts
# ═══════════════════════════════════════════════════════════
JS_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script>
// Fix gallery share → open in new tab
(function() {
    var _origCanShare = navigator.canShare ? navigator.canShare.bind(navigator) : null;
    var _origShare = navigator.share ? navigator.share.bind(navigator) : null;
    navigator.canShare = function(data) {
        if (data && data.files && data.files.length > 0 &&
            data.files[0].type && data.files[0].type.startsWith('image/')) return true;
        return _origCanShare ? _origCanShare(data) : false;
    };
    navigator.share = async function(data) {
        if (data && data.files && data.files.length > 0 &&
            data.files[0].type && data.files[0].type.startsWith('image/')) {
            var url = URL.createObjectURL(data.files[0]);
            window.open(url, '_blank');
            return;
        }
        if (_origShare) return _origShare(data);
    };
})();

// Boost brush/eraser max slider from ~100 to 300
function boostBrushMax() {
    document.querySelectorAll('input[type="range"]').forEach(function(slider) {
        if (parseFloat(slider.max) > 10 && parseFloat(slider.max) <= 110) {
            var parent = slider.closest('.image-editor, .image_editor');
            if (!parent) {
                var labels = slider.closest('.block, .wrap, div');
                if (labels && labels.querySelector('canvas')) parent = labels;
            }
            if (parent || slider.closest('[data-testid]')) {
                slider.max = 300;
                slider.setAttribute('max', '300');
            }
        }
    });
}
setInterval(boostBrushMax, 2000);
setTimeout(boostBrushMax, 1000);
</script>
"""


# ═══════════════════════════════════════════════════════════
#  GRADIO UI — Gemini-like conversational interface
# ═══════════════════════════════════════════════════════════

ffs_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
)


with gr.Blocks(title="FreeFakeStudio") as demo:

    # ── Session State ──────────────────────────────────────
    chat_history = gr.State([])           # list of {role, content, ...}
    attached_image = gr.State(None)       # PIL Image or None
    last_gen_settings = gr.State(None)    # for Regenerate
    selected_result_idx = gr.State(0)

    # ═══════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════
    with gr.Row(elem_classes="ffs-header"):
        gr.HTML("""
            <div class="ffs-logo">
                <span class="ffs-logo-icon">🎭</span>
                <span class="ffs-logo-gradient">FreeFakeStudio</span>
            </div>
        """)
        with gr.Column(scale=0, min_width=200):
            model_selector = gr.Dropdown(
                choices=model_manager.MODEL_NAMES,
                value="Z-Image Turbo",
                label="Model",
                container=False,
                scale=0,
                min_width=180,
            )
        with gr.Column(scale=0, min_width=100):
            model_status_display = gr.HTML(
                '<span class="ffs-model-badge">Not loaded</span>'
            )
        with gr.Column(scale=0, min_width=80):
            new_chat_btn = gr.Button("✨ New", size="sm", variant="secondary")

    # ═══════════════════════════════════════════════════════
    # MAIN CONTENT AREA
    # ═══════════════════════════════════════════════════════
    with gr.Column(elem_classes="ffs-chat-area"):

        conversation_display = gr.HTML(
            value='<div class="ffs-history"></div>',
            elem_classes="ffs-history",
        )

        # Status display (streaming updates)
        status_display = gr.HTML(
            value="""
            <div class="ffs-empty">
                <div class="ffs-empty-icon">🎨</div>
                <div class="ffs-empty-title">What would you like to create?</div>
                <div class="ffs-empty-sub">
                    Describe an image, attach a photo to edit, or select a model and start creating.
                </div>
            </div>
            """,
            elem_id="ffs-status",
        )

        # Results gallery
        _gallery_kwargs = dict(
            label="Results",
            columns=2,
            height=520,
            object_fit="contain",
            preview=True,
            visible=False,
            elem_classes="ffs-result-gallery",
        )
        result_gallery = gr.Gallery(**_gallery_kwargs)

        # Download all
        result_files = gr.File(
            label="Download All",
            file_count="multiple",
            visible=False,
        )

        # Seed display
        seed_display = gr.Textbox(
            label="Seed Used", interactive=False, visible=False,
        )

        # ── Action Buttons Row ─────────────────────────────
        with gr.Row(visible=False) as action_row:
            add_to_prompt_btn = gr.Button("📎 Add to Prompt", size="sm")
            regenerate_btn = gr.Button("🔄 Regenerate", size="sm")

    # ═══════════════════════════════════════════════════════
    # EDITING PANEL (for mask/inpaint)
    # ═══════════════════════════════════════════════════════
    with gr.Accordion("🎨 Mask / Edit Tools", open=False, visible=False,
                       elem_classes="ffs-settings") as edit_panel:
        with gr.Column(elem_classes="ffs-settings-panel"):
            mask_mode = gr.Radio(
                choices=["None", "Manual Paint", "Background Only", "Everything Except Face"],
                value="None",
                label="Mask Mode",
            )
            with gr.Group(visible=False) as manual_mask_group:
                mask_editor = gr.ImageEditor(
                    label="Paint Mask (white = change)",
                    type="pil",
                    canvas_size=(2048, 2048),
                    brush=gr.Brush(colors=["#ffffff"], default_size=40, default_color="#ffffff"),
                    eraser=gr.Eraser(default_size=40),
                    sources=["upload"], transforms=[],
                    layers=False,
                )
            with gr.Group(visible=False) as auto_mask_group:
                mask_preview = gr.Image(
                    label="Mask Preview (white = will be changed)",
                    height=300, interactive=False,
                )
                edit_mask_btn = gr.Button("✏️ Edit This Mask Manually", variant="secondary")

    # ═══════════════════════════════════════════════════════
    # SETTINGS PANEL
    # ═══════════════════════════════════════════════════════
    with gr.Accordion("⚙️ Generation Settings", open=False,
                       elem_classes="ffs-settings") as settings_panel:
        with gr.Column(elem_classes="ffs-settings-panel"):
            with gr.Row():
                aspect_ratio = gr.Dropdown(
                    ASPECTS, value="1024x1024 (1:1)",
                    label="Aspect Ratio",
                )
                num_images = gr.Slider(1, 8, value=1, step=1, label="Images")
            with gr.Row():
                gen_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                gen_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
            with gr.Row():
                gen_cfg = gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="CFG")
                gen_denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
            negative_prompt = gr.Textbox(
                DEFAULT_NEG, label="Negative Prompt", lines=2,
            )

    # ═══════════════════════════════════════════════════════
    # COMPOSER (bottom bar)
    # ═══════════════════════════════════════════════════════
    with gr.Row(elem_classes="ffs-composer-wrap"):
        with gr.Column(elem_classes="ffs-settings-panel"):
            # Attachment display
            attachment_display = gr.Image(
                label="Attached Image",
                type="pil",
                height=80,
                visible=False,
                sources=["upload"],
                interactive=True,
            )
            with gr.Row():
                attach_btn = gr.UploadButton(
                    "📎",
                    file_types=["image"],
                    size="sm",
                    min_width=40,
                )
                prompt_input = gr.Textbox(
                    placeholder="Describe what you want to create or edit...",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                    scale=10,
                    container=False,
                )
                send_btn = gr.Button(
                    "➤",
                    variant="primary",
                    size="sm",
                    min_width=50,
                    elem_classes="ffs-btn-primary",
                )

    # ═══════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════

    # ── Attachment handling ─────────────────────────────────
    def handle_upload(file):
        if file is None:
            return gr.update(visible=False, value=None), None, gr.update(visible=True)
        img = Image.open(file).convert("RGB")
        return (gr.update(visible=True, value=img), img,
                gr.update(visible=True))

    attach_btn.upload(
        handle_upload,
        inputs=[attach_btn],
        outputs=[attachment_display, attached_image, edit_panel],
    )

    def clear_attachment():
        return gr.update(visible=False, value=None), None, gr.update(visible=False)

    # ── Mask mode toggle ───────────────────────────────────
    def toggle_mask_ui(mode):
        manual = mode == "Manual Paint"
        auto = mode in ("Background Only", "Everything Except Face")
        return (
            gr.update(visible=manual),  # manual_mask_group
            gr.update(visible=auto),    # auto_mask_group
        )

    mask_mode.change(
        toggle_mask_ui,
        inputs=[mask_mode],
        outputs=[manual_mask_group, auto_mask_group],
    )

    # ── Auto mask preview ──────────────────────────────────
    def update_mask_preview(image, mode):
        if image is None or mode in ("None", "Manual Paint"):
            return None
        return generate_auto_mask_preview(image, mode)

    mask_mode.change(
        update_mask_preview,
        inputs=[attached_image, mask_mode],
        outputs=[mask_preview],
    )

    attachment_display.change(
        lambda img: img,
        inputs=[attachment_display],
        outputs=[attached_image],
    )

    # ── Edit mask manually button ──────────────────────────
    def do_edit_mask_manually(image, mode):
        if image is None:
            raise gr.Error("Attach an image first!")
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        original = image.convert("RGB")
        w, h = original.size
        if mode == "Background Only":
            mask_np = auto_mask_background(original)
        elif mode == "Everything Except Face":
            mask_np = auto_mask_except_face(original)
        else:
            raise gr.Error("Select Background Only or Everything Except Face first.")
        mask_layer = np.zeros((h, w, 4), dtype=np.uint8)
        mask_layer[:, :, 0] = 255
        mask_layer[:, :, 1] = 255
        mask_layer[:, :, 2] = 255
        mask_layer[:, :, 3] = mask_np
        mask_layer_pil = Image.fromarray(mask_layer, "RGBA")
        editor_value = {"background": original, "layers": [mask_layer_pil], "composite": original}
        return (
            editor_value,
            "Manual Paint",
            gr.update(visible=True),
            gr.update(visible=False),
        )

    edit_mask_btn.click(
        do_edit_mask_manually,
        inputs=[attached_image, mask_mode],
        outputs=[mask_editor, mask_mode, manual_mask_group, auto_mask_group],
    )

    # ── Model selector change ──────────────────────────────
    def on_model_change(model_name):
        status = model_manager.get_model_status()
        s = status.get(model_name, "available")
        defaults = model_manager.get_defaults(model_name)
        if s == "ready":
            badge = '<span class="ffs-model-badge ready">● Ready</span>'
        elif s == "missing":
            badge = '<span class="ffs-model-badge" style="color: var(--ffs-danger)">✗ Missing</span>'
        else:
            badge = '<span class="ffs-model-badge">○ Available</span>'

        return (
            badge,
            gr.update(value=defaults.get("steps", 8)),
            gr.update(value=defaults.get("cfg", 1.0)),
            gr.update(value=defaults.get("denoise", 1.0)),
        )

    model_selector.change(
        on_model_change,
        inputs=[model_selector],
        outputs=[model_status_display, gen_steps, gen_cfg, gen_denoise],
    )

    # ── SEND (main generation) ─────────────────────────────
    def on_send(model_name, prompt, image, mask_m, editor_data,
                aspect, seed, steps, cfg, denoise, n_images, neg, history):
        """Main generation handler. Yields streaming updates."""
        history = list(history or [])
        if not prompt.strip() and image is None:
            yield (
                _status_html("error", "Please enter a prompt or attach an image."),
                gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
                None,
                _render_history(history), history,
            )
            return

        # Determine effective mask mode
        effective_mask = mask_m if mask_m != "None" else None
        request_history = history + [_request_html(prompt, model_name, image is not None, mask_m)]

        # Save settings for regenerate
        settings = {
            "model": model_name, "prompt": prompt, "neg": neg,
            "aspect": aspect, "seed": seed, "steps": steps,
            "cfg": cfg, "denoise": denoise, "n_images": n_images,
            "mask_mode": mask_m,
        }

        for status_html, images, paths, seed_str in do_generate(
            model_name, prompt, neg, aspect,
            seed, cfg, denoise, n_images, steps,
            input_image=image, mask_mode=effective_mask,
            editor_data=editor_data,
        ):
            if images:
                final_history = request_history + [_assistant_html(paths, seed_str)]
                # Show results
                yield (
                    status_html,
                    gr.update(visible=True, value=paths),
                    gr.update(visible=True, value=paths),
                    gr.update(visible=True, value=seed_str),
                    gr.update(visible=True),  # action_row
                    '<span class="ffs-model-badge ready">● Ready</span>',
                    settings,
                    _render_history(final_history),
                    final_history,
                )
            else:
                yield (
                    status_html,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    settings if "error" not in status_html.lower() else None,
                    _render_history(request_history),
                    request_history,
                )

    send_btn.click(
        on_send,
        inputs=[
            model_selector, prompt_input, attached_image,
            mask_mode, mask_editor,
            aspect_ratio, gen_seed, gen_steps, gen_cfg, gen_denoise,
            num_images, negative_prompt, chat_history,
        ],
        outputs=[
            status_display, result_gallery, result_files,
            seed_display, action_row,
            model_status_display, last_gen_settings,
            conversation_display, chat_history,
        ],
    )

    # Enter key submit
    prompt_input.submit(
        on_send,
        inputs=[
            model_selector, prompt_input, attached_image,
            mask_mode, mask_editor,
            aspect_ratio, gen_seed, gen_steps, gen_cfg, gen_denoise,
            num_images, negative_prompt, chat_history,
        ],
        outputs=[
            status_display, result_gallery, result_files,
            seed_display, action_row,
            model_status_display, last_gen_settings,
            conversation_display, chat_history,
        ],
    )

    # ── Add to Prompt ──────────────────────────────────────
    def on_add_to_prompt(gallery_data, selected_idx):
        """Take the selected/first result and set it as attachment."""
        if not gallery_data:
            raise gr.Error("No results to add!")
        idx = min(int(selected_idx or 0), len(gallery_data) - 1)
        img_data = gallery_data[idx]
        if isinstance(img_data, tuple):
            img_data = img_data[0]
        if isinstance(img_data, str):
            img_data = Image.open(img_data)
        # Check if current model supports editing
        return (
            gr.update(visible=True, value=img_data),  # attachment_display
            img_data,                                  # attached_image
            gr.update(visible=True),                  # edit_panel
            gr.update(value=""),                       # clear prompt
        )

    add_to_prompt_btn.click(
        on_add_to_prompt,
        inputs=[result_gallery, selected_result_idx],
        outputs=[attachment_display, attached_image, edit_panel, prompt_input],
    )

    def _on_gallery_select(evt: gr.SelectData):
        return evt.index

    result_gallery.select(
        _on_gallery_select,
        outputs=[selected_result_idx],
    )

    # ── Regenerate ─────────────────────────────────────────
    def on_regenerate(settings, image, history):
        """Re-run the last generation with a new random seed."""
        history = list(history or [])
        if settings is None:
            raise gr.Error("No previous generation to regenerate!")
        request_history = history + [_request_html(
            f"Regenerate: {settings['prompt']}",
            settings["model"],
            image is not None,
            settings.get("mask_mode", "None"),
        )]
        # Force new random seed
        for status_html, images, paths, seed_str in do_generate(
            settings["model"], settings["prompt"], settings["neg"],
            settings["aspect"], 0,  # 0 = random seed
            settings["cfg"], settings["denoise"],
            settings["n_images"], settings["steps"],
            input_image=image,
            mask_mode=settings.get("mask_mode") if settings.get("mask_mode") != "None" else None,
        ):
            if images:
                final_history = request_history + [_assistant_html(paths, seed_str)]
                yield (
                    status_html,
                    gr.update(visible=True, value=paths),
                    gr.update(visible=True, value=paths),
                    gr.update(visible=True, value=seed_str),
                    _render_history(final_history),
                    final_history,
                )
            else:
                yield (
                    status_html,
                    gr.update(), gr.update(), gr.update(),
                    _render_history(request_history),
                    request_history,
                )

    regenerate_btn.click(
        on_regenerate,
        inputs=[last_gen_settings, attached_image, chat_history],
        outputs=[
            status_display, result_gallery, result_files, seed_display,
            conversation_display, chat_history,
        ],
    )

    # ── New Chat ───────────────────────────────────────────
    def on_new_chat():
        return (
            # status_display
            """<div class="ffs-empty">
                <div class="ffs-empty-icon">🎨</div>
                <div class="ffs-empty-title">What would you like to create?</div>
                <div class="ffs-empty-sub">
                    Describe an image, attach a photo to edit, or select a model and start creating.
                </div>
            </div>""",
            gr.update(visible=False, value=None),   # result_gallery
            gr.update(visible=False, value=None),   # result_files
            gr.update(visible=False, value=""),      # seed_display
            gr.update(visible=False),                # action_row
            gr.update(visible=False, value=None),    # attachment_display
            None,                                     # attached_image
            gr.update(value=""),                      # prompt_input
            None,                                     # last_gen_settings
            gr.update(visible=False),                # edit_panel
            "None",                                   # mask_mode
            '<div class="ffs-history"></div>',        # conversation_display
            [],                                       # chat_history
        )

    new_chat_btn.click(
        on_new_chat,
        outputs=[
            status_display, result_gallery, result_files,
            seed_display, action_row,
            attachment_display, attached_image, prompt_input,
            last_gen_settings, edit_panel, mask_mode,
            conversation_display, chat_history,
        ],
    )


# ═══════════════════════════════════════════════════════════
#  STARTUP — NO model preload
# ═══════════════════════════════════════════════════════════
if DEV_MODE:
    print("\n" + "=" * 50)
    print("  FreeFakeStudio — Local Development Mode")
    print("  Real AI models are disabled.")
    print("  UI-only testing is enabled.")
    print("=" * 50 + "\n")
    # Mark all models as available in dev mode
    for name in model_manager.MODEL_NAMES:
        model_manager.set_model_availability(name, True)
else:
    # Check which models are installed (but do NOT load any)
    model_manager.check_model_files(os.environ.get("COMFYUI_ROOT", "/content/ComfyUI"))
    status = model_manager.get_model_status()
    print("\n🎭 FreeFakeStudio — Model Status:")
    for name, st in status.items():
        icon = "✓" if st != "missing" else "✗"
        print(f"  {icon} {name}: {st}")
    print()

# Configure Gradio queue for single GPU concurrency
demo.queue(default_concurrency_limit=1)


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# The Colab launcher supplies an absolute HTTPS root for its selected proxy.
if __name__ == "__main__":
    demo.launch(
        share=_env_flag("FREEFAKESTUDIO_SHARE", IS_COLAB),
        debug=True,
        show_error=True,
        inline=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=ffs_theme,
        css=CSS,
        head=JS_HEAD,
        root_path=os.environ.get("FREEFAKESTUDIO_PUBLIC_URL") or None,
        allowed_paths=[os.path.abspath(SAVE_DIR)],
    )
