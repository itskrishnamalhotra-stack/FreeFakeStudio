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
            action = "Generating" if mode == "generate" else "Editing" if mode == "img2img" else "Inpainting"
            yield (
                _status_html("active", f"{action} image {i + 1} of {int(num_images)}"),
                [], [], str(seed),
            )
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
    """Generate an accessible, animated generation status surface."""
    safe_message = html.escape(str(message))
    if state == "active":
        return (
            '<section class="ffs-generation-stage" aria-live="polite">'
            '<div class="ffs-generation-preview" aria-hidden="true">'
            '<span class="ffs-gen-tile ffs-gen-tile-a"></span>'
            '<span class="ffs-gen-tile ffs-gen-tile-b"></span>'
            '<span class="ffs-gen-tile ffs-gen-tile-c"></span>'
            '<span class="ffs-gen-scan"></span>'
            '</div>'
            '<div class="ffs-generation-copy">'
            '<div class="ffs-kicker"><span class="ffs-live-dot"></span>Creating</div>'
            f'<div class="ffs-generation-title">{safe_message}</div>'
            '<div class="ffs-progress-track"><span></span></div>'
            '<div class="ffs-generation-note">The first model load takes the longest.</div>'
            '</div>'
            '</section>'
        )
    if state == "done":
        return (
            '<div class="ffs-notice ffs-notice-success" aria-live="polite">'
            '<span class="ffs-notice-icon">&#10003;</span>'
            f'<span>{safe_message}</span>'
            '</div>'
        )
    if state == "error":
        return (
            '<div class="ffs-notice ffs-notice-error" role="alert">'
            '<span class="ffs-notice-icon">!</span>'
            f'<span>{safe_message}</span>'
            '</div>'
        )
    return f'<div class="ffs-notice">{safe_message}</div>'


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

# Product shell overrides. These use stable element IDs instead of Gradio's
# generated component numbers so the layout survives Gradio upgrades.
CSS += """
:root {
    --ffs-bg: #f4f5f7;
    --ffs-surface: #ffffff;
    --ffs-surface-2: #eceff2;
    --ffs-text: #15171a;
    --ffs-text-2: #606770;
    --ffs-border: #dfe3e8;
    --ffs-accent: #e65235;
    --ffs-accent-2: #147d72;
    --ffs-accent-bg: rgba(230, 82, 53, 0.08);
    --ffs-success: #147d72;
    --ffs-danger: #c93c35;
    --ffs-max-width: 1040px;
    --ffs-shadow: 0 1px 2px rgba(19, 24, 31, 0.06);
    --ffs-shadow-lg: 0 18px 50px rgba(25, 30, 38, 0.14), 0 2px 8px rgba(25, 30, 38, 0.08);
}

@media (prefers-color-scheme: dark) {
    :root {
        --ffs-bg: #111315;
        --ffs-surface: #1a1d20;
        --ffs-surface-2: #24282c;
        --ffs-text: #f1f3f5;
        --ffs-text-2: #a3aab2;
        --ffs-border: #30353a;
        --ffs-accent: #ff7558;
        --ffs-accent-2: #4ec9b9;
        --ffs-accent-bg: rgba(255, 117, 88, 0.1);
        --ffs-success: #4ec9b9;
        --ffs-danger: #ff736c;
        --ffs-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
        --ffs-shadow-lg: 0 22px 60px rgba(0, 0, 0, 0.42), 0 2px 8px rgba(0, 0, 0, 0.3);
    }
}

.dark {
    --ffs-bg: #111315;
    --ffs-surface: #1a1d20;
    --ffs-surface-2: #24282c;
    --ffs-text: #f1f3f5;
    --ffs-text-2: #a3aab2;
    --ffs-border: #30353a;
    --ffs-accent: #ff7558;
    --ffs-accent-2: #4ec9b9;
    --ffs-accent-bg: rgba(255, 117, 88, 0.1);
    --ffs-success: #4ec9b9;
    --ffs-danger: #ff736c;
}

html { scroll-padding-top: 88px; }
body { background: var(--ffs-bg) !important; }
.gradio-container {
    min-height: 100vh !important;
    color: var(--ffs-text) !important;
    padding-bottom: 190px !important;
}
.gradio-container button,
.gradio-container input,
.gradio-container textarea { letter-spacing: 0 !important; }

/* Header */
#ffs-app-header {
    position: fixed !important;
    inset: 0 0 auto 0;
    z-index: 1200;
    min-height: 72px;
    padding: 10px max(18px, calc((100vw - var(--ffs-max-width)) / 2)) !important;
    align-items: center !important;
    gap: 18px !important;
    background: color-mix(in srgb, var(--ffs-surface) 92%, transparent) !important;
    border: 0 !important;
    border-bottom: 1px solid var(--ffs-border) !important;
    backdrop-filter: blur(18px) saturate(140%);
    box-shadow: 0 1px 0 rgba(255,255,255,0.35);
    flex-wrap: nowrap !important;
}
#ffs-brand { flex: 1 1 auto !important; min-width: 230px !important; padding: 0 !important; }
#ffs-brand .html-container { padding: 0 !important; }
.ffs-brand-lockup { display: flex; align-items: center; gap: 11px; min-height: 48px; }
.ffs-brand-mark {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    flex: 0 0 38px;
    border-radius: 8px;
    background: #15171a;
    color: #ffffff;
    font-size: 12px;
    font-weight: 800;
}
.ffs-wordmark { color: var(--ffs-text); font-size: 18px; font-weight: 800; line-height: 1.05; }
.ffs-wordmark span { color: var(--ffs-accent); }
.ffs-brand-sub { margin-top: 4px; color: var(--ffs-text-2); font-size: 10px; font-weight: 700; text-transform: uppercase; }
#ffs-model-select { min-width: 210px !important; }
#ffs-model-select > div { border-radius: 7px !important; }
#ffs-model-state { min-width: 88px !important; padding: 0 !important; }
#ffs-model-state .html-container { padding: 0 !important; }
#ffs-new-session { min-width: 78px !important; border-radius: 7px !important; }

/* Workspace */
#ffs-workspace {
    width: min(var(--ffs-max-width), calc(100% - 32px)) !important;
    margin: 0 auto !important;
    padding: 104px 0 34px !important;
    min-height: calc(100vh - 160px);
}
#ffs-conversation { padding: 0 !important; }
#ffs-status { min-height: 0; }
.ffs-chat-area { max-width: none !important; min-height: 0 !important; padding: 0 !important; }
.ffs-history { gap: 18px !important; }
.ffs-turn { max-width: 760px; }
.ffs-turn-user { margin-left: auto; }
.ffs-turn-assistant { margin-right: auto; }
.ffs-role { text-transform: uppercase; font-size: 10px !important; letter-spacing: 0 !important; }
.ffs-bubble { border-radius: 8px !important; box-shadow: none !important; padding: 13px 15px !important; }
.ffs-turn-user .ffs-bubble { background: var(--ffs-surface-2) !important; }

/* Empty canvas */
.ffs-empty { min-height: 44vh; display: grid; place-content: center; padding: 70px 20px !important; }
.ffs-empty-icon {
    width: 62px;
    height: 62px;
    margin: 0 auto 22px !important;
    border-radius: 8px;
    background: var(--ffs-surface);
    border: 1px solid var(--ffs-border);
    box-shadow: var(--ffs-shadow);
    color: var(--ffs-accent);
    display: grid;
    place-items: center;
    font-size: 18px !important;
    font-weight: 800;
    opacity: 1 !important;
}
.ffs-empty-title { font-size: 30px !important; font-weight: 800 !important; }
.ffs-empty-sub { margin-top: 10px !important; font-size: 14px !important; }
#ffs-starter-prompts {
    max-width: 720px !important;
    margin: -42px auto 26px !important;
    gap: 8px !important;
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr));
}
#ffs-starter-prompts > * { min-width: 0 !important; width: 100% !important; }
#ffs-starter-prompts button {
    width: 100% !important;
    min-width: 0 !important;
    min-height: 38px !important;
    border-radius: 7px !important;
    border-color: var(--ffs-border) !important;
    background: var(--ffs-surface) !important;
    color: var(--ffs-text-2) !important;
    box-shadow: var(--ffs-shadow);
}
#ffs-starter-prompts button:hover { color: var(--ffs-text) !important; border-color: var(--ffs-accent) !important; }
#ffs-workspace:has(#ffs-conversation .ffs-turn) #ffs-starter-prompts { display: none !important; }

/* Generation motion */
.ffs-generation-stage {
    display: grid;
    grid-template-columns: minmax(260px, 0.9fr) minmax(280px, 1.1fr);
    gap: 34px;
    align-items: center;
    margin: 30px 0 22px;
    padding: 26px;
    background: var(--ffs-surface);
    border: 1px solid var(--ffs-border);
    border-radius: 8px;
    box-shadow: var(--ffs-shadow);
    overflow: hidden;
}
.ffs-generation-preview {
    position: relative;
    height: 220px;
    overflow: hidden;
    border-radius: 7px;
    background: #202326;
}
.ffs-gen-tile { position: absolute; border-radius: 5px; animation: ffs-tile 2.8s ease-in-out infinite; }
.ffs-gen-tile-a { inset: 24px 42% 46% 24px; background: #e65235; }
.ffs-gen-tile-b { inset: 46% 24px 24px 35%; background: #147d72; animation-delay: -0.9s; }
.ffs-gen-tile-c { inset: 34px 25px 50% 62%; background: #e4b84f; animation-delay: -1.8s; }
.ffs-gen-scan {
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg, transparent 35%, rgba(255,255,255,0.26) 50%, transparent 65%);
    transform: translateX(-100%);
    animation: ffs-scan 1.8s linear infinite;
}
@keyframes ffs-tile { 0%,100% { transform: scale(1); opacity: .72; } 50% { transform: scale(1.045); opacity: 1; } }
@keyframes ffs-scan { to { transform: translateX(100%); } }
.ffs-kicker { display: flex; align-items: center; gap: 8px; color: var(--ffs-accent); font-size: 11px; font-weight: 800; text-transform: uppercase; }
.ffs-live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ffs-accent); animation: ffs-pulse 1.1s ease-in-out infinite; }
.ffs-generation-title { margin-top: 12px; color: var(--ffs-text); font-size: 22px; font-weight: 750; line-height: 1.25; }
.ffs-progress-track { height: 4px; margin-top: 22px; overflow: hidden; border-radius: 4px; background: var(--ffs-surface-2); }
.ffs-progress-track span { display: block; width: 42%; height: 100%; background: var(--ffs-accent-2); animation: ffs-progress 1.6s ease-in-out infinite; }
@keyframes ffs-progress { 0% { transform: translateX(-110%); } 100% { transform: translateX(340%); } }
.ffs-generation-note { margin-top: 11px; color: var(--ffs-text-2); font-size: 12px; }
@media (prefers-reduced-motion: reduce) {
    .ffs-gen-tile, .ffs-gen-scan, .ffs-live-dot, .ffs-progress-track span { animation-duration: 5s !important; }
}

.ffs-notice { display: flex; align-items: flex-start; gap: 10px; margin: 18px 0; padding: 12px 14px; border: 1px solid var(--ffs-border); border-radius: 7px; background: var(--ffs-surface); color: var(--ffs-text-2); }
.ffs-notice-icon { display: grid; place-items: center; flex: 0 0 22px; height: 22px; border-radius: 50%; font-size: 12px; font-weight: 800; }
.ffs-notice-success { color: var(--ffs-success); border-color: color-mix(in srgb, var(--ffs-success) 30%, var(--ffs-border)); }
.ffs-notice-success .ffs-notice-icon { background: var(--ffs-success); color: #fff; }
.ffs-notice-error { color: var(--ffs-danger); border-color: color-mix(in srgb, var(--ffs-danger) 35%, var(--ffs-border)); background: color-mix(in srgb, var(--ffs-danger) 5%, var(--ffs-surface)); overflow-wrap: anywhere; }
.ffs-notice-error .ffs-notice-icon { background: var(--ffs-danger); color: #fff; }

/* Results */
#ffs-result-gallery { margin-top: 12px; background: transparent !important; border: 0 !important; }
#ffs-result-gallery .gallery-item { border-radius: 7px !important; border: 1px solid var(--ffs-border) !important; box-shadow: var(--ffs-shadow); overflow: hidden; }
#ffs-result-actions { justify-content: flex-start !important; gap: 8px !important; margin-top: 10px; }
#ffs-result-actions button { min-height: 38px; border-radius: 7px !important; }
#ffs-seed { max-width: 260px; min-height: 38px !important; }
#ffs-seed textarea { min-height: 38px !important; height: 38px !important; padding: 8px 10px !important; }
#ffs-downloads { display: none !important; }

/* Tool shelf */
#ffs-controls-shell { width: min(var(--ffs-max-width), calc(100% - 32px)) !important; margin: 0 auto 26px !important; }
#ffs-controls-shell .ffs-settings { border: 1px solid var(--ffs-border) !important; border-radius: 7px !important; overflow: hidden; background: var(--ffs-surface) !important; }
#ffs-controls-shell .ffs-settings + .ffs-settings { margin-top: 9px; }
#ffs-controls-shell .label-wrap { min-height: 48px; border-radius: 0 !important; font-weight: 700; }
#ffs-controls-shell .ffs-settings-panel { padding: 14px !important; }
#ffs-edit-panel, #ffs-settings-panel {
    width: min(var(--ffs-max-width), calc(100% - 32px)) !important;
    margin: 0 auto 9px !important;
    border: 1px solid var(--ffs-border) !important;
    border-radius: 7px !important;
    overflow: clip !important;
    box-sizing: border-box !important;
    background: var(--ffs-surface) !important;
}
#ffs-edit-panel .label-wrap, #ffs-settings-panel .label-wrap {
    min-height: 48px;
    border-radius: 0 !important;
    font-weight: 700;
}
#ffs-edit-panel .ffs-settings-panel, #ffs-settings-panel .ffs-settings-panel { padding: 14px !important; }
#ffs-edit-panel > div, #ffs-settings-panel > div { max-width: 100% !important; overflow-x: clip !important; }
#ffs-settings-panel > .label-wrap {
    position: sticky;
    top: 0;
    z-index: 2;
    padding: 0 14px !important;
    background: var(--ffs-surface) !important;
    border-bottom: 1px solid var(--ffs-border) !important;
    color: var(--ffs-text) !important;
    font-size: 13px !important;
}
#ffs-settings-panel .form,
#ffs-settings-panel .gradio-row {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
#ffs-settings-panel .form { padding: 0 !important; }
#ffs-settings-panel .gradio-row { gap: 9px !important; }
#ffs-settings-panel .block {
    min-width: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    border-bottom: 1px solid var(--ffs-border) !important;
    border-radius: 0 !important;
    padding: 13px 0 15px !important;
    background: transparent !important;
    box-shadow: none !important;
}
#ffs-settings-panel .block:last-child { border-bottom: 0 !important; }
#ffs-settings-panel label > span,
#ffs-settings-panel .block > label > span {
    color: var(--ffs-text-2) !important;
    font-size: 11px !important;
    font-weight: 750 !important;
    text-transform: uppercase;
}
#ffs-settings-panel input[type="range"] {
    --ffs-range-progress: 0%;
    width: 100% !important;
    min-height: 30px !important;
    margin: 3px 0 0 !important;
    padding: 0 !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    background: transparent !important;
    cursor: pointer;
}
#ffs-settings-panel input[type="range"]::-webkit-slider-runnable-track {
    height: 5px;
    border-radius: 5px;
    background: linear-gradient(to right,
        var(--ffs-accent) 0,
        var(--ffs-accent) var(--ffs-range-progress),
        var(--ffs-surface-2) var(--ffs-range-progress),
        var(--ffs-surface-2) 100%);
}
#ffs-settings-panel input[type="range"]::-webkit-slider-thumb {
    width: 18px;
    height: 18px;
    margin-top: -6.5px;
    appearance: none;
    -webkit-appearance: none;
    border: 3px solid var(--ffs-surface);
    border-radius: 50%;
    background: var(--ffs-accent);
    box-shadow: 0 0 0 1px var(--ffs-accent), 0 2px 5px rgba(20, 24, 29, .18);
}
#ffs-settings-panel input[type="range"]::-moz-range-track {
    height: 5px;
    border-radius: 5px;
    background: var(--ffs-surface-2);
}
#ffs-settings-panel input[type="range"]::-moz-range-progress {
    height: 5px;
    border-radius: 5px;
    background: var(--ffs-accent);
}
#ffs-settings-panel input[type="range"]::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border: 3px solid var(--ffs-surface);
    border-radius: 50%;
    background: var(--ffs-accent);
    box-shadow: 0 0 0 1px var(--ffs-accent), 0 2px 5px rgba(20, 24, 29, .18);
}
#ffs-settings-panel input[type="range"]:focus-visible::-webkit-slider-thumb {
    box-shadow: 0 0 0 4px var(--ffs-accent-bg), 0 0 0 1px var(--ffs-accent);
}
#ffs-settings-panel input[type="number"] { min-height: 42px !important; border-radius: 6px !important; }
#ffs-settings-panel textarea { min-height: 82px !important; border-radius: 6px !important; line-height: 1.45 !important; }
#ffs-settings-panel input[type="number"]:focus,
#ffs-settings-panel textarea:focus { border-color: var(--ffs-accent) !important; box-shadow: 0 0 0 3px var(--ffs-accent-bg) !important; }
#ffs-settings-panel button[aria-label="Reset to default value"] { min-width: 38px !important; min-height: 38px !important; }
#ffs-settings-panel::-webkit-scrollbar { width: 7px; }
#ffs-settings-panel::-webkit-scrollbar-track { background: transparent; }
#ffs-settings-panel::-webkit-scrollbar-thumb { background: var(--ffs-border); border-radius: 7px; }

/* Floating composer */
#ffs-composer-dock {
    position: fixed !important;
    z-index: 1100;
    inset: auto 0 0 0;
    margin: 0 !important;
    padding: 12px 18px max(16px, env(safe-area-inset-bottom)) !important;
    background: linear-gradient(to bottom, transparent, var(--ffs-bg) 28%) !important;
    border: 0 !important;
}
#ffs-composer-inner { width: min(920px, 100%) !important; margin: 0 auto !important; gap: 8px !important; }
#ffs-attachment-row {
    width: 100% !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 8px 10px !important;
    border: 1px solid var(--ffs-border) !important;
    border-radius: 7px !important;
    background: var(--ffs-surface) !important;
    box-shadow: var(--ffs-shadow);
}
#ffs-attachment-preview { height: 64px !important; min-height: 64px !important; max-width: 120px !important; border: 0 !important; }
#ffs-remove-attachment { min-width: 78px !important; border-radius: 7px !important; }
#ffs-composer {
    align-items: flex-end !important;
    gap: 8px !important;
    padding: 8px !important;
    border: 1px solid color-mix(in srgb, var(--ffs-border) 75%, var(--ffs-text)) !important;
    border-radius: 8px !important;
    background: var(--ffs-surface) !important;
    box-shadow: var(--ffs-shadow-lg);
    transition: border-color .18s ease, box-shadow .18s ease;
}
#ffs-composer:focus-within { border-color: var(--ffs-accent) !important; box-shadow: 0 18px 50px rgba(25,30,38,.16), 0 0 0 3px var(--ffs-accent-bg); }
#ffs-prompt { min-height: 46px !important; }
#ffs-prompt textarea { min-height: 46px !important; padding: 12px 10px !important; line-height: 22px !important; background: transparent !important; color: var(--ffs-text) !important; }
#ffs-attach, #ffs-generate { min-height: 46px !important; height: 46px !important; border-radius: 7px !important; }
#ffs-attach { min-width: 46px !important; width: 46px !important; font-size: 20px !important; }
#ffs-generate { min-width: 104px !important; background: var(--ffs-accent) !important; color: #fff !important; border-color: var(--ffs-accent) !important; font-weight: 750 !important; }
#ffs-generate:hover { filter: brightness(.96); transform: none !important; }

@media (max-width: 720px) {
    .gradio-container { padding-bottom: 170px !important; }
    #ffs-app-header { min-height: 64px; padding: 8px 12px !important; gap: 8px !important; }
    #ffs-brand { min-width: 46px !important; flex: 0 0 46px !important; }
    .ffs-wordmark, .ffs-brand-sub { display: none; }
    #ffs-model-select { min-width: 150px !important; }
    #ffs-model-state { display: none !important; }
    #ffs-new-session { min-width: 58px !important; }
    #ffs-workspace { width: calc(100% - 20px) !important; padding-top: 18px !important; }
    .ffs-empty { min-height: 42vh; padding: 40px 12px !important; }
    .ffs-empty-title {
        width: 100%;
        max-width: 100%;
        font-size: 24px !important;
        line-height: 1.15 !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
    }
    .ffs-empty-sub {
        width: 100%;
        max-width: 100% !important;
        line-height: 1.45 !important;
        white-space: normal !important;
    }
    .ffs-generation-stage { grid-template-columns: 1fr; gap: 20px; padding: 14px; }
    .ffs-generation-preview { height: 160px; }
    #ffs-controls-shell { width: calc(100% - 20px) !important; }
    #ffs-edit-panel, #ffs-settings-panel { width: calc(100% - 20px) !important; }
    #ffs-composer-dock { padding: 8px 10px max(10px, env(safe-area-inset-bottom)) !important; }
    #ffs-generate { min-width: 48px !important; width: 48px !important; }
    #ffs-generate, #ffs-generate * { font-size: 0 !important; }
    #ffs-generate::after { content: '↑'; font-size: 22px; line-height: 1; }
    #ffs-result-actions { flex-direction: column !important; }
    #ffs-starter-prompts { margin-top: -18px !important; display: grid !important; grid-template-columns: 1fr 1fr; }
}

@media (max-width: 420px) {
    .ffs-empty-title { font-size: 22px !important; }
    .ffs-empty-sub { font-size: 13px !important; }
}

/* Desktop creation-tool layout, following the established image-AI pattern:
   persistent controls on the left, creation feed on the right. */
@media (min-width: 901px) {
    .gradio-container { padding-left: 0 !important; }
    #ffs-app-header {
        padding-left: 22px !important;
        padding-right: 22px !important;
    }
    #ffs-brand { min-width: 236px !important; flex: 0 0 236px !important; }
    #ffs-workspace {
        width: min(var(--ffs-max-width), calc(100% - 322px)) !important;
        margin-left: max(300px, calc((100% - var(--ffs-max-width) + 280px) / 2)) !important;
        margin-right: 22px !important;
        padding-top: 104px !important;
    }
    #ffs-settings-panel {
        position: fixed !important;
        z-index: 900;
        left: 14px;
        top: 86px;
        bottom: 18px;
        width: 252px !important;
        margin: 0 !important;
        overflow-y: auto !important;
        overflow-x: clip !important;
        border-color: var(--ffs-border) !important;
        box-shadow: var(--ffs-shadow);
    }
    #ffs-settings-panel > .wrap,
    #ffs-settings-panel > div { overflow-x: clip !important; }
    #ffs-edit-panel {
        width: min(var(--ffs-max-width), calc(100% - 322px)) !important;
        margin-left: max(300px, calc((100% - var(--ffs-max-width) + 280px) / 2)) !important;
        margin-right: 22px !important;
    }
    #ffs-composer-dock { left: 280px !important; }
    #ffs-composer-dock { width: auto !important; }
}

.ffs-canvas-heading {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 0 0 18px;
    color: var(--ffs-text);
    font-size: 20px;
    font-weight: 800;
}
.ffs-canvas-heading i { height: 1px; flex: 1; background: var(--ffs-border); }
#ffs-canvas-heading, #ffs-canvas-heading .html-container { padding: 0 !important; width: 100% !important; }

@media (max-width: 900px) {
    .gradio-container { padding-left: 0 !important; }
    #ffs-workspace { margin-left: auto !important; margin-right: auto !important; }
    #ffs-settings-panel {
        position: static !important;
        width: calc(100% - 20px) !important;
        margin: 82px auto 0 !important;
        max-height: min(58vh, 520px);
        overflow-y: auto !important;
    }
    #ffs-composer-dock { left: 0 !important; }
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

function syncRangeFill(slider) {
    var min = parseFloat(slider.min || '0');
    var max = parseFloat(slider.max || '100');
    var value = parseFloat(slider.value || String(min));
    var progress = max === min ? 0 : ((value - min) / (max - min)) * 100;
    slider.style.setProperty('--ffs-range-progress', Math.max(0, Math.min(100, progress)) + '%');
}

function prepareGenerationSliders() {
    document.querySelectorAll('#ffs-settings-panel input[type="range"]').forEach(function(slider) {
        syncRangeFill(slider);
        if (!slider.dataset.ffsRangeReady) {
            slider.dataset.ffsRangeReady = 'true';
            slider.addEventListener('input', function() { syncRangeFill(slider); });
            slider.addEventListener('change', function() { syncRangeFill(slider); });
        }
    });
}

function enhanceStudioUI() {
    var attach = document.querySelector('#ffs-attach button, #ffs-attach');
    var generate = document.querySelector('#ffs-generate button, #ffs-generate');
    var fresh = document.querySelector('#ffs-new-session button, #ffs-new-session');
    if (attach) attach.setAttribute('title', 'Attach image');
    if (generate) generate.setAttribute('title', 'Generate image');
    if (fresh) fresh.setAttribute('title', 'Start a new session');

    prepareGenerationSliders();

    var settings = document.querySelector('#ffs-settings-panel');
    if (settings && window.innerWidth <= 900 && !settings.dataset.ffsMobileCollapsed) {
        var settingsToggle = settings.querySelector('button.label-wrap, .label-wrap button');
        if (settingsToggle) {
            settings.dataset.ffsMobileCollapsed = 'true';
            settingsToggle.click();
        }
    }

    var status = document.querySelector('#ffs-status');
    if (status && !status.dataset.ffsObserved) {
        status.dataset.ffsObserved = 'true';
        new MutationObserver(function() {
            var active = status.querySelector('.ffs-generation-stage');
            if (active) active.scrollIntoView({behavior: 'smooth', block: 'center'});
        }).observe(status, {childList: true, subtree: true});
    }
}
setInterval(enhanceStudioUI, 1200);
setTimeout(enhanceStudioUI, 500);
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


def _ui_trace(message):
    if DEBUG_MODE:
        print(f"[ui] {message}", flush=True)


_ui_trace("building interface")
with gr.Blocks(title="FreeFakeStudio") as demo:

    # ── Session State ──────────────────────────────────────
    chat_history = gr.State([])           # list of {role, content, ...}
    attached_image = gr.State(None)       # PIL Image or None
    last_gen_settings = gr.State(None)    # for Regenerate
    selected_result_idx = gr.State(0)

    # ═══════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════
    with gr.Row(elem_id="ffs-app-header"):
        gr.HTML("""
            <div class="ffs-brand-lockup">
                <span class="ffs-brand-mark">FF</span>
                <span>
                    <span class="ffs-wordmark">FreeFake<span>Studio</span></span>
                    <span class="ffs-brand-sub">AI image workspace</span>
                </span>
            </div>
        """, elem_id="ffs-brand")
        model_selector = gr.Dropdown(
            choices=model_manager.MODEL_NAMES,
            value="Z-Image Turbo",
            show_label=False,
            container=False,
            scale=0,
            min_width=210,
            elem_id="ffs-model-select",
        )
        model_status_display = gr.HTML(
            '<span class="ffs-model-badge">Available</span>',
            scale=0,
            min_width=92,
            elem_id="ffs-model-state",
        )
        new_chat_btn = gr.Button(
            "New", size="sm", variant="secondary", scale=0,
            min_width=78, elem_id="ffs-new-session",
        )

    # ═══════════════════════════════════════════════════════
    # MAIN CONTENT AREA
    # ═══════════════════════════════════════════════════════
    with gr.Accordion(
        "Generation controls",
        open=True,
        elem_classes="ffs-settings",
        elem_id="ffs-settings-panel",
    ) as settings_panel:
        with gr.Column(elem_classes="ffs-settings-panel"):
            aspect_ratio = gr.Dropdown(
                ASPECTS,
                value="1024x1024 (1:1)",
                label="Aspect Ratio",
            )
            num_images = gr.Slider(1, 8, value=1, step=1, label="Images")
            gen_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
            gen_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
            gen_cfg = gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="CFG")
            gen_denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
            negative_prompt = gr.Textbox(
                DEFAULT_NEG,
                label="Negative Prompt",
                lines=2,
            )

    with gr.Column(elem_classes="ffs-chat-area", elem_id="ffs-workspace"):

        gr.HTML(
            '<div class="ffs-canvas-heading"><span>Create</span><i></i></div>',
            elem_id="ffs-canvas-heading",
        )

        conversation_display = gr.HTML(
            value='<div class="ffs-history"></div>',
            elem_classes="ffs-history",
            elem_id="ffs-conversation",
        )

        # Status display (streaming updates)
        status_display = gr.HTML(
            value="""
            <div class="ffs-empty">
                <div class="ffs-empty-icon">FF</div>
                <div class="ffs-empty-title">What will you make?</div>
                <div class="ffs-empty-sub">Portrait. Product. Editorial. Concept.</div>
            </div>
            """,
            elem_id="ffs-status",
        )

        with gr.Row(elem_id="ffs-starter-prompts"):
            starter_portrait = gr.Button("Cinematic portrait", size="sm")
            starter_product = gr.Button("Product campaign", size="sm")
            starter_editorial = gr.Button("Editorial fashion", size="sm")
            starter_concept = gr.Button("Surreal concept", size="sm")

        # Results gallery
        _gallery_kwargs = dict(
            show_label=False,
            columns=2,
            height="auto",
            object_fit="contain",
            preview=False,
            allow_preview=True,
            buttons=["download", "download_all", "fullscreen"],
            visible=False,
            elem_classes="ffs-result-gallery",
            elem_id="ffs-result-gallery",
        )
        result_gallery = gr.Gallery(**_gallery_kwargs)

        # Download all
        result_files = gr.File(
            label="Download All",
            file_count="multiple",
            visible=False,
            elem_id="ffs-downloads",
        )

        # ── Action Buttons Row ─────────────────────────────
        with gr.Row(visible=False, elem_id="ffs-result-actions") as action_row:
            add_to_prompt_btn = gr.Button("Use as input", size="sm")
            regenerate_btn = gr.Button("Regenerate", size="sm")
            seed_display = gr.Textbox(
                interactive=False, visible=False, show_label=False,
                container=False, elem_id="ffs-seed",
            )

    # ═══════════════════════════════════════════════════════
    # EDITING PANEL (for mask/inpaint)
    # ═══════════════════════════════════════════════════════
    with gr.Accordion("Mask and edit", open=False, visible=False,
                       elem_classes="ffs-settings", elem_id="ffs-edit-panel") as edit_panel:
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
                edit_mask_btn = gr.Button("Edit mask manually", variant="secondary")

    # ═══════════════════════════════════════════════════════
    # SETTINGS PANEL
    # ═══════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════
    # COMPOSER (bottom bar)
    # ═══════════════════════════════════════════════════════
    with gr.Row(elem_id="ffs-composer-dock"):
        with gr.Column(elem_id="ffs-composer-inner"):
            with gr.Row(visible=False, elem_id="ffs-attachment-row") as attachment_row:
                attachment_display = gr.Image(
                    show_label=False,
                    type="pil",
                    height=64,
                    interactive=False,
                    elem_id="ffs-attachment-preview",
                )
                remove_attachment_btn = gr.Button(
                    "Remove", size="sm", variant="secondary", elem_id="ffs-remove-attachment"
                )
            with gr.Row(elem_id="ffs-composer"):
                attach_btn = gr.UploadButton(
                    "+",
                    file_types=["image"],
                    size="sm",
                    min_width=40,
                    elem_id="ffs-attach",
                )
                prompt_input = gr.Textbox(
                    placeholder="Describe what you want to create or edit...",
                    show_label=False,
                    lines=1,
                    max_lines=4,
                    scale=10,
                    container=False,
                    elem_id="ffs-prompt",
                )
                send_btn = gr.Button(
                    "Generate",
                    variant="primary",
                    size="sm",
                    min_width=50,
                    elem_id="ffs-generate",
                )

    # ═══════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════

    # ── Attachment handling ─────────────────────────────────
    starter_examples = {
        starter_portrait: "A cinematic portrait with soft window light, natural skin texture, 85mm photography",
        starter_product: "A premium product campaign photograph, precise studio lighting, bold art direction",
        starter_editorial: "An editorial fashion photograph, sculptural styling, dramatic location, magazine quality",
        starter_concept: "A surreal architectural landscape at blue hour, dreamlike scale, intricate detail",
    }
    for starter_button, starter_prompt in starter_examples.items():
        starter_button.click(
            lambda value=starter_prompt: value,
            outputs=[prompt_input],
            show_progress="hidden",
        )

    def handle_upload(file):
        if file is None:
            return gr.update(visible=False), gr.update(value=None), None, gr.update(visible=False)
        img = Image.open(file).convert("RGB")
        return (
            gr.update(visible=True), gr.update(value=img), img,
            gr.update(visible=True),
        )

    attach_btn.upload(
        handle_upload,
        inputs=[attach_btn],
        outputs=[attachment_row, attachment_display, attached_image, edit_panel],
    )

    def clear_attachment():
        return (
            gr.update(visible=False), gr.update(value=None), None,
            gr.update(visible=False), "None", None, None,
        )

    remove_attachment_btn.click(
        clear_attachment,
        outputs=[
            attachment_row, attachment_display, attached_image,
            edit_panel, mask_mode, mask_editor, mask_preview,
        ],
    )

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
                gr.update(),
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
                    gr.update(value=""),
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
                    gr.update(value=""),
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
            conversation_display, chat_history, prompt_input,
        ],
        show_progress="hidden",
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
            conversation_display, chat_history, prompt_input,
        ],
        show_progress="hidden",
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
            gr.update(visible=True),                  # attachment_row
            gr.update(value=img_data),                # attachment_display
            img_data,                                  # attached_image
            gr.update(visible=True),                  # edit_panel
            gr.update(value=""),                       # clear prompt
        )

    add_to_prompt_btn.click(
        on_add_to_prompt,
        inputs=[result_gallery, selected_result_idx],
        outputs=[attachment_row, attachment_display, attached_image, edit_panel, prompt_input],
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
                <div class="ffs-empty-icon">FF</div>
                <div class="ffs-empty-title">What will you make?</div>
                <div class="ffs-empty-sub">Portrait. Product. Editorial. Concept.</div>
            </div>""",
            gr.update(visible=False, value=None),   # result_gallery
            gr.update(visible=False, value=None),   # result_files
            gr.update(visible=False, value=""),      # seed_display
            gr.update(visible=False),                # action_row
            gr.update(visible=False),                # attachment_row
            gr.update(value=None),                   # attachment_display
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
            attachment_row, attachment_display, attached_image, prompt_input,
            last_gen_settings, edit_panel, mask_mode,
            conversation_display, chat_history,
        ],
    )

_ui_trace("interface ready")


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
