# ============================================================
#  FreeFakeStudio — Gemini-like AI Image Studio
#  Conversational image generation & editing interface
#  Models: Z-Image Turbo · FLUX.2-klein 4B · ERNIE-Image-Turbo
#  Built for Google Colab T4 (15GB VRAM)
# ============================================================

import os, random, time, sys, gc, re, uuid, json, base64, traceback, html
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO
from datetime import datetime
from urllib.parse import quote

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

import gradio as gr

# ── Save directory ─────────────────────────────────────────
import workspace as ws
import avatar_gallery
import avatar_gallery_jobs
import avatar_studio
import avatar_guide
import avatar_design
SAVE_DIR = ws.get_save_dir()
CHAT_HISTORY_PATH = os.path.join(SAVE_DIR, "_sessions", "current_chat.json")

def get_save_path(prefix="img"):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)[:20]
    uid = uuid.uuid4().hex[:6]
    return os.path.join(SAVE_DIR, f"{safe}_{uid}.png")


def _load_chat_history():
    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as history_file:
            payload = json.load(history_file)
        history = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
            raise ValueError("Saved chat history has an invalid format.")
        return history[-200:]
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"[history] Could not restore saved chat: {exc}", flush=True)
        return []


def _save_chat_history(history):
    history = [item for item in (history or []) if isinstance(item, str)][-200:]
    session_dir = os.path.dirname(CHAT_HISTORY_PATH)
    os.makedirs(session_dir, exist_ok=True)
    temporary_path = CHAT_HISTORY_PATH + f".{uuid.uuid4().hex}.tmp"
    payload = {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "history": history,
    }
    try:
        with open(temporary_path, "w", encoding="utf-8") as history_file:
            json.dump(payload, history_file, ensure_ascii=False)
        os.replace(temporary_path, CHAT_HISTORY_PATH)
    except Exception as exc:
        try:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        except OSError:
            pass
        print(f"[history] Could not save chat: {exc}", flush=True)
    return history


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
    h, w = img_np.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    faces = []
    try:
        import cv2

        if not all(hasattr(cv2, name) for name in ("CascadeClassifier", "cvtColor", "ellipse")):
            raise ImportError("OpenCV face detection is unavailable")
        cascade_root = getattr(getattr(cv2, "data", None), "haarcascades", "")
        cascade_path = os.path.join(cascade_root, "haarcascade_frontalface_default.xml")
        if not cascade_root or not os.path.isfile(cascade_path):
            raise ImportError("OpenCV Haar cascade data is unavailable")
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise RuntimeError("OpenCV could not load its face cascade")
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
    except (AttributeError, ImportError, RuntimeError) as exc:
        print(f"[mask] OpenCV face detector unavailable; using geometric fallback: {exc}")

    if len(faces) > 0:
        import cv2

        for (fx, fy, fw, fh) in faces:
            pad_w, pad_h = int(fw * 0.3), int(fh * 0.3)
            x1, y1 = max(0, fx - pad_w), max(0, fy - pad_h)
            x2, y2 = min(w, fx + fw + pad_w), min(h, fy + fh + pad_h)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = (x2 - x1) // 2, (y2 - y1) // 2
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 0, -1)
    else:
        # Estimate the face from the upper portion of the segmented foreground.
        # This keeps image editing functional even when Colab's cv2 package is partial.
        foreground = auto_mask_background(image_pil) < 127
        ys, xs = np.where(foreground)
        if len(xs) and len(ys):
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            fw, fh = max(1, x2 - x1), max(1, y2 - y1)
            cx = x1 + fw // 2
            cy = y1 + int(fh * 0.20)
            rx = max(20, int(fw * 0.18))
            ry = max(24, int(fh * 0.16))
            pil_mask = Image.fromarray(mask)
            ImageDraw.Draw(pil_mask).ellipse(
                (cx - rx, cy - ry, cx + rx, cy + ry), fill=0
            )
            mask = np.array(pil_mask)
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
    clothing_keywords = r'\b(clothes|clothing|outfit|dress|shirt|jacket|pants|trousers|body|garment|wearing)\b'
    if re.search(clothing_keywords, p):
        except_face = auto_mask_except_face(image_pil)
        background = auto_mask_background(image_pil)
        clothing_mask = np.where(
            (except_face > 127) & (background < 127), 255, 0
        ).astype(np.uint8)
        return clothing_mask, prompt, 0.75

    # Ordinary attached-image requests use native img2img. Masking is only
    # introduced when the prompt or the explicit mask control asks for it.
    return None, prompt, 1.0


# ═══════════════════════════════════════════════════════════
#  GENERATION CORE — yields status updates for streaming
# ═══════════════════════════════════════════════════════════
def do_generate(model_name, prompt, negative, aspect_ratio,
                seed, cfg, denoise, num_images, steps,
                input_image=None, mask_mode=None, editor_data=None,
                canvas_index=0):
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
    input_images = _as_image_list(input_image)
    canvas_idx = canvas_index if canvas_index is not None and input_images else None
    if canvas_idx is not None:
        if not isinstance(canvas_idx, int) or not 0 <= canvas_idx < len(input_images):
            yield (_status_html("error", "The selected canvas is no longer available. Select it again."),
                   [], [], str(seed))
            return
        # The FLUX engine uses the first input as its editing base.
        input_images = [input_images[canvas_idx]] + [
            image for index, image in enumerate(input_images) if index != canvas_idx
        ]
    primary_image = input_images[0] if canvas_idx is not None else None

    # Determine mode
    if input_images:
        if canvas_idx is None:
            mode = "ref_generate"  # references only, no canvas
        elif mask_mode and mask_mode != "None":
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
    elif mode == "ref_generate":
        ref_label = f"{len(input_images)} reference image" + ("s" if len(input_images) != 1 else "")
        yield (_status_html("active", f"Generating from {ref_label} with {model_name}"), [], [], str(seed))
    elif mode == "img2img":
        ref_label = f"{len(input_images)} reference image" + ("s" if len(input_images) != 1 else "")
        yield (_status_html("active", f"Editing with {ref_label} in {model_name}"), [], [], str(seed))
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
            elif mode == "ref_generate":
                # Reference-only: clean canvas + reference conditioning
                img = engine.generate_with_references(
                    input_images, prompt, negative, w, h,
                    seed + i, cfg, denoise, int(steps))
            elif mode == "img2img":
                # For FLUX models, use smart mask selection
                if model_name == "FLUX.2-klein 4B":
                    mask, img_prompt, effective_denoise = _select_mask_for_prompt(prompt, primary_image)
                    img = engine.img2img(input_images, img_prompt, negative,
                                         seed + i, cfg, effective_denoise, int(steps), mask=mask)
                else:
                    img = engine.img2img(primary_image, prompt, negative,
                                         seed + i, cfg, denoise, int(steps))
            elif mode == "inpaint":
                mask_combined = _resolve_mask(primary_image, mask_mode, editor_data)
                if mask_combined is None:
                    yield (_status_html("error", "No mask detected. Paint or select a mask mode."),
                           [], [], str(seed))
                    return
                if model_name == "FLUX.2-klein 4B":
                    img = engine.inpaint(
                        primary_image, mask_combined, prompt, negative,
                        seed + i, cfg, denoise, int(steps), references=input_images,
                    )
                else:
                    img = engine.inpaint(primary_image, mask_combined, prompt, negative,
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


def _image_thumbnail_data_uri(image, max_size=220):
    if image is None:
        return ""
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        preview = image.convert("RGB").copy()
        preview.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        preview.save(buffer, format="JPEG", quality=78, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as exc:
        _ui_trace(f"could not render attachment thumbnail: {exc}")
        return ""


def _request_html(prompt, model_name, input_image, mask_mode):
    prompt = html.escape(prompt or "(image edit)")
    input_images = _as_image_list(input_image)
    has_image = bool(input_images)
    mode = "image edit" if has_image else "text to image"
    explicit_mask = mask_mode not in (None, "None", "Smart")
    edit_method = mask_mode if explicit_mask else "Smart edit" if has_image else ""
    mask_note = f" / {html.escape(edit_method)}" if edit_method else ""
    attachment_items = []
    for index, image in enumerate(input_images):
        attachment_src = _image_thumbnail_data_uri(image, max_size=640)
        if attachment_src:
            role = "Canvas" if index == 0 else "Reference"
            attachment_items.append(
                '<div class="ffs-request-attachment">'
                f'<img src="{attachment_src}" alt="Attached image {index + 1}">'
                f'<span><strong>Image {index + 1}</strong><small>{role}</small></span>'
                '</div>'
            )
    attachment_html = (
        f'<div class="ffs-request-attachments">{"".join(attachment_items)}</div>'
        if attachment_items else ""
    )
    return (
        '<div class="ffs-turn ffs-turn-user">'
        '<div class="ffs-role">You</div>'
        f'<div class="ffs-bubble">{prompt}</div>'
        f'{attachment_html}'
        f'<div class="ffs-meta">{html.escape(model_name)} / {mode}{mask_note}</div>'
        '</div>'
    )


def _assistant_html(paths, seed_str):
    count = len(paths or [])
    images = []
    for index, path in enumerate(paths or []):
        absolute = os.path.abspath(path)
        src = "/gradio_api/file=" + quote(absolute, safe="/:")
        images.append(
            '<a class="ffs-chat-image" href="{src}" target="_blank" '
            'rel="noopener" aria-label="Open generated image {number}">'
            '<img src="{src}" alt="Generated image {number}" loading="lazy">'
            '</a>'.format(src=html.escape(src, quote=True), number=index + 1)
        )
    return (
        '<div class="ffs-turn ffs-turn-assistant">'
        '<div class="ffs-role">FreeFakeStudio</div>'
        f'<div class="ffs-chat-images count-{min(count, 4)}">{"".join(images)}</div>'
        f'<div class="ffs-assistant-meta">Seed {html.escape(str(seed_str))}</div>'
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
MAX_FLUX_REFERENCES = 4


def _as_image_list(value):
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [item for item in values if item is not None]


def _primary_image(value):
    images = _as_image_list(value)
    return images[0] if images else None


def _attachment_status_html(count, canvas_index=0):
    if count <= 0:
        return ""
    noun = "image" if count == 1 else "images"
    canvas_data = -1 if canvas_index is None else int(canvas_index)
    if canvas_index is None:
        references = f"{count} references (no canvas) / FLUX reference-only"
    elif count == 1:
        references = "Canvas only"
    else:
        references = f"Image {canvas_index + 1} canvas / {count - 1} additional references"
    return (
        f'<div class="ffs-attachment-copy" data-canvas-index="{canvas_data}" data-image-count="{count}">'
        f'<strong>{count} {noun} attached</strong>'
        f'<span>{references} / FLUX edit</span>'
        '<small>Refer to them as image 1, image 2, and so on in your prompt.</small>'
        '</div>'
    )


def _append_attachment_images(existing, additions):
    images = _as_image_list(existing)
    additions = _as_image_list(additions)
    if len(images) + len(additions) > MAX_FLUX_REFERENCES:
        raise ValueError(
            f"FLUX.2 Klein accepts up to {MAX_FLUX_REFERENCES} reference images. "
            f"You already have {len(images)} and selected {len(additions)} more."
        )
    return images + additions


def _apply_attachment_action(images, index, action, canvas_index=0):
    images = _as_image_list(images)
    if not images:
        return [], None
    index = max(0, min(int(index or 0), len(images) - 1))
    if canvas_index is not None:
        canvas_index = max(0, min(int(canvas_index), len(images) - 1))
    if action == "canvas":
        # Toggle: click canvas on current canvas → un-canvas; on another → make it canvas
        if index == canvas_index:
            canvas_index = None  # un-canvas (all become references)
        else:
            canvas_index = index  # make this one the canvas
    elif action == "left" and index > 0:
        # Adjust canvas_index if the swap affects it
        if canvas_index == index:
            canvas_index = index - 1
        elif canvas_index == index - 1:
            canvas_index = index
        images[index - 1], images[index] = images[index], images[index - 1]
    elif action == "right" and index < len(images) - 1:
        if canvas_index == index:
            canvas_index = index + 1
        elif canvas_index == index + 1:
            canvas_index = index
        images[index + 1], images[index] = images[index], images[index + 1]
    elif action == "remove":
        if canvas_index is not None:
            if index == canvas_index:
                canvas_index = None
            elif index < canvas_index:
                canvas_index -= 1
        images.pop(index)
    if not images:
        canvas_index = None
    return images, canvas_index


def _gallery_item_to_pil(item):
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (list, tuple)) and item:
        item = item[0]
    if isinstance(item, dict):
        nested = item.get("image")
        item = (
            item.get("path")
            or (nested.get("path") if isinstance(nested, dict) else None)
            or nested
        )
    path = getattr(item, "path", None) or item
    if isinstance(path, Image.Image):
        return path.convert("RGB")
    if isinstance(path, (str, os.PathLike)):
        return Image.open(path).convert("RGB")
    raise ValueError("The selected generated image could not be read.")


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
    --ffs-header-h: 84px;
    --ffs-page-top: 116px;
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
    overflow: visible !important;
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
    padding: var(--ffs-page-top) 0 34px !important;
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
#ffs-settings-panel {
    width: min(var(--ffs-max-width), calc(100% - 32px)) !important;
    margin: 0 auto 9px !important;
    border: 1px solid var(--ffs-border) !important;
    border-radius: 7px !important;
    overflow: clip !important;
    box-sizing: border-box !important;
    background: var(--ffs-surface) !important;
}
#ffs-settings-panel .label-wrap {
    min-height: 48px;
    border-radius: 0 !important;
    font-weight: 700;
}
#ffs-settings-panel .ffs-settings-panel { padding: 14px !important; }
#ffs-settings-panel > div { max-width: 100% !important; overflow-x: clip !important; }
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
#ffs-attachment-preview { height: 88px !important; min-height: 88px !important; max-width: 320px !important; border: 0 !important; }
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
    :root { --ffs-header-h: 118px; --ffs-page-top: 140px; }
    .gradio-container { padding-bottom: 170px !important; }
    #ffs-app-header {
        min-height: var(--ffs-header-h);
        padding: 8px 12px !important;
        gap: 8px !important;
        align-content: center !important;
        flex-wrap: wrap !important;
    }
    #ffs-brand { min-width: 46px !important; flex: 0 0 46px !important; }
    .ffs-wordmark, .ffs-brand-sub { display: none; }
    #ffs-model-select { min-width: 150px !important; flex: 1 1 calc(100% - 166px) !important; }
    #ffs-model-state { display: none !important; }
    #ffs-new-session { min-width: 58px !important; }
    #ffs-workspace { width: calc(100% - 20px) !important; padding-top: var(--ffs-page-top) !important; }
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
    #ffs-settings-panel { width: calc(100% - 20px) !important; }
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
        padding-top: var(--ffs-page-top) !important;
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

/* Final studio polish layer. Kept ID-scoped so Gradio internals can change
   without turning the interface into a selector lottery. */
:root {
    --ffs-ink: #111418;
    --ffs-muted: #747b84;
    --ffs-line-strong: #cfd5dc;
    --ffs-coral: #ee5b3f;
    --ffs-coral-dark: #d94a30;
    --ffs-teal: #138477;
    --ffs-gold: #d7a437;
    --ffs-elevated: 0 10px 30px rgba(20, 27, 36, .07), 0 2px 8px rgba(20, 27, 36, .05);
}

/* App chrome */
.gradio-container {
    background:
        linear-gradient(rgba(255,255,255,.46), rgba(255,255,255,.46)),
        var(--ffs-bg) !important;
}
.dark .gradio-container {
    background: var(--ffs-bg) !important;
}
#ffs-app-header::before {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, var(--ffs-coral) 0 34%, var(--ffs-gold) 34% 52%, var(--ffs-teal) 52% 100%);
}
#ffs-app-header {
    box-shadow: 0 8px 24px rgba(24, 31, 40, .045) !important;
}
.ffs-brand-mark {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.1), 0 5px 12px rgba(17,20,24,.16);
}
.ffs-brand-mark::after {
    content: '';
    position: absolute;
    z-index: -1;
    right: -7px;
    bottom: -8px;
    width: 25px;
    height: 25px;
    background: var(--ffs-coral);
    transform: rotate(28deg);
}
.ffs-wordmark { letter-spacing: 0 !important; }
.ffs-brand-sub { color: var(--ffs-muted) !important; letter-spacing: .06em !important; }
#ffs-model-select > div {
    min-height: 42px !important;
    border: 1px solid var(--ffs-border) !important;
    background: var(--ffs-surface-2) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.7) !important;
    transition: border-color .18s ease, background-color .18s ease, box-shadow .18s ease;
}
#ffs-model-select > div:hover,
#ffs-model-select > div:focus-within {
    border-color: var(--ffs-line-strong) !important;
    background: var(--ffs-surface) !important;
    box-shadow: 0 0 0 3px var(--ffs-accent-bg) !important;
}
.ffs-model-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    min-height: 28px;
    padding: 5px 11px;
    border: 1px solid color-mix(in srgb, var(--ffs-success) 22%, var(--ffs-border));
    border-radius: 999px;
    background: color-mix(in srgb, var(--ffs-success) 7%, var(--ffs-surface));
    color: var(--ffs-success);
    font-size: 11px;
    font-weight: 750;
}
.ffs-model-badge::before {
    content: '';
    width: 7px;
    height: 7px;
    flex: 0 0 7px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 15%, transparent);
}
#ffs-new-session {
    min-height: 38px !important;
    border: 1px solid var(--ffs-border) !important;
    background: var(--ffs-surface) !important;
    color: var(--ffs-text) !important;
    box-shadow: var(--ffs-shadow) !important;
    transition: border-color .16s ease, background-color .16s ease, transform .16s ease !important;
}
#ffs-new-session:hover {
    border-color: var(--ffs-line-strong) !important;
    background: var(--ffs-surface-2) !important;
    transform: translateY(-1px) !important;
}

/* Settings rail */
#ffs-settings-panel {
    border-color: color-mix(in srgb, var(--ffs-border) 88%, var(--ffs-text)) !important;
    box-shadow: var(--ffs-elevated) !important;
}
#ffs-settings-panel > .label-wrap {
    min-height: 58px !important;
    padding: 0 17px !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    box-shadow: 0 1px 0 var(--ffs-border);
}
#ffs-settings-panel > .label-wrap:hover { background: var(--ffs-surface-2) !important; }
#ffs-settings-panel .ffs-settings-panel { padding: 4px 17px 16px !important; }
#ffs-settings-panel .block { padding: 15px 0 17px !important; }
#ffs-settings-panel label > span,
#ffs-settings-panel .block > label > span {
    color: var(--ffs-muted) !important;
    font-size: 10px !important;
    font-weight: 800 !important;
    letter-spacing: .065em !important;
}
#ffs-settings-panel input[type="number"],
#ffs-settings-panel textarea,
#ffs-settings-panel [role="combobox"] {
    border-color: transparent !important;
    background: var(--ffs-surface-2) !important;
    color: var(--ffs-text) !important;
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--ffs-border) 78%, transparent) !important;
    transition: box-shadow .16s ease, background-color .16s ease !important;
}
#ffs-settings-panel input[type="number"]:hover,
#ffs-settings-panel textarea:hover,
#ffs-settings-panel [role="combobox"]:hover {
    background: color-mix(in srgb, var(--ffs-surface-2) 70%, var(--ffs-surface)) !important;
    box-shadow: inset 0 0 0 1px var(--ffs-line-strong) !important;
}
#ffs-settings-panel input[type="number"]:focus,
#ffs-settings-panel textarea:focus,
#ffs-settings-panel [role="combobox"]:focus-within {
    background: var(--ffs-surface) !important;
    box-shadow: 0 0 0 3px var(--ffs-accent-bg), inset 0 0 0 1px var(--ffs-accent) !important;
}
#ffs-settings-panel button[aria-label="Reset to default value"] {
    border-color: transparent !important;
    background: var(--ffs-surface-2) !important;
    color: var(--ffs-muted) !important;
    transition: color .16s ease, background-color .16s ease !important;
}
#ffs-settings-panel button[aria-label="Reset to default value"]:hover {
    background: var(--ffs-accent-bg) !important;
    color: var(--ffs-accent) !important;
}
#ffs-settings-panel input[type="range"]::-webkit-slider-runnable-track {
    height: 6px;
    box-shadow: inset 0 1px 2px rgba(20,27,36,.08);
}
#ffs-settings-panel input[type="range"]::-webkit-slider-thumb {
    width: 20px;
    height: 20px;
    margin-top: -7px;
    border-width: 4px;
    box-shadow: 0 0 0 1px var(--ffs-accent), 0 4px 9px rgba(230,82,53,.24);
    transition: transform .12s ease, box-shadow .12s ease;
}
#ffs-settings-panel input[type="range"]:hover::-webkit-slider-thumb {
    transform: scale(1.08);
    box-shadow: 0 0 0 1px var(--ffs-accent), 0 5px 12px rgba(230,82,53,.3);
}

/* Creation canvas */
.ffs-canvas-heading {
    margin-bottom: 24px !important;
    font-size: 22px !important;
    letter-spacing: 0 !important;
}
.ffs-canvas-heading span {
    display: inline-flex;
    align-items: center;
    gap: 9px;
}
.ffs-canvas-heading span::before {
    content: '';
    width: 9px;
    height: 9px;
    border-radius: 2px;
    background: var(--ffs-coral);
    box-shadow: 5px 5px 0 color-mix(in srgb, var(--ffs-teal) 88%, transparent);
    transform: translateY(-2px);
}
.ffs-canvas-heading i {
    background: linear-gradient(90deg, var(--ffs-line-strong), transparent) !important;
}
.ffs-empty {
    position: relative;
    isolation: isolate;
}
.ffs-empty::before {
    content: '';
    position: absolute;
    z-index: -1;
    left: 50%;
    top: 50%;
    width: min(540px, 86%);
    height: 250px;
    transform: translate(-50%, -47%);
    border: 1px dashed color-mix(in srgb, var(--ffs-border) 78%, transparent);
    border-radius: 8px;
    background: color-mix(in srgb, var(--ffs-surface) 48%, transparent);
}
.ffs-empty-icon {
    position: relative;
    width: 72px !important;
    height: 72px !important;
    border: 0 !important;
    background: var(--ffs-ink) !important;
    color: #fff !important;
    box-shadow: 8px 8px 0 var(--ffs-coral), -8px -8px 0 var(--ffs-teal), var(--ffs-elevated) !important;
    transform: rotate(-2deg);
}
.ffs-empty-title {
    margin-top: 5px !important;
    color: var(--ffs-ink) !important;
    font-size: 34px !important;
    line-height: 1.08 !important;
}
.dark .ffs-empty-title { color: var(--ffs-text) !important; }
.ffs-empty-sub { color: var(--ffs-muted) !important; }
#ffs-starter-prompts { gap: 10px !important; }
#ffs-starter-prompts button {
    position: relative;
    min-height: 44px !important;
    padding: 0 13px !important;
    overflow: hidden;
    font-size: 12px !important;
    font-weight: 650 !important;
    box-shadow: 0 5px 16px rgba(20,27,36,.045) !important;
    transition: transform .16s ease, border-color .16s ease, color .16s ease, box-shadow .16s ease !important;
}
#ffs-starter-prompts button::before {
    content: '';
    width: 6px;
    height: 6px;
    margin-right: 8px;
    flex: 0 0 6px;
    border-radius: 2px;
    background: var(--ffs-coral);
}
#ffs-starter-prompts button:nth-child(2)::before { background: var(--ffs-gold); }
#ffs-starter-prompts button:nth-child(3)::before { background: var(--ffs-teal); }
#ffs-starter-prompts button:nth-child(4)::before { background: var(--ffs-ink); }
#ffs-starter-prompts button:hover {
    transform: translateY(-2px) !important;
    border-color: var(--ffs-line-strong) !important;
    box-shadow: 0 10px 22px rgba(20,27,36,.08) !important;
}

/* Conversation and results */
.ffs-role {
    margin-bottom: 7px !important;
    color: var(--ffs-muted) !important;
    font-weight: 800 !important;
    letter-spacing: .07em !important;
}
.ffs-bubble {
    border: 1px solid var(--ffs-border) !important;
    background: var(--ffs-surface) !important;
    box-shadow: 0 3px 12px rgba(20,27,36,.04) !important;
}
.ffs-turn-user .ffs-bubble {
    border-color: color-mix(in srgb, var(--ffs-teal) 18%, var(--ffs-border)) !important;
    background: color-mix(in srgb, var(--ffs-teal) 6%, var(--ffs-surface)) !important;
}
.ffs-notice {
    min-height: 48px;
    align-items: center !important;
    box-shadow: 0 5px 18px rgba(20,27,36,.045);
}
#ffs-result-gallery .grid-wrap { gap: 14px !important; }
#ffs-result-gallery .gallery-item {
    position: relative;
    border-color: color-mix(in srgb, var(--ffs-border) 82%, var(--ffs-text)) !important;
    background: var(--ffs-surface) !important;
    box-shadow: 0 8px 24px rgba(20,27,36,.07) !important;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease !important;
}
#ffs-result-gallery .gallery-item:hover {
    z-index: 1;
    transform: translateY(-3px);
    border-color: var(--ffs-line-strong) !important;
    box-shadow: 0 16px 34px rgba(20,27,36,.12) !important;
}
#ffs-result-gallery .gallery-item img {
    transition: transform .35s ease !important;
}
#ffs-result-gallery .gallery-item:hover img { transform: scale(1.012); }
#ffs-result-actions {
    padding: 10px 0 4px !important;
    border-top: 1px solid var(--ffs-border) !important;
}
#ffs-result-actions button {
    border-color: var(--ffs-border) !important;
    background: var(--ffs-surface) !important;
    color: var(--ffs-text) !important;
    font-weight: 700 !important;
    transition: transform .16s ease, border-color .16s ease, background-color .16s ease !important;
}
#ffs-result-actions button:hover {
    transform: translateY(-1px) !important;
    border-color: var(--ffs-line-strong) !important;
    background: var(--ffs-surface-2) !important;
}

/* Generation stage */
.ffs-generation-stage {
    position: relative;
    border-color: color-mix(in srgb, var(--ffs-border) 80%, var(--ffs-text)) !important;
    box-shadow: var(--ffs-elevated) !important;
}
.ffs-generation-stage::after {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, var(--ffs-coral), var(--ffs-gold), var(--ffs-teal));
}
.ffs-generation-preview {
    background: #16191d !important;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.07);
}
.ffs-gen-tile { box-shadow: inset 0 0 0 1px rgba(255,255,255,.14), 0 12px 25px rgba(0,0,0,.24); }
.ffs-gen-tile-a { transform: rotate(-2deg); }
.ffs-gen-tile-b { transform: rotate(2deg); }
.ffs-gen-tile-c { transform: rotate(-1deg); }
.ffs-generation-title { font-size: 24px !important; font-weight: 800 !important; }
.ffs-progress-track { height: 6px !important; }
.ffs-progress-track span {
    background: linear-gradient(90deg, var(--ffs-coral), var(--ffs-gold), var(--ffs-teal)) !important;
}

/* Composer */
#ffs-composer-dock {
    padding-top: 22px !important;
    background: linear-gradient(to bottom, transparent 0, color-mix(in srgb, var(--ffs-bg) 93%, transparent) 30%, var(--ffs-bg) 68%) !important;
}
#ffs-composer {
    min-height: 64px !important;
    padding: 8px 9px !important;
    border-color: color-mix(in srgb, var(--ffs-border) 66%, var(--ffs-text)) !important;
    box-shadow: 0 22px 55px rgba(20,27,36,.16), 0 4px 13px rgba(20,27,36,.08) !important;
}
#ffs-composer:focus-within {
    border-color: var(--ffs-coral) !important;
    box-shadow: 0 24px 60px rgba(20,27,36,.18), 0 0 0 4px var(--ffs-accent-bg) !important;
}
#ffs-prompt textarea {
    font-size: 14px !important;
    font-weight: 500 !important;
}
#ffs-prompt textarea::placeholder { color: color-mix(in srgb, var(--ffs-muted) 78%, transparent) !important; }
#ffs-attach {
    border-color: transparent !important;
    background: var(--ffs-surface-2) !important;
    color: var(--ffs-text) !important;
    transition: transform .16s ease, color .16s ease, background-color .16s ease !important;
}
#ffs-attach:hover {
    transform: translateY(-1px) !important;
    background: var(--ffs-accent-bg) !important;
    color: var(--ffs-coral) !important;
}
#ffs-generate {
    min-width: 112px !important;
    background: var(--ffs-coral) !important;
    border-color: var(--ffs-coral) !important;
    box-shadow: 0 7px 15px rgba(230,82,53,.22) !important;
    transition: transform .16s ease, background-color .16s ease, box-shadow .16s ease !important;
}
#ffs-generate:hover {
    transform: translateY(-1px) !important;
    background: var(--ffs-coral-dark) !important;
    box-shadow: 0 10px 20px rgba(230,82,53,.28) !important;
}
#ffs-generate:active { transform: translateY(0) !important; box-shadow: 0 4px 9px rgba(230,82,53,.2) !important; }
#ffs-attachment-row { box-shadow: var(--ffs-elevated) !important; }

/* Accessible motion and focus */
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container [role="combobox"]:focus-visible {
    outline: 2px solid var(--ffs-accent) !important;
    outline-offset: 2px !important;
}
::selection { background: color-mix(in srgb, var(--ffs-coral) 25%, transparent); color: var(--ffs-text); }

@media (max-width: 900px) {
    #ffs-settings-panel { box-shadow: 0 7px 22px rgba(20,27,36,.07) !important; }
    #ffs-settings-panel > .label-wrap { min-height: 56px !important; }
    .ffs-empty::before { width: min(500px, 92%); }
}

@media (max-width: 720px) {
    #ffs-app-header { box-shadow: 0 6px 18px rgba(20,27,36,.06) !important; }
    .ffs-brand-mark { width: 40px; height: 40px; flex-basis: 40px; }
    #ffs-settings-panel { border-radius: 8px !important; }
    .ffs-canvas-heading { margin-bottom: 14px !important; font-size: 20px !important; }
    .ffs-empty {
        width: 100% !important;
        min-height: 39vh !important;
        display: flex !important;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    .ffs-empty::before { height: 220px; width: calc(100% - 8px); }
    .ffs-empty-icon {
        width: 58px !important;
        height: 58px !important;
        box-shadow: 6px 6px 0 var(--ffs-coral), -6px -6px 0 var(--ffs-teal), var(--ffs-elevated) !important;
    }
    .ffs-empty-title {
        width: 100% !important;
        max-width: 100% !important;
        font-size: 25px !important;
        text-align: center;
        white-space: normal !important;
    }
    .ffs-empty-sub {
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 8px;
        text-align: center;
        white-space: normal !important;
    }
    #ffs-settings-panel > .label-wrap .icon {
        display: block !important;
        width: 18px !important;
        margin-left: auto;
        color: var(--ffs-text) !important;
        opacity: .72;
    }
    #ffs-starter-prompts { gap: 8px !important; }
    #ffs-starter-prompts button { min-height: 43px !important; padding: 0 9px !important; font-size: 11px !important; }
    #ffs-starter-prompts button::before { margin-right: 6px; }
    #ffs-composer-dock { padding-top: 16px !important; }
    #ffs-composer { min-height: 62px !important; border-radius: 8px !important; }
    #ffs-generate { min-width: 48px !important; width: 48px !important; box-shadow: 0 6px 14px rgba(230,82,53,.23) !important; }
    .ffs-generation-title { font-size: 20px !important; }
    #ffs-result-gallery .grid-wrap { gap: 8px !important; }
    #ffs-result-actions { gap: 7px !important; }
}

@media (max-width: 420px) {
    #ffs-model-select { min-width: 118px !important; flex: 1 1 calc(100% - 150px) !important; }
    #ffs-new-session { min-width: 54px !important; padding-left: 10px !important; padding-right: 10px !important; }
    .ffs-empty-title { font-size: 23px !important; }
    #ffs-starter-prompts button { min-height: 42px !important; }
}

@media (prefers-reduced-motion: reduce) {
    #ffs-starter-prompts button,
    #ffs-result-gallery .gallery-item,
    #ffs-result-gallery .gallery-item img,
    #ffs-generate,
    #ffs-attach,
    #ffs-new-session { transition: none !important; }
}

/* Theme control, model profile, and final control alignment */
html[data-ffs-theme="light"] {
    color-scheme: light;
    --ffs-bg: #f4f5f7;
    --ffs-surface: #ffffff;
    --ffs-surface-2: #eceff2;
    --ffs-text: #15171a;
    --ffs-text-2: #606770;
    --ffs-border: #dfe3e8;
    --ffs-ink: #111418;
    --ffs-muted: #747b84;
    --ffs-line-strong: #cfd5dc;
    --ffs-accent: #e65235;
    --ffs-accent-bg: rgba(230, 82, 53, .08);
    --ffs-success: #147d72;
}
html[data-ffs-theme="dark"] {
    color-scheme: dark;
    --ffs-bg: #101214;
    --ffs-surface: #191c1f;
    --ffs-surface-2: #24282c;
    --ffs-text: #f2f4f6;
    --ffs-text-2: #adb3ba;
    --ffs-border: #32373c;
    --ffs-ink: #f6f7f8;
    --ffs-muted: #969da5;
    --ffs-line-strong: #495057;
    --ffs-accent: #ff7558;
    --ffs-accent-bg: rgba(255, 117, 88, .11);
    --ffs-success: #4ec9b9;
    --ffs-shadow: 0 1px 2px rgba(0, 0, 0, .28);
    --ffs-shadow-lg: 0 22px 60px rgba(0, 0, 0, .42), 0 2px 8px rgba(0, 0, 0, .3);
    --ffs-elevated: 0 14px 34px rgba(0, 0, 0, .25), 0 2px 8px rgba(0, 0, 0, .2);
}
html[data-ffs-theme="dark"] body,
html[data-ffs-theme="dark"] .gradio-container { background: var(--ffs-bg) !important; }
#ffs-theme-toggle {
    width: 40px !important;
    min-width: 40px !important;
    height: 38px !important;
    min-height: 38px !important;
    padding: 0 !important;
    border: 1px solid var(--ffs-border) !important;
    border-radius: 7px !important;
    background: var(--ffs-surface) !important;
    color: var(--ffs-text) !important;
    box-shadow: var(--ffs-shadow) !important;
    font-size: 18px !important;
    line-height: 1 !important;
}
#ffs-theme-toggle:hover {
    border-color: var(--ffs-line-strong) !important;
    background: var(--ffs-surface-2) !important;
}
#ffs-model-profile { padding: 0 !important; }
#ffs-model-profile .html-container { padding: 0 !important; }
.ffs-model-profile {
    padding: 14px 0 15px;
    border-bottom: 1px solid var(--ffs-border);
}
.ffs-model-profile strong,
.ffs-model-profile small { display: block; }
.ffs-encoder-status {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    min-height: 36px;
    padding: 9px 10px;
    border: 1px solid var(--ffs-line);
    background: color-mix(in srgb, var(--ffs-panel) 76%, var(--ffs-canvas));
    font-size: 11px;
    flex-wrap: wrap;
}
.ffs-encoder-status strong { color: var(--ffs-text); font-size: 11px; }
.ffs-encoder-file {
    max-width: 100%;
    overflow: hidden;
    color: var(--ffs-teal);
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ffs-encoder-file.muted { color: var(--ffs-muted); }
.ffs-flux-encoder-panel { gap: 9px !important; }
.ffs-flux-encoder-panel .wrap { gap: 7px !important; }
#ffs-apply-encoder { min-height: 36px !important; }
.ffs-model-profile strong { color: var(--ffs-text); font-size: 13px; font-weight: 800; }
.ffs-model-profile small { margin-top: 3px; color: var(--ffs-muted); font-size: 11px; line-height: 1.35; }
.ffs-model-profile > div { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.ffs-model-profile span {
    padding: 4px 7px;
    border: 1px solid var(--ffs-border);
    border-radius: 5px;
    background: var(--ffs-surface-2);
    color: var(--ffs-text-2);
    font-size: 9px;
    font-weight: 750;
}
#ffs-settings-panel input[type="number"] {
    box-sizing: border-box !important;
    height: 42px !important;
    min-height: 42px !important;
    padding: 0 12px !important;
    line-height: 40px !important;
}
#ffs-settings-panel .block:has(input[type="range"]) input[type="number"] {
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 9px !important;
    line-height: 32px !important;
}
#ffs-settings-panel .block:has(input[type="range"]) button[aria-label="Reset to default value"] {
    height: 34px !important;
    min-height: 34px !important;
}
.ffs-generation-stage { animation: ffs-stage-enter .36s ease-out both; }
@keyframes ffs-stage-enter {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (min-width: 901px) {
    #ffs-settings-panel > .label-wrap {
        pointer-events: none !important;
        cursor: default !important;
    }
    #ffs-settings-panel > .label-wrap > span:last-child,
    #ffs-settings-panel > .label-wrap svg { display: none !important; }
}

@media (max-width: 720px) {
    #ffs-theme-toggle { width: 38px !important; min-width: 38px !important; }
    .ffs-encoder-status {
        align-items: flex-start;
        flex-direction: column;
    }
    .ffs-encoder-file { width: 100%; }
}

@media (max-width: 420px) {
    #ffs-app-header { gap: 6px !important; padding-left: 8px !important; padding-right: 8px !important; }
    #ffs-brand { min-width: 42px !important; flex-basis: 42px !important; }
    #ffs-model-select { min-width: 118px !important; flex: 1 1 calc(100% - 150px) !important; }
    #ffs-theme-toggle { width: 36px !important; min-width: 36px !important; }
    #ffs-new-session { min-width: 48px !important; padding-left: 8px !important; padding-right: 8px !important; }
}

/* Header-triggered settings drawer */
#ffs-settings-toggle {
    width: 40px !important;
    min-width: 40px !important;
    height: 38px !important;
    min-height: 38px !important;
    padding: 0 !important;
    border: 1px solid var(--ffs-border) !important;
    border-radius: 7px !important;
    background: var(--ffs-surface) !important;
    color: var(--ffs-text) !important;
    box-shadow: var(--ffs-shadow) !important;
    font-size: 17px !important;
    line-height: 1 !important;
}
#ffs-settings-toggle:hover,
html[data-ffs-settings="open"] #ffs-settings-toggle {
    border-color: var(--ffs-accent) !important;
    background: var(--ffs-accent-bg) !important;
    color: var(--ffs-accent) !important;
}
#ffs-settings-toggle.ffs-settings-open {
    border-color: var(--ffs-accent) !important;
    background: var(--ffs-accent-bg) !important;
    color: var(--ffs-accent) !important;
}
#ffs-settings-panel {
    position: fixed !important;
    z-index: 1160 !important;
    left: 14px !important;
    top: 86px !important;
    bottom: 18px !important;
    width: 280px !important;
    box-sizing: border-box !important;
    max-height: none !important;
    margin: 0 !important;
    display: flex !important;
    overflow-y: auto !important;
    overflow-x: clip !important;
    border: 1px solid color-mix(in srgb, var(--ffs-border) 82%, var(--ffs-text)) !important;
    border-radius: 8px !important;
    background: var(--ffs-surface) !important;
    box-shadow: 0 24px 70px rgba(20,27,36,.22), 0 4px 14px rgba(20,27,36,.1) !important;
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transform: translateX(calc(-100% - 28px));
    transition: transform .24s cubic-bezier(.2,.75,.2,1), opacity .18s ease, visibility .18s ease !important;
}
html[data-ffs-settings="open"] #ffs-settings-panel {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transform: translateX(0);
}
#ffs-settings-panel.ffs-settings-open {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transform: translateX(0);
}
#ffs-settings-heading {
    position: sticky;
    z-index: 4;
    top: 0;
    width: 100% !important;
    padding: 0 !important;
    background: var(--ffs-surface) !important;
}
#ffs-settings-heading .html-container { padding: 0 !important; }
.ffs-settings-heading {
    min-height: 58px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 12px 0 17px;
    border-bottom: 1px solid var(--ffs-border);
    color: var(--ffs-text);
    font-size: 14px;
    font-weight: 800;
}
.ffs-settings-close {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    margin-left: auto;
    padding: 0;
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: var(--ffs-muted);
    font: inherit;
    font-size: 21px;
    cursor: pointer;
}
.ffs-settings-close:hover { background: var(--ffs-surface-2); color: var(--ffs-text); }
#ffs-settings-panel > .ffs-settings-panel { width: 100% !important; }
#ffs-settings-backdrop {
    position: fixed;
    z-index: 1140;
    inset: 72px 0 0 0;
    border: 0;
    background: rgba(10,13,17,.36);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    backdrop-filter: blur(2px);
    transition: opacity .18s ease, visibility .18s ease;
}
html[data-ffs-settings="open"] #ffs-settings-backdrop {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
}
#ffs-settings-backdrop.ffs-settings-open {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
}

/* Conversational result feed */
.ffs-chat-images {
    display: grid;
    width: min(760px, 100%);
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}
.ffs-chat-images.count-1 {
    width: min(680px, 100%);
    grid-template-columns: minmax(0, 1fr);
}
.ffs-chat-image {
    display: block;
    min-width: 0;
    overflow: hidden;
    border: 1px solid var(--ffs-border);
    border-radius: 8px;
    background: var(--ffs-surface-2);
    box-shadow: var(--ffs-shadow);
}
.ffs-chat-image img {
    display: block;
    width: 100%;
    height: auto;
    max-height: 720px;
    object-fit: contain;
}
.ffs-assistant-meta {
    margin-top: 6px;
    color: var(--ffs-muted);
    font-size: 11px;
    font-weight: 650;
}
.ffs-request-attachment {
    display: flex;
    align-items: center;
    gap: 9px;
    width: max-content;
    max-width: 100%;
    margin-top: 7px;
    padding: 6px;
    border: 1px solid var(--ffs-border);
    border-radius: 7px;
    background: var(--ffs-surface);
}
.ffs-request-attachments {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 7px;
}
.ffs-request-attachments .ffs-request-attachment { margin-top: 0; }
.ffs-request-attachment img {
    width: 54px;
    height: 54px;
    flex: 0 0 54px;
    border-radius: 5px;
    object-fit: cover;
}
.ffs-request-attachment span,
.ffs-request-attachment strong,
.ffs-request-attachment small { display: block; }
.ffs-request-attachment strong { color: var(--ffs-text); font-size: 11px; }
.ffs-request-attachment small { margin-top: 2px; color: var(--ffs-muted); font-size: 10px; }

/* The input image and its edit tools belong to the same request. */
#ffs-attachment-row {
    align-items: stretch !important;
    gap: 10px !important;
    min-height: 112px !important;
    padding: 8px !important;
    border: 1px solid var(--ffs-border) !important;
    border-bottom: 0 !important;
    border-radius: 8px 8px 0 0 !important;
    background: var(--ffs-surface) !important;
}
#ffs-attachment-manager { gap: 7px !important; min-width: 0 !important; }
#ffs-attachment-footer { align-items: center !important; gap: 7px !important; }
#ffs-attachment-copy,
#ffs-attachment-copy .html-container { min-width: 0 !important; padding: 0 !important; }
.ffs-attachment-copy strong,
.ffs-attachment-copy span,
.ffs-attachment-copy small { display: block; overflow-wrap: anywhere; }
.ffs-attachment-copy strong { color: var(--ffs-text); font-size: 12px; font-weight: 800; }
.ffs-attachment-copy span { margin-top: 2px; color: var(--ffs-muted); font-size: 10px; }
.ffs-attachment-copy small { margin-top: 4px; color: var(--ffs-muted); font-size: 9px; line-height: 1.35; }
#ffs-attachment-preview {
    width: 100% !important;
    max-width: none !important;
    height: 112px !important;
    min-height: 112px !important;
    border: 0 !important;
    overflow: hidden !important;
}
#ffs-attachment-preview .grid-wrap {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(94px, 1fr)) !important;
    gap: 7px !important;
    overflow-x: auto !important;
    scrollbar-width: thin;
}
#ffs-attachment-preview .gallery-item {
    position: relative !important;
    min-width: 94px !important;
    height: 104px !important;
    overflow: hidden !important;
    border: 1px solid var(--ffs-border) !important;
    border-radius: 7px !important;
    background: var(--ffs-surface-2) !important;
}
#ffs-attachment-preview .gallery-item img { cursor: zoom-in !important; }
.ffs-attachment-role {
    position: absolute;
    z-index: 4;
    top: 6px;
    left: 6px;
    max-width: calc(100% - 12px);
    padding: 3px 6px;
    border-radius: 5px;
    background: rgba(12, 16, 22, .78);
    color: #fff;
    font-size: 9px;
    font-weight: 800;
    line-height: 1.15;
    pointer-events: none;
    backdrop-filter: blur(7px);
}
.ffs-attachment-actions {
    position: absolute;
    z-index: 5;
    inset: auto 5px 5px 5px;
    display: flex;
    justify-content: center;
    gap: 3px;
    opacity: 0;
    transform: translateY(5px);
    transition: opacity .16s ease, transform .16s ease;
}
#ffs-attachment-preview .gallery-item:hover .ffs-attachment-actions,
#ffs-attachment-preview .gallery-item:focus-within .ffs-attachment-actions {
    opacity: 1;
    transform: translateY(0);
}
.ffs-attachment-action {
    width: 25px;
    min-width: 25px;
    height: 25px;
    min-height: 25px;
    padding: 0;
    border: 1px solid rgba(255,255,255,.16);
    border-radius: 5px;
    background: rgba(12,16,22,.86);
    color: #fff;
    font: 800 12px/1 Inter, sans-serif;
    cursor: pointer;
    backdrop-filter: blur(7px);
}
.ffs-attachment-action:hover { background: var(--ffs-coral); }
.ffs-attachment-action:disabled { opacity: .38; cursor: default; }
#ffs-open-mask-settings,
#ffs-clear-attachments { flex: 0 0 auto !important; min-width: 86px !important; }
.ffs-internal-control { display: none !important; }

/* Shared full-screen image viewer */
#ffs-lightbox {
    position: fixed;
    z-index: 5000;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 28px;
    background: rgba(7, 10, 14, .9);
    backdrop-filter: blur(12px);
}
#ffs-lightbox.ffs-open { display: flex; }
#ffs-lightbox img {
    display: block;
    max-width: min(94vw, 1600px);
    max-height: 92vh;
    object-fit: contain;
    border-radius: 7px;
    box-shadow: 0 26px 80px rgba(0,0,0,.5);
}
#ffs-lightbox button {
    position: absolute;
    top: max(16px, env(safe-area-inset-top));
    right: 18px;
    width: 40px;
    min-width: 40px;
    height: 40px;
    padding: 0;
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 7px;
    background: rgba(12,16,22,.82);
    color: #fff;
    font-size: 22px;
    cursor: pointer;
}

#ffs-settings-panel #ffs-edit-panel {
    width: 100% !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 15px 0 2px !important;
    overflow: visible !important;
    border: 0 !important;
    border-top: 1px solid var(--ffs-border) !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
#ffs-edit-heading,
#ffs-edit-heading .html-container { padding: 0 !important; }
.ffs-section-heading {
    margin-bottom: 10px;
    color: var(--ffs-text);
    font-size: 13px;
    font-weight: 800;
}
.ffs-section-heading small {
    display: block;
    margin-top: 3px;
    color: var(--ffs-muted);
    font-size: 9px;
    font-weight: 600;
    line-height: 1.35;
}
#ffs-edit-panel .block {
    padding-top: 8px !important;
    padding-bottom: 10px !important;
}
#ffs-edit-panel .gradio-radio .wrap { gap: 6px !important; }
#ffs-edit-panel .gradio-radio label {
    min-height: 34px !important;
    padding: 7px 8px !important;
    border: 1px solid var(--ffs-border) !important;
    border-radius: 6px !important;
    background: var(--ffs-surface-2) !important;
    font-size: 11px !important;
}
#ffs-edit-panel .image-container { border-radius: 6px !important; overflow: hidden !important; }

/* The drawer overlays compact layouts and shifts the canvas on wide screens. */
@media (min-width: 901px) {
    #ffs-workspace {
        width: min(var(--ffs-max-width), calc(100% - 44px)) !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    #ffs-composer-dock { left: 0 !important; }
    html[data-ffs-settings="open"] #ffs-workspace {
        width: min(var(--ffs-max-width), calc(100% - 342px)) !important;
        margin-left: max(320px, calc((100% - var(--ffs-max-width) + 300px) / 2)) !important;
        margin-right: 22px !important;
    }
    html[data-ffs-settings="open"] #ffs-composer-dock { left: 300px !important; }
    #ffs-workspace.ffs-settings-open {
        width: min(var(--ffs-max-width), calc(100% - 342px)) !important;
        margin-left: max(320px, calc((100% - var(--ffs-max-width) + 300px) / 2)) !important;
        margin-right: 22px !important;
    }
    #ffs-composer-dock.ffs-settings-open { left: 300px !important; }
    #ffs-settings-backdrop { display: none; }
}

@media (max-width: 900px) {
    #ffs-settings-panel {
        left: 12px !important;
        top: 76px !important;
        bottom: 12px !important;
        width: min(340px, calc(100% - 24px)) !important;
    }
    #ffs-workspace { padding-top: var(--ffs-page-top) !important; }
    #ffs-settings-backdrop { top: 65px; }
}

@media (max-width: 720px) {
    #ffs-settings-toggle,
    #ffs-theme-toggle {
        width: 36px !important;
        min-width: 36px !important;
    }
    .ffs-chat-images { grid-template-columns: minmax(0, 1fr); }
    #ffs-attachment-copy { flex: 1 1 auto !important; }
    #ffs-attachment-preview .grid-wrap { grid-template-columns: repeat(4, 98px) !important; }
    .ffs-attachment-actions { opacity: 1; transform: none; }
    #ffs-open-mask-settings,
    #ffs-clear-attachments { min-width: 72px !important; }
}

@media (max-width: 420px) {
    :root { --ffs-header-h: 128px; --ffs-page-top: 150px; }
    #ffs-app-header { gap: 4px !important; }
    #ffs-workspace-tabs {
        width: 218px !important;
        min-width: 0 !important;
        flex: 0 1 218px !important;
    }
    #ffs-model-select { min-width: 118px !important; flex: 1 1 calc(100% - 150px) !important; }
    #ffs-settings-toggle,
    #ffs-theme-toggle { width: 34px !important; min-width: 34px !important; }
    #ffs-new-session { min-width: 44px !important; }
}

/* Avatar Studio */
#ffs-workspace-tabs {
    min-height: 38px !important;
    border: 1px solid var(--ffs-border) !important;
    border-radius: 8px !important;
    padding: 3px !important;
    background: var(--ffs-panel) !important;
}
#ffs-workspace-tabs .wrap {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 3px !important;
}
#ffs-workspace-tabs label {
    margin: 0 !important;
    border-radius: 6px !important;
    min-height: 30px !important;
    justify-content: center !important;
    font-size: 12px !important;
}
#ffs-avatar-studio {
    max-width: 1240px !important;
    width: min(100%, 1240px) !important;
    margin: 0 auto !important;
    padding: var(--ffs-page-top) 16px 36px !important;
    min-height: calc(100vh - var(--ffs-header-h));
}
#ffs-avatar-layout { align-items: flex-start !important; gap: 14px !important; }
#ffs-avatar-sidebar {
    position: sticky !important;
    top: calc(var(--ffs-header-h) + 14px) !important;
    border: 1px solid var(--ffs-border) !important;
    background: var(--ffs-panel) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: var(--ffs-elevated) !important;
}
.ffs-avatar-panel-title strong,
.ffs-avatar-panel-title span { display: block; }
.ffs-avatar-panel-title strong { color: var(--ffs-text); font-size: 16px; }
.ffs-avatar-panel-title span { color: var(--ffs-muted); font-size: 12px; margin-top: 3px; }
.ffs-avatar-status,
.ffs-avatar-empty {
    border: 1px solid var(--ffs-border) !important;
    background: color-mix(in srgb, var(--ffs-panel) 78%, var(--ffs-bg)) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    margin-top: 8px !important;
}
.ffs-avatar-status strong,
.ffs-avatar-status span,
.ffs-avatar-status small,
.ffs-avatar-empty strong,
.ffs-avatar-empty span { display: block; }
.ffs-avatar-status strong { color: var(--ffs-text); font-size: 14px; }
.ffs-avatar-status span { color: var(--ffs-coral); font-size: 12px; margin-top: 4px; }
.ffs-avatar-status small,
.ffs-avatar-empty span { color: var(--ffs-muted); font-size: 11px; margin-top: 4px; }
#ffs-avatar-main { gap: 12px !important; min-width: 0 !important; }
.ffs-avatar-step {
    border: 1px solid var(--ffs-border) !important;
    background: var(--ffs-panel) !important;
    border-radius: 8px !important;
    padding: 20px !important;
    box-shadow: var(--ffs-elevated) !important;
    min-width: 0 !important;
}
.ffs-avatar-step-title {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: baseline;
    margin-bottom: 10px;
}
.ffs-avatar-step-title strong { color: var(--ffs-text); font-size: 18px; }
.ffs-avatar-step-title span { color: var(--ffs-muted); font-size: 12px; text-align: right; max-width: 470px; }
.ffs-avatar-reference-row { align-items: stretch !important; gap: 14px !important; }
.ffs-avatar-reference-row > * { min-width: 0 !important; }
.ffs-avatar-reference-row textarea { min-height: 118px !important; }
.ffs-avatar-action-row { gap: 8px !important; }
.ffs-avatar-action-row button { min-height: 42px !important; }
.ffs-avatar-control-row,
.ffs-avatar-extra-row,
.ffs-avatar-gallery-controls { gap: 10px !important; align-items: end !important; }
#ffs-avatar-sidebar fieldset { margin-top: 8px !important; }
#ffs-avatar-sidebar fieldset .wrap { gap: 3px !important; }
#ffs-avatar-sidebar fieldset label {
    border-radius: 6px !important;
    min-height: 34px !important;
    justify-content: center !important;
}
.ffs-avatar-specs {
    display: grid;
    gap: 7px;
    margin-top: 10px;
}
.ffs-avatar-specs > strong {
    color: var(--ffs-text);
    font-size: 13px;
}
.ffs-avatar-spec-row {
    display: grid;
    grid-template-columns: minmax(120px, 1fr) minmax(120px, 1fr) 92px;
    gap: 8px;
    align-items: center;
    border-bottom: 1px solid var(--ffs-border);
    padding: 7px 0;
}
.ffs-avatar-spec-row span { color: var(--ffs-muted); font-size: 11px; }
.ffs-avatar-spec-row strong { color: var(--ffs-text); font-size: 12px; font-weight: 650; }
.ffs-avatar-spec-row em { color: var(--ffs-muted); font-size: 10px; font-style: normal; text-align: right; }
.ffs-avatar-chat-log { display: grid; gap: 10px; margin-top: 12px; }
.ffs-avatar-chat-turn {
    border: 1px solid var(--ffs-border);
    border-radius: 8px;
    padding: 10px;
    background: color-mix(in srgb, var(--ffs-panel) 80%, var(--ffs-bg));
}
.ffs-avatar-chat-turn strong,
.ffs-avatar-chat-turn span { display: block; }
.ffs-avatar-chat-turn strong { color: var(--ffs-coral); font-size: 12px; margin-bottom: 5px; }
.ffs-avatar-chat-turn span { color: var(--ffs-text); font-size: 13px; line-height: 1.45; }
.ffs-avatar-chat-turn.user {
    width: min(88%, 720px);
    margin-left: auto;
    background: color-mix(in srgb, var(--ffs-coral) 9%, var(--ffs-panel));
    border-color: color-mix(in srgb, var(--ffs-coral) 34%, var(--ffs-border));
}
.ffs-avatar-chat-turn.assistant { width: min(94%, 820px); }
.ffs-avatar-chat-turn .ffs-chat-image { margin-top: 9px; }
.ffs-avatar-chat-empty,
.ffs-avatar-gallery-empty {
    color: var(--ffs-muted);
    border: 1px dashed var(--ffs-border);
    border-radius: 8px;
    padding: 18px;
    text-align: center;
}
.ffs-avatar-gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(128px, 1fr));
    gap: 8px;
    margin-top: 12px;
}
.ffs-avatar-gallery-tile {
    display: block;
    aspect-ratio: 1 / 1;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--ffs-border);
    background: var(--ffs-bg);
}
.ffs-avatar-gallery-tile img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
#ffs-avatar-gallery {
    border: 0 !important;
    background: transparent !important;
    margin-top: 12px !important;
}
#ffs-avatar-gallery .grid-wrap { gap: 8px !important; }
#ffs-avatar-gallery .gallery-item {
    aspect-ratio: 3 / 4 !important;
    border-radius: 8px !important;
    border: 1px solid var(--ffs-border) !important;
    overflow: hidden !important;
    background: var(--ffs-bg) !important;
}
#ffs-avatar-gallery .gallery-item img { width: 100% !important; height: 100% !important; object-fit: cover !important; }
.ffs-avatar-gallery-selection {
    border: 1px solid var(--ffs-border);
    border-radius: 8px;
    background: color-mix(in srgb, var(--ffs-panel) 82%, var(--ffs-bg));
    padding: 12px;
    margin: 10px 0;
}
.ffs-avatar-gallery-selection strong,
.ffs-avatar-gallery-selection span,
.ffs-avatar-gallery-selection p,
.ffs-avatar-gallery-selection small { display: block; }
.ffs-avatar-gallery-selection strong { color: var(--ffs-text); font-size: 13px; }
.ffs-avatar-gallery-selection span { color: var(--ffs-coral); font-size: 11px; margin-top: 3px; }
.ffs-avatar-gallery-selection p { color: var(--ffs-text); font-size: 12px; line-height: 1.45; margin: 8px 0; }
.ffs-avatar-gallery-selection small { color: var(--ffs-muted); font-size: 11px; }
@media (max-width: 820px) {
    #ffs-avatar-layout { flex-direction: column !important; }
    #ffs-avatar-sidebar { position: static !important; width: 100% !important; }
    #ffs-avatar-studio { padding-left: 10px !important; padding-right: 10px !important; }
    .ffs-avatar-step { padding: 14px !important; }
    .ffs-avatar-reference-row,
    .ffs-avatar-control-row,
    .ffs-avatar-extra-row,
    .ffs-avatar-gallery-controls { flex-direction: column !important; align-items: stretch !important; }
    .ffs-avatar-reference-row > *,
    .ffs-avatar-control-row > *,
    .ffs-avatar-extra-row > *,
    .ffs-avatar-gallery-controls > * { width: 100% !important; min-width: 0 !important; }
    .ffs-avatar-step-title { display: block; }
    .ffs-avatar-step-title span { display: block; text-align: left; margin-top: 3px; }
    .ffs-avatar-spec-row { grid-template-columns: 1fr; gap: 3px; }
    .ffs-avatar-spec-row em { text-align: left; }
    #ffs-avatar-gallery .grid-wrap { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    .ffs-avatar-chat-turn.user,
    .ffs-avatar-chat-turn.assistant { width: 100%; }
}
@media (max-width: 440px) {
    .ffs-avatar-action-row { flex-direction: column !important; }
    .ffs-avatar-action-row > * { width: 100% !important; }
    #ffs-avatar-gallery .grid-wrap { grid-template-columns: 1fr !important; }
}
"""

# ═══════════════════════════════════════════════════════════
#  JAVASCRIPT — brush size boost, keyboard shortcuts
# ═══════════════════════════════════════════════════════════
CSS += avatar_design.CSS

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
    document.querySelectorAll('#ffs-settings-panel input[type="range"], #ffs-avatar-studio input[type="range"]').forEach(function(slider) {
        syncRangeFill(slider);
        if (!slider.dataset.ffsRangeReady) {
            slider.dataset.ffsRangeReady = 'true';
            slider.addEventListener('input', function() { syncRangeFill(slider); });
            slider.addEventListener('change', function() { syncRangeFill(slider); });
        }
    });
}

function applyStudioTheme(theme) {
    var resolved = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.dataset.ffsTheme = resolved;
    document.documentElement.classList.toggle('dark', resolved === 'dark');
    var toggle = document.querySelector('#ffs-theme-toggle button, #ffs-theme-toggle');
    if (toggle) {
        toggle.textContent = resolved === 'dark' ? '☀' : '☾';
        toggle.setAttribute('title', resolved === 'dark' ? 'Use light theme' : 'Use dark theme');
        toggle.setAttribute('aria-label', resolved === 'dark' ? 'Use light theme' : 'Use dark theme');
    }
}

function prepareThemeToggle() {
    var toggle = document.querySelector('#ffs-theme-toggle button, #ffs-theme-toggle');
    if (!document.documentElement.dataset.ffsTheme) {
        var stored = localStorage.getItem('ffs-theme');
        var preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        applyStudioTheme(stored || preferred);
    }
    if (toggle && !toggle.dataset.ffsThemeReady) {
        toggle.dataset.ffsThemeReady = 'true';
        toggle.addEventListener('click', function() {
            var next = document.documentElement.dataset.ffsTheme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('ffs-theme', next);
            applyStudioTheme(next);
        });
        applyStudioTheme(document.documentElement.dataset.ffsTheme);
    }
}

function setSettingsDrawer(open) {
    document.documentElement.dataset.ffsSettings = open ? 'open' : 'closed';
    var toggle = document.querySelector('#ffs-settings-toggle button, #ffs-settings-toggle');
    var panel = document.querySelector('#ffs-settings-panel');
    var workspace = document.querySelector('#ffs-workspace');
    var composer = document.querySelector('#ffs-composer-dock');
    var backdrop = document.querySelector('#ffs-settings-backdrop');
    [toggle, panel, workspace, composer, backdrop].forEach(function(element) {
        if (element) element.classList.toggle('ffs-settings-open', open);
    });
    if (panel) {
        panel.style.opacity = open ? '1' : '0';
        panel.style.visibility = open ? 'visible' : 'hidden';
        panel.style.pointerEvents = open ? 'auto' : 'none';
        panel.style.transform = open ? 'translateX(0)' : 'translateX(calc(-100% - 28px))';
    }
    if (backdrop) {
        backdrop.style.opacity = open ? '1' : '0';
        backdrop.style.visibility = open ? 'visible' : 'hidden';
        backdrop.style.pointerEvents = open ? 'auto' : 'none';
    }
    if (toggle) {
        toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        toggle.setAttribute('title', open ? 'Close generation controls' : 'Open generation controls');
        toggle.setAttribute('aria-label', open ? 'Close generation controls' : 'Open generation controls');
    }
}

function prepareSettingsDrawer() {
    var toggle = document.querySelector('#ffs-settings-toggle button, #ffs-settings-toggle');
    var close = document.querySelector('.ffs-settings-close');
    var backdrop = document.querySelector('#ffs-settings-backdrop');
    if (!backdrop) {
        backdrop = document.createElement('button');
        backdrop.id = 'ffs-settings-backdrop';
        backdrop.type = 'button';
        backdrop.setAttribute('aria-label', 'Close settings');
        document.body.appendChild(backdrop);
    }
    if (!document.documentElement.dataset.ffsSettings) {
        setSettingsDrawer(window.innerWidth >= 1100);
    }
    if (toggle && !toggle.dataset.ffsSettingsReady) {
        toggle.dataset.ffsSettingsReady = 'true';
        toggle.addEventListener('click', function() {
            setSettingsDrawer(document.documentElement.dataset.ffsSettings !== 'open');
        });
    }
    if (close && !close.dataset.ffsSettingsReady) {
        close.dataset.ffsSettingsReady = 'true';
        close.addEventListener('click', function() { setSettingsDrawer(false); });
    }
    if (!backdrop.dataset.ffsSettingsReady) {
        backdrop.dataset.ffsSettingsReady = 'true';
        backdrop.addEventListener('click', function() { setSettingsDrawer(false); });
    }
    if (!document.documentElement.dataset.ffsSettingsEscape) {
        document.documentElement.dataset.ffsSettingsEscape = 'true';
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') setSettingsDrawer(false);
        });
    }
    var avatarTab = document.querySelector('#ffs-workspace-tabs input[value="Avatar Studio"]:checked');
    setSettingsDrawer(!avatarTab && document.documentElement.dataset.ffsSettings === 'open');
}

function openMaskSettings() {
    setSettingsDrawer(true);
    window.setTimeout(function() {
        var panel = document.querySelector('#ffs-edit-panel');
        if (panel) panel.scrollIntoView({behavior: 'smooth', block: 'center'});
    }, 180);
}

function setStudioControlValue(selector, value) {
    var input = document.querySelector(selector);
    if (!input) return false;
    var descriptor = Object.getOwnPropertyDescriptor(
        input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,
        'value'
    );
    if (descriptor && descriptor.set) descriptor.set.call(input, String(value));
    else input.value = String(value);
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
    return true;
}

function triggerAttachmentAction(index, action) {
    var indexReady = setStudioControlValue('#ffs-attachment-action-index input', index);
    var actionReady = setStudioControlValue(
        '#ffs-attachment-action-name textarea, #ffs-attachment-action-name input', action
    );
    if (!indexReady || !actionReady) return;
    window.setTimeout(function() {
        var run = document.querySelector('#ffs-attachment-action-run button, #ffs-attachment-action-run');
        if (run) run.click();
    }, 60);
}

function ensureLightbox() {
    var lightbox = document.querySelector('#ffs-lightbox');
    if (lightbox) return lightbox;
    lightbox = document.createElement('div');
    lightbox.id = 'ffs-lightbox';
    lightbox.setAttribute('role', 'dialog');
    lightbox.setAttribute('aria-modal', 'true');
    lightbox.setAttribute('aria-label', 'Image preview');
    lightbox.innerHTML = '<button type="button" aria-label="Close image preview">&times;</button><img alt="Full-screen preview">';
    document.body.appendChild(lightbox);
    lightbox.addEventListener('click', function(event) {
        if (event.target === lightbox || event.target.closest('button')) closeLightbox();
    });
    return lightbox;
}

function openLightbox(src, alt) {
    if (!src) return;
    var lightbox = ensureLightbox();
    var image = lightbox.querySelector('img');
    image.src = src;
    image.alt = alt || 'Full-screen preview';
    lightbox.classList.add('ffs-open');
    document.documentElement.style.overflow = 'hidden';
    lightbox.querySelector('button').focus();
}

function closeLightbox() {
    var lightbox = document.querySelector('#ffs-lightbox');
    if (!lightbox) return;
    lightbox.classList.remove('ffs-open');
    document.documentElement.style.overflow = '';
}

function makeAttachmentAction(label, title, index, action, disabled) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'ffs-attachment-action';
    button.textContent = label;
    button.title = title;
    button.setAttribute('aria-label', title);
    button.disabled = Boolean(disabled);
    button.addEventListener('click', function(event) {
        event.preventDefault();
        event.stopPropagation();
        if (action === 'preview') {
            var image = button.closest('.gallery-item').querySelector('img');
            openLightbox(image.currentSrc || image.src, image.alt);
        } else if (action === 'mask') {
            openMaskSettings();
        } else {
            triggerAttachmentAction(index, action);
        }
    });
    return button;
}

function prepareAttachmentCards() {
    var items = Array.from(document.querySelectorAll('#ffs-attachment-preview .gallery-item'));
    /* Read the backend canvas state rendered into the attachment summary. */
    var statusEl = document.querySelector('#ffs-attachment-copy .ffs-attachment-copy');
    var canvasIndex = -1; /* -1 = no canvas */
    if (statusEl && statusEl.dataset.canvasIndex !== undefined) {
        canvasIndex = parseInt(statusEl.dataset.canvasIndex, 10);
        if (Number.isNaN(canvasIndex)) canvasIndex = -1;
    }
    items.forEach(function(item, index) {
        var image = item.querySelector('img');
        if (!image) return;
        var isCanvas = index === canvasIndex;
        var renderKey = [index, items.length, canvasIndex].join(':');
        if (item.dataset.ffsAttachmentKey === renderKey) return;
        item.dataset.ffsAttachmentKey = renderKey;
        item.querySelectorAll('.ffs-attachment-role, .ffs-attachment-actions').forEach(function(node) {
            node.remove();
        });
        image.setAttribute('title', 'Open image ' + (index + 1) + ' full screen');
        var role = document.createElement('span');
        role.className = 'ffs-attachment-role';
        role.textContent = isCanvas ? 'Image ' + (index + 1) + ' / Canvas' : 'Image ' + (index + 1) + ' / Reference';
        item.appendChild(role);

        var actions = document.createElement('div');
        actions.className = 'ffs-attachment-actions';
        actions.appendChild(makeAttachmentAction('[]', 'Open full-screen preview', index, 'preview'));
        actions.appendChild(makeAttachmentAction(isCanvas ? 'C✓' : 'C', isCanvas ? 'Remove canvas (toggle)' : 'Make this image the canvas', index, 'canvas'));
        actions.lastElementChild.textContent = isCanvas ? 'C*' : 'C';
        actions.lastElementChild.title = isCanvas ? 'Remove canvas and use all images as references' : 'Make this image the canvas';
        actions.lastElementChild.setAttribute('aria-label', actions.lastElementChild.title);
        actions.appendChild(makeAttachmentAction('<', 'Move image left', index, 'left', index === 0));
        actions.appendChild(makeAttachmentAction('>', 'Move image right', index, 'right', index === items.length - 1));
        if (isCanvas) actions.appendChild(makeAttachmentAction('M', 'Open canvas mask settings', index, 'mask'));
        actions.appendChild(makeAttachmentAction('x', 'Remove this image', index, 'remove'));
        item.appendChild(actions);
    });
}

function prepareImagePreviews() {
    document.querySelectorAll('.ffs-chat-image img, .ffs-request-attachment img, .ffs-avatar-gallery-tile img').forEach(function(image) {
        image.setAttribute('title', 'Open full-screen preview');
    });
    if (document.documentElement.dataset.ffsPreviewDelegated) return;
    document.documentElement.dataset.ffsPreviewDelegated = 'true';
    document.addEventListener('click', function(event) {
        var image = event.target.closest(
            '#ffs-attachment-preview .gallery-item img, .ffs-chat-image img, .ffs-request-attachment img, .ffs-avatar-gallery-tile img'
        );
        if (!image) return;
        event.preventDefault();
        event.stopPropagation();
        openLightbox(image.currentSrc || image.src, image.alt);
    }, true);
}

var avatarPendingUploads = new Map();
var avatarUploadOwnedDisable = false;
function releaseAvatarUploads() {
    avatarPendingUploads.forEach(function(token, slot) {
        var marker = document.querySelector('#av-extra-ready-' + slot + ' [data-token]');
        if (marker && marker.dataset.token !== token) avatarPendingUploads.delete(slot);
    });
    var compose = document.querySelector('#av-console-compose');
    var send = document.querySelector('#av-console-input button.primary');
    if (avatarPendingUploads.size || !compose || !send) return;
    compose.removeAttribute('data-ffs-uploading');
    send.removeAttribute('aria-busy');
    if (avatarUploadOwnedDisable && !compose.querySelector('.ffs-status-active, .ffs-generation-stage')) send.disabled = false;
    avatarUploadOwnedDisable = false;
}
document.addEventListener('change', function(event) {
    var input = event.target;
    if (!input.matches('.ffs-avatar-extra-row input[type="file"]') || !input.files.length) return;
    var slot = input.closest('#av-console-extra-1') ? '1' : '2';
    var marker = document.querySelector('#av-extra-ready-' + slot + ' [data-token]');
    avatarPendingUploads.set(slot, marker ? marker.dataset.token : '');
    var compose = document.querySelector('#av-console-compose');
    var send = document.querySelector('#av-console-input button.primary');
    if (compose && send) {
        if (!send.disabled) avatarUploadOwnedDisable = true;
        send.disabled = true;
        send.setAttribute('aria-busy', 'true');
        compose.setAttribute('data-ffs-uploading', 'true');
    }
}, true);

function enhanceStudioUI() {
    var attach = document.querySelector('#ffs-attach button, #ffs-attach');
    var generate = document.querySelector('#ffs-generate button, #ffs-generate');
    var fresh = document.querySelector('#ffs-new-session button, #ffs-new-session');
    if (attach) attach.setAttribute('title', 'Attach up to four FLUX reference images');
    if (generate) generate.setAttribute('title', 'Generate image');
    if (fresh) fresh.setAttribute('title', 'Start a new session');

    var maskSettings = document.querySelector('#ffs-open-mask-settings button, #ffs-open-mask-settings');
    if (maskSettings && !maskSettings.dataset.ffsMaskReady) {
        maskSettings.dataset.ffsMaskReady = 'true';
        maskSettings.addEventListener('click', openMaskSettings);
    }

    prepareGenerationSliders();
    prepareThemeToggle();
    prepareSettingsDrawer();
    prepareAttachmentCards();
    prepareImagePreviews();
    releaseAvatarUploads();
    document.querySelectorAll('#av-extra-ready-1, #av-extra-ready-2').forEach(function(marker) {
        if (!marker.dataset.ffsObserved) {
            marker.dataset.ffsObserved = 'true';
            new MutationObserver(releaseAvatarUploads).observe(marker, {childList:true, subtree:true});
        }
    });
    var backToLibrary = document.querySelector('#av-back-library');
    if (backToLibrary) {
        backToLibrary.setAttribute('aria-label', 'All avatars');
        backToLibrary.setAttribute('title', 'All avatars');
    }
    var avatarRoot = document.querySelector('#ffs-avatar-studio');
    if (avatarRoot && avatarRoot.getClientRects().length) {
        var library = document.querySelector('#av-library-view');
        var creation = document.querySelector('#av-creation');
        var view = library && library.getClientRects().length ? 'library' : creation && creation.getClientRects().length ? 'creation' : 'workbench';
        if (document.documentElement.dataset.ffsAvatarView !== view) {
            document.documentElement.dataset.ffsAvatarView = view;
            window.scrollTo({top:0, behavior:'auto'});
        }
    }
    var avatarChat = document.querySelector('#av-console-chat');
    if (avatarChat && !avatarChat.dataset.ffsObserved) {
        avatarChat.dataset.ffsObserved = 'true';
        var scrollAvatarChat = function() {
            window.requestAnimationFrame(function() { avatarChat.scrollTop = avatarChat.scrollHeight; });
        };
        new MutationObserver(scrollAvatarChat).observe(avatarChat, {childList:true, subtree:true});
        new ResizeObserver(scrollAvatarChat).observe(avatarChat);
        avatarChat.addEventListener('load', scrollAvatarChat, true);
        scrollAvatarChat();
    }
    var heading = document.querySelector('#av-guide .av-guide-heading h2');
    if (heading && heading.getClientRects().length && heading.textContent !== document.documentElement.dataset.ffsGuideHeading) {
        document.documentElement.dataset.ffsGuideHeading = heading.textContent;
        var top = heading.closest('.av-guide-heading').getBoundingClientRect().top;
        if (top < 90 || top > 240) {
            window.scrollTo({top: Math.max(0, window.scrollY + top - 104), behavior: 'auto'});
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
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') closeLightbox();
});
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


def _model_capabilities_html(model_name):
    """Render compact capability metadata from the model registry."""
    info = model_manager.MODEL_REGISTRY.get(model_name, {})
    capabilities = set(info.get("capabilities", []))
    labels = ["Text to image"]
    if "img2img" in capabilities:
        labels.append("Image edit")
    if "inpaint" in capabilities:
        labels.append("Mask edit")
    if model_name == model_manager.FLUX_MODEL_NAME:
        labels.append("Up to 4 references")
    chips = "".join(f'<span>{html.escape(label)}</span>' for label in labels)
    description = html.escape(info.get("description", "Image generation"))
    return (
        '<div class="ffs-model-profile">'
        f'<strong>{html.escape(model_name)}</strong>'
        f'<small>{description}</small>'
        f'<div>{chips}</div>'
        '</div>'
    )


def _flux_encoder_status_html():
    config = model_manager.get_flux_encoder_config()
    active = "Custom" if config["mode"] == "custom" else "Official FP4"
    if config["custom_available"]:
        size = config["custom_size"] / 1024**3 if config["custom_size"] else 0
        detail = config["custom_format"].upper() or "CUSTOM"
        if size:
            detail += f" / {size:.2f} GiB"
        custom = f'<span class="ffs-encoder-file">{html.escape(detail)}</span>'
    else:
        custom = '<span class="ffs-encoder-file muted">No custom file configured</span>'
    return (
        '<div class="ffs-encoder-status">'
        f'<strong>{html.escape(active)} active</strong>'
        f'{custom}'
        '</div>'
    )


def _ui_trace(message):
    if DEBUG_MODE:
        print(f"[ui] {message}", flush=True)


def _avatar_first_id():
    selected = avatar_studio.selected_or_first(SAVE_DIR)
    return selected["id"] if selected else None


def _avatar_choices_update(value=None):
    choices = avatar_studio.avatar_choices(SAVE_DIR)
    selected = value or (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=selected)


def _avatar_image_value(avatar, key):
    kind = key.removesuffix("_image")
    path = (avatar.get(f"{kind}_candidate") or (avatar.get(key) if avatar.get(f"{kind}_locked") else None)) if avatar else None
    return path if path and os.path.exists(path) else avatar_guide.draft(avatar, key.removesuffix("_image"))


def _avatar_chat_html(avatar):
    if not avatar:
        return '<div class="ffs-avatar-chat-empty">Create or select an avatar first.</div>'
    turns = []
    for item in avatar.get("chat_history", [])[-40:]:
        role = html.escape(item.get("role", "assistant"))
        content = html.escape(item.get("content", ""))
        image_path = item.get("image_path")
        image_html = ""
        references = []
        for path in (item.get("metadata") or {}).get("extra_references", []):
            if os.path.isfile(path):
                reference_src = "/gradio_api/file=" + quote(os.path.abspath(path).replace("\\", "/"))
                references.append(f'<a class="ffs-chat-image" href="{reference_src}" target="_blank"><img src="{reference_src}" alt="Attached avatar reference"></a>')
        if image_path and os.path.exists(image_path):
            src = "/gradio_api/file=" + quote(os.path.abspath(image_path).replace("\\", "/"))
            image_html = f'<a class="ffs-chat-image" href="{src}" target="_blank"><img src="{src}" alt="Avatar result"></a>'
        turns.append(
            f'<div class="ffs-avatar-chat-turn {role}">'
            f'<strong>{"You" if role == "user" else "Avatar Studio"}</strong>'
            f'<span>{content}</span><div class="av-turn-references">{"".join(references)}</div>{image_html}</div>'
        )
    return '<div class="ffs-avatar-chat-log">' + "".join(turns) + '</div>' if turns else (
        '<div class="ffs-avatar-chat-empty">No creations yet.</div>'
    )


def _avatar_gallery_html(avatar):
    if not avatar:
        return '<div class="ffs-avatar-gallery-empty">No avatar selected.</div>'
    gallery_dir = avatar_studio.avatar_path(SAVE_DIR, avatar["id"]) / "gallery" / "images"
    paths = sorted(gallery_dir.glob("*.png"))
    if not paths:
        return '<div class="ffs-avatar-gallery-empty">Gallery images generated here will be saved separately from chat.</div>'
    tiles = []
    for path in paths[-80:]:
        src = quote(os.path.abspath(path).replace("\\", "/"))
        tiles.append(f'<a class="ffs-avatar-gallery-tile" href="{src}" target="_blank"><img src="{src}" alt="Gallery image"></a>')
    return '<div class="ffs-avatar-gallery-grid">' + "".join(tiles) + '</div>'


def _avatar_available_steps(avatar):
    if avatar_guide.active(avatar):
        return ["Guide", "Console", "Gallery"] if avatar_guide.stage(avatar) == "complete" else ["Guide"]
    steps = (["Guide"] if avatar_guide.active(avatar) else []) + ["Face"]
    if avatar and avatar.get("face_locked"):
        steps.append("Body")
    if avatar and avatar.get("face_locked") and avatar.get("body_locked"):
        steps.extend(["Console", "Gallery"])
    return steps


def _avatar_step_state(avatar, requested=None):
    available = _avatar_available_steps(avatar)
    current = str(requested or (avatar or {}).get("current_step", "face")).title()
    if current not in available:
        current = available[-1]
    return current, gr.update(value=current, choices=available, interactive=True, visible=bool(avatar) and len(available) > 1)


def _avatar_step_visibility(step):
    current = (step or "Face").lower()
    return tuple(gr.update(visible=current == name) for name in ("face", "body", "console", "gallery"))


def _avatar_gallery_value(avatar):
    if not avatar:
        return []
    values = []
    for item in avatar_studio.visible_gallery_items(SAVE_DIR, avatar["id"]):
        path = item.get("generated_path")
        if path and os.path.exists(path):
            status = item.get("status", "saved").replace("_", " ").title()
            score = (item.get("validation") or {}).get("score")
            caption = f"{status} / Score {score}" if score is not None else status
            values.append((path, caption))
    return values


def _avatar_gallery_details(avatar, index=None):
    if not avatar:
        return '<div class="ffs-avatar-gallery-empty">Create or select an avatar first.</div>'
    item = avatar_studio.gallery_item_by_index(SAVE_DIR, avatar["id"], index)
    if not item:
        status = avatar_gallery.configuration_status() if not DEV_MODE else {"ready": True, "message": "Local preview uses mock search and validation."}
        return (
            '<div class="ffs-avatar-gallery-empty">'
            f'<strong>{html.escape(status["message"])}</strong>'
            '<span>Select a generated image to inspect or regenerate it.</span>'
            '</div>'
        )
    validation = item.get("validation") or {}
    reasons = validation.get("reasons") or [validation.get("reason")] if validation else []
    reasons = [str(reason) for reason in reasons if reason]
    return (
        '<div class="ffs-avatar-gallery-selection">'
        f'<strong>{html.escape(item.get("status", "saved").replace("_", " ").title())}</strong>'
        f'<span>Attempt {int(item.get("attempt", 1))} / Score {html.escape(str(validation.get("score", "not checked")))}</span>'
        f'<p>{html.escape(item.get("prompt", ""))}</p>'
        f'<small>{html.escape("; ".join(reasons) or "No validation problems recorded.")}</small>'
        '</div>'
    )


def _avatar_ui_payload(avatar_id=None, step=None, *, creating=False, library_view=False, refresh_library=True):
    avatar = None if creating else avatar_studio.selected_or_first(SAVE_DIR, avatar_id)
    current, navigation = _avatar_step_state(avatar, step)
    ready = bool(avatar and avatar.get("face_locked") and avatar.get("body_locked"))
    guide_stage = avatar_guide.stage(avatar)
    face_stage = guide_stage.startswith("face")
    guide_kind = "face" if face_stage else "body"
    face_prompt, body_prompt, scene_one, scene_two = avatar_guide.options(avatar)
    items = avatar_studio.load_gallery_items(SAVE_DIR, avatar["id"]) if avatar else []
    starters = [(item["generated_path"], item.get("status", "saved").replace("_", " ").title())
                for item in items if item.get("origin") == "guided" and os.path.isfile(item.get("generated_path") or "")]
    face_value = _avatar_image_value(avatar, "face_image")
    body_value = _avatar_image_value(avatar, "body_image")
    api_ready = DEV_MODE or avatar_gallery.configuration_status()["ready"]
    gallery_data = avatar_gallery_jobs.safe_snapshot(SAVE_DIR, avatar["id"] if avatar else None)
    pending_batch = gallery_data.get("job") and gallery_data["job"]["status"] != "complete"
    values = {
        selected_avatar_id: avatar["id"] if avatar else None,
        avatar_selector: gr.update(choices=avatar_studio.avatar_choices(SAVE_DIR), value=avatar["id"] if avatar else None, visible=bool(avatar), interactive=True),
        avatar_back_library: gr.update(interactive=True),
        avatar_status: avatar_studio.status_html(avatar),
        avatar_step: navigation,
        face_image: face_value,
        body_image: body_value,
        face_details: (avatar or {}).get("face_prompt", face_prompt),
        body_details: (avatar or {}).get("body_prompt", body_prompt),
        face_specs: avatar_studio.specs_html((avatar or {}).get("face_specs"), "Face Specs"),
        body_specs: avatar_studio.specs_html((avatar or {}).get("body_specs"), "Body Specs"),
        lock_face_btn: gr.update(interactive=face_value is not None),
        lock_body_btn: gr.update(interactive=body_value is not None),
        avatar_chat_display: _avatar_chat_html(avatar),
        avatar_gallery_display: _avatar_gallery_value(avatar),
        avatar_gallery_details: _avatar_gallery_details(avatar),
        selected_avatar_gallery_index: None,
        regenerate_gallery_btn: gr.update(interactive=False),
        gallery_rejection_reason: gr.update(value="", interactive=True),
        gallery_plan_btn: gr.update(interactive=ready and api_ready, value="Resume batch" if pending_batch else "Generate gallery"),
        gallery_pipeline_status: avatar_design.gallery_pipeline(gallery_data),
        gallery_reference_display: gallery_data["images"],
        gallery_reference_table: avatar_design.reference_table(gallery_data),
        gallery_reference_filter: "All findings",
        gallery_poll_cache: {},
        gallery_new_search: gr.update(interactive=bool(pending_batch and not gallery_data["job"].get("item_ids"))),
        gallery_require_face: gr.update(value=gallery_data["job"]["require_face"] if pending_batch else False),
        gallery_quantity: gr.update(value=gallery_data["job"]["quantity"]) if pending_batch else gr.update(),
        gallery_aspect: gr.update(value=gallery_data["job"]["aspect"]) if pending_batch else gr.update(),
        gallery_theme: gr.update(value=gallery_data["job"]["theme"]) if pending_batch else gr.update(),
        avatar_face_step: gr.update(visible=bool(avatar) and current == "Face"),
        avatar_body_step: gr.update(visible=bool(avatar) and current == "Body"),
        avatar_console_step: gr.update(visible=bool(avatar) and current == "Console"),
        avatar_gallery_step: gr.update(visible=bool(avatar) and current == "Gallery"),
        avatar_dashboard: gr.update(value=avatar_design.dashboard(avatar), visible=bool(avatar)),
        avatar_creation: gr.update(visible=creating),
        avatar_cancel_new: gr.update(visible=True),
        avatar_guide_group: gr.update(visible=bool(avatar) and current == "Guide"),
        avatar_library_view: gr.update(visible=library_view),
        avatar_workspace_layout: gr.update(visible=not library_view),
        guide_heading: avatar_design.guide_header(avatar),
        guide_context: avatar_design.reference_context(avatar, guide_kind),
        guide_brief: avatar_studio.build_reference_prompt(guide_kind, face_prompt if face_stage else body_prompt, avatar),
        face_brief: avatar_studio.build_reference_prompt("face", (avatar or {}).get("face_prompt", face_prompt), avatar),
        body_brief: avatar_studio.build_reference_prompt("body", (avatar or {}).get("body_prompt", body_prompt), avatar),
        body_context: avatar_design.reference_context(avatar, "body"),
        guide_image: gr.update(value=avatar_guide.draft(avatar, guide_kind), label="Face draft" if face_stage else "Body draft"),
        guide_face_prompt: gr.update(value=face_prompt, visible=face_stage),
        guide_body_prompt: gr.update(value=body_prompt, visible=not face_stage),
        guide_scene_one: scene_one,
        guide_scene_two: scene_two,
        guide_scenes: gr.update(visible=guide_stage in {"body_review", "starters"}),
        guide_next_btn: gr.update(value=avatar_guide.next_label(avatar), visible=guide_stage != "complete", interactive=True),
        guide_regenerate_btn: gr.update(value="Try another face" if face_stage else "Try another body", visible=guide_stage in {"face_review", "body_review"}, interactive=True),
        guide_upload: gr.update(visible=guide_stage not in {"starters", "complete"}, interactive=True),
        guide_back_btn: gr.update(visible=guide_stage not in {"face", "face_review"}, interactive=True),
        guide_custom_btn: gr.update(interactive=True),
        guide_stop_btn: gr.update(visible=False),
        guide_results: gr.update(value=starters, visible=bool(starters)),
        guide_console_btn: gr.update(visible=guide_stage == "complete"),
        guide_gallery_btn: gr.update(visible=guide_stage == "complete"),
        guide_work: gr.update(visible=guide_stage not in {"starters", "complete"}),
        guide_status: "",
        avatar_rename_panel: gr.update(visible=False),
        avatar_rename_name: (avatar or {}).get("name", ""),
        avatar_extra_1: None,
        avatar_extra_2: None,
        avatar_extra_panel: gr.update(label="Additional references (optional)"),
        avatar_prompt: "",
        avatar_name_input: gr.update(),
        avatar_creation_mode: gr.update(),
        avatar_preset: gr.update(),
        avatar_rename_btn: gr.update(visible=bool(avatar), interactive=True),
        avatar_task_status: gr.update(value="", visible=False),
        avatar_stop_btn: gr.update(interactive=False, visible=False),
        avatar_generate_btn: gr.update(interactive=ready, visible=True),
        gallery_stop_btn: gr.update(interactive=False, visible=False),
        generate_face_btn: gr.update(interactive=bool(avatar)),
        generate_body_btn: gr.update(interactive=bool(avatar and avatar.get("face_locked"))),
        avatar_console_status: "",
        gallery_status: "" if api_ready else _status_html("error", avatar_gallery.configuration_status()["message"]),
        gallery_selected_panel: gr.update(visible=False),
    }
    if refresh_library:
        avatars = avatar_design.library(SAVE_DIR)
        values.update({
            avatar_library: gr.update(value=avatar_design.library_values(SAVE_DIR), visible=bool(avatars)),
            avatar_library_ids: [item["id"] for item in avatars],
            avatar_search: "",
            avatar_filter: "All avatars",
            avatar_sort: "Recent",
            avatar_library_empty: gr.update(value=avatar_design.library_empty(SAVE_DIR), visible=not bool(avatars)),
        })
    return values


_initial_chat_history = _load_chat_history()
_initial_avatar = avatar_studio.selected_or_first(SAVE_DIR)
_initial_avatar_step = str((_initial_avatar or {}).get("current_step", "face")).title()
if _initial_avatar_step not in _avatar_available_steps(_initial_avatar):
    _initial_avatar_step = _avatar_available_steps(_initial_avatar)[-1]
_ui_trace("building interface")
with gr.Blocks(title="FreeFakeStudio") as demo:

    # ── Session State ──────────────────────────────────────
    chat_history = gr.State(_initial_chat_history)  # rendered conversation turns
    attached_image = gr.State(None)       # PIL Image or None
    canvas_index = gr.State(0)            # which image is the canvas (None = all references)
    last_gen_settings = gr.State(None)    # for Regenerate
    selected_result_idx = gr.State(0)
    latest_result_paths = gr.State([])
    selected_avatar_id = gr.State(_avatar_first_id())

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
        workspace_tab = gr.Radio(
            choices=["Create", "Avatar Studio"],
            value="Create",
            show_label=False,
            container=False,
            scale=0,
            min_width=230,
            elem_id="ffs-workspace-tabs",
        )
        settings_btn = gr.Button(
            "⚙",
            size="sm",
            variant="secondary",
            scale=0,
            min_width=40,
            elem_id="ffs-settings-toggle",
        )
        theme_btn = gr.Button(
            "◐",
            size="sm",
            variant="secondary",
            scale=0,
            min_width=40,
            elem_id="ffs-theme-toggle",
        )
        new_chat_btn = gr.Button(
            "New", size="sm", variant="secondary", scale=0,
            min_width=78, elem_id="ffs-new-session",
        )

    # ═══════════════════════════════════════════════════════
    # MAIN CONTENT AREA
    # ═══════════════════════════════════════════════════════
    with gr.Column(elem_classes="ffs-settings", elem_id="ffs-settings-panel") as settings_panel:
        gr.HTML(
            '<div class="ffs-settings-heading">'
            '<span>Generation controls</span>'
            '<button type="button" class="ffs-settings-close" aria-label="Close settings">×</button>'
            '</div>',
            elem_id="ffs-settings-heading",
        )
        with gr.Column(elem_classes="ffs-settings-panel"):
            model_capabilities_display = gr.HTML(
                _model_capabilities_html("Z-Image Turbo"),
                elem_id="ffs-model-profile",
            )
            with gr.Column(
                visible=False,
                scale=0,
                elem_classes="ffs-flux-encoder-panel",
            ) as flux_encoder_panel:
                flux_encoder_choice = gr.Radio(
                    choices=model_manager.get_flux_encoder_choices(),
                    value=(
                        "Custom"
                        if model_manager.get_flux_encoder_config()["mode"] == "custom"
                        else "Official"
                    ),
                    label="FLUX Text Encoder",
                )
                flux_encoder_status = gr.HTML(
                    _flux_encoder_status_html(),
                    elem_id="ffs-flux-encoder-status",
                )
                apply_flux_encoder_btn = gr.Button(
                    "Apply encoder",
                    size="sm",
                    variant="secondary",
                    elem_id="ffs-apply-encoder",
                )
            aspect_ratio = gr.Dropdown(
                ASPECTS,
                value="1024x1024 (1:1)",
                label="Aspect Ratio",
            )
            num_images = gr.Slider(1, 8, value=1, step=1, label="Images")
            gen_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
            gen_steps = gr.Slider(1, 8, value=8, step=1, label="Steps")
            gen_cfg = gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="CFG")
            gen_denoise = gr.Slider(
                0.1, 1.0, value=1.0, step=0.05,
                label="Denoise", visible=False,
            )
            negative_prompt = gr.Textbox(
                DEFAULT_NEG,
                label="Negative Prompt",
                lines=2,
            )
            with gr.Column(
                visible=False,
                scale=0,
                elem_classes="ffs-image-edit-panel",
                elem_id="ffs-edit-panel",
            ) as edit_panel:
                gr.HTML(
                    '<div class="ffs-section-heading">Image edit'
                    '<small>Mask tools apply to image 1, the canvas.</small></div>',
                    elem_id="ffs-edit-heading",
                )
                mask_mode = gr.Radio(
                    choices=[
                        ("Smart", "Smart"),
                        ("Paint", "Manual Paint"),
                        ("Background", "Background Only"),
                        ("Protect face", "Everything Except Face"),
                    ],
                    value="Smart",
                    label="Edit area",
                )
                with gr.Column(visible=False, scale=0) as manual_mask_group:
                    mask_editor = gr.ImageEditor(
                        label="Paint area to change",
                        type="pil",
                        height=280,
                        canvas_size=(768, 768),
                        brush=gr.Brush(
                            colors=["#ffffff"],
                            default_size=40,
                            default_color="#ffffff",
                        ),
                        eraser=gr.Eraser(default_size=40),
                        sources=["upload"],
                        transforms=[],
                        layers=False,
                    )
                with gr.Column(visible=False, scale=0) as auto_mask_group:
                    mask_preview = gr.Image(
                        label="Area to change",
                        height=190,
                        interactive=False,
                    )
                    edit_mask_btn = gr.Button(
                        "Paint this mask",
                        size="sm",
                        variant="secondary",
                    )

    with gr.Column(elem_classes="ffs-chat-area", elem_id="ffs-workspace") as create_workspace:

        gr.HTML(
            '<div class="ffs-canvas-heading"><span>Create</span><i></i></div>',
            elem_id="ffs-canvas-heading",
        )

        conversation_display = gr.HTML(
            value=_render_history(_initial_chat_history),
            elem_classes="ffs-history",
            elem_id="ffs-conversation",
        )

        # Status display (streaming updates)
        status_display = gr.HTML(
            value=(
                '<div class="ffs-notice ffs-notice-success">Saved conversation restored</div>'
                if _initial_chat_history else """
            <div class="ffs-empty">
                <div class="ffs-empty-icon">FF</div>
                <div class="ffs-empty-title">What will you make?</div>
                <div class="ffs-empty-sub">Portrait. Product. Editorial. Concept.</div>
            </div>
            """
            ),
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
            add_to_prompt_btn = gr.Button("Use result as input", size="sm", visible=False)
            regenerate_btn = gr.Button("Regenerate", size="sm")
            seed_display = gr.Textbox(
                interactive=False, visible=False, show_label=False,
                container=False, elem_id="ffs-seed",
            )

    # ═══════════════════════════════════════════════════════
    # SETTINGS PANEL
    # ═══════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════
    # COMPOSER (bottom bar)
    # ═══════════════════════════════════════════════════════
    with gr.Row(elem_id="ffs-composer-dock") as composer_dock:
        with gr.Column(elem_id="ffs-composer-inner"):
            with gr.Row(visible=False, elem_id="ffs-attachment-row") as attachment_row:
                with gr.Column(elem_id="ffs-attachment-manager"):
                    attachment_display = gr.Gallery(
                        show_label=False,
                        columns=4,
                        rows=1,
                        height=112,
                        object_fit="cover",
                        preview=False,
                        allow_preview=False,
                        elem_id="ffs-attachment-preview",
                    )
                    with gr.Row(elem_id="ffs-attachment-footer"):
                        attachment_summary = gr.HTML(
                            "",
                            elem_id="ffs-attachment-copy",
                        )
                        open_mask_settings_btn = gr.Button(
                            "Edit canvas",
                            size="sm",
                            variant="secondary",
                            elem_id="ffs-open-mask-settings",
                        )
                        clear_attachments_btn = gr.Button(
                            "Clear all",
                            size="sm",
                            variant="secondary",
                            elem_id="ffs-clear-attachments",
                        )
                attachment_action_index = gr.Number(
                    value=0,
                    precision=0,
                    show_label=False,
                    container=False,
                    elem_id="ffs-attachment-action-index",
                    elem_classes="ffs-internal-control",
                )
                attachment_action_name = gr.Textbox(
                    value="",
                    show_label=False,
                    container=False,
                    elem_id="ffs-attachment-action-name",
                    elem_classes="ffs-internal-control",
                )
                attachment_action_btn = gr.Button(
                    "Apply attachment action",
                    elem_id="ffs-attachment-action-run",
                    elem_classes="ffs-internal-control",
                )
            with gr.Row(elem_id="ffs-composer"):
                attach_btn = gr.UploadButton(
                    "+",
                    file_types=["image"],
                    file_count="multiple",
                    size="sm",
                    min_width=40,
                    visible=False,
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

    with gr.Column(visible=False, elem_classes="ffs-avatar-studio", elem_id="ffs-avatar-studio") as avatar_workspace:
        with gr.Column(elem_id="av-library-view") as avatar_library_view:
            with gr.Row(elem_id="av-library-top"):
                gr.HTML('<div class="av-library-title"><div><span class="av-eyebrow">CHARACTER WORKSPACE</span><h1>Avatar Studio</h1><p>Your identities, references, and creations.</p></div></div>')
                avatar_new_btn = gr.Button("New avatar", icon=avatar_design.icon("plus"), elem_id="av-new", variant="primary", scale=0)
            with gr.Row(elem_id="av-library-tools"):
                avatar_search = gr.Textbox(placeholder="Search avatars", label="Search", show_label=False, container=False, scale=3)
                avatar_filter = gr.Dropdown(choices=["All avatars", "Ready", "In progress"], value="All avatars", label="Status", scale=1)
                avatar_sort = gr.Dropdown(choices=["Recent", "Name"], value="Recent", label="Sort", scale=1)
                refresh_avatar_btn = gr.Button("Refresh", icon=avatar_design.icon("refresh-cw"), size="sm", variant="secondary", scale=0)
            avatar_library_ids = gr.State([item["id"] for item in avatar_design.library(SAVE_DIR)])
            avatar_library = gr.Gallery(
                value=avatar_design.library_values(SAVE_DIR), columns=4, height="auto",
                visible=bool(_initial_avatar), show_label=False, allow_preview=False,
                interactive=False, object_fit="cover", buttons=[], elem_id="av-library",
            )
            avatar_library_empty = gr.HTML(avatar_design.library_empty(SAVE_DIR), visible=not bool(_initial_avatar))
            avatar_status = gr.HTML(avatar_studio.status_html(_initial_avatar), visible=False)

        with gr.Column(visible=False, elem_id="ffs-avatar-layout") as avatar_workspace_layout:
            with gr.Column(scale=5, elem_id="ffs-avatar-main"):
                with gr.Row(elem_id="av-workspace-tools"):
                    avatar_back_library = gr.Button("All avatars", icon=avatar_design.icon("arrow-left"), size="sm", scale=0, elem_id="av-back-library")
                    avatar_dashboard = gr.HTML(avatar_design.dashboard(_initial_avatar), elem_id="av-dashboard")
                    avatar_selector = gr.Dropdown(choices=avatar_studio.avatar_choices(SAVE_DIR), value=_avatar_first_id(), label="Switch avatar", show_label=False, container=False, scale=0, elem_id="av-switch")
                    avatar_rename_btn = gr.Button("Rename", size="sm", scale=0, elem_id="av-rename")
                with gr.Group(visible=False, elem_id="av-rename-panel") as avatar_rename_panel:
                    avatar_rename_name = gr.Textbox(label="Avatar name")
                    with gr.Row():
                        avatar_rename_save = gr.Button("Save name", variant="primary")
                        avatar_rename_cancel = gr.Button("Cancel")
                with gr.Group(visible=not bool(_initial_avatar), elem_id="av-creation") as avatar_creation:
                    gr.HTML('<h2>Create an avatar</h2>')
                    avatar_creation_mode = gr.Radio(
                        choices=[("Guided setup", "Guided"), ("Custom avatar", "Custom")],
                        value="Guided", label="Start your way", elem_id="av-creation-mode",
                    )
                    with gr.Row():
                        avatar_name_input = gr.Textbox(label="Avatar name", placeholder="e.g. Maya", lines=1, scale=2)
                        avatar_preset = gr.Dropdown(choices=list(avatar_guide.PRESETS), value="Studio creator", label="Starting style", scale=1)
                    with gr.Row():
                        create_avatar_btn = gr.Button("Create avatar", variant="primary")
                        avatar_cancel_new = gr.Button("Cancel", variant="secondary")
                avatar_task_status = gr.HTML("", visible=False, elem_id="av-task-status")
                avatar_step = gr.Radio(
                    choices=_avatar_available_steps(_initial_avatar),
                    value=_initial_avatar_step,
                    show_label=False, interactive=True, visible=bool(_initial_avatar) and len(_avatar_available_steps(_initial_avatar)) > 1, elem_id="av-navigation",
                )
                with gr.Group(visible=_initial_avatar_step == "Guide", elem_id="av-guide") as avatar_guide_group:
                    guide_heading = gr.HTML(avatar_design.guide_header(_initial_avatar))
                    with gr.Row(visible=avatar_guide.stage(_initial_avatar) not in {"complete", "starters"}, elem_id="av-guide-work") as guide_work:
                        with gr.Column(scale=3, elem_id="av-guide-canvas"):
                            guide_image = gr.Image(
                                value=avatar_guide.draft(_initial_avatar, "face" if avatar_guide.stage(_initial_avatar).startswith("face") else "body"),
                                label="Identity draft", type="pil", sources=["upload"], height=450, elem_id="av-guide-image",
                            )
                        with gr.Column(scale=1, elem_id="av-guide-prompts"):
                            guide_context = gr.HTML(avatar_design.reference_context(_initial_avatar, "face"))
                            guide_face_prompt = gr.Textbox(label="Face direction", value=avatar_guide.options(_initial_avatar)[0], lines=6)
                            guide_body_prompt = gr.Textbox(label="Body & outfit direction", value=avatar_guide.options(_initial_avatar)[1], lines=6)
                            with gr.Row(elem_id="av-preview-tools"):
                                guide_regenerate_btn = gr.Button("Try another face", icon=avatar_design.icon("sparkles"), size="sm")
                                guide_upload = gr.UploadButton("Upload image", file_types=["image"], type="filepath", size="sm")
                            with gr.Accordion("Generation brief", open=False, elem_id="av-guide-brief"):
                                guide_brief = gr.Textbox(label="Resolved FLUX prompt", interactive=False, lines=8)
                            guide_custom_btn = gr.Button("Use custom controls", size="sm", elem_id="av-custom-handoff")
                    with gr.Accordion("Your first two scenes", open=False, visible=False) as guide_scenes:
                        guide_scene_one = gr.Textbox(label="Portrait", value=avatar_guide.options(_initial_avatar)[2], lines=3)
                        guide_scene_two = gr.Textbox(label="Full-body scene", value=avatar_guide.options(_initial_avatar)[3], lines=3)
                    guide_status = gr.HTML("", elem_id="av-guide-status")
                    guide_results = gr.Gallery(
                        value=[item for item in _avatar_gallery_value(_initial_avatar)][:2] if avatar_guide.stage(_initial_avatar) == "complete" else [],
                        label="Starter collection", show_label=False, columns=2, height="auto",
                        object_fit="contain", buttons=["download", "fullscreen"], elem_id="av-guide-results",
                        visible=avatar_guide.stage(_initial_avatar) == "complete",
                    )
                    with gr.Row(elem_id="av-guide-actions"):
                        guide_back_btn = gr.Button("Back", icon=avatar_design.icon("arrow-left"), size="sm", scale=0)
                        guide_stop_btn = gr.Button("Stop", variant="stop", visible=False, scale=0)
                        guide_next_btn = gr.Button(avatar_guide.next_label(_initial_avatar), icon=avatar_design.icon("arrow-right"), variant="primary", scale=2, elem_id="av-guide-next", visible=avatar_guide.stage(_initial_avatar) != "complete")
                        guide_console_btn = gr.Button("Open avatar console", variant="primary", visible=avatar_guide.stage(_initial_avatar) == "complete", scale=2)
                        guide_gallery_btn = gr.Button("View gallery", visible=False, scale=1)
                with gr.Group(
                    visible=bool(_initial_avatar) and _initial_avatar_step == "Face",
                    elem_classes="ffs-avatar-step",
                ) as avatar_face_step:
                    gr.HTML(
                        '<div class="ffs-avatar-step-title"><strong>1. Face Reference</strong>'
                        '<span>Generate or upload a clean headshot, then lock it.</span></div>'
                    )
                    with gr.Row(elem_classes="ffs-avatar-reference-row"):
                        face_image = gr.Image(value=_avatar_image_value(_initial_avatar, "face_image"), label="Face candidate", type="pil", sources=["upload"], height=450, scale=3)
                        with gr.Column(scale=1, elem_classes="av-direction-panel"):
                            gr.HTML(avatar_design.reference_context(None, "face"))
                            face_details = gr.Textbox(label="Face details", placeholder="Example: dark hair, soft makeup, oval face", lines=6)
                            with gr.Row(elem_classes="ffs-avatar-action-row"):
                                generate_face_btn = gr.Button("Generate face", icon=avatar_design.icon("sparkles"), variant="secondary")
                                lock_face_btn = gr.Button("Confirm face & next", icon=avatar_design.icon("arrow-right"), variant="primary", interactive=bool(_avatar_image_value(_initial_avatar, "face_image")))
                            with gr.Accordion("Generation brief", open=False):
                                face_brief = gr.Textbox(label="Resolved face prompt", interactive=False, lines=8)
                    with gr.Accordion("Face identity details", open=False):
                        face_specs = gr.HTML(avatar_studio.specs_html((_initial_avatar or {}).get("face_specs"), "Face Specs"))

                with gr.Group(
                    visible=_initial_avatar_step == "Body",
                    elem_classes="ffs-avatar-step",
                ) as avatar_body_step:
                    gr.HTML(
                        '<div class="ffs-avatar-step-title"><strong>2. Body Reference</strong>'
                        '<span>Generate or upload a clean full-body reference, then lock it.</span></div>'
                    )
                    with gr.Row(elem_classes="ffs-avatar-reference-row"):
                        body_image = gr.Image(value=_avatar_image_value(_initial_avatar, "body_image"), label="Body candidate", type="pil", sources=["upload"], height=450, scale=3)
                        with gr.Column(scale=1, elem_classes="av-direction-panel"):
                            body_context = gr.HTML(avatar_design.reference_context(_initial_avatar, "body"))
                            body_details = gr.Textbox(label="Body details", placeholder="Example: full body, neutral fitted outfit, clear proportions", lines=6)
                            with gr.Row(elem_classes="ffs-avatar-action-row"):
                                generate_body_btn = gr.Button("Generate body", icon=avatar_design.icon("sparkles"), variant="secondary")
                                lock_body_btn = gr.Button("Confirm body & next", icon=avatar_design.icon("arrow-right"), variant="primary", interactive=bool(_avatar_image_value(_initial_avatar, "body_image")))
                            with gr.Accordion("Generation brief", open=False):
                                body_brief = gr.Textbox(label="Resolved body prompt", interactive=False, lines=8)
                    with gr.Accordion("Body identity details", open=False):
                        body_specs = gr.HTML(avatar_studio.specs_html((_initial_avatar or {}).get("body_specs"), "Body Specs"))

                with gr.Group(
                    visible=_initial_avatar_step == "Console",
                    elem_classes="ffs-avatar-step",
                ) as avatar_console_step:
                    gr.HTML(
                        '<div class="ffs-avatar-step-title"><strong>Avatar Console</strong></div>'
                    )
                    avatar_chat_display = gr.HTML(_avatar_chat_html(avatar_studio.selected_or_first(SAVE_DIR)), elem_id="av-console-chat")
                    with gr.Column(elem_id="av-console-compose"):
                        with gr.Accordion("Generation settings", open=False):
                            with gr.Row(elem_classes="ffs-avatar-control-row"):
                                avatar_mode = gr.Dropdown(choices=["Identity Strict", "Outfit Focus", "Pose Focus", "Scene Focus", "Group Image"], value="Identity Strict", label="Mode")
                                avatar_aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect")
                                avatar_steps = gr.Slider(1, 8, value=4, step=1, label="Steps")
                        with gr.Accordion("Additional references (optional)", open=False) as avatar_extra_panel:
                            with gr.Row(elem_classes="ffs-avatar-extra-row"):
                                avatar_extra_1 = gr.Image(label="Extra reference 1", type="pil", sources=["upload"], height=180, elem_id="av-console-extra-1")
                                avatar_extra_2 = gr.Image(label="Extra reference 2", type="pil", sources=["upload"], height=180, elem_id="av-console-extra-2")
                            avatar_extra_ready_1 = gr.HTML('<span data-token="initial"></span>', elem_id="av-extra-ready-1", elem_classes="av-internal")
                            avatar_extra_ready_2 = gr.HTML('<span data-token="initial"></span>', elem_id="av-extra-ready-2", elem_classes="av-internal")
                        avatar_console_status = gr.HTML("")
                        with gr.Row(elem_id="av-console-input"):
                            avatar_prompt = gr.Textbox(label="Prompt", show_label=False, container=False, placeholder="Describe the scene, outfit, pose, or action...", lines=2, max_lines=5, scale=8)
                            avatar_generate_btn = gr.Button("Generate image", icon=avatar_design.icon("sparkles"), variant="primary", scale=0)
                            avatar_stop_btn = gr.Button("Stop", variant="stop", scale=0, visible=False)

                with gr.Group(
                    visible=_initial_avatar_step == "Gallery",
                    elem_classes="ffs-avatar-step",
                ) as avatar_gallery_step:
                    gr.HTML(
                        '<div class="ffs-avatar-step-title"><strong>Auto Gallery</strong></div>'
                    )
                    with gr.Row(elem_classes="ffs-avatar-gallery-controls"):
                        gallery_quantity = gr.Slider(1, 20, value=6, step=1, label="Quantity")
                        gallery_aspect = gr.Dropdown(ASPECTS, value="864x1152 (3:4)", label="Aspect")
                        gallery_theme = gr.Textbox(label="Gallery theme", placeholder="Street fashion in daylight")
                    with gr.Row(elem_id="av-gallery-actions"):
                        gallery_plan_btn = gr.Button("Generate gallery", icon=avatar_design.icon("sparkles"), variant="primary", scale=0, min_width=176)
                        gallery_stop_btn = gr.Button("Stop", variant="stop", scale=0, visible=False)
                        gallery_new_search = gr.Button("New search", variant="secondary", scale=0, interactive=False)
                    gallery_require_face = gr.Checkbox(value=False, label="Require visible face")
                    gallery_status = gr.HTML("" if DEV_MODE or avatar_gallery.configuration_status()["ready"] else _status_html("error", avatar_gallery.configuration_status()["message"]))
                    gallery_pipeline_status = gr.HTML("")
                    with gr.Tabs(selected="gallery-results") as gallery_views:
                        with gr.Tab("Results", id="gallery-results"):
                            with gr.Row(elem_id="av-gallery-work"):
                                with gr.Column(scale=4):
                                    avatar_gallery_display = gr.Gallery(
                                        value=_avatar_gallery_value(_initial_avatar), show_label=False,
                                        columns=4, height="auto", object_fit="cover", preview=False,
                                        allow_preview=True, buttons=["download", "fullscreen"], elem_id="ffs-avatar-gallery",
                                    )
                                with gr.Column(scale=1, visible=False, elem_id="av-gallery-inspector") as gallery_selected_panel:
                                    avatar_gallery_details = gr.HTML(_avatar_gallery_details(_initial_avatar))
                                    gallery_rejection_reason = gr.Textbox(
                                        label="What should change?", placeholder="Wrong face, outfit, pose, or another issue",
                                        lines=2, max_lines=4,
                                    )
                                    regenerate_gallery_btn = gr.Button("Mark Selected Failed and Regenerate", variant="secondary", interactive=False)
                        with gr.Tab("References", id="gallery-references"):
                            gallery_reference_filter = gr.Radio(
                                ["All findings", "Approved", "Rejected", "Awaiting checks"],
                                value="All findings", label="Reference status",
                            )
                            gallery_reference_display = gr.Gallery(columns=4, height="auto", show_label=False,
                                                                   elem_id="av-reference-gallery")
                            gallery_reference_table = gr.HTML("")
                    gallery_poll_cache = gr.State({})
                    selected_avatar_gallery_index = gr.State(None)
    gallery_poll = gr.Timer(2.0, active=False)
    # ═══════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════

    # ── Attachment handling ─────────────────────────────────
    def switch_workspace_tab(tab_name):
        avatar_open = tab_name == "Avatar Studio"
        return (
            gr.update(visible=not avatar_open),
            gr.update(visible=not avatar_open),
            gr.update(visible=not avatar_open),
            gr.update(visible=avatar_open),
            gr.update(active=avatar_open),
        )

    workspace_tab.change(
        switch_workspace_tab,
        inputs=[workspace_tab],
        outputs=[settings_panel, create_workspace, composer_dock, avatar_workspace, gallery_poll],
        show_progress="hidden",
    )

    avatar_outputs = list(_avatar_ui_payload(_avatar_first_id()).keys())

    def poll_gallery_pipeline(avatar_id, filter_value, previous, step):
        if not avatar_id or step != "Gallery":
            return [gr.skip()] * 6
        data = avatar_gallery_jobs.safe_snapshot(SAVE_DIR, avatar_id, filter_value)
        summary = json.dumps({"job": data["job"], "counts": data["counts"], "error": data.get("error")}, sort_keys=True)
        refs = json.dumps(data["rows"], sort_keys=True)
        job = data.get("job") or {}
        label = "Resume batch" if job and job.get("status") != "complete" else "Generate gallery"
        previous = previous or {}
        return (
            avatar_design.gallery_pipeline(data) if previous.get("summary") != summary else gr.skip(),
            gr.update(value=data["images"]) if previous.get("refs") != refs else gr.skip(),
            avatar_design.reference_table(data) if previous.get("refs") != refs else gr.skip(),
            gr.update(value=label) if previous.get("label") != label else gr.skip(),
            {"summary": summary, "refs": refs, "label": label},
            gr.update(interactive=bool(job.get("status") in {"paused", "stopped"} and not job.get("item_ids"))),
        )

    for event in (gallery_poll.tick, gallery_reference_filter.change):
        event(poll_gallery_pipeline, inputs=[selected_avatar_id, gallery_reference_filter, gallery_poll_cache, avatar_step],
              outputs=[gallery_pipeline_status, gallery_reference_display, gallery_reference_table, gallery_plan_btn, gallery_poll_cache, gallery_new_search],
              queue=False, show_progress="hidden")

    def start_new_gallery_search(avatar_id):
        avatar_gallery_jobs.new_search(SAVE_DIR, avatar_id)
        return _avatar_ui_payload(avatar_id, "Gallery")

    gallery_new_search.click(start_new_gallery_search, inputs=[selected_avatar_id], outputs=avatar_outputs, show_progress="hidden")

    def avatar_reference_ready(image, other_image):
        count = int(image is not None) + int(other_image is not None)
        label = f"Additional references ({count}/2)" if count else "Additional references (optional)"
        return f'<span data-token="{uuid.uuid4().hex}"></span>', gr.update(label=label)

    avatar_extra_1.change(avatar_reference_ready, inputs=[avatar_extra_1, avatar_extra_2], outputs=[avatar_extra_ready_1, avatar_extra_panel], show_progress="hidden")
    avatar_extra_2.change(avatar_reference_ready, inputs=[avatar_extra_2, avatar_extra_1], outputs=[avatar_extra_ready_2, avatar_extra_panel], show_progress="hidden")

    def update_guide_brief(avatar_id, face, body):
        avatar = avatar_studio.selected_or_first(SAVE_DIR, avatar_id)
        kind = "face" if avatar_guide.stage(avatar).startswith("face") else "body"
        return avatar_studio.build_reference_prompt(kind, face if kind == "face" else body, avatar)

    for direction in (guide_face_prompt, guide_body_prompt):
        direction.input(update_guide_brief, inputs=[selected_avatar_id, guide_face_prompt, guide_body_prompt], outputs=[guide_brief], show_progress="hidden", trigger_mode="always_last")
    face_details.input(lambda details: avatar_studio.build_reference_prompt("face", details), inputs=[face_details], outputs=[face_brief], show_progress="hidden", trigger_mode="always_last")
    body_details.input(lambda avatar_id, details: avatar_studio.build_reference_prompt("body", details, avatar_studio.selected_or_first(SAVE_DIR, avatar_id)), inputs=[selected_avatar_id, body_details], outputs=[body_brief], show_progress="hidden", trigger_mode="always_last")

    def change_avatar_step(avatar_id, step):
        avatar = avatar_studio.selected_or_first(SAVE_DIR, avatar_id)
        current, _ = _avatar_step_state(avatar, step)
        if avatar and current.lower() != avatar.get("current_step"):
            avatar["current_step"] = current.lower()
            avatar_studio.save_avatar(SAVE_DIR, avatar)
        return _avatar_ui_payload(avatar_id, current)

    avatar_step.input(change_avatar_step, inputs=[selected_avatar_id, avatar_step], outputs=avatar_outputs, show_progress="hidden")

    def create_avatar_handler(name, mode, preset):
        avatar = avatar_guide.create(SAVE_DIR, name, preset, mode)
        payload = _avatar_ui_payload(avatar["id"], "Guide" if mode == "Guided" else "Face")
        payload[avatar_name_input] = ""
        return payload

    create_avatar_btn.click(create_avatar_handler, inputs=[avatar_name_input, avatar_creation_mode, avatar_preset], outputs=avatar_outputs, show_progress="minimal")

    def refresh_avatar_handler(avatar_id):
        if avatar_id:
            avatar_studio.select_avatar(SAVE_DIR, avatar_id)
        return _avatar_ui_payload(avatar_id)

    avatar_selector.input(refresh_avatar_handler, inputs=[avatar_selector], outputs=avatar_outputs, show_progress="hidden")

    def search_avatar_library(query, status, sort):
        avatars = avatar_design.library(SAVE_DIR, query, status, sort)
        return (
            gr.update(value=avatar_design.library_values(SAVE_DIR, query, status, sort), visible=bool(avatars)),
            [avatar["id"] for avatar in avatars],
            gr.update(value=avatar_design.library_empty(SAVE_DIR, query, status), visible=not bool(avatars)),
        )

    library_inputs = [avatar_search, avatar_filter, avatar_sort]
    library_outputs = [avatar_library, avatar_library_ids, avatar_library_empty]
    for control in library_inputs:
        control.input(search_avatar_library, inputs=library_inputs, outputs=library_outputs, show_progress="hidden")
    refresh_avatar_btn.click(search_avatar_library, inputs=library_inputs, outputs=library_outputs, show_progress="hidden")

    def show_avatar_library(avatar_id, query, status, sort):
        payload = _avatar_ui_payload(avatar_id, library_view=True, refresh_library=False)
        values = search_avatar_library(query, status, sort)
        payload.update(dict(zip(library_outputs, values)))
        return payload

    avatar_back_library.click(show_avatar_library, inputs=[selected_avatar_id, *library_inputs], outputs=avatar_outputs, show_progress="hidden")
    avatar_cancel_new.click(show_avatar_library, inputs=[selected_avatar_id, *library_inputs], outputs=avatar_outputs, show_progress="hidden")

    def start_new_avatar():
        payload = _avatar_ui_payload(creating=True)
        payload[avatar_name_input] = ""
        payload[avatar_creation_mode] = "Guided"
        payload[avatar_preset] = gr.update(value="Studio creator", visible=True)
        return payload

    avatar_new_btn.click(start_new_avatar, outputs=avatar_outputs, show_progress="hidden")
    avatar_creation_mode.input(lambda mode: gr.update(visible=mode == "Guided"), inputs=[avatar_creation_mode], outputs=[avatar_preset], show_progress="hidden")

    def select_avatar_from_library(ids, evt: gr.SelectData):
        index = int(evt.index)
        if not 0 <= index < len(ids):
            raise gr.Error("Refresh the library and select the avatar again.")
        avatar_studio.select_avatar(SAVE_DIR, ids[index])
        return _avatar_ui_payload(ids[index])

    avatar_library.select(select_avatar_from_library, inputs=[avatar_library_ids], outputs=avatar_outputs, show_progress="hidden")

    def refresh_avatar_overview(avatar_id, query):
        avatar = avatar_studio.load_avatar(SAVE_DIR, avatar_id) if avatar_id else None
        avatars = avatar_design.library(SAVE_DIR, query)
        return (
            avatar_design.dashboard(avatar),
            gr.update(value=avatar_design.library_values(SAVE_DIR, query), visible=bool(avatars)),
            [avatar["id"] for avatar in avatars],
        )

    avatar_rename_btn.click(lambda: gr.update(visible=True), outputs=[avatar_rename_panel], show_progress="hidden")
    avatar_rename_cancel.click(lambda: gr.update(visible=False), outputs=[avatar_rename_panel], show_progress="hidden")

    def rename_selected_avatar(avatar_id, name):
        avatar_studio.rename_avatar(SAVE_DIR, avatar_id, name)
        return _avatar_ui_payload(avatar_id)

    avatar_rename_save.click(rename_selected_avatar, inputs=[selected_avatar_id, avatar_rename_name], outputs=avatar_outputs, show_progress="hidden")

    def make_avatar_reference(kind, details, avatar=None):
        prompt = avatar_studio.build_reference_prompt(kind, details, avatar)
        if DEV_MODE:
            return avatar_studio.make_dev_reference(kind, details)
        aspect = "1024x1024 (1:1)" if kind == "face" else "864x1152 (3:4)"
        references = None
        if kind == "body" and avatar and avatar.get("face_locked"):
            references = avatar_studio.reference_images(SAVE_DIR, {"face_image": avatar.get("face_image")})
        final_image = None
        last_status = ""
        for last_status, images, _, _ in do_generate(
            model_manager.FLUX_MODEL_NAME, prompt, DEFAULT_NEG, aspect,
            0, 1.0, 1.0, 1, 4, input_image=references, canvas_index=None,
        ):
            if images:
                final_image = images[0]
        if final_image is None:
            raise gr.Error(re.sub("<[^>]+>", "", last_status) or "FLUX returned no image. Check results/_debug.")
        return final_image

    def make_custom_reference(avatar_id, kind, details):
        if not avatar_id:
            raise gr.Error("Create an avatar first.")
        avatar = avatar_studio.load_avatar(SAVE_DIR, avatar_id)
        avatar[f"{kind}_prompt"] = details or ""
        avatar_studio.save_avatar(SAVE_DIR, avatar)
        payload = _avatar_ui_payload(avatar_id, kind.title(), refresh_library=False)
        payload[avatar_task_status] = gr.update(value='<div class="av-busy" role="status"><i></i><span>Generating identity draft</span></div>', visible=True)
        payload[generate_face_btn if kind == "face" else generate_body_btn] = gr.update(interactive=False)
        for control in (avatar_selector, avatar_back_library, avatar_step, avatar_rename_btn, lock_face_btn, lock_body_btn):
            payload[control] = gr.update(interactive=False)
        yield payload
        try:
            image = make_avatar_reference(kind, details, avatar)
            avatar_studio.save_candidate(SAVE_DIR, avatar_id, kind, image, details)
            yield _avatar_ui_payload(avatar_id, kind.title())
        except Exception as exc:
            _write_runtime_error("avatar reference draft", exc)
            payload = _avatar_ui_payload(avatar_id, kind.title())
            payload[avatar_task_status] = gr.update(value=_status_html("error", str(exc)), visible=True)
            yield payload

    def generate_custom_face(avatar_id, details):
        yield from make_custom_reference(avatar_id, "face", details)

    def generate_custom_body(avatar_id, details):
        yield from make_custom_reference(avatar_id, "body", details)

    generate_face_btn.click(generate_custom_face, inputs=[selected_avatar_id, face_details], outputs=avatar_outputs, show_progress="hidden", trigger_mode="once")
    generate_body_btn.click(generate_custom_body, inputs=[selected_avatar_id, body_details], outputs=avatar_outputs, show_progress="hidden", trigger_mode="once")

    def upload_custom_reference(avatar_id, kind, image, details):
        avatar_studio.save_candidate(SAVE_DIR, avatar_id, kind, image, details)
        return _avatar_ui_payload(avatar_id, kind.title())

    face_image.upload(lambda avatar_id, image, details: upload_custom_reference(avatar_id, "face", image, details), inputs=[selected_avatar_id, face_image, face_details], outputs=avatar_outputs, show_progress="minimal")
    body_image.upload(lambda avatar_id, image, details: upload_custom_reference(avatar_id, "body", image, details), inputs=[selected_avatar_id, body_image, body_details], outputs=avatar_outputs, show_progress="minimal")
    face_image.clear(lambda: gr.update(interactive=False), outputs=[lock_face_btn], show_progress="hidden")
    body_image.clear(lambda: gr.update(interactive=False), outputs=[lock_body_btn], show_progress="hidden")

    def lock_avatar_reference(avatar_id, kind, image, details):
        if not avatar_id:
            raise gr.Error("Create or select an avatar first.")
        avatar = avatar_studio.load_avatar(SAVE_DIR, avatar_id)
        avatar[f"{kind}_prompt"] = details or ""
        avatar_studio.save_avatar(SAVE_DIR, avatar)
        avatar_studio.lock_reference(SAVE_DIR, avatar_id, kind, image, dev_mode=DEV_MODE)
        step = "Body" if kind == "face" else "Console"
        return _avatar_ui_payload(avatar_id, step)

    lock_face_btn.click(
        lambda avatar_id, image, details: lock_avatar_reference(avatar_id, "face", image, details),
        inputs=[selected_avatar_id, face_image, face_details],
        outputs=avatar_outputs,
        show_progress="minimal",
    )
    lock_body_btn.click(
        lambda avatar_id, image, details: lock_avatar_reference(avatar_id, "body", image, details),
        inputs=[selected_avatar_id, body_image, body_details],
        outputs=avatar_outputs,
        show_progress="minimal",
    )

    def generate_guided_image(kind, prompt, avatar):
        if kind in {"face", "body"}:
            return make_avatar_reference(kind, prompt, avatar)
        references = avatar_studio.reference_images(SAVE_DIR, avatar)
        result = None
        last_status = ""
        for last_status, images, _, _ in do_generate(
            model_manager.FLUX_MODEL_NAME, prompt, DEFAULT_NEG,
            "864x1152 (3:4)", 0, 1.0, 1.0, 1, 4,
            input_image=references, canvas_index=None,
        ):
            if images:
                result = images[0]
        if result is None:
            raise RuntimeError(re.sub("<[^>]+>", "", last_status) or "FLUX returned no starter image.")
        return result

    def guide_action(avatar_id, image, face, body, scene_one, scene_two, regenerate=False):
        if not avatar_id:
            raise gr.Error("Create a guided avatar first.")
        avatar = avatar_guide.save_options(SAVE_DIR, avatar_id, face, body, scene_one, scene_two)

        def progress(message, error=False):
            payload = _avatar_ui_payload(avatar_id, "Guide", refresh_library=False)
            if not error:
                for button in (guide_next_btn, guide_regenerate_btn, guide_back_btn, guide_upload, guide_custom_btn, avatar_selector, avatar_back_library, avatar_step, avatar_rename_btn):
                    payload[button]["interactive"] = False
                payload[guide_next_btn]["visible"] = False
                payload[guide_stop_btn] = gr.update(visible=True)
            status = _status_html("error", message) if error else (
                '<div class="av-busy" role="status"><i aria-hidden="true"></i>'
                f'<span>{html.escape(message)}</span></div>'
            )
            payload[guide_status] = status
            return payload

        try:
            if regenerate:
                kind = "face" if avatar_guide.stage(avatar).startswith("face") else "body"
                yield progress(f"Creating another {kind} draft")
                candidate = generate_guided_image(kind, avatar_guide.state(avatar)[f"{kind}_prompt"], avatar)
                avatar_guide.save_draft(SAVE_DIR, avatar, kind, candidate)
            else:
                validate = lambda candidate, saved, prompt: avatar_studio.validate_generated_image(
                    candidate, saved, prompt, dev_mode=DEV_MODE
                )
                for _, message in avatar_guide.advance(
                    SAVE_DIR, avatar_id, image, generate_guided_image, validate, dev_mode=DEV_MODE,
                ):
                    yield progress(message)
            yield _avatar_ui_payload(avatar_id, "Guide")
        except Exception as exc:
            _write_runtime_error("avatar guided setup", exc)
            yield progress(str(exc) + " Your completed steps are saved. Retry to continue.", error=True)

    guide_inputs = [selected_avatar_id, guide_image, guide_face_prompt, guide_body_prompt, guide_scene_one, guide_scene_two]
    guide_next_event = guide_next_btn.click(guide_action, inputs=guide_inputs, outputs=avatar_outputs, show_progress="hidden", trigger_mode="once")

    def regenerate_guided_draft(avatar_id, image, face, body, scene_one, scene_two):
        yield from guide_action(avatar_id, image, face, body, scene_one, scene_two, regenerate=True)

    guide_regenerate_event = guide_regenerate_btn.click(regenerate_guided_draft, inputs=guide_inputs, outputs=avatar_outputs, show_progress="hidden", trigger_mode="once")

    def upload_guided_draft(avatar_id, path, face, body, scene_one, scene_two):
        avatar = avatar_guide.save_options(SAVE_DIR, avatar_id, face, body, scene_one, scene_two)
        kind = "face" if avatar_guide.stage(avatar).startswith("face") else "body"
        avatar_guide.save_draft(SAVE_DIR, avatar, kind, path)
        return _avatar_ui_payload(avatar_id, "Guide")

    guide_upload.upload(upload_guided_draft, inputs=[selected_avatar_id, guide_upload, guide_face_prompt, guide_body_prompt, guide_scene_one, guide_scene_two], outputs=avatar_outputs, show_progress="minimal")
    guide_image.upload(upload_guided_draft, inputs=[selected_avatar_id, guide_image, guide_face_prompt, guide_body_prompt, guide_scene_one, guide_scene_two], outputs=avatar_outputs, show_progress="minimal")

    def clear_guided_draft(avatar_id):
        avatar_guide.clear_draft(SAVE_DIR, avatar_id)
        return _avatar_ui_payload(avatar_id, "Guide", refresh_library=False)

    guide_image.clear(clear_guided_draft, inputs=[selected_avatar_id], outputs=avatar_outputs, show_progress="hidden")

    def back_in_guide(avatar_id, image, face, body, scene_one, scene_two):
        avatar_guide.save_options(SAVE_DIR, avatar_id, face, body, scene_one, scene_two)
        avatar_guide.back(SAVE_DIR, avatar_id)
        return _avatar_ui_payload(avatar_id, "Guide")

    guide_back_btn.click(back_in_guide, inputs=guide_inputs, outputs=avatar_outputs, show_progress="hidden")

    def stop_guide(avatar_id):
        payload = _avatar_ui_payload(avatar_id, "Guide")
        payload[guide_status] = _status_html("done", "Stopping after the current step. Saved work can be resumed.")
        return payload

    guide_stop_btn.click(stop_guide, inputs=[selected_avatar_id], outputs=avatar_outputs, cancels=[guide_next_event, guide_regenerate_event], queue=False, show_progress="hidden")

    def leave_guide(avatar_id):
        avatar = avatar_guide.use_custom(SAVE_DIR, avatar_id)
        return _avatar_ui_payload(avatar_id, avatar["current_step"].title())

    guide_custom_btn.click(leave_guide, inputs=[selected_avatar_id], outputs=avatar_outputs, show_progress="hidden")
    guide_console_btn.click(lambda avatar_id: change_avatar_step(avatar_id, "Console"), inputs=[selected_avatar_id], outputs=avatar_outputs, show_progress="hidden")
    guide_gallery_btn.click(lambda avatar_id: change_avatar_step(avatar_id, "Gallery"), inputs=[selected_avatar_id], outputs=avatar_outputs, show_progress="hidden")

    def _avatar_console_generate(avatar_id, mode, prompt, aspect, steps, extra_1, extra_2):
        if not avatar_id:
            raise gr.Error("Create or select an avatar first.")
        avatar = avatar_studio.load_avatar(SAVE_DIR, avatar_id)
        if not avatar.get("face_locked") or not avatar.get("body_locked"):
            raise gr.Error("Lock the face and body references before using the console.")
        if not (prompt or "").strip():
            raise gr.Error("Describe what you want the avatar to do or wear.")

        extra_paths = avatar_studio.save_console_references(SAVE_DIR, avatar_id, [extra_1, extra_2])
        avatar = avatar_studio.append_avatar_chat(
            SAVE_DIR, avatar_id, "user", prompt,
            metadata={"extra_references": extra_paths, "mode": mode},
        )
        references = avatar_studio.reference_images(SAVE_DIR, avatar, [extra_1, extra_2])
        built_prompt = avatar_studio.build_generation_prompt(avatar, prompt, mode)
        yield (
            _status_html("active", "Preparing avatar references"),
            _avatar_chat_html(avatar),
            _avatar_gallery_value(avatar),
            avatar_id,
            gr.update(),
        )

        final_image = None
        final_seed = "0"
        for status_html, images, _, seed_str in do_generate(
            model_manager.FLUX_MODEL_NAME, built_prompt, DEFAULT_NEG, aspect,
            0, 1.0, 1.0, 1, int(steps),
            input_image=references, canvas_index=None,
        ):
            final_seed = seed_str
            if images:
                final_image = images[0]
            current_avatar = avatar_studio.load_avatar(SAVE_DIR, avatar_id)
            yield (
                status_html,
                _avatar_chat_html(current_avatar),
                _avatar_gallery_value(current_avatar),
                avatar_id,
                gr.update(),
            )

        if final_image is not None:
            image_path = avatar_studio.save_chat_image(SAVE_DIR, avatar_id, final_image)
            avatar = avatar_studio.append_avatar_chat(
                SAVE_DIR,
                avatar_id,
                "assistant",
                f"Generated with {mode}. Seed {final_seed}.",
                image_path=image_path,
                metadata={"mode": mode, "prompt": built_prompt, "seed": final_seed},
            )
            yield (
                _status_html("done", "Avatar image saved"),
                _avatar_chat_html(avatar),
                _avatar_gallery_value(avatar),
                avatar_id,
                gr.update(value=""),
            )

    avatar_task_controls = [avatar_generate_btn, avatar_stop_btn, gallery_plan_btn, gallery_stop_btn,
                            regenerate_gallery_btn, avatar_selector, avatar_step, avatar_back_library, avatar_rename_btn,
                            avatar_extra_1, avatar_extra_2, gallery_rejection_reason,
                            gallery_require_face, gallery_quantity, gallery_aspect, gallery_theme, gallery_new_search]

    def avatar_task_state(avatar_id, busy, kind, selected_index=None):
        avatar = avatar_studio.selected_or_first(SAVE_DIR, avatar_id)
        ready = bool(avatar and avatar.get("face_locked") and avatar.get("body_locked"))
        api_ready = DEV_MODE or avatar_gallery.configuration_status()["ready"]
        selected_item = avatar_studio.gallery_item_by_index(SAVE_DIR, avatar_id, selected_index) if avatar and not busy else None
        return {
            avatar_generate_btn: gr.update(interactive=ready and not busy, visible=not busy),
            avatar_stop_btn: gr.update(interactive=busy and kind == "console", visible=busy and kind == "console"),
            gallery_plan_btn: gr.update(interactive=ready and api_ready and not busy),
            gallery_stop_btn: gr.update(interactive=busy and kind == "gallery", visible=busy and kind == "gallery"),
            regenerate_gallery_btn: gr.update(interactive=ready and bool(selected_item and selected_item.get("id"))),
            gallery_rejection_reason: gr.update(interactive=not busy),
            gallery_new_search: gr.update(interactive=False),
            **{component: gr.update(interactive=not busy) for component in
               (gallery_require_face, gallery_quantity, gallery_aspect, gallery_theme)},
            avatar_extra_1: gr.update(interactive=not busy),
            avatar_extra_2: gr.update(interactive=not busy),
            **{component: gr.update(interactive=not busy) for component in
               (avatar_selector, avatar_step, avatar_back_library, avatar_rename_btn)},
        }

    def run_avatar_task(iterator, columns, avatar_id, kind, selected_index=None):
        status_component = avatar_console_status if kind == "console" else gallery_status
        yield avatar_task_state(avatar_id, True, kind)
        try:
            for row in iterator:
                yield dict(zip(columns, row))
        except Exception as exc:
            _write_runtime_error(f"avatar {kind}", exc)
            payload = {status_component: _status_html("error", f"{exc} Saved work is preserved; retry when ready.")}
            if kind == "gallery":
                saved = avatar_studio.selected_or_first(SAVE_DIR, avatar_id)
                payload[avatar_gallery_display] = _avatar_gallery_value(saved)
                payload[avatar_gallery_details] = _avatar_gallery_details(saved)
            yield payload
        yield avatar_task_state(avatar_id, False, kind, selected_index)

    console_columns = [avatar_console_status, avatar_chat_display, avatar_gallery_display, selected_avatar_id, avatar_prompt]

    def avatar_console_generate(avatar_id, mode, prompt, aspect, steps, extra_1, extra_2):
        yield from run_avatar_task(
            _avatar_console_generate(avatar_id, mode, prompt, aspect, steps, extra_1, extra_2),
            console_columns, avatar_id, "console",
        )

    avatar_generation_event = avatar_generate_btn.click(
        avatar_console_generate,
        inputs=[selected_avatar_id, avatar_mode, avatar_prompt, avatar_aspect, avatar_steps, avatar_extra_1, avatar_extra_2],
        outputs=console_columns + avatar_task_controls,
        trigger_mode="once",
        show_progress="hidden",
    )
    avatar_generation_event.then(refresh_avatar_overview, inputs=[selected_avatar_id, avatar_search], outputs=[avatar_dashboard, avatar_library, avatar_library_ids], show_progress="hidden")

    def stop_avatar_console(avatar_id):
        payload = avatar_task_state(avatar_id, False, "console")
        payload[avatar_console_status] = _status_html("done", "Stopping after the current model operation. Saved chat is preserved.")
        return payload

    avatar_stop_btn.click(stop_avatar_console, inputs=[selected_avatar_id],
                          outputs=[avatar_console_status] + avatar_task_controls,
                          cancels=[avatar_generation_event], queue=False, show_progress="hidden")

    def _dev_gallery_references(avatar_id, quantity, theme):
        references = []
        for index in range(int(quantity)):
            image = avatar_studio.make_dev_reference("body", f"{theme or 'gallery'} #{index + 1}")
            path = avatar_studio.save_gallery_reference(SAVE_DIR, avatar_id, image, f"dev-{index + 1}")
            references.append(
                {
                    "path": path,
                    "image_url": "dev://mock-reference",
                    "source_url": "dev://local-preview",
                    "title": f"Mock gallery reference {index + 1}",
                    "description": theme or "editorial avatar gallery",
                    "validation": {"accept": True, "score": 100, "reason": "local mock"},
                }
            )
        return references, {"query": "local mock", "selected": len(references), "approved": len(references)}

    def _gallery_prompt(avatar, theme, reference):
        if DEV_MODE:
            return avatar_studio.build_generation_prompt(
                avatar,
                f"{theme or 'editorial gallery'}; use the extra reference for outfit, pose, and composition",
                "Outfit Focus",
            )
        return avatar_gallery.create_generation_prompt(
            avatar.get("name", "Avatar"),
            theme,
            reference,
            avatar_studio.summarize_specs(avatar.get("face_specs"), limit=10),
            avatar_studio.summarize_specs(avatar.get("body_specs"), limit=10),
        )

    def _run_gallery_generation(avatar, reference, prompt, aspect, item_id, attempt, identity_only=False, validate_output=True):
        reference_path = reference.get("path")
        if identity_only:
            references = avatar_studio.reference_images(SAVE_DIR, avatar)
        elif not reference_path or not os.path.exists(reference_path):
            raise RuntimeError("The selected gallery reference image is missing.")
        else:
            with Image.open(reference_path) as reference_image:
                references = avatar_studio.reference_images(SAVE_DIR, avatar, [reference_image.convert("RGB")])
        final_image = None
        final_seed = "0"
        for _, images, _, seed_str in do_generate(
            model_manager.FLUX_MODEL_NAME,
            prompt,
            DEFAULT_NEG,
            aspect,
            0,
            1.0,
            1.0,
            1,
            4,
            input_image=references,
            canvas_index=None,
        ):
            final_seed = seed_str
            if images:
                final_image = images[0]
        if final_image is None:
            raise RuntimeError("FLUX did not return a gallery image. Check the newest runtime debug log.")
        path = avatar_studio.save_gallery_image(SAVE_DIR, avatar["id"], item_id, final_image, attempt)
        avatar_studio.update_gallery_item(
            SAVE_DIR, avatar["id"], item_id, generated_path=path,
            attempt=attempt, seed=final_seed, status="checking",
        )
        validation = avatar_studio.validate_generated_image(final_image, avatar, prompt, dev_mode=DEV_MODE) if validate_output else None
        return path, validation, final_seed

    def _create_avatar_gallery(avatar_id, quantity, aspect, theme, require_face=False):
        if not avatar_id:
            raise gr.Error("Create or select an avatar first.")
        avatar = avatar_studio.load_avatar(SAVE_DIR, avatar_id)
        if not avatar.get("face_locked") or not avatar.get("body_locked"):
            raise gr.Error("Lock the face and body before generating the gallery.")
        avatar["current_step"] = "gallery"
        avatar_studio.save_avatar(SAVE_DIR, avatar)
        validate = lambda image, saved, prompt: avatar_studio.validate_generated_image(
            image, saved, prompt, dev_mode=DEV_MODE,
        )
        generate = lambda *args: _run_gallery_generation(*args, validate_output=False)
        repair = (lambda prompt, result: prompt + " Fix identity and anatomy while preserving the requested scene.") if DEV_MODE else avatar_gallery.repair_generation_prompt
        iterator = avatar_gallery_jobs.run(
            SAVE_DIR, avatar_id, quantity, aspect, theme, require_face,
            _gallery_prompt, generate, validate, repair,
            mock_discover=_dev_gallery_references if DEV_MODE else None,
        )
        showed_results = False
        for job in iterator:
            saved = avatar_studio.load_avatar(SAVE_DIR, avatar_id)
            visible = avatar_studio.visible_gallery_items(SAVE_DIR, avatar_id)
            batch_visible = [index for index, row in enumerate(visible) if row.get("batch_id") == job["id"]]
            selected_index = batch_visible[-1] if batch_visible else None
            tab = gr.skip()
            if not showed_results:
                if job["phase"] in {"generating", "checking", "repairing", "complete"}:
                    tab = gr.update(selected="gallery-results")
                    showed_results = True
                else:
                    tab = gr.update(selected="gallery-references")
            yield (
                _status_html("done" if job["status"] == "complete" else "active", job["message"]),
                _avatar_gallery_value(saved), avatar_studio.status_html(saved),
                _avatar_gallery_details(saved, selected_index), selected_index, tab,
            )

    gallery_columns = [gallery_status, avatar_gallery_display, avatar_status,
                       avatar_gallery_details, selected_avatar_gallery_index, gallery_views]

    def create_avatar_gallery(avatar_id, quantity, aspect, theme, require_face):
        yield from run_avatar_task(_create_avatar_gallery(avatar_id, quantity, aspect, theme, require_face),
                                   gallery_columns, avatar_id, "gallery")

    avatar_gallery_event = gallery_plan_btn.click(
        create_avatar_gallery,
        inputs=[selected_avatar_id, gallery_quantity, gallery_aspect, gallery_theme, gallery_require_face],
        outputs=gallery_columns + avatar_task_controls,
        trigger_mode="once",
        show_progress="hidden",
    )
    avatar_gallery_event.then(refresh_avatar_overview, inputs=[selected_avatar_id, avatar_search], outputs=[avatar_dashboard, avatar_library, avatar_library_ids], show_progress="hidden")

    def select_avatar_gallery_item(avatar_id, evt: gr.SelectData):
        avatar = avatar_studio.selected_or_first(SAVE_DIR, avatar_id)
        index = int(evt.index) if evt.index is not None else None
        item = avatar_studio.gallery_item_by_index(SAVE_DIR, avatar_id, index) if avatar else None
        return index, _avatar_gallery_details(avatar, index), gr.update(interactive=bool(item and item.get("id"))), gr.update(visible=bool(item)), gr.update(value="")

    avatar_gallery_display.select(
        select_avatar_gallery_item,
        inputs=[selected_avatar_id],
        outputs=[selected_avatar_gallery_index, avatar_gallery_details, regenerate_gallery_btn, gallery_selected_panel, gallery_rejection_reason],
        show_progress="hidden",
    )

    def _regenerate_avatar_gallery_item(avatar_id, selected_index, rejection_reason=""):
        if not avatar_id or selected_index is None:
            raise gr.Error("Select a gallery image first.")
        avatar = avatar_studio.load_avatar(SAVE_DIR, avatar_id)
        item = avatar_studio.gallery_item_by_index(SAVE_DIR, avatar_id, selected_index)
        if not item or not item.get("id"):
            raise gr.Error("That gallery item is no longer available. Refresh Avatar Studio.")
        reference = {
            "path": item.get("reference_path"),
            "image_url": item.get("reference_url"),
            "source_url": item.get("reference_source"),
        }
        item = avatar_studio.reject_gallery_item(SAVE_DIR, avatar_id, item["id"], rejection_reason)
        previous_validation = item["validation"]
        prompt = item.get("prompt", "")
        prompt = (
            prompt + " User rejected the previous image. " + previous_validation["repair_instruction"]
            if DEV_MODE or item.get("origin") == "guided"
            else avatar_gallery.repair_generation_prompt(prompt, previous_validation)
        )
        attempt = int(item.get("attempt", 1)) + 1
        avatar_studio.update_gallery_item(
            SAVE_DIR, avatar_id, item["id"], status="regenerating", prompt=prompt, attempt=attempt
        )
        yield (
            _status_html("active", "Regenerating the selected gallery image"),
            _avatar_gallery_value(avatar),
            _avatar_gallery_details(avatar, selected_index),
        )
        try:
            path, validation, seed = _run_gallery_generation(
                avatar, reference, prompt, item.get("aspect") or "864x1152 (3:4)", item["id"], attempt,
                identity_only=item.get("origin") == "guided",
            )
        except Exception as exc:
            debug_path = _write_runtime_error("avatar gallery regeneration", exc)
            avatar_studio.update_gallery_item(
                SAVE_DIR,
                avatar_id,
                item["id"],
                status="failed",
                validation={
                    "pass": False,
                    "score": 0,
                    "reasons": [str(exc)],
                    "debug_path": debug_path,
                },
            )
            raise gr.Error("Regeneration failed. Check results/_debug for the full traceback.") from exc
        status = "passed" if validation.get("pass") else "needs_review"
        avatar_studio.update_gallery_item(
            SAVE_DIR,
            avatar_id,
            item["id"],
            status=status,
            generated_path=path,
            validation=validation,
            seed=seed,
        )
        avatar = avatar_studio.load_avatar(SAVE_DIR, avatar_id)
        yield (
            _status_html("done", "Selected gallery image regenerated and checked."),
            _avatar_gallery_value(avatar),
            _avatar_gallery_details(avatar, selected_index),
        )

    regenerate_columns = [gallery_status, avatar_gallery_display, avatar_gallery_details]

    def regenerate_avatar_gallery_item(avatar_id, selected_index, rejection_reason):
        yield from run_avatar_task(_regenerate_avatar_gallery_item(avatar_id, selected_index, rejection_reason),
                                   regenerate_columns, avatar_id, "gallery", selected_index)

    avatar_regenerate_event = regenerate_gallery_btn.click(
        regenerate_avatar_gallery_item,
        inputs=[selected_avatar_id, selected_avatar_gallery_index, gallery_rejection_reason],
        outputs=regenerate_columns + avatar_task_controls,
        trigger_mode="once",
        show_progress="hidden",
    )

    def stop_avatar_gallery(avatar_id):
        avatar_gallery_jobs.stop_batch(SAVE_DIR, avatar_id)
        payload = avatar_task_state(avatar_id, False, "gallery")
        payload[gallery_status] = _status_html("done", "Stopping after the current operation. Completed gallery images remain saved.")
        return payload

    gallery_stop_btn.click(stop_avatar_gallery, inputs=[selected_avatar_id],
                          outputs=[gallery_status] + avatar_task_controls,
                          cancels=[avatar_gallery_event, avatar_regenerate_event], queue=False, show_progress="hidden")

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

    def _attachment_updates(images, model_name, reset_mask=True, canvas_idx=0):
        images = _as_image_list(images)
        defaults = model_manager.get_defaults(model_name)
        if not images:
            return (
                gr.update(visible=False), gr.update(value=None), None,
                "", gr.update(visible=False), "Smart", None, None,
                gr.update(visible=False, value=defaults.get("denoise", 1.0)),
                gr.update(visible=False), gr.update(visible=False),
                0,  # canvas_index reset
            )
        has_canvas = canvas_idx is not None
        return (
            gr.update(visible=True), gr.update(value=images), images,
            _attachment_status_html(len(images), canvas_idx), gr.update(visible=has_canvas),
            "Smart" if reset_mask else gr.update(),
            None if reset_mask else gr.update(),
            None if reset_mask else gr.update(),
            gr.update(
                visible=True,
                value=defaults.get("img2img_denoise", 1.0),
            ),
            gr.update(visible=False) if reset_mask else gr.update(),
            gr.update(visible=False) if reset_mask else gr.update(),
            canvas_idx,  # canvas_index state; None means all images are references
        )

    def handle_upload(file, model_name, existing_images):
        files = _as_image_list(file)
        if not files:
            return (*_attachment_updates(existing_images, model_name, reset_mask=False), gr.update(value=None))
        if not model_manager.supports_img2img(model_name):
            raise gr.Error(
                f"{model_name} supports text-to-image only. "
                "Choose FLUX.2-klein 4B to edit an image."
            )
        existing = _as_image_list(existing_images)
        if len(existing) + len(files) > MAX_FLUX_REFERENCES:
            raise gr.Error(
                f"FLUX.2 Klein accepts up to {MAX_FLUX_REFERENCES} reference images. "
                f"You already have {len(existing)} and selected {len(files)} more."
            )
        additions = [Image.open(path).convert("RGB") for path in files]
        images = _append_attachment_images(existing, additions)
        return (
            *_attachment_updates(images, model_name, reset_mask=not existing),
            gr.update(value=None),
        )

    attach_btn.upload(
        handle_upload,
        inputs=[attach_btn, model_selector, attached_image],
        outputs=[
            attachment_row, attachment_display, attached_image,
            attachment_summary, edit_panel, mask_mode, mask_editor, mask_preview,
            gen_denoise, manual_mask_group, auto_mask_group, canvas_index, attach_btn,
        ],
    )

    def clear_attachment(model_name):
        return _attachment_updates([], model_name)

    clear_attachments_btn.click(
        clear_attachment,
        inputs=[model_selector],
        outputs=[
            attachment_row, attachment_display, attached_image,
            attachment_summary, edit_panel, mask_mode, mask_editor, mask_preview,
            gen_denoise, manual_mask_group, auto_mask_group, canvas_index,
        ],
    )

    def apply_attachment_action(images, index, action, model_name, current_canvas_idx):
        images = _as_image_list(images)
        if not images:
            return _attachment_updates([], model_name)
        if action not in ("canvas", "left", "right", "remove"):
            return _attachment_updates(images, model_name, reset_mask=False, canvas_idx=current_canvas_idx)
        prev_canvas = current_canvas_idx
        updated, new_canvas_idx = _apply_attachment_action(images, index, action, current_canvas_idx)
        canvas_changed = prev_canvas != new_canvas_idx
        return _attachment_updates(updated, model_name, reset_mask=canvas_changed, canvas_idx=new_canvas_idx)

    attachment_action_btn.click(
        apply_attachment_action,
        inputs=[
            attached_image, attachment_action_index,
            attachment_action_name, model_selector, canvas_index,
        ],
        outputs=[
            attachment_row, attachment_display, attached_image,
            attachment_summary, edit_panel, mask_mode, mask_editor, mask_preview,
            gen_denoise, manual_mask_group, auto_mask_group, canvas_index,
        ],
        show_progress="hidden",
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
        image = _primary_image(image)
        if image is None or mode in ("None", "Smart", "Manual Paint"):
            return None
        return generate_auto_mask_preview(image, mode)

    mask_mode.change(
        update_mask_preview,
        inputs=[attached_image, mask_mode],
        outputs=[mask_preview],
    )

    # ── Edit mask manually button ──────────────────────────
    def do_edit_mask_manually(image, mode):
        image = _primary_image(image)
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

    # ── FLUX encoder selection ─────────────────────────────
    def apply_flux_encoder(selection, model_name):
        if model_name != model_manager.FLUX_MODEL_NAME:
            raise gr.Error("The text-encoder selector applies only to FLUX.2 Klein 4B.")
        try:
            mode = model_manager.set_flux_encoder_mode(selection)
        except Exception as exc:
            _write_runtime_error("FLUX encoder selection", exc)
            raise gr.Error(str(exc)) from exc
        label = "Custom" if mode == "custom" else "Official"
        _ui_trace(f"FLUX encoder applied: {mode}")
        return (
            gr.update(value=label, choices=model_manager.get_flux_encoder_choices()),
            _flux_encoder_status_html(),
            '<span class="ffs-model-badge">○ Available</span>',
        )

    apply_flux_encoder_btn.click(
        apply_flux_encoder,
        inputs=[flux_encoder_choice, model_selector],
        outputs=[flux_encoder_choice, flux_encoder_status, model_status_display],
        show_progress="minimal",
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

        supports_image = model_manager.supports_img2img(model_name)
        encoder_config = model_manager.get_flux_encoder_config()
        return (
            badge,
            _model_capabilities_html(model_name),
            gr.update(
                value=defaults.get("steps", 8),
                maximum=defaults.get("max_steps", 50),
            ),
            gr.update(value=defaults.get("cfg", 1.0)),
            gr.update(value=defaults.get("denoise", 1.0), visible=False),
            gr.update(visible=supports_image),
            gr.update(visible=False),
            gr.update(value=None),
            None,
            "",
            gr.update(visible=False),
            "Smart",
            None,
            None,
            gr.update(visible=supports_image),
            gr.update(visible=model_name == model_manager.FLUX_MODEL_NAME),
            gr.update(
                choices=model_manager.get_flux_encoder_choices(),
                value="Custom" if encoder_config["mode"] == "custom" else "Official",
            ),
            _flux_encoder_status_html(),
            gr.update(visible=False),
            gr.update(visible=False),
            0,  # canvas_index reset
        )

    model_selector.change(
        on_model_change,
        inputs=[model_selector],
        outputs=[
            model_status_display, model_capabilities_display,
            gen_steps, gen_cfg, gen_denoise, attach_btn,
            attachment_row, attachment_display, attached_image,
            attachment_summary, edit_panel, mask_mode, mask_editor, mask_preview,
            add_to_prompt_btn,
            flux_encoder_panel, flux_encoder_choice, flux_encoder_status,
            manual_mask_group, auto_mask_group, canvas_index,
        ],
    )

    # ── SEND (main generation) ─────────────────────────────
    def on_send(model_name, prompt, image, mask_m, editor_data,
                aspect, seed, steps, cfg, denoise, n_images, neg, history, canvas_idx):
        """Main generation handler. Yields streaming updates."""
        history = list(history or [])
        if not prompt.strip() and image is None:
            yield (
                _status_html("error", "Please enter a prompt or attach an image."),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(),
                None,
                _render_history(history), history,
                gr.update(),
            )
            return

        # Determine effective mask mode
        effective_mask = mask_m if mask_m not in (None, "None", "Smart") else None
        request_history = history + [_request_html(prompt, model_name, image, mask_m)]

        # Save settings for regenerate
        settings = {
            "model": model_name, "prompt": prompt, "neg": neg,
            "aspect": aspect, "seed": seed, "steps": steps,
            "cfg": cfg, "denoise": denoise, "n_images": n_images,
            "mask_mode": mask_m,
        }

        request_saved = False
        for status_html, images, paths, seed_str in do_generate(
            model_name, prompt, neg, aspect,
            seed, cfg, denoise, n_images, steps,
            input_image=image, mask_mode=effective_mask,
            editor_data=editor_data, canvas_index=canvas_idx,
        ):
            if images:
                final_history = request_history + [_assistant_html(paths, seed_str)]
                _save_chat_history(final_history)
                # Show results
                yield (
                    status_html,
                    gr.update(visible=False, value=paths),
                    paths,
                    gr.update(visible=False, value=paths),
                    gr.update(visible=True, value=seed_str),
                    gr.update(visible=True),  # action_row
                    '<span class="ffs-model-badge ready">● Ready</span>',
                    settings,
                    _render_history(final_history),
                    final_history,
                    gr.update(value=""),
                )
            else:
                if not request_saved:
                    _save_chat_history(request_history)
                    request_saved = True
                yield (
                    status_html,
                    gr.update(),
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
            num_images, negative_prompt, chat_history, canvas_index,
        ],
        outputs=[
            status_display, result_gallery, latest_result_paths, result_files,
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
            canvas_index,
        ],
        outputs=[
            status_display, result_gallery, latest_result_paths, result_files,
            seed_display, action_row,
            model_status_display, last_gen_settings,
            conversation_display, chat_history, prompt_input,
        ],
        show_progress="hidden",
    )

    # ── Add to Prompt ──────────────────────────────────────
    def on_add_to_prompt(gallery_data, selected_idx, model_name, existing_images):
        """Append the selected latest result to the FLUX reference queue."""
        if not model_manager.supports_img2img(model_name):
            raise gr.Error(f"{model_name} cannot use an image as input.")
        if not gallery_data:
            raise gr.Error("No results to add!")
        existing = _as_image_list(existing_images)
        if len(existing) >= MAX_FLUX_REFERENCES:
            raise gr.Error(
                f"Remove a reference before adding another. FLUX supports up to {MAX_FLUX_REFERENCES}."
            )
        idx = min(int(selected_idx or 0), len(gallery_data) - 1)
        img_data = _gallery_item_to_pil(gallery_data[idx])
        images = _append_attachment_images(existing, [img_data])
        updates = _attachment_updates(images, model_name, reset_mask=not existing)
        return (*updates[:5], gr.update(), *updates[5:])

    add_to_prompt_btn.click(
        on_add_to_prompt,
        inputs=[latest_result_paths, selected_result_idx, model_selector, attached_image],
        outputs=[
            attachment_row, attachment_display, attached_image,
            attachment_summary, edit_panel, prompt_input,
            mask_mode, mask_editor, mask_preview, gen_denoise,
            manual_mask_group, auto_mask_group, canvas_index,
        ],
    )

    def _on_gallery_select(evt: gr.SelectData):
        return evt.index

    result_gallery.select(
        _on_gallery_select,
        outputs=[selected_result_idx],
    )

    # ── Regenerate ─────────────────────────────────────────
    def on_regenerate(settings, image, history, canvas_idx):
        """Re-run the last generation with a new random seed."""
        history = list(history or [])
        if settings is None:
            raise gr.Error("No previous generation to regenerate!")
        request_history = history + [_request_html(
            f"Regenerate: {settings['prompt']}",
            settings["model"],
            image,
            settings.get("mask_mode", "None"),
        )]
        request_saved = False
        # Force new random seed
        for status_html, images, paths, seed_str in do_generate(
            settings["model"], settings["prompt"], settings["neg"],
            settings["aspect"], 0,  # 0 = random seed
            settings["cfg"], settings["denoise"],
            settings["n_images"], settings["steps"],
            input_image=image,
            mask_mode=(
                settings.get("mask_mode")
                if settings.get("mask_mode") not in (None, "None", "Smart")
                else None
            ),
            canvas_index=canvas_idx,
        ):
            if images:
                final_history = request_history + [_assistant_html(paths, seed_str)]
                _save_chat_history(final_history)
                yield (
                    status_html,
                    gr.update(visible=False, value=paths),
                    paths,
                    gr.update(visible=False, value=paths),
                    gr.update(visible=True, value=seed_str),
                    _render_history(final_history),
                    final_history,
                )
            else:
                if not request_saved:
                    _save_chat_history(request_history)
                    request_saved = True
                yield (
                    status_html,
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    _render_history(request_history),
                    request_history,
                )

    regenerate_btn.click(
        on_regenerate,
        inputs=[last_gen_settings, attached_image, chat_history, canvas_index],
        outputs=[
            status_display, result_gallery, latest_result_paths, result_files, seed_display,
            conversation_display, chat_history,
        ],
    )

    # ── New Chat ───────────────────────────────────────────
    def on_new_chat():
        _save_chat_history([])
        return (
            # status_display
            """<div class="ffs-empty">
                <div class="ffs-empty-icon">FF</div>
                <div class="ffs-empty-title">What will you make?</div>
                <div class="ffs-empty-sub">Portrait. Product. Editorial. Concept.</div>
            </div>""",
            gr.update(visible=False, value=None),   # result_gallery
            [],                                      # latest_result_paths
            gr.update(visible=False, value=None),   # result_files
            gr.update(visible=False, value=""),      # seed_display
            gr.update(visible=False),                # action_row
            gr.update(visible=False),                # attachment_row
            gr.update(value=None),                   # attachment_display
            None,                                     # attached_image
            "",                                       # attachment_summary
            gr.update(value=""),                      # prompt_input
            None,                                     # last_gen_settings
            gr.update(visible=False),                # edit_panel
            "Smart",                                  # mask_mode
            '<div class="ffs-history"></div>',        # conversation_display
            [],                                       # chat_history
            gr.update(visible=False),                 # gen_denoise
            gr.update(visible=False),                 # manual_mask_group
            gr.update(visible=False),                 # auto_mask_group
        )

    new_chat_btn.click(
        on_new_chat,
        outputs=[
            status_display, result_gallery, latest_result_paths, result_files,
            seed_display, action_row,
            attachment_row, attachment_display, attached_image, attachment_summary,
            prompt_input,
            last_gen_settings, edit_panel, mask_mode,
            conversation_display, chat_history, gen_denoise,
            manual_mask_group, auto_mask_group,
        ],
    )

    def restore_saved_chat():
        history = _load_chat_history()
        return _render_history(history), history

    demo.load(
        restore_saved_chat,
        outputs=[conversation_display, chat_history],
        show_progress="hidden",
    )

_ui_trace("interface ready")


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# ═══════════════════════════════════════════════════════════
#  STARTUP — optional FLUX preload
# ═══════════════════════════════════════════════════════════
_ready_state_path = os.environ.get("FFS_READY_STATE_PATH", "").strip()


def _record_startup_ready(**values):
    if not _ready_state_path:
        return
    import startup_restore

    startup_restore.write_readiness(
        _ready_state_path,
        fingerprint=os.environ.get("FFS_RUNTIME_FINGERPRINT_ID", "unknown"),
        **values,
    )


_record_startup_ready(
    ready=False, phase="starting", model=None, warmup="pending", vision="pending"
)
if DEV_MODE:
    print("\n" + "=" * 50)
    print("  FreeFakeStudio — Local Development Mode")
    print("  Real AI models are disabled.")
    print("  UI-only testing is enabled.")
    print("=" * 50 + "\n")
    # Mark all models as available in dev mode
    for name in model_manager.MODEL_NAMES:
        model_manager.set_model_availability(name, True)
    _record_startup_ready(
        ready=True,
        phase="development",
        model="mock",
        warmup="disabled",
        vision="disabled",
    )
else:
    model_manager.check_model_files(os.environ.get("COMFYUI_ROOT", "/content/ComfyUI"))
    preload_flux = _env_flag("FFS_PRELOAD_FLUX", False)
    preload_avatar_vision = _env_flag("FFS_PRELOAD_AVATAR_VISION", True)
    warmup_enabled = _env_flag("FFS_WARMUP_ENABLED", True)
    expose_after_warmup = _env_flag("FFS_EXPOSE_AFTER_WARMUP", True)
    if expose_after_warmup and not preload_flux:
        raise RuntimeError(
            "EXPOSE_AFTER_WARMUP requires PRELOAD_FLUX=True. Enable FLUX preload or "
            "disable the strict ready gate in the Colab startup settings."
        )
    if preload_flux:
        print("\n⏳ Preloading FLUX.2-klein 4B before opening FreeFakeStudio...")
        try:
            model_manager.ensure_model(
                model_manager.FLUX_MODEL_NAME,
                status_callback=lambda message: print(message, flush=True),
            )
            print("✓ FLUX.2-klein 4B preloaded and ready.")
            warmup_state = "disabled"
            if warmup_enabled:
                warmup_size = int(os.environ.get("FFS_WARMUP_SIZE", "256"))
                print(
                    f"⏳ Running internal FLUX warmup ({warmup_size}x{warmup_size}, 1 step)...",
                    flush=True,
                )
                engine = model_manager.ensure_model(model_manager.FLUX_MODEL_NAME)
                warmup_report = engine.warmup(warmup_size)
                warmup_state = "complete"
                print(
                    "✓ FLUX warmup complete; discarded calibration output. "
                    f"Pipeline={warmup_report['pipeline']}",
                    flush=True,
                )
            vision_state = "disabled"
            if preload_avatar_vision:
                try:
                    import avatar_vision

                    if warmup_enabled:
                        print("⏳ Preloading and warming Avatar Studio SmolVLM...", flush=True)
                        vision_report = avatar_vision.warmup(128)
                        vision_state = "complete"
                        print(
                            "✓ Avatar Studio SmolVLM warmup complete; "
                            f"model={vision_report['model']}",
                            flush=True,
                        )
                    else:
                        vision_report = avatar_vision.preload()
                        vision_state = "loaded"
                        print(
                            f"✓ Avatar Studio analyzer loaded; model={vision_report['model']}",
                            flush=True,
                        )
                except Exception as vision_exc:
                    vision_state = "deferred"
                    _write_runtime_error("avatar vision preload", vision_exc)
                    print(
                        "[startup warning] Avatar Studio vision preload failed; "
                        "the ready FLUX interface will still open. "
                        f"Details: {vision_exc}",
                        flush=True,
                    )
            _record_startup_ready(
                ready=True,
                phase="model-ready",
                model=model_manager.FLUX_MODEL_NAME,
                warmup=warmup_state,
                warmup_size=warmup_size if warmup_enabled else None,
                vision=vision_state,
            )
        except Exception as exc:
            _record_startup_ready(
                ready=False,
                phase="startup-failed",
                model=model_manager.FLUX_MODEL_NAME,
                warmup="failed",
                vision="failed",
                error=str(exc),
            )
            _write_runtime_error("startup preload", exc)
            raise
    else:
        _record_startup_ready(
            ready=True,
            phase="lazy-model",
            model=None,
            warmup="disabled",
            vision="disabled",
        )
    status = model_manager.get_model_status()
    print("\n🎭 FreeFakeStudio — Model Status:")
    for name, st in status.items():
        icon = "✓" if st != "missing" else "✗"
        print(f"  {icon} {name}: {st}")
    print()

# Configure Gradio queue for single GPU concurrency
demo.queue(default_concurrency_limit=1)


# The Colab launcher supplies an absolute HTTPS root for its selected proxy.
if __name__ == "__main__":
    demo.launch(
        share=_env_flag("FREEFAKESTUDIO_SHARE", IS_COLAB),
        debug=True,
        show_error=True,
        inline=False,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("FREEFAKESTUDIO_PORT", "7860")),
        theme=ffs_theme,
        css=CSS,
        head=JS_HEAD,
        root_path=os.environ.get("FREEFAKESTUDIO_PUBLIC_URL") or None,
        allowed_paths=[os.path.abspath(SAVE_DIR)],
    )
