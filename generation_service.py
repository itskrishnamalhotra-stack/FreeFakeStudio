"""UI-independent generation and mask processing; no Gradio imports."""
import os
from pathlib import Path
import re
import secrets

import numpy as np
from PIL import Image, ImageDraw

import model_manager
from runtime_environment import mock_mode

DEFAULT_NEGATIVE = "low quality, blurry, pixelated, noise, watermark, text, logo"
ASPECTS = ["1024x1024 (1:1)", "864x1152 (3:4)", "1152x864 (4:3)", "768x1344 (9:16)", "1344x768 (16:9)"]
MODES = ["Identity Strict", "Outfit Focus", "Pose Focus", "Scene Focus", "Group Image"]
MASK_MODES = ["None", "Smart", "Manual Paint", "Background Only", "Everything Except Face"]
_rembg_session = None


def dimensions(aspect):
    match = re.match(r"^(\d+)x(\d+)", str(aspect))
    if not match:
        raise ValueError("Invalid output aspect")
    width, height = map(int, match.groups())
    if min(width, height) < 64 or max(width, height) > 1536 or width * height > 1600000:
        raise ValueError("Choose an output size within the runtime memory budget")
    return width, height


def background_mask(image):
    global _rembg_session
    if mock_mode():
        width, height = image.size
        result = np.full((height, width), 255, dtype=np.uint8)
        result[height // 4:3 * height // 4, width // 4:3 * width // 4] = 0
        return result
    from rembg import new_session, remove
    if _rembg_session is None:
        _rembg_session = new_session("u2net")
    return 255 - np.asarray(remove(image, session=_rembg_session, only_mask=True))


def except_face_mask(image):
    width, height = image.size
    mask = Image.new("L", image.size, 255)
    faces = []
    if not mock_mode():
        try:
            import cv2
            root = getattr(getattr(cv2, "data", None), "haarcascades", "")
            if not root or not hasattr(cv2, "CascadeClassifier"):
                raise ImportError("Face detector unavailable")
            cascade = cv2.CascadeClassifier(os.path.join(root, "haarcascade_frontalface_default.xml"))
            if cascade.empty():
                raise ImportError("Face detector data unavailable")
            faces = cascade.detectMultiScale(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY), 1.1, 5, minSize=(50, 50))
        except (ImportError, AttributeError, RuntimeError):
            faces = []
    draw = ImageDraw.Draw(mask)
    if len(faces):
        for x, y, fw, fh in faces:
            draw.ellipse((max(0, x - .3 * fw), max(0, y - .3 * fh), min(width, x + 1.3 * fw), min(height, y + 1.3 * fh)), fill=0)
    else:
        foreground = background_mask(image) < 127
        ys, xs = np.where(foreground)
        if len(xs):
            x, y, fw, fh = xs.min(), ys.min(), max(1, xs.max() - xs.min()), max(1, ys.max() - ys.min())
            cx, cy, rx, ry = x + fw / 2, y + fh * .2, max(20, fw * .18), max(24, fh * .16)
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=0)
    return np.asarray(mask)


def resolve_mask(image, mode, manual=None, prompt=""):
    if mode == "Smart":
        if re.search(r"\b(background|backdrop|bg|behind|surroundings|scenery|scene)\b", prompt.lower()):
            return background_mask(image)
        if re.search(r"\b(clothes|clothing|outfit|dress|shirt|jacket|pants|trousers|body|garment|wearing)\b", prompt.lower()):
            return np.where((except_face_mask(image) > 127) & (background_mask(image) < 127), 255, 0).astype(np.uint8)
        return None
    if mode == "Background Only":
        return background_mask(image)
    if mode == "Everything Except Face":
        return except_face_mask(image)
    if mode == "Manual Paint":
        if manual is None:
            raise ValueError("Paint a mask before generating")
        raw = Image.open(manual).convert("RGBA") if isinstance(manual, (str, Path)) else manual.convert("RGBA")
        # The browser exports white strokes on transparent pixels.
        mask = np.asarray(raw.resize(image.size, Image.Resampling.NEAREST))[:, :, 3]
        if not np.any(mask):
            raise ValueError("The manual mask is empty")
        return mask
    return None


def generate(model, prompt, negative=DEFAULT_NEGATIVE, aspect=ASPECTS[0], seed=0, cfg=1.0,
             denoise=1.0, steps=4, references=None, canvas=None, mask_mode="None", mask=None, on_status=None):
    if model not in model_manager.MODEL_REGISTRY:
        raise ValueError("Unknown model")
    if not str(prompt).strip():
        raise ValueError("Write a prompt first")
    references = [image.convert("RGB") for image in (references or [])]
    if len(references) > 4:
        raise ValueError("FLUX supports at most four input images")
    capabilities = model_manager.get_capabilities(model)
    if references and "img2img" not in capabilities:
        raise ValueError("The selected model does not support image input")
    if canvas is not None and not 0 <= canvas < len(references):
        raise ValueError("The selected canvas is no longer available")
    if mask_mode not in MASK_MODES:
        raise ValueError("Unknown mask mode")
    if mask_mode != "None" and canvas is None:
        raise ValueError("Select a canvas before masking")
    width, height = dimensions(aspect)
    seed = int(seed) if int(seed) > 0 else secrets.randbelow(2**63)
    engine = model_manager.ensure_model(model, status_callback=on_status)
    try:
        if not references:
            image = engine.generate(prompt, negative, width, height, seed, cfg, denoise, int(steps))
        elif canvas is None:
            image = engine.generate_with_references(references, prompt, negative, width, height, seed, cfg, denoise, int(steps))
        else:
            refs = [references[canvas], *[item for index, item in enumerate(references) if index != canvas]]
            selected_mask = resolve_mask(refs[0], mask_mode, mask, prompt)
            if selected_mask is not None:
                if model_manager.DEV_MODE:
                    image = engine.img2img(refs, prompt, negative, seed, cfg, denoise, int(steps), mask=selected_mask)
                else:
                    image = engine.inpaint(refs[0], selected_mask, prompt, negative, seed, cfg, denoise, int(steps), references=refs)
            else:
                image = engine.img2img(refs, prompt, negative, seed, cfg, denoise, int(steps))
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            model_manager.unload_current()
            raise RuntimeError("GPU memory exhausted. Model unloaded; use fewer references or a smaller output.") from exc
        raise
    if not isinstance(image, Image.Image):
        raise RuntimeError("The model returned no image")
    return image, str(seed)
