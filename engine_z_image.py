# ============================================================
#  Engine: Z-Image Turbo GGUF Q3_K_M
#  Fast generation (8 steps), img2img, mask-based inpaint
#  Uses ComfyUI nodes
# ============================================================

import gc, os, sys, torch, numpy as np
from PIL import Image, ImageFilter
from gguf_nodes import load_gguf_node_mappings

_loaded = False
_unet = None
_clip = None
_vae = None
_nodes = {}

Z_IMAGE_TEXT_ENCODER = "qwen_3_4b_fp4_mixed.safetensors"
Z_IMAGE_UNET = "z_image_turbo-Q3_K_M.gguf"


def _read_kib(path, key):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{key}:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _memory_status(stage):
    available_kib = _read_kib("/proc/meminfo", "MemAvailable")
    rss_kib = _read_kib("/proc/self/status", "VmRSS")
    parts = [f"RAM {stage}"]
    if available_kib is not None:
        parts.append(f"available={available_kib / 1024 / 1024:.1f} GB")
    if rss_kib is not None:
        parts.append(f"process={rss_kib / 1024 / 1024:.1f} GB")
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        parts.append(f"VRAM free={free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GB")
    print("[memory] " + " | ".join(parts), flush=True)
    return available_kib


def _require_host_headroom(stage, minimum_gib):
    available_kib = _memory_status(stage)
    if available_kib is not None and available_kib < minimum_gib * 1024 * 1024:
        raise RuntimeError(
            f"Not enough Colab host RAM before {stage}: "
            f"{available_kib / 1024 / 1024:.1f} GB available, {minimum_gib:.1f} GB required. "
            "Restart the Colab session to clear stale model processes, then run the cell once."
        )


def _configure_comfy_memory():
    """Set embedded ComfyUI defaults before model_management is imported."""
    from comfy.cli_args import args

    args.cache_none = True
    args.cache_classic = False
    args.cache_lru = 0
    args.high_ram = False
    args.enable_dynamic_vram = True
    args.disable_dynamic_vram = False
    args.disable_pinned_memory = True

# ── Node references (set once) ─────────────────────────────
def _get_nodes():
    global _nodes
    if not _nodes:
        comfyui_root = os.environ.get("COMFYUI_ROOT", "/content/ComfyUI")
        if comfyui_root not in sys.path:
            sys.path.insert(0, comfyui_root)
        _configure_comfy_memory()
        from nodes import NODE_CLASS_MAPPINGS

        gguf_mappings = load_gguf_node_mappings(comfyui_root)

        _nodes = {
            "UnetLoaderGGUF":   gguf_mappings["UnetLoaderGGUF"](),
            "CLIPLoader":       NODE_CLASS_MAPPINGS["CLIPLoader"](),
            "VAELoader":        NODE_CLASS_MAPPINGS["VAELoader"](),
            "CLIPTextEncode":   NODE_CLASS_MAPPINGS["CLIPTextEncode"](),
            "KSampler":         NODE_CLASS_MAPPINGS["KSampler"](),
            "VAEDecode":        NODE_CLASS_MAPPINGS["VAEDecode"](),
            "VAEEncode":        NODE_CLASS_MAPPINGS["VAEEncode"](),
            "SetLatentNoiseMask": NODE_CLASS_MAPPINGS["SetLatentNoiseMask"](),
        }
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
        from comfy_extras.nodes_sd3 import EmptySD3LatentImage

        _nodes["ModelSamplingAuraFlow"] = ModelSamplingAuraFlow()
        _nodes["EmptySD3LatentImage"] = EmptySD3LatentImage()
    return _nodes

# ── Load / Unload ──────────────────────────────────────────
def load():
    global _loaded, _unet, _clip, _vae
    if _loaded:
        return
    n = _get_nodes()
    print("⏳ Loading Z-Image Turbo GGUF Q3_K_M...")
    with torch.inference_mode():
        _require_host_headroom("Z-Image GGUF diffusion model", 4.5)
        raw_unet = n["UnetLoaderGGUF"].load_unet(Z_IMAGE_UNET)[0]
        _memory_status("after GGUF diffusion model")
        _unet = n["ModelSamplingAuraFlow"].patch_aura(raw_unet, 3.0)[0]
        _require_host_headroom("Z-Image FP4 text encoder", 3.8)
        _clip = n["CLIPLoader"].load_clip(Z_IMAGE_TEXT_ENCODER, type="lumina2")[0]
        _memory_status("after text encoder")
        _vae  = n["VAELoader"].load_vae("ae.safetensors")[0]
        _memory_status("after VAE")
    _loaded = True
    print("✅ Z-Image Turbo loaded!")

def unload():
    global _loaded, _unet, _clip, _vae
    _unet = None
    _clip = None
    _vae = None
    _loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
    print("🗑️ Z-Image Turbo unloaded")

def is_loaded():
    return _loaded

# ── Helpers ────────────────────────────────────────────────
def _pil_to_tensor(img):
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)

def _resize_to_multiple(img, multiple=64, max_dim=1024):
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = max(multiple, int(w * scale) // multiple * multiple)
    new_h = max(multiple, int(h * scale) // multiple * multiple)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ── Generate ───────────────────────────────────────────────
@torch.inference_mode()
def generate(prompt, negative, width, height, seed, cfg, denoise, steps=8):
    n = _get_nodes()
    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    latent = n["EmptySD3LatentImage"].generate(width, height, batch_size=1)[0]
    samples = n["KSampler"].sample(
        _unet, seed, min(steps, 8), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

# ── Img2Img ────────────────────────────────────────────────
@torch.inference_mode()
def img2img(input_image, prompt, negative, seed, cfg, denoise, steps=8, mask=None):
    """img2img with optional mask support.
    If mask is provided, routes through inpaint to preserve unmasked regions.
    """
    if mask is not None:
        return inpaint(input_image, mask, prompt, negative, seed, cfg, float(denoise), int(steps))

    n = _get_nodes()
    input_image = _resize_to_multiple(input_image)
    img_tensor = _pil_to_tensor(input_image)
    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    latent = n["VAEEncode"].encode(_vae, img_tensor)[0]
    samples = n["KSampler"].sample(
        _unet, seed, min(steps, 8), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

# ── Inpaint (Fooocus-style) ───────────────────────────────
def _compute_crop_region(mask_np, padding=0.30):
    indices = np.where(mask_np > 0)
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        return None
    a, b = np.min(indices[0]), np.max(indices[0])
    c, d = np.min(indices[1]), np.max(indices[1])
    h_center, h_half = (b + a) // 2, (b - a) // 2
    w_center, w_half = (d + c) // 2, (d - c) // 2
    size = int(max(h_half, w_half) * (1.0 + padding))
    a = max(0, h_center - size)
    b = min(mask_np.shape[0], h_center + size + 1)
    c = max(0, w_center - size)
    d = min(mask_np.shape[1], w_center + size + 1)
    return (a, b, c, d)

def _fooocus_fill(image_np, mask_np):
    current = image_np.copy()
    raw = image_np.copy()
    area = np.where(mask_np < 127)
    store = raw[area]
    for k, repeats in [(512,2),(256,2),(128,4),(64,4),(33,8),(15,8),(5,16),(3,16)]:
        for _ in range(repeats):
            pil_img = Image.fromarray(current)
            pil_img = pil_img.filter(ImageFilter.BoxBlur(k))
            current = np.array(pil_img)
            current[area] = store
    return current

@torch.inference_mode()
def inpaint(original, mask_combined, prompt, negative, seed, cfg, denoise, steps=8):
    """original: PIL Image, mask_combined: numpy uint8 array (255=masked)"""
    n = _get_nodes()
    prompt_enhanced = _enhance_prompt(prompt)

    crop = _compute_crop_region(mask_combined)
    if crop is None:
        raise ValueError("No mask detected.")
    a, b, c, d = crop

    cropped_img = np.array(original)[a:b, c:d]
    cropped_mask = mask_combined[a:b, c:d]

    crop_pil = Image.fromarray(cropped_img)
    crop_pil = _resize_to_multiple(crop_pil, multiple=64, max_dim=1024)
    cw, ch = crop_pil.size

    mask_pil = Image.fromarray(cropped_mask).resize((cw, ch), Image.NEAREST)
    mask_resized = np.array(mask_pil)

    filled = _fooocus_fill(np.array(crop_pil), mask_resized)
    filled_pil = Image.fromarray(filled)

    filled_tensor = _pil_to_tensor(filled_pil)
    latent_base = n["VAEEncode"].encode(_vae, filled_tensor)[0]
    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0).unsqueeze(0)

    pos = n["CLIPTextEncode"].encode(_clip, prompt_enhanced)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]

    latent = n["SetLatentNoiseMask"].set_mask(latent_base, mask_tensor)[0]
    samples = n["KSampler"].sample(
        _unet, seed, min(steps, 8), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]

    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    result_crop = np.array(decoded * 255, dtype=np.uint8)[0]
    result_crop_pil = Image.fromarray(result_crop).resize((d - c, b - a), Image.LANCZOS)

    # Composite back
    result = np.array(original).copy()
    result_crop_np = np.array(result_crop_pil)
    mask_composite = Image.fromarray(cropped_mask).resize((d - c, b - a), Image.LANCZOS)
    mask_float = np.array(mask_composite).astype(np.float32)[:, :, None] / 255.0

    mask_blur = Image.fromarray((mask_float[:, :, 0] * 255).astype(np.uint8))
    mask_blur = mask_blur.filter(ImageFilter.GaussianBlur(3))
    mask_float = np.array(mask_blur).astype(np.float32)[:, :, None] / 255.0

    old_region = result[a:b, c:d].astype(np.float32)
    new_region = result_crop_np.astype(np.float32)
    blended = new_region * mask_float + old_region * (1 - mask_float)
    result[a:b, c:d] = blended.clip(0, 255).astype(np.uint8)

    return Image.fromarray(result)

def _enhance_prompt(prompt):
    boosters = "photorealistic, high quality, detailed, natural lighting, 8K"
    if any(b in prompt.lower() for b in ["realistic", "quality", "detailed", "8k", "4k"]):
        return prompt
    return f"{prompt}, {boosters}"
