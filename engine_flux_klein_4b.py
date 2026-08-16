# ============================================================
#  Engine: FLUX.2-klein 4B
#  Generation, img2img, mask-based inpaint
#  Uses ComfyUI nodes — smaller model, fits T4 better
#  UNET: flux-2-klein-4b.safetensors (7.75 GB)
#  CLIP: qwen_3_4b_fp4_flux2.safetensors (FLUX.2-specific encoder)
# ============================================================

import gc, os, sys, torch, numpy as np
from PIL import Image, ImageFilter

_loaded = False
_unet = None
_clip = None
_vae = None
_nodes = {}
_loaded_encoder = None

# ── Node references ────────────────────────────────────────
def _get_nodes():
    global _nodes
    if not _nodes:
        comfyui_root = os.environ.get("COMFYUI_ROOT", "/content/ComfyUI")
        if comfyui_root not in sys.path:
            sys.path.insert(0, comfyui_root)
        from nodes import NODE_CLASS_MAPPINGS
        _nodes = {
            "UNETLoader":       NODE_CLASS_MAPPINGS["UNETLoader"](),
            "CLIPLoader":       NODE_CLASS_MAPPINGS["CLIPLoader"](),
            "VAELoader":        NODE_CLASS_MAPPINGS["VAELoader"](),
            "CLIPTextEncode":   NODE_CLASS_MAPPINGS["CLIPTextEncode"](),
            "KSampler":         NODE_CLASS_MAPPINGS["KSampler"](),
            "VAEDecode":        NODE_CLASS_MAPPINGS["VAEDecode"](),
            "VAEEncode":        NODE_CLASS_MAPPINGS["VAEEncode"](),
            "EmptyLatentImage": NODE_CLASS_MAPPINGS["EmptyLatentImage"](),
            "SetLatentNoiseMask": NODE_CLASS_MAPPINGS["SetLatentNoiseMask"](),
        }
        
        try:
            import nodes
            if hasattr(nodes, "init_extra_nodes"):
                nodes.init_extra_nodes()
            from nodes import NODE_CLASS_MAPPINGS as ALL_NODES
            if "DifferentialDiffusion" in ALL_NODES:
                _nodes["DifferentialDiffusion"] = ALL_NODES["DifferentialDiffusion"]()
        except:
            pass
            
        if "DifferentialDiffusion" not in _nodes:
            try:
                from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
                _nodes["DifferentialDiffusion"] = DifferentialDiffusion()
            except ImportError:
                pass

        try:
            import node_helpers
            _nodes["conditioning_set_values"] = node_helpers.conditioning_set_values
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "ComfyUI reference-conditioning support is missing. "
                "Update ComfyUI and restart the Colab runtime."
            ) from exc

    return _nodes

# ── Load / Unload ──────────────────────────────────────────
def load():
    global _loaded, _unet, _clip, _vae, _loaded_encoder
    if _loaded:
        return
    n = _get_nodes()
    print("⏳ Loading FLUX.2-klein 4B...")
    with torch.inference_mode():
        _unet = n["UNETLoader"].load_unet(
            "flux-2-klein-4b.safetensors", "fp8_e4m3fn_fast"
        )[0]
        mode = os.environ.get("FFS_FLUX_ENCODER_MODE", "official").strip().lower()
        custom_name = os.environ.get("FFS_FLUX_CUSTOM_ENCODER_FILE", "").strip()
        if mode == "custom":
            if not custom_name:
                raise RuntimeError(
                    "Custom FLUX encoder was selected but the launcher did not configure a file."
                )
            custom_path = os.path.join(
                os.environ.get("COMFYUI_ROOT", "/content/ComfyUI"),
                "models", "text_encoders", custom_name,
            )
            if not os.path.isfile(custom_path):
                raise RuntimeError(f"Custom FLUX encoder is missing: {custom_path}")
            if custom_name.lower().endswith(".gguf"):
                from gguf_nodes import load_gguf_node_mappings

                gguf_nodes = load_gguf_node_mappings(
                    os.environ.get("COMFYUI_ROOT", "/content/ComfyUI")
                )
                print(f"[memory] FLUX text encoder | custom GGUF | {custom_name}")
                _clip = gguf_nodes["CLIPLoaderGGUF"]().load_clip(custom_name, type="flux2")[0]
            elif custom_name.lower().endswith(".safetensors"):
                print(f"[memory] FLUX text encoder | custom Safetensors | {custom_name}")
                _clip = n["CLIPLoader"].load_clip(custom_name, type="flux2")[0]
            else:
                raise RuntimeError("Custom FLUX encoder must be GGUF or Safetensors.")
            _loaded_encoder = "custom"
        else:
            # FLUX.2-specific Qwen3 4B encoder; do not substitute the Z-Image encoder.
            print("[memory] FLUX text encoder | official FP4")
            _clip = n["CLIPLoader"].load_clip(
                "qwen_3_4b_fp4_flux2.safetensors", type="flux2"
            )[0]
            _loaded_encoder = "official"
        _vae  = n["VAELoader"].load_vae("flux2-vae.safetensors")[0]
    _loaded = True
    print("✅ FLUX.2-klein 4B loaded!")

def unload():
    global _loaded, _unet, _clip, _vae, _loaded_encoder
    _unet = None
    _clip = None
    _vae = None
    _loaded_encoder = None
    _loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
    print("🗑️ FLUX.2-klein 4B unloaded")

def is_loaded():
    return _loaded


def get_loaded_encoder():
    return _loaded_encoder

# ── Helpers ────────────────────────────────────────────────
def _pil_to_tensor(img):
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)

def _resize_to_multiple(img, multiple=64, max_dim=1024):
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = max(multiple, int(w * scale) // multiple * multiple)
    new_h = max(multiple, int(h * scale) // multiple * multiple)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _normalize_references(input_images):
    references = input_images if isinstance(input_images, (list, tuple)) else [input_images]
    references = [image.convert("RGB") for image in references if image is not None]
    if not references:
        raise ValueError("At least one FLUX reference image is required.")
    if len(references) > 4:
        raise ValueError("FLUX.2 Klein supports at most 4 reference images in this app.")
    return references


def _resize_reference(img, max_pixels, multiple=64):
    width, height = img.size
    scale = min((max_pixels / max(1, width * height)) ** 0.5, 1.0)
    resized_width = max(multiple, int(width * scale) // multiple * multiple)
    resized_height = max(multiple, int(height * scale) // multiple * multiple)
    return img.resize((resized_width, resized_height), Image.LANCZOS)


def _add_reference_conditioning(nodes, conditioning, references):
    # Keep the total reference-token budget close to one 1024px image. This
    # makes 2-4 reference editing practical on a free Colab T4.
    pixels_per_reference = (1024 * 1024) // len(references)
    print(
        f"[flux-reference] count={len(references)} "
        f"pixel_budget_each={pixels_per_reference}"
    )
    for index, image in enumerate(references, start=1):
        resized = _resize_reference(image, pixels_per_reference)
        tensor = _pil_to_tensor(resized)
        latent = nodes["VAEEncode"].encode(_vae, tensor)[0]
        conditioning = nodes["conditioning_set_values"](
            conditioning,
            {"reference_latents": [latent["samples"]]},
            append=True,
        )
        print(f"[flux-reference] image={index} encoded={resized.width}x{resized.height}")
    return conditioning

# ── Generate ───────────────────────────────────────────────
@torch.inference_mode()
def generate(prompt, negative, width, height, seed, cfg, denoise, steps=4):
    n = _get_nodes()
    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    latent = n["EmptyLatentImage"].generate(width, height, batch_size=1)[0]
    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

# ── Generate with References ──────────────────────────────
@torch.inference_mode()
def generate_with_references(input_images, prompt, negative, width, height,
                             seed, cfg, denoise, steps=4):
    """Text-to-image on a clean canvas, conditioned by reference images.
    Unlike img2img, this starts from an EmptyLatentImage so no input image's
    structure is used as a base. The reference images influence style and
    content through conditioning only. Supports 1-4 reference images.
    """
    references = _normalize_references(input_images)
    n = _get_nodes()
    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    pos = _add_reference_conditioning(n, pos, references)
    latent = n["EmptyLatentImage"].generate(width, height, batch_size=1)[0]
    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

# ── Img2Img ────────────────────────────────────────────────
@torch.inference_mode()
def img2img(input_image, prompt, negative, seed, cfg, denoise, steps=4, mask=None):
    """Native FLUX.2 reference editing with optional mask support.
    If mask is provided (numpy uint8, 255=areas to regenerate), routes through
    the inpaint pipeline which preserves unmasked regions pixel-perfectly.
    """
    references = _normalize_references(input_image)
    primary = references[0]
    if mask is not None:
        return inpaint(
            primary, mask, prompt, negative, seed, cfg, float(denoise), steps,
            references=references,
        )

    n = _get_nodes()
    primary = _resize_to_multiple(primary)
    img_tensor = _pil_to_tensor(primary)
    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    pos = _add_reference_conditioning(n, pos, references)
    latent = n["VAEEncode"].encode(_vae, img_tensor)[0]
    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

# ── Inpaint (Fooocus-style mask-based) ────────────────────
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
def inpaint(original, mask_combined, prompt, negative, seed, cfg, denoise, steps=4,
            references=None):
    """original: PIL Image, mask_combined: numpy uint8 array (255=masked)"""
    n = _get_nodes()

    # Pass the full image to maintain aspect ratio and body proportions
    crop_pil = _resize_to_multiple(original, multiple=64, max_dim=1024)
    cw, ch = crop_pil.size

    mask_pil = Image.fromarray(mask_combined).resize((cw, ch), Image.NEAREST)
    mask_resized = np.array(mask_pil)

    # Encode raw unadulterated pixels so DifferentialDiffusion retains full structural context under the denoise
    crop_tensor = _pil_to_tensor(crop_pil)
    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0).unsqueeze(0)
    latent_base = n["VAEEncode"].encode(_vae, crop_tensor)[0]

    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    if references:
        pos = _add_reference_conditioning(n, pos, _normalize_references(references))

    latent = n["SetLatentNoiseMask"].set_mask(latent_base, mask_tensor)[0]
    
    model_to_sample = _unet
    if "DifferentialDiffusion" in n:
        try:
            diff_node = n["DifferentialDiffusion"]
            if hasattr(diff_node, "apply"):
                res = diff_node.apply(model_to_sample)
            elif hasattr(diff_node.__class__, "execute"):
                res = diff_node.__class__.execute(model_to_sample)
            elif hasattr(diff_node, "execute"):
                res = diff_node.execute(model_to_sample)
            else:
                res = model_to_sample
                
            if isinstance(res, tuple):
                model_to_sample = res[0]
            elif hasattr(res, "__class__") and "NodeOutput" in res.__class__.__name__:
                model_to_sample = res.args[0] if hasattr(res, "args") else res[0]
            else:
                model_to_sample = res
        except Exception as e:
            print(f"⚠️ DifferentialDiffusion ignored due to API change: {e}")
        
    samples = n["KSampler"].sample(
        model_to_sample, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]

    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    result_full_np = np.array(decoded * 255, dtype=np.uint8)[0]
    result_full_pil = Image.fromarray(result_full_np).resize(original.size, Image.LANCZOS)
    
    # Composite the generated full image back over the original image using the mask
    result = np.array(original).copy()
    mask_composite = Image.fromarray(mask_combined).resize(original.size, Image.LANCZOS)
    mask_float = np.array(mask_composite).astype(np.float32)[:, :, None] / 255.0

    mask_blur = Image.fromarray((mask_float[:, :, 0] * 255).astype(np.uint8))
    mask_blur = mask_blur.filter(ImageFilter.GaussianBlur(8))
    mask_float = np.array(mask_blur).astype(np.float32)[:, :, None] / 255.0

    old_region = result.astype(np.float32)
    new_region = np.array(result_full_pil).astype(np.float32)
    blended = new_region * mask_float + old_region * (1 - mask_float)
    
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
