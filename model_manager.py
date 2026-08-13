# ============================================================
#  FreeFakeStudio — Thread-Safe Model Manager
#  Handles loading/unloading of AI models one at a time.
#  Supports mock mode for local development without GPU.
# ============================================================

import os
import gc
import sys
import threading
import time
import random

# ── Environment Detection ──────────────────────────────────
def running_in_colab():
    """Detect if running inside Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Dev mode: enabled when NOT in Colab, or forced via environment variable
DEV_MODE = os.environ.get("FREEFAKESTUDIO_DEV", "").lower() in ("1", "true", "yes") or not running_in_colab()

# ── Model Registry ─────────────────────────────────────────
# Only the 3 required models
MODEL_REGISTRY = {
    "Z-Image Turbo": {
        "engine_module": "engine_z_image",
        "description": "Fast generation",
        "capabilities": ["generate", "img2img", "inpaint"],
        "default_steps": 8,
        "default_cfg": 1.0,
        "default_denoise": 1.0,
        "model_file": "z_image_turbo-Q3_K_M.gguf",
        "required_files": [
            ("diffusion_models", "z_image_turbo-Q3_K_M.gguf"),
            ("text_encoders", "qwen_3_4b_fp4_mixed.safetensors"),
            ("vae", "ae.safetensors"),
        ],
    },
    "FLUX.2-klein 4B": {
        "engine_module": "engine_flux_klein_4b",
        "description": "Generate + image editing",
        "capabilities": ["generate", "img2img", "inpaint"],
        "default_steps": 20,
        "default_cfg": 1.0,
        "default_denoise": 1.0,
        "img2img_denoise": 0.45,
        "inpaint_denoise": 0.75,
        "model_file": "flux-2-klein-4b.safetensors",
        "required_files": [
            ("diffusion_models", "flux-2-klein-4b.safetensors"),
            ("text_encoders", "qwen_3_4b_fp4_flux2.safetensors"),
            ("vae", "flux2-vae.safetensors"),
        ],
    },
    "ERNIE-Image-Turbo": {
        "engine_module": "engine_ernie_image_turbo",
        "description": "Fast generation + text handling",
        "capabilities": ["generate"],
        "default_steps": 8,
        "default_cfg": 1.0,
        "default_denoise": 1.0,
        "model_file": "ernie-image-turbo-Q6_K.gguf",
        "required_files": [
            ("diffusion_models", "ernie-image-turbo-Q6_K.gguf"),
            ("text_encoders", "ministral-3-3b.safetensors"),
            ("vae", "flux2-vae.safetensors"),
        ],
    },
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())

# ── State ──────────────────────────────────────────────────
_current_model = None
_engines = {}  # name -> engine module (lazy loaded)
_lock = threading.Lock()
_model_availability = {}  # name -> bool, set during startup check
_model_file_report = {}  # name -> list of component validation rows


def _configure_comfy_runtime():
    """Apply low-memory defaults before ComfyUI model management initializes."""
    comfyui_root = os.environ.get("COMFYUI_ROOT", "/content/ComfyUI")
    if comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)
    from comfy.cli_args import args

    args.cache_none = True
    args.cache_classic = False
    args.cache_lru = 0
    args.high_ram = False
    args.enable_dynamic_vram = True
    args.disable_dynamic_vram = False
    args.disable_pinned_memory = True


def _get_engine(model_name):
    """Lazily import the engine module."""
    global _engines
    if model_name in _engines:
        return _engines[model_name]

    if DEV_MODE:
        engine = MockEngine(model_name)
        _engines[model_name] = engine
        return engine

    info = MODEL_REGISTRY[model_name]
    import importlib
    _configure_comfy_runtime()
    engine = importlib.import_module(info["engine_module"])
    _engines[model_name] = engine
    return engine


def get_current_model():
    """Return the name of the currently loaded model, or None."""
    return _current_model


def get_model_status():
    """Return dict of model_name -> status string."""
    result = {}
    for name in MODEL_NAMES:
        if not _model_availability.get(name, True):
            result[name] = "missing"
        elif _current_model == name:
            result[name] = "ready"
        else:
            result[name] = "available"
    return result


def set_model_availability(name, available):
    """Set whether a model's files are installed."""
    _model_availability[name] = available


def _minimum_size(filename):
    """Fast sanity thresholds. These catch interrupted tiny downloads."""
    lower = filename.lower()
    if lower.endswith((".gguf", ".safetensors")):
        if filename in {"ae.safetensors", "flux2-vae.safetensors"}:
            return 50 * 1024 * 1024
        return 500 * 1024 * 1024
    return 1024


def check_model_files(comfyui_root=None):
    """Check which models have their required files present.

    The launcher points /content/ComfyUI at the persistent Drive copy. This
    function follows symlinks and performs only fast size checks so startup does
    not hash multi-GB checkpoints every session.
    """
    if comfyui_root is None:
        comfyui_root = os.environ.get("COMFYUI_ROOT", "/content/ComfyUI")
    _model_file_report.clear()
    for name, info in MODEL_REGISTRY.items():
        all_present = True
        rows = []
        for subdir, filename in info["required_files"]:
            path = os.path.join(comfyui_root, "models", subdir, filename)
            min_size = _minimum_size(filename)
            exists = os.path.isfile(path)
            size = os.path.getsize(path) if exists else 0
            ok = exists and size >= min_size
            rows.append({
                "subdir": subdir,
                "filename": filename,
                "path": path,
                "exists": exists,
                "size": size,
                "min_size": min_size,
                "ok": ok,
            })
            if not ok:
                all_present = False
        _model_availability[name] = all_present
        _model_file_report[name] = rows
    return dict(_model_availability)


def get_model_file_report():
    """Return details from the last check_model_files call."""
    return dict(_model_file_report)


def ensure_model(model_name, status_callback=None):
    """Load the requested model, unloading any other model first.
    
    Thread-safe. Only one model can be loaded at a time.
    
    Args:
        model_name: One of MODEL_NAMES
        status_callback: Optional callable(stage_text) for UI updates
    
    Returns:
        The engine module
    
    Raises:
        ValueError: if model_name is unknown
        RuntimeError: if model files are missing or load fails
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {MODEL_NAMES}")

    if not _model_availability.get(model_name, True):
        raise RuntimeError(
            f"{model_name} is not installed. "
            f"Run the setup notebook to download the required model files."
        )

    with _lock:
        return _ensure_model_locked(model_name, status_callback)


def _ensure_model_locked(model_name, status_callback=None):
    """Internal model loading with lock already held."""
    global _current_model

    def _status(msg):
        if status_callback:
            status_callback(msg)
        print(msg)

    engine = _get_engine(model_name)

    # Already loaded and marked active.
    if _current_model == model_name and hasattr(engine, 'is_loaded') and engine.is_loaded():
        _status(f"[ok] {model_name} ready")
        return engine

    # Unload the tracked previous model first.
    if _current_model and _current_model != model_name:
        old_engine = _get_engine(_current_model)
        _status(f"[..] Unloading {_current_model}")
        try:
            old_engine.unload()
        except Exception as e:
            print(f"Warning: error unloading {_current_model}: {e}")
        _current_model = None

        _status("[..] Clearing GPU memory")
        _clear_memory()

    # Defensive cleanup: if a cached engine reports loaded but is not the
    # tracked active model, unload it before loading the requested one.
    for other_name, other_engine in list(_engines.items()):
        if other_name == model_name:
            continue
        try:
            if hasattr(other_engine, "is_loaded") and other_engine.is_loaded():
                _status(f"[..] Unloading {other_name}")
                other_engine.unload()
                _clear_memory()
        except Exception as e:
            print(f"Warning: error checking {other_name}: {e}")

    # Load requested model
    _status(f"[..] Loading {model_name}")
    try:
        engine.load()
        _current_model = model_name
        _status(f"[ok] {model_name} loaded")
        return engine
    except Exception as e:
        # Failed to load — clean up
        _current_model = None
        try:
            engine.unload()
        except Exception:
            pass
        _clear_memory()
        raise RuntimeError(f"Failed to load {model_name}: {e}") from e


def unload_current(status_callback=None):
    """Unload the currently loaded model (for OOM recovery)."""
    global _current_model
    with _lock:
        if _current_model:
            engine = _get_engine(_current_model)
            name = _current_model
            if status_callback:
                status_callback(f"[..] Unloading {name}")
            try:
                engine.unload()
            except Exception:
                pass
            _current_model = None
            _clear_memory()
            if status_callback:
                status_callback(f"[ok] {name} unloaded")


def _clear_memory():
    """Release engine references plus ComfyUI's internal model registry."""
    gc.collect()
    try:
        import torch
        try:
            import comfy.model_management as comfy_memory
            if hasattr(comfy_memory, "unload_all_models"):
                comfy_memory.unload_all_models()
            if hasattr(comfy_memory, "soft_empty_cache"):
                comfy_memory.soft_empty_cache()
        except Exception as exc:
            print(f"Warning: ComfyUI memory cleanup failed: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
    except ImportError:
        pass


def get_capabilities(model_name):
    """Return list of capabilities for a model."""
    if model_name not in MODEL_REGISTRY:
        return []
    return MODEL_REGISTRY[model_name]["capabilities"]


def supports_img2img(model_name):
    return "img2img" in get_capabilities(model_name)


def supports_inpaint(model_name):
    return "inpaint" in get_capabilities(model_name)


def get_defaults(model_name):
    """Return default generation parameters for a model."""
    if model_name not in MODEL_REGISTRY:
        return {}
    info = MODEL_REGISTRY[model_name]
    return {
        "steps": info.get("default_steps", 20),
        "cfg": info.get("default_cfg", 1.0),
        "denoise": info.get("default_denoise", 1.0),
        "img2img_denoise": info.get("img2img_denoise", 0.45),
        "inpaint_denoise": info.get("inpaint_denoise", 0.75),
    }


# ── Mock Engine for Local Development ──────────────────────
class MockEngine:
    """Simulates an engine for UI testing without GPU/models."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._loaded = False

    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True
        print(f"[MOCK] {self.model_name} loaded")

    def unload(self):
        self._loaded = False
        print(f"[MOCK] {self.model_name} unloaded")

    def _make_placeholder(self, width=512, height=512):
        """Create a placeholder image with model name."""
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (width, height), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)
        # Draw gradient-like stripes
        for y in range(height):
            r = int(30 + 40 * (y / height))
            g = int(20 + 60 * (y / height))
            b = int(50 + 80 * (y / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        # Add text
        text = f"[DEV] {self.model_name}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width - tw) // 2
        y = (height - th) // 2
        draw.text((x, y), text, fill=(200, 200, 255), font=font)
        draw.text((x, y + th + 10), "Mock Mode", fill=(150, 150, 180), font=font)
        return img

    def generate(self, prompt, negative, width, height, seed, cfg, denoise, steps=8):
        return self._make_placeholder(width, height)

    def img2img(self, input_image, prompt, negative, seed, cfg, denoise, steps=20, mask=None):
        if input_image:
            w, h = input_image.size
        else:
            w, h = 512, 512
        return self._make_placeholder(w, h)

    def inpaint(self, original, mask_combined, prompt, negative, seed, cfg, denoise, steps=20):
        if original:
            w, h = original.size
        else:
            w, h = 512, 512
        return self._make_placeholder(w, h)
