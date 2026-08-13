# ============================================================
#  FreeFakeStudio — Workspace & Path Configuration
#  Manages persistent paths and environment detection.
# ============================================================

import os
import sys


def running_in_colab():
    """Detect if running inside Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


# ── Workspace Resolution ───────────────────────────────────
# In Colab: set by notebook before importing app
# Locally: defaults to current directory
WORKSPACE_DIR = os.environ.get("FREEFAKESTUDIO_WORKSPACE", ".")
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", "/content/ComfyUI")


def configure_workspace(workspace_dir=None, comfyui_root=None):
    """Configure workspace paths. Called by the notebook before app launch."""
    global WORKSPACE_DIR, COMFYUI_ROOT
    if workspace_dir:
        WORKSPACE_DIR = workspace_dir
    if comfyui_root:
        COMFYUI_ROOT = comfyui_root


def get_save_dir():
    """Return the persistent results directory."""
    if running_in_colab():
        save_dir = os.path.join(WORKSPACE_DIR, "results")
    else:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def ensure_comfyui_path():
    """Add ComfyUI to sys.path if not already present."""
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)


def get_model_dir(subdir):
    """Get path to a model subdirectory under ComfyUI."""
    return os.path.join(COMFYUI_ROOT, "models", subdir)
