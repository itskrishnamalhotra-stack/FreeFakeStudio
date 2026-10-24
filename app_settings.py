"""Workspace-wide preferences, independent of avatar identity versions."""
import json
from pathlib import Path

import workspace
from avatar_studio import _atomic_json


def load(root=None):
    path = Path(root or workspace.get_save_dir()) / "_settings" / "app.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"gpu_memory_management": False}
    except (ValueError, OSError) as exc:
        raise ValueError("Cannot read app settings. Restore results/_settings/app.json from backup.") from exc
    if not isinstance(data, dict) or type(data.get("gpu_memory_management", False)) is not bool:
        raise ValueError("Invalid app GPU memory setting; expected true or false.")
    return {"gpu_memory_management": data.get("gpu_memory_management", False)}


def save(root, gpu_memory_management):
    if type(gpu_memory_management) is not bool:
        raise ValueError("GPU memory management must be true or false.")
    data = {"gpu_memory_management": gpu_memory_management}
    _atomic_json(Path(root) / "_settings" / "app.json", data)
    return data


def memory_management_enabled(root=None):
    return load(root)["gpu_memory_management"]


def prepare_vision(root=None):
    if memory_management_enabled(root):
        import model_manager
        print("[memory] App-wide management ON: releasing image model before vision checks")
        model_manager.unload_current()


def prepare_generation(root=None):
    if memory_management_enabled(root):
        import avatar_vision
        print("[memory] App-wide management ON: releasing vision model before generation")
        avatar_vision.unload()
