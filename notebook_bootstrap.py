"""Notebook-only bootstrap. No model work is performed on a local workstation."""
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys


ALIASES = {"NGROK_AUTH_TOKEN": "FFS_NGROK_AUTHTOKEN", "NGROK_AUTHTOKEN": "FFS_NGROK_AUTHTOKEN", "HUGGINGFACE_TOKEN": "HF_TOKEN",
           "HUGGING_FACE_TOKEN": "HF_TOKEN", "FLUX_ENCODER": "FFS_FLUX_ENCODER_MODE"}
DIRECT = {"GEMINI_API_KEY", "TAVILY_API_KEY", "HF_TOKEN", "GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET", "GOOGLE_DRIVE_REFRESH_TOKEN", "GOOGLE_DRIVE_WORKSPACE_FOLDER_ID", "INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_USER_ID", "INSTAGRAM_GRAPH_VERSION", "INSTAGRAM_LOGIN_MODE"}
SETTINGS = {"PUBLIC_ROUTE", "PRELOAD_FLUX", "PRELOAD_AVATAR_VISION", "FAST_RESTORE", "STAGE_FLUX_TO_SSD", "WARMUP_ENABLED", "WARMUP_SIZE", "EXPOSE_AFTER_WARMUP", "FORCE_REBUILD_BUNDLES", "FORCE_CACHE_REFRESH", "STARTUP_VERBOSE", "FLUX_CUSTOM_ENCODER_URL", "GEMINI_MODEL", "AVATAR_REFERENCE_DOMAINS", "AVATAR_REFERENCE_TIME_RANGE", "AVATAR_SEARCH_ROUNDS", "AVATAR_GALLERY_RETRIES", "AVATAR_MAX_CANDIDATE_DOWNLOADS", "AVATAR_VISION_MAX_EDGE", "AVATAR_VISION_MAX_TOKENS", "MODEL_SET"}


def parse_keys(text):
    values = {}
    for line in text.lstrip("\ufeff").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ";", "//")):
            continue
        line = re.sub(r"^export\s+", "", line)
        pair = re.split(r"[=:]", line, maxsplit=1)
        if len(pair) != 2:
            continue
        name, value = pair[0].strip().upper(), pair[1].strip().strip('"').strip("'")
        target = ALIASES.get(name, "FFS_" + name if name in SETTINGS else name)
        allowed = DIRECT | set(ALIASES.values()) | {"FFS_" + key for key in SETTINGS}
        if target in allowed and value:
            values[target] = value
    return values


def start(form, workspace, app_dir, kaggle):
    workspace, app_dir = Path(workspace), Path(app_dir)
    os.environ["FFS_PROVIDER"] = "kaggle" if kaggle else "colab"
    os.environ["FFS_WORKSPACE"] = str(workspace)
    os.environ["FREEFAKESTUDIO_WORKSPACE"] = str(workspace)
    os.environ["FFS_LOCAL_RUNTIME"] = "/tmp/freefakestudio_runtime" if kaggle else "/content/freefakestudio_runtime"
    import importlib
    import runtime_process
    # The notebook kernel survives reruns; use the freshly updated shutdown helper.
    importlib.reload(runtime_process).stop_previous(workspace, os.environ["FFS_LOCAL_RUNTIME"])
    os.environ["FFS_ACCESS_TOKEN"] = secrets.token_urlsafe(32)
    os.environ.pop("FREEFAKESTUDIO_DEV", None)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        memory = [int(value.strip()) for value in gpu.stdout.splitlines() if value.strip().isdigit()]
        if gpu.returncode or not memory or max(memory) < 12000:
            raise RuntimeError("Enable a GPU runtime with at least 12 GB VRAM before running setup. No models were downloaded.")
    except FileNotFoundError:
        raise RuntimeError("No NVIDIA GPU is attached. Enable GPU in the notebook session settings first.") from None
    config = workspace / "config"
    config.mkdir(exist_ok=True)
    saved_keys = config / "freefakestudio_keys.txt"
    path_text = str(form.get("KEYS_TXT_PATH", "")).strip()
    if form.get("UPLOAD_KEYS_TXT"):
        if kaggle:
            raise RuntimeError("On Kaggle attach a PRIVATE keys dataset and set KEYS_TXT_PATH, or use Add-ons > Secrets. Colab's upload widget is not available.")
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise RuntimeError("Upload exactly one private keys text file")
        saved_keys.write_bytes(next(iter(uploaded.values())))
        saved_keys.chmod(0o600)
    values = {}
    candidate = Path(path_text).expanduser() if path_text else saved_keys
    if path_text and not candidate.is_file():
        raise RuntimeError("KEYS_TXT_PATH does not exist in this cloud runtime. A Windows path cannot be used here.")
    if candidate.is_file():
        values.update(parse_keys(candidate.read_text(encoding="utf-8")))
    names = DIRECT | {"FFS_NGROK_AUTHTOKEN"}
    if kaggle:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for name in names:
            if not values.get(name):
                try:
                    value = client.get_secret(name)
                    if value:
                        values[name] = value
                except Exception:
                    pass
    else:
        from google.colab import userdata
        for name in names:
            if not values.get(name):
                try:
                    value = userdata.get(name)
                    if value:
                        values[name] = value
                except Exception:
                    pass
    # Explicit secret overrides win; nonsecret form controls win over saved settings.
    for name in DIRECT | set(ALIASES):
        value = str(form.get(name, "") or "").strip()
        if value:
            values[ALIASES.get(name, name)] = value
    for name in SETTINGS:
        if name in form:
            value = form[name]
            if name in {"FLUX_CUSTOM_ENCODER_URL", "GEMINI_MODEL"} and not str(value).strip():
                continue
            values["FFS_" + name] = ("1" if value else "0") if isinstance(value, bool) else str(value)
    values["FFS_FLUX_ENCODER_MODE"] = str(form.get("FLUX_ENCODER", "Official")).lower()
    os.environ.update(values)
    os.environ["FFS_UPDATE"] = "1" if form.get("UPDATE_APP") else ""
    os.environ["FFS_REPAIR"] = "1" if form.get("REPAIR_INSTALL") else ""
    if kaggle:
        if not values.get("FFS_NGROK_AUTHTOKEN"):
            raise RuntimeError("Kaggle requires NGROK_AUTH_TOKEN in the private keys file (or FFS_NGROK_AUTHTOKEN in Secrets)")
        os.environ["FFS_PUBLIC_ROUTE"] = "ngrok"
        os.environ["FFS_STORAGE_MODE"] = "drive_api"
        from drive_store import DriveStore, restore_selection
        store = DriveStore.from_environment()
        # Clone the backend before restoring its model directory, without overwriting user data.
        comfy = workspace / "ComfyUI"
        if not (comfy / ".git").exists():
            subprocess.run(["git", "clone", "--depth", "1", "--branch", "v0.28.0", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy)], check=True)
        print("Connecting privately to the existing Google Drive workspace...", flush=True)
        store.restore(restore_selection)
        if saved_keys.is_file():
            for name, value in parse_keys(saved_keys.read_text(encoding="utf-8")).items():
                if not os.environ.get(name):
                    os.environ[name] = value
        # Optional encoder files may have arbitrary filenames; the saved manifest identifies them.
        import json
        encoder = config / "flux_encoder.json"
        if encoder.exists():
            for value in json.loads(encoder.read_text()).values():
                if isinstance(value, str) and value.endswith((".gguf", ".safetensors")):
                    relative = "ComfyUI/models/text_encoders/" + Path(value).name
                    if relative in store.index and not (workspace / relative).exists():
                        store.download(relative, store.index[relative])
        print("Drive restored. Results will sync at durable job checkpoints. Do not run a second writer against this folder.", flush=True)
    else:
        os.environ["FFS_STORAGE_MODE"] = "direct"
    os.environ["FFS_UI"] = "web"
    exec(compile((app_dir / "launch.py").read_text(encoding="utf-8"), str(app_dir / "launch.py"), "exec"), {"__name__": "__main__", "__file__": str(app_dir / "launch.py")})
