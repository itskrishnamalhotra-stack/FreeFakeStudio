# @title Start FreeFakeStudio - Colab or Kaggle
# @markdown Colab renders these as form controls. In Kaggle edit their values in this same cell.
UPDATE_APP = False  # @param {type:"boolean"}
REPAIR_INSTALL = False  # @param {type:"boolean"}
WORKSPACE_DIR = "/content/drive/MyDrive/FreeFakeStudio"  # @param {type:"string"}
KEYS_TXT_PATH = ""  # @param {type:"string"}
UPLOAD_KEYS_TXT = False  # @param {type:"boolean"}
PUBLIC_ROUTE = "Colab proxy"  # @param ["Auto", "Colab proxy", "ngrok"]
MODEL_SET = "flux"  # @param ["flux", "all"]
PRELOAD_FLUX = True  # @param {type:"boolean"}
PRELOAD_AVATAR_VISION = True  # @param {type:"boolean"}
FAST_RESTORE = True  # @param {type:"boolean"}
STAGE_FLUX_TO_SSD = True  # @param {type:"boolean"}
WARMUP_ENABLED = True  # @param {type:"boolean"}
WARMUP_SIZE = 256  # @param [128, 192, 256, 320, 384, 512]
EXPOSE_AFTER_WARMUP = True  # @param {type:"boolean"}
FORCE_REBUILD_BUNDLES = False  # @param {type:"boolean"}
FORCE_CACHE_REFRESH = False  # @param {type:"boolean"}
STARTUP_VERBOSE = True  # @param {type:"boolean"}
FLUX_ENCODER = "Official"  # @param ["Official", "Custom"]
FLUX_CUSTOM_ENCODER_URL = ""  # @param {type:"string"}
# @markdown Keep credentials in a private keys file or notebook Secrets, never in a published notebook.
NGROK_AUTH_TOKEN = ""  # @param {type:"string"}
HUGGINGFACE_TOKEN = ""  # @param {type:"string"}
GEMINI_API_KEY = ""  # @param {type:"string"}
TAVILY_API_KEY = ""  # @param {type:"string"}
GOOGLE_DRIVE_WORKSPACE_FOLDER_ID = ""  # @param {type:"string"}
GOOGLE_DRIVE_CLIENT_ID = ""  # @param {type:"string"}
GOOGLE_DRIVE_CLIENT_SECRET = ""  # @param {type:"string"}
GOOGLE_DRIVE_REFRESH_TOKEN = ""  # @param {type:"string"}
GEMINI_MODEL = ""  # @param {type:"string"}
AVATAR_REFERENCE_DOMAINS = "instagram.com"  # @param {type:"string"}
AVATAR_REFERENCE_TIME_RANGE = "month"  # @param ["", "day", "week", "month", "year"]
AVATAR_SEARCH_ROUNDS = 3  # @param {type:"integer"}
AVATAR_GALLERY_RETRIES = 2  # @param {type:"integer"}
AVATAR_MAX_CANDIDATE_DOWNLOADS = 60  # @param {type:"integer"}
AVATAR_VISION_MAX_EDGE = 768  # @param [768, 1024, 1536]
AVATAR_VISION_MAX_TOKENS = 900  # @param {type:"integer"}
PROJECT_REPO_URL = "https://github.com/itskrishnamalhotra-stack/FreeFakeStudio.git"

import os
from pathlib import Path
import subprocess
import sys

is_kaggle = Path("/kaggle/working").is_dir() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
if not is_kaggle:
    try:
        from google.colab import drive
    except ImportError:
        raise RuntimeError("This setup cell only runs on Colab or Kaggle. Use web_server.py for local mock previews.") from None
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
workspace = Path("/tmp/freefakestudio_workspace" if is_kaggle else WORKSPACE_DIR).resolve()
workspace.mkdir(parents=True, exist_ok=True)
app_dir = workspace / "app"

def run(args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-1800:])
    return result.stdout

if not (app_dir / ".git").exists():
    if app_dir.exists() and any(app_dir.iterdir()):
        raise RuntimeError("App folder is not a Git checkout; it was left untouched. Choose another workspace.")
    run(["git", "clone", "--depth", "1", PROJECT_REPO_URL, str(app_dir)])
elif UPDATE_APP:
    if run(["git", "-C", str(app_dir), "status", "--porcelain"]).strip():
        from datetime import datetime
        label = "FreeFakeStudio cloud auto-stash before update " + datetime.now().strftime("%Y%m%d-%H%M%S")
        run(["git", "-C", str(app_dir), "stash", "push", "-u", "-m", label])
        print("Cloud source edits preserved in a Git stash before updating.")
    run(["git", "-C", str(app_dir), "pull", "--ff-only"])
sys.path.insert(0, str(app_dir))
# Reload bootstrap modules after updating within an existing notebook kernel.
for module in ("notebook_bootstrap", "drive_store", "runtime_environment"):
    sys.modules.pop(module, None)
from notebook_bootstrap import start
start(globals(), workspace, app_dir, is_kaggle)
