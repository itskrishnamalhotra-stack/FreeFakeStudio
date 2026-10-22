"""Platform detection shared by notebook, server and model engines."""
import os
from pathlib import Path


def provider():
    forced = os.environ.get("FFS_PROVIDER", "auto").strip().lower()
    if forced in {"colab", "kaggle", "local"}:
        return forced
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return "colab"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


def cloud_runtime():
    return provider() in {"colab", "kaggle"}


def mock_mode():
    return os.environ.get("FREEFAKESTUDIO_DEV", "").lower() in {"1", "true", "yes"} or not cloud_runtime()


def local_runtime_root():
    default = "/tmp/freefakestudio_runtime" if provider() == "kaggle" else "/content/freefakestudio_runtime"
    return Path(os.environ.get("FFS_LOCAL_RUNTIME", default)).resolve()
