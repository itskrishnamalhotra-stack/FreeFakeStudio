"""Single-port React/FastAPI studio; local runs always default to mock models."""
import os
from pathlib import Path
import traceback

from runtime_environment import cloud_runtime
import model_manager
import startup_restore
import workspace


def flag(name, default=True):
    return os.environ.get(name, "1" if default else "0").lower() in {"1", "true", "yes"}


def prepare_models():
    path = os.environ.get("FFS_READY_STATE_PATH")
    def state(**values):
        if path:
            startup_restore.write_readiness(path, fingerprint=os.environ.get("FFS_RUNTIME_FINGERPRINT_ID", "unknown"), **values)
    state(ready=False, phase="starting", model=None, warmup="pending", vision="pending")
    if model_manager.DEV_MODE:
        for name in model_manager.MODEL_NAMES:
            model_manager.set_model_availability(name, True)
        state(ready=True, phase="development", model="mock", warmup="disabled", vision="disabled")
        print("Local mock preview: real AI inference and model downloads are disabled.", flush=True)
        return
    model_manager.check_model_files(os.environ.get("COMFYUI_ROOT", "/content/ComfyUI"))
    warmup, vision = "disabled", "disabled"
    try:
        if flag("FFS_EXPOSE_AFTER_WARMUP") and not flag("FFS_PRELOAD_FLUX"):
            raise RuntimeError("Enable PRELOAD_FLUX when EXPOSE_AFTER_WARMUP is enabled")
        if flag("FFS_PRELOAD_FLUX"):
            print("Preloading FLUX.2-klein 4B before publishing the interface...", flush=True)
            engine = model_manager.ensure_model(model_manager.FLUX_MODEL_NAME, status_callback=lambda message: print(message, flush=True))
            if flag("FFS_WARMUP_ENABLED"):
                engine.warmup(int(os.environ.get("FFS_WARMUP_SIZE", "256")))
                warmup = "complete"
            if flag("FFS_PRELOAD_AVATAR_VISION"):
                try:
                    import avatar_vision
                    if flag("FFS_WARMUP_ENABLED"):
                        avatar_vision.warmup(128)
                        vision = "complete"
                    else:
                        avatar_vision.preload()
                        vision = "loaded"
                except Exception:
                    vision = "deferred"
                    debug = Path(workspace.get_save_dir()) / "_debug"
                    debug.mkdir(parents=True, exist_ok=True)
                    (debug / "vision-startup.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    print("Vision preload deferred; details saved in results/_debug/vision-startup.txt", flush=True)
        state(ready=True, phase="model-ready", model=model_manager.get_current_model(), warmup=warmup, vision=vision)
    except Exception as exc:
        state(ready=False, phase="startup-failed", model=model_manager.get_current_model(), warmup="failed", vision=vision, error=str(exc))
        raise


def main():
    if cloud_runtime() and not os.environ.get("FFS_ACCESS_TOKEN"):
        raise RuntimeError("A private access token is required. Start through the notebook launcher.")
    bundle = Path(__file__).parent / "frontend" / "dist" / "index.html"
    if not bundle.is_file():
        raise RuntimeError("The prebuilt frontend is missing. Update the application checkout.")
    prepare_models()
    persist = None
    if os.environ.get("FFS_STORAGE_MODE") == "drive_api":
        from drive_store import DriveStore
        store = DriveStore.from_environment()
        persist = store.sync
    from web_api import create_app
    import uvicorn
    app = create_app(persist=persist)
    port = int(os.environ.get("FREEFAKESTUDIO_PORT", "7860"))
    host = "0.0.0.0" if cloud_runtime() else "127.0.0.1"
    print(f"Running on local URL: http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, access_log=False, proxy_headers=True, forwarded_allow_ips="127.0.0.1")


if __name__ == "__main__":
    main()
