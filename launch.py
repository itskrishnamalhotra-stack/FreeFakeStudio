# ============================================================
# FreeFakeStudio Colab launcher
# Persistent Drive setup, dependency checks, model validation, launch.
# ============================================================

import importlib
import importlib.metadata
import importlib.util
import gc
import json
import os
import platform
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import unquote, urlparse

from IPython.display import HTML, clear_output, display


def running_in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


if not running_in_colab() and os.environ.get("FFS_ALLOW_LOCAL_SETUP") != "1":
    raise RuntimeError(
        "launch.py is Colab-only because it can install packages and download large models. "
        "Run app.py locally for mock UI testing."
    )


WS = Path(os.environ.get("FFS_WORKSPACE", "/content/drive/MyDrive/FreeFakeStudio")).resolve()
REPAIR = os.environ.get("FFS_REPAIR", "") == "1"
UPDATE_APP = os.environ.get("FFS_UPDATE", "") == "1"
COMFY_TAG = os.environ.get("FFS_COMFY_TAG", "v0.28.0")
DEBUG = os.environ.get("FFS_DEBUG", "1").lower() not in ("0", "false", "no", "off")


def load_colab_secret(name):
    if os.environ.get(name, "").strip():
        return
    try:
        from google.colab import userdata

        value = (userdata.get(name) or "").strip()
    except Exception:
        value = ""
    if value:
        os.environ[name] = value


for _secret_name in ("GEMINI_API_KEY", "TAVILY_API_KEY"):
    load_colab_secret(_secret_name)
FLUX_CUSTOM_MAX_BYTES = 3 * 1024**3
FLUX_CUSTOM_MIN_BYTES = 500 * 1024**2
FLUX_ENCODER_CONFIG = WS / "config" / "flux_encoder.json"
PRIVATE_SETTINGS_FILE = WS / "config" / "private_settings.json"
APP_PID_FILE = WS / "config" / "app.pid"

COMFYUI = WS / "ComfyUI"
CACHE = WS / "cache"
APP = WS / "app"
RESULTS = WS / "results"
DIAGNOSTICS = WS / "diagnostics"

for directory in [
    CACHE / "huggingface",
    CACHE / "pip",
    RESULTS,
    DIAGNOSTICS,
    FLUX_ENCODER_CONFIG.parent,
]:
    directory.mkdir(parents=True, exist_ok=True)


def load_private_settings():
    if not PRIVATE_SETTINGS_FILE.is_file():
        return {}
    try:
        data = json.loads(PRIVATE_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def private_value(settings, env_name, key, default=""):
    """Prefer this run's form value, then reuse the private Drive value."""
    current = os.environ.get(env_name, "").strip()
    if current:
        return current
    return str(settings.get(key, default) or default).strip()


def normalize_public_route(value):
    raw = str(value or "colab_proxy").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "colab": "colab_proxy",
        "proxy": "colab_proxy",
        "colabproxy": "colab_proxy",
        "colab_proxy": "colab_proxy",
        "ngrok": "ngrok",
        "ngrok_tunnel": "ngrok",
        "auto": "auto",
    }
    if raw not in aliases:
        raise RuntimeError(
            f"Unsupported PUBLIC_ROUTE value {value!r}. Use Colab proxy, ngrok, or Auto."
        )
    return aliases[raw]


def normalize_bool(value, default=False):
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _private_env(name):
    return os.environ.get(name, "").strip()


def save_private_settings(settings):
    """Atomically persist stable single-user settings outside the Git checkout."""
    values = {
        "public_route": PUBLIC_ROUTE_MODE,
        "preload_flux": "1" if PRELOAD_FLUX else "0",
        "ngrok_auth_token": NGROK_AUTHTOKEN,
        "flux_encoder_mode": FLUX_ENCODER_MODE,
        "flux_custom_encoder_url": FLUX_CUSTOM_ENCODER_URL,
        "huggingface_token": _private_env("HF_TOKEN"),
        "gemini_api_key": _private_env("GEMINI_API_KEY"),
        "tavily_api_key": _private_env("TAVILY_API_KEY"),
        "gemini_model": _private_env("FFS_GEMINI_MODEL"),
        "avatar_reference_domains": _private_env("FFS_AVATAR_REFERENCE_DOMAINS"),
        "avatar_reference_time_range": _private_env("FFS_AVATAR_REFERENCE_TIME_RANGE"),
        "avatar_search_rounds": _private_env("FFS_AVATAR_SEARCH_ROUNDS"),
        "avatar_gallery_retries": _private_env("FFS_AVATAR_GALLERY_RETRIES"),
        "avatar_max_candidate_downloads": _private_env("FFS_AVATAR_MAX_CANDIDATE_DOWNLOADS"),
        "avatar_vision_model": _private_env("FFS_AVATAR_VISION_MODEL"),
        "avatar_vision_max_edge": _private_env("FFS_AVATAR_VISION_MAX_EDGE"),
        "avatar_vision_max_tokens": _private_env("FFS_AVATAR_VISION_MAX_TOKENS"),
    }
    merged = {**settings, **{key: value for key, value in values.items() if value}}
    partial = PRIVATE_SETTINGS_FILE.with_suffix(".json.part")
    partial.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    try:
        partial.chmod(0o600)
    except OSError:
        pass
    os.replace(str(partial), str(PRIVATE_SETTINGS_FILE))
    try:
        PRIVATE_SETTINGS_FILE.chmod(0o600)
    except OSError:
        pass


_private_settings = load_private_settings()
PUBLIC_ROUTE_MODE = normalize_public_route(
    private_value(_private_settings, "FFS_PUBLIC_ROUTE", "public_route", "colab_proxy")
)
PRELOAD_FLUX = normalize_bool(
    private_value(_private_settings, "FFS_PRELOAD_FLUX", "preload_flux", "1"), True
)
NGROK_AUTHTOKEN = private_value(
    _private_settings, "FFS_NGROK_AUTHTOKEN", "ngrok_auth_token"
)
FLUX_ENCODER_MODE = private_value(
    _private_settings, "FFS_FLUX_ENCODER_MODE", "flux_encoder_mode", "official"
).lower()
FLUX_CUSTOM_ENCODER_URL = private_value(
    _private_settings, "FFS_FLUX_CUSTOM_ENCODER_URL", "flux_custom_encoder_url"
)
_hf_token = private_value(_private_settings, "HF_TOKEN", "huggingface_token")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
for _env_name, _private_key in (
    ("GEMINI_API_KEY", "gemini_api_key"),
    ("TAVILY_API_KEY", "tavily_api_key"),
    ("FFS_GEMINI_MODEL", "gemini_model"),
    ("FFS_AVATAR_REFERENCE_DOMAINS", "avatar_reference_domains"),
    ("FFS_AVATAR_REFERENCE_TIME_RANGE", "avatar_reference_time_range"),
    ("FFS_AVATAR_SEARCH_ROUNDS", "avatar_search_rounds"),
    ("FFS_AVATAR_GALLERY_RETRIES", "avatar_gallery_retries"),
    ("FFS_AVATAR_MAX_CANDIDATE_DOWNLOADS", "avatar_max_candidate_downloads"),
    ("FFS_AVATAR_VISION_MODEL", "avatar_vision_model"),
    ("FFS_AVATAR_VISION_MAX_EDGE", "avatar_vision_max_edge"),
    ("FFS_AVATAR_VISION_MAX_TOKENS", "avatar_vision_max_tokens"),
):
    _value = private_value(_private_settings, _env_name, _private_key)
    if _value:
        os.environ[_env_name] = _value
save_private_settings(_private_settings)


def _use_ngrok_route():
    if PUBLIC_ROUTE_MODE == "ngrok":
        return True
    if PUBLIC_ROUTE_MODE == "colab_proxy":
        return False
    return bool(NGROK_AUTHTOKEN)

os.environ["HF_HOME"] = str(CACHE / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE / "huggingface")
os.environ["PIP_CACHE_DIR"] = str(CACHE / "pip")
os.environ["COMFYUI_ROOT"] = str(COMFYUI)
os.environ["FREEFAKESTUDIO_WORKSPACE"] = str(WS)
os.environ["GRADIO_TEMP_DIR"] = str(WS / "gradio_tmp")
os.environ["GRADIO_SSR_MODE"] = "false"
os.environ["FREEFAKESTUDIO_SHARE"] = "0"
os.environ["FFS_DEBUG"] = "1" if DEBUG else "0"
os.environ["FFS_PRELOAD_FLUX"] = "1" if PRELOAD_FLUX else "0"


_steps = []
_t0 = time.time()


def _render(final=False):
    rows = []
    for item in _steps:
        cls = item["status"]
        icon = {"ok": "✓", "err": "✗", "run": "●", "info": "•"}.get(cls, "•")
        rows.append(
            f'<div class="row {cls}"><span class="icon">{icon}</span>'
            f'<span>{item["title"]}</span><span class="detail">{item["detail"]}</span></div>'
        )
    html = f"""
    <style>
      .ffs-setup{{font-family:Inter,system-ui,-apple-system,sans-serif;max-width:680px;color:#e8e8ee}}
      .ffs-head{{background:#10131c;border:1px solid #262b3a;border-radius:10px;padding:18px 20px;margin-bottom:12px}}
      .ffs-title{{font-size:24px;font-weight:800;color:#f5f7ff}}
      .ffs-sub{{font-size:13px;color:#9ca3af;margin-top:4px}}
      .ffs-box{{background:#0f1117;border:1px solid #252936;border-radius:10px;overflow:hidden}}
      .row{{display:flex;gap:10px;align-items:center;padding:10px 14px;border-bottom:1px solid #232734}}
      .row:last-child{{border-bottom:none}}
      .icon{{width:18px;text-align:center}}
      .ok .icon{{color:#34d399}} .err .icon{{color:#f87171}} .run .icon{{color:#60a5fa}}
      .detail{{margin-left:auto;color:#8b93a5;font-size:12px;text-align:right}}
      .foot{{display:flex;justify-content:space-between;color:#6b7280;font-size:12px;margin-top:8px}}
    </style>
    <div class="ffs-setup">
      <div class="ffs-head"><div class="ffs-title">FreeFakeStudio</div>
      <div class="ffs-sub">Persistent Colab setup and launch</div></div>
      <div class="ffs-box">{''.join(rows)}</div>
      <div class="foot"><span>{time.time() - _t0:.0f}s</span><span>{WS}</span></div>
    </div>
    """
    if not final:
        clear_output(wait=True)
    display(HTML(html))


def step(title, detail="", status="run"):
    _steps.append({"title": title, "detail": detail, "status": status})
    _render()


def done(title=None, detail=None):
    if _steps:
        _steps[-1]["status"] = "ok"
        if title is not None:
            _steps[-1]["title"] = title
        if detail is not None:
            _steps[-1]["detail"] = detail
    _render()


def fail(detail):
    if _steps:
        _steps[-1]["status"] = "err"
        _steps[-1]["detail"] = detail
    _render(final=True)


def run_cmd(args, check=True, quiet=True, cwd=None):
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=quiet,
    )
    if check and result.returncode != 0:
        msg = result.stderr or result.stdout or "command failed"
        raise RuntimeError(msg[-1500:])
    return result


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(path):
    if not (Path(path) / ".git").exists():
        return None
    result = run_cmd(["git", "-C", str(path), "rev-parse", "HEAD"], check=False, quiet=True)
    return result.stdout.strip() if result.returncode == 0 else None


def safe_nvidia_smi():
    try:
        result = subprocess.run(["nvidia-smi"], text=True, capture_output=True)
        if result.returncode != 0:
            return result.stderr[-1000:] if result.stderr else "nvidia-smi unavailable"
        return result.stdout[-4000:]
    except FileNotFoundError:
        return "nvidia-smi not found"


def host_memory():
    values = {}
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return values


def gpu_summary():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError((result.stderr or "No CUDA GPU detected").strip())
    name, memory_mib = [part.strip() for part in result.stdout.splitlines()[0].rsplit(",", 1)]
    return name, int(memory_mib)


def required_model_targets():
    targets = [
        COMFYUI / "models" / "diffusion_models" / "z_image_turbo-Q3_K_M.gguf",
        COMFYUI / "models" / "text_encoders" / "qwen_3_4b_fp4_mixed.safetensors",
        COMFYUI / "models" / "vae" / "ae.safetensors",
        COMFYUI / "models" / "diffusion_models" / "flux-2-klein-4b.safetensors",
        COMFYUI / "models" / "text_encoders" / "qwen_3_4b_fp4_flux2.safetensors",
        COMFYUI / "models" / "vae" / "flux2-vae.safetensors",
        COMFYUI / "models" / "diffusion_models" / "ernie-image-turbo-Q6_K.gguf",
        COMFYUI / "models" / "text_encoders" / "ministral-3-3b.safetensors",
    ]
    manifest = load_flux_encoder_manifest()
    if manifest:
        targets.append(COMFYUI / "models" / "text_encoders" / manifest["local_name"])
    return targets


def write_debug_report(stage, exc=None):
    if not DEBUG:
        return None
    report = {
        "stage": stage,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "workspace": str(WS),
        "app": str(APP),
        "comfyui": str(COMFYUI),
        "comfyui_required_tag": COMFY_TAG,
        "comfyui_revision": git_revision(COMFYUI),
        "cache": str(CACHE),
        "results": str(RESULTS),
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
            "PIP_CACHE_DIR": os.environ.get("PIP_CACHE_DIR"),
            "COMFYUI_ROOT": os.environ.get("COMFYUI_ROOT"),
            "FREEFAKESTUDIO_WORKSPACE": os.environ.get("FREEFAKESTUDIO_WORKSPACE"),
            "FFS_PUBLIC_ROUTE": PUBLIC_ROUTE_MODE,
            "FFS_PUBLIC_ROUTE_EFFECTIVE": "ngrok" if _use_ngrok_route() else "colab_proxy",
            "FFS_PRELOAD_FLUX": "1" if PRELOAD_FLUX else "0",
            "FFS_FLUX_ENCODER_MODE": os.environ.get("FFS_FLUX_ENCODER_MODE"),
            "FFS_FLUX_CUSTOM_ENCODER_FILE": os.environ.get("FFS_FLUX_CUSTOM_ENCODER_FILE"),
            "FFS_FLUX_CUSTOM_ENCODER_FORMAT": os.environ.get("FFS_FLUX_CUSTOM_ENCODER_FORMAT"),
            "FFS_FLUX_CUSTOM_ENCODER_SIZE": os.environ.get("FFS_FLUX_CUSTOM_ENCODER_SIZE"),
            "FFS_FLUX_CUSTOM_ENCODER_SOURCE": os.environ.get("FFS_FLUX_CUSTOM_ENCODER_SOURCE"),
            "FFS_GEMINI_MODEL": os.environ.get("FFS_GEMINI_MODEL"),
            "FFS_AVATAR_REFERENCE_DOMAINS": os.environ.get("FFS_AVATAR_REFERENCE_DOMAINS"),
            "FFS_AVATAR_REFERENCE_TIME_RANGE": os.environ.get("FFS_AVATAR_REFERENCE_TIME_RANGE"),
            "FFS_AVATAR_SEARCH_ROUNDS": os.environ.get("FFS_AVATAR_SEARCH_ROUNDS"),
            "FFS_AVATAR_GALLERY_RETRIES": os.environ.get("FFS_AVATAR_GALLERY_RETRIES"),
            "FFS_AVATAR_MAX_CANDIDATE_DOWNLOADS": os.environ.get("FFS_AVATAR_MAX_CANDIDATE_DOWNLOADS"),
            "FFS_AVATAR_VISION_MODEL": os.environ.get("FFS_AVATAR_VISION_MODEL"),
            "FFS_AVATAR_VISION_MAX_EDGE": os.environ.get("FFS_AVATAR_VISION_MAX_EDGE"),
            "FFS_AVATAR_VISION_MAX_TOKENS": os.environ.get("FFS_AVATAR_VISION_MAX_TOKENS"),
            "GEMINI_API_KEY_present": bool(os.environ.get("GEMINI_API_KEY")),
            "TAVILY_API_KEY_present": bool(os.environ.get("TAVILY_API_KEY")),
        },
        "packages": {
            name: package_version(name)
            for name in [
                "numpy",
                "torch",
                "torchsde",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "sentencepiece",
                "safetensors",
                "gradio",
                "huggingface_hub",
                "Pillow",
                "opencv-python-headless",
                "onnxruntime-gpu",
                "rembg",
            ]
        },
        "paths": {
            "app_py": (APP / "app.py").exists(),
            "launch_py": (APP / "launch.py").exists(),
            "comfyui_main": (COMFYUI / "main.py").exists(),
            "gguf_nodes": (COMFYUI / "custom_nodes" / "ComfyUI-GGUF" / "nodes.py").exists(),
            "/content/ComfyUI_is_symlink": Path("/content/ComfyUI").is_symlink(),
        },
        "model_files": [
            {
                "path": str(path),
                "exists": path.exists(),
                "is_symlink": path.is_symlink(),
                "size": path.stat().st_size if path.exists() else 0,
                "target": str(path.resolve()) if path.exists() else None,
            }
            for path in required_model_targets()
        ],
    }
    if exc is not None:
        report["exception"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    report["nvidia_smi"] = safe_nvidia_smi()
    report["host_memory"] = host_memory()
    path = DIAGNOSTICS / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = DIAGNOSTICS / "latest.json"
    latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def launch_app_process(timeout_seconds=180):
    terminate_previous_app_process()
    log_path = DIAGNOSTICS / "app_launch_latest.log"
    history = []
    url_pattern = re.compile(r"https://[^\s]+\.gradio\.live|Running on public URL:\s*(https://[^\s]+)")
    public_url = None
    public_label = None
    tunnel = None

    if _use_ngrok_route():
        if not NGROK_AUTHTOKEN:
            raise RuntimeError(
                "PUBLIC_ROUTE is set to ngrok, but NGROK_AUTH_TOKEN is blank. "
                "Set PUBLIC_ROUTE to Colab proxy or add your ngrok token."
            )
        try:
            from pyngrok import ngrok

            ngrok.set_auth_token(NGROK_AUTHTOKEN)
            tunnel = ngrok.connect(7860, proto="http")
            public_url = tunnel.public_url.rstrip("/")
            public_label = "ngrok"
            if not public_url.startswith("https://"):
                raise RuntimeError(f"ngrok returned a non-HTTPS URL: {public_url}")
        except Exception as exc:
            raise RuntimeError(
                "ngrok tunnel setup failed. Check NGROK_AUTH_TOKEN and your ngrok account. "
                f"Details: {exc}"
            ) from exc
    else:
        try:
            from google.colab.output import eval_js

            public_url = str(eval_js("google.colab.kernel.proxyPort(7860)")).rstrip("/")
            public_label = "Colab proxy"
            if not public_url.startswith("https://"):
                raise RuntimeError(f"Colab returned a non-HTTPS proxy URL: {public_url}")
        except Exception as exc:
            raise RuntimeError(
                "Could not create the Colab HTTPS proxy URL. Set PUBLIC_ROUTE=ngrok "
                f"and add NGROK_AUTH_TOKEN if you need an ngrok tunnel. Details: {exc}"
            ) from exc

    # Gradio otherwise discovers Colab's internal HTTP host and emits mixed-content
    # asset/API URLs. An absolute root path makes every browser request use the
    # selected external HTTPS origin.
    env = {
        **os.environ,
        "PYTHONPATH": str(APP),
        "PYTHONUNBUFFERED": "1",
        "MALLOC_ARENA_MAX": "2",
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
        "TOKENIZERS_PARALLELISM": "false",
        "FREEFAKESTUDIO_PUBLIC_URL": public_url,
        "GRADIO_ROOT_PATH": public_url,
        "FREEFAKESTUDIO_SHARE": "0",
        "FFS_PRELOAD_FLUX": "1" if PRELOAD_FLUX else "0",
    }
    cmd = [sys.executable, "-u", str(APP / "app.py")]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write(f"Working directory: {APP}\n\n")
        log.write(f"Public route: {public_label} ({public_url})\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(APP),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        APP_PID_FILE.write_text(str(proc.pid), encoding="ascii")
        started = time.time()
        local_started = False
        output_queue = queue.Queue()
        assert proc.stdout is not None

        def reader():
            for item in proc.stdout:
                output_queue.put(item)

        threading.Thread(target=reader, daemon=True).start()
        while True:
            try:
                line = output_queue.get(timeout=0.2)
                print(line, end="")
                log.write(line)
                log.flush()
                history.append(line.rstrip())
                history = history[-80:]
                match = url_pattern.search(line)
                if "Running on local URL:" in line:
                    local_started = True
                    print("\n" + "=" * 72)
                    print(f"OPEN FREEFAKESTUDIO ({public_label}):")
                    print(public_url)
                    print("=" * 72 + "\n")
                if match:
                    gradio_share_url = match.group(1) or match.group(0)
                    print(f"\nUnexpected Gradio share URL: {gradio_share_url}\n")
            except queue.Empty:
                line = None

            if proc.poll() is not None:
                while not output_queue.empty():
                    extra = output_queue.get_nowait()
                    print(extra, end="")
                    log.write(extra)
                    history.append(extra.rstrip())
                log.flush()
                code = proc.returncode
                if code != 0:
                    tail = "\n".join(history[-40:])
                    raise RuntimeError(f"app.py exited with code {code}. Last output:\n{tail}")
                return public_url

            if not local_started and time.time() - started > timeout_seconds:
                tail = "\n".join(history[-40:]) or "(no app output yet)"
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError(
                    f"Gradio did not start its local server within {timeout_seconds}s. "
                    f"App log: {log_path}\nLast output:\n{tail}"
                )


def terminate_previous_app_process():
    """Stop only a previous app process recorded for this Drive workspace."""
    if not APP_PID_FILE.is_file():
        return
    try:
        pid = int(APP_PID_FILE.read_text(encoding="ascii").strip())
        command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (OSError, ValueError):
        APP_PID_FILE.unlink(missing_ok=True)
        return
    expected = str(APP / "app.py")
    if expected not in command_line:
        APP_PID_FILE.unlink(missing_ok=True)
        return
    print(f"Stopping previous FreeFakeStudio process ({pid})...", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.2)
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        APP_PID_FILE.unlink(missing_ok=True)


def pip_install(*packages, force=False):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--cache-dir", str(CACHE / "pip")]
    if force:
        cmd.append("--force-reinstall")
    cmd.extend(packages)
    run_cmd(cmd, quiet=True)


def package_ok(module_name):
    return importlib.util.find_spec(module_name) is not None


def verify_comfy_runtime():
    code = (
        "import inspect, sys; "
        f"sys.path.insert(0, {str(COMFYUI)!r}); "
        "import torchsde; "
        "import comfy.samplers; "
        "import comfy.sd; "
        "import comfy.quant_ops as quant_ops; "
        "import comfy.model_detection as model_detection; "
        "import comfy.text_encoders.z_image; "
        "import comfy.text_encoders.ernie; "
        "source = inspect.getsource(model_detection.detect_unet_config); "
        "assert 'z_image_modulation' in source, 'ComfyUI has no Z-Image model detection'; "
        "assert '\"flux2\"' in source, 'ComfyUI has no FLUX.2 model detection'; "
        "assert quant_ops._CK_AVAILABLE, 'comfy-kitchen is unavailable; mixed-FP4 cannot load'; "
        "assert 'nvfp4' in quant_ops.QUANT_ALGOS, 'ComfyUI has no mixed-FP4 support'; "
        "print('dependencies + Z-Image/FLUX.2/ERNIE + mixed-FP4 support: OK')"
    )
    probe = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "ComfyUI import probe failed")[-3000:]
        raise RuntimeError(f"ComfyUI runtime import check failed:\n{detail}")
    return probe.stdout.strip()


def verify_engine_nodes():
    required = (
        "UnetLoaderGGUF",
        "CLIPLoader",
        "VAELoader",
        "CLIPTextEncode",
        "KSampler",
        "VAEDecode",
        "VAEEncode",
        "EmptySD3LatentImage",
        "SetLatentNoiseMask",
        "ModelSamplingAuraFlow",
    )
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(APP)!r}); "
        f"sys.path.insert(0, {str(COMFYUI)!r}); "
        "import engine_z_image; "
        "node_map = engine_z_image._get_nodes(); "
        f"required = {required!r}; "
        "missing = [name for name in required if name not in node_map]; "
        "assert not missing, 'Missing Z-Image engine nodes: ' + ', '.join(missing); "
        "print('Z-Image engine nodes: OK')"
    )
    probe = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "Engine node probe failed")[-3000:]
        raise RuntimeError(f"Engine node construction check failed:\n{detail}")
    return probe.stdout.strip()


def verify_z_image_checkpoint():
    from safetensors import safe_open

    path = COMFYUI / "models" / "diffusion_models" / "z_image_turbo-Q3_K_M.gguf"
    text_path = COMFYUI / "models" / "text_encoders" / "qwen_3_4b_fp4_mixed.safetensors"
    if not file_ok(path):
        raise RuntimeError(f"Z-Image checkpoint is missing or incomplete: {path}")
    if not file_ok(text_path):
        raise RuntimeError(f"Z-Image FP4 text encoder is missing or incomplete: {text_path}")
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise RuntimeError(f"Z-Image diffusion model has an invalid GGUF header: {path}")
    with safe_open(str(text_path), framework="numpy", device="cpu") as handle:
        text_keys = tuple(handle.keys())
    quantized_keys = [key for key in text_keys if "weight_scale" in key]
    if not quantized_keys:
        raise RuntimeError(
            "Z-Image text encoder is not the expected mixed-FP4 checkpoint: "
            "no quantization scale tensors were found."
        )
    return f"Headers OK / diffusion=GGUF Q3_K_M / FP4 encoder={len(text_keys)} tensors"


def ensure_numpy():
    step("NumPy", "Checking binary consistency")
    code = "import numpy; import numpy._core.strings; print(numpy.__version__)"
    probe = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if probe.returncode == 0:
        done("NumPy", probe.stdout.strip())
        return

    detail = (probe.stderr or probe.stdout)[-240:].replace("\n", " ")
    step("NumPy repair", detail)
    pip_install("numpy==1.26.4", force=True)
    probe = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    if probe.returncode != 0:
        fail("Restart the Colab runtime once, then run this cell again.")
        raise RuntimeError(probe.stderr or probe.stdout)
    done("NumPy repair", probe.stdout.strip())


def ensure_repo(path, repo, tag=None, update=False):
    if not (path / ".git").exists():
        if path.exists() and any(path.iterdir()):
            entries = list(path.rglob("*"))
            has_files = any(item.is_file() or item.is_symlink() for item in entries)
            if has_files:
                raise RuntimeError(
                    f"{path} contains files but is not a git repository. "
                    "The launcher left it untouched to protect persistent data."
                )
            for directory in sorted(
                (item for item in entries if item.is_dir()),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                directory.rmdir()
            path.rmdir()
        args = ["git", "clone", "--depth", "1"]
        if tag:
            args.extend(["--branch", tag])
        args.extend([repo, str(path)])
        run_cmd(args, quiet=True)
        return "installed"
    if tag:
        head = git_revision(path)
        target = run_cmd(
            ["git", "-C", str(path), "rev-parse", f"{tag}^{{commit}}"],
            check=False,
            quiet=True,
        )
        tag_revision = target.stdout.strip() if target.returncode == 0 else None
        update = update or not tag_revision or head != tag_revision
    if update:
        if tag:
            run_cmd(
                ["git", "-C", str(path), "fetch", "--depth", "1", "origin", "tag", tag],
                quiet=True,
            )
            run_cmd(["git", "-C", str(path), "checkout", "--force", tag], quiet=True)
            return f"updated to {tag}"
        run_cmd(["git", "-C", str(path), "pull", "--ff-only"], quiet=True)
        return "updated"
    return "cached"


def ensure_comfy_directories():
    for directory in [
        COMFYUI / "models" / "diffusion_models",
        COMFYUI / "models" / "text_encoders",
        COMFYUI / "models" / "clip",
        COMFYUI / "models" / "vae",
        COMFYUI / "custom_nodes",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_symlink(link, target):
    link = Path(link)
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        if link.name == "ComfyUI" and str(link).startswith("/content/"):
            shutil.rmtree(str(link))
        else:
            raise RuntimeError(f"Refusing to replace non-symlink path: {link}")
    os.symlink(str(target), str(link))


def min_bytes(filename):
    if filename in {"ae.safetensors", "flux2-vae.safetensors"}:
        return 50 * 1024 * 1024
    if filename.endswith((".safetensors", ".gguf")):
        return 500 * 1024 * 1024
    return 1024


def file_ok(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size >= min_bytes(path.name)


def hub_download(repo, filename, dest_dir, dest_name=None, force=False, revision=None):
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (dest_name or Path(filename).name)
    if file_ok(target) and not REPAIR and not force:
        return target
    cached = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        cache_dir=str(CACHE / "huggingface"),
        force_download=REPAIR or force,
        token=os.environ.get("HF_TOKEN") or None,
    )
    if not file_ok(cached):
        raise RuntimeError(f"Downloaded file looks incomplete: {cached}")
    if target.exists() or target.is_symlink():
        target.unlink()
    cached = Path(cached)
    cached_is_symlink = cached.is_symlink()
    material = cached.resolve(strict=True) if cached_is_symlink else cached
    partial = target.with_name(f"{target.name}.part")
    if partial.exists() or partial.is_symlink():
        partial.unlink()
    try:
        os.replace(str(material), str(target))
    except OSError:
        shutil.copyfile(str(material), str(partial))
        if not file_ok(partial):
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Could not materialize persistent model file: {target}")
        os.replace(str(partial), str(target))
    else:
        if cached_is_symlink:
            cached.unlink(missing_ok=True)
    if not file_ok(target):
        raise RuntimeError(f"Persistent model file looks incomplete: {target}")
    return target


def parse_huggingface_file_url(url):
    """Return repo, revision and filename from a Hugging Face file URL."""
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or parsed.netloc.lower() not in {
        "huggingface.co", "www.huggingface.co"
    }:
        raise RuntimeError("Custom FLUX encoder must be an https://huggingface.co file URL.")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] not in {"blob", "resolve"}:
        raise RuntimeError(
            "Use the full Hugging Face file URL, for example "
            "https://huggingface.co/owner/repo/blob/main/encoder.gguf"
        )
    repo = "/".join(parts[:2])
    revision = parts[3]
    filename = "/".join(parts[4:])
    suffix = Path(filename).suffix.lower()
    if suffix not in {".gguf", ".safetensors"}:
        raise RuntimeError("Custom FLUX encoder must end in .gguf or .safetensors.")
    return repo, revision, filename, suffix


def huggingface_file_size(repo, revision, filename):
    from huggingface_hub import HfApi

    info = HfApi().model_info(
        repo_id=repo,
        revision=revision,
        files_metadata=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    sibling = next((item for item in info.siblings if item.rfilename == filename), None)
    if sibling is None:
        raise RuntimeError(f"Encoder file was not found in {repo}: {filename}")
    size = getattr(sibling, "size", None)
    if not size:
        raise RuntimeError("Hugging Face did not report the encoder file size; download stopped.")
    if size < FLUX_CUSTOM_MIN_BYTES:
        raise RuntimeError(
            f"Custom encoder is only {size / 1024**2:.0f} MiB; expected a complete Qwen3-4B file."
        )
    if size > FLUX_CUSTOM_MAX_BYTES:
        raise RuntimeError(
            f"Custom encoder is {size / 1024**3:.2f} GiB. Free Colab limit is 3.00 GiB; "
            "use a Q4 file around 2.0-2.7 GiB."
        )
    return int(size)


def validate_flux_encoder_file(path):
    """Perform cheap structural checks before exposing a custom file to the app."""
    path = Path(path)
    size = path.stat().st_size
    if not FLUX_CUSTOM_MIN_BYTES <= size <= FLUX_CUSTOM_MAX_BYTES:
        raise RuntimeError(f"Custom FLUX encoder has an unsupported size: {size} bytes")
    if path.suffix.lower() == ".gguf":
        from gguf import GGUFReader

        reader = GGUFReader(str(path), mode="r")
        field = reader.fields.get("general.architecture")
        arch = None
        if field is not None and field.parts:
            value = field.parts[field.data[-1]]
            arch = bytes(value).decode("utf-8", errors="replace")
        if arch != "qwen3":
            raise RuntimeError(
                f"Custom GGUF architecture is {arch or 'unknown'}, expected qwen3 for FLUX.2 Klein 4B."
            )
        return {"format": "gguf", "architecture": arch}

    from safetensors import safe_open

    with safe_open(str(path), framework="numpy", device="cpu") as handle:
        keys = tuple(handle.keys())
    qwen_markers = ("model.layers.", "transformer.h.", "text_encoders.qwen")
    if len(keys) < 100 or not any(any(marker in key for marker in qwen_markers) for key in keys):
        raise RuntimeError(
            "Custom Safetensors file does not look like a complete Qwen3-4B text encoder."
        )
    return {"format": "safetensors", "architecture": "qwen3-compatible", "tensors": len(keys)}


def load_flux_encoder_manifest():
    if not FLUX_ENCODER_CONFIG.is_file():
        return None
    try:
        data = json.loads(FLUX_ENCODER_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    target = COMFYUI / "models" / "text_encoders" / data.get("local_name", "")
    if not target.name or not file_ok(target):
        return None
    data["path"] = str(target)
    return data


def ensure_custom_flux_encoder():
    """Download/validate an optional single-file encoder and publish app config."""
    global FLUX_ENCODER_MODE

    manifest = load_flux_encoder_manifest()
    if FLUX_CUSTOM_ENCODER_URL:
        repo, revision, filename, suffix = parse_huggingface_file_url(FLUX_CUSTOM_ENCODER_URL)
        size = huggingface_file_size(repo, revision, filename)
        source = f"https://huggingface.co/{repo}/blob/{revision}/{filename}"
        local_name = f"flux2-klein-custom-encoder{suffix}"
        changed = not manifest or manifest.get("source") != source
        step("FLUX custom encoder", f"Verified metadata / {size / 1024**3:.2f} GiB")
        target = hub_download(
            repo,
            filename,
            COMFYUI / "models" / "text_encoders",
            local_name,
            force=changed,
            revision=revision,
        )
        details = validate_flux_encoder_file(target)
        manifest = {
            "source": source,
            "repo": repo,
            "revision": revision,
            "remote_filename": filename,
            "local_name": local_name,
            "size": target.stat().st_size,
            **details,
        }
        FLUX_ENCODER_CONFIG.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        done("FLUX custom encoder", f"Ready / {details['format'].upper()}")

    if FLUX_ENCODER_MODE not in {"official", "custom"}:
        raise RuntimeError("FLUX_ENCODER must be Official or Custom.")
    if FLUX_ENCODER_MODE == "custom" and not manifest:
        raise RuntimeError(
            "FLUX_ENCODER is Custom but no validated custom encoder exists. "
            "Paste a compatible Hugging Face file URL or choose Official."
        )

    os.environ["FFS_FLUX_ENCODER_MODE"] = FLUX_ENCODER_MODE
    if manifest:
        os.environ["FFS_FLUX_CUSTOM_ENCODER_FILE"] = manifest["local_name"]
        os.environ["FFS_FLUX_CUSTOM_ENCODER_FORMAT"] = manifest["format"]
        os.environ["FFS_FLUX_CUSTOM_ENCODER_SIZE"] = str(manifest["size"])
        os.environ["FFS_FLUX_CUSTOM_ENCODER_SOURCE"] = manifest["source"]
    else:
        for name in (
            "FFS_FLUX_CUSTOM_ENCODER_FILE",
            "FFS_FLUX_CUSTOM_ENCODER_FORMAT",
            "FFS_FLUX_CUSTOM_ENCODER_SIZE",
            "FFS_FLUX_CUSTOM_ENCODER_SOURCE",
        ):
            os.environ.pop(name, None)
    return manifest


def ensure_models():
    diff = COMFYUI / "models" / "diffusion_models"
    text = COMFYUI / "models" / "text_encoders"
    vae = COMFYUI / "models" / "vae"

    specs = {
        "Z-Image Turbo": [
            ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q3_K_M.gguf", diff, None),
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors", text, "qwen_3_4b_fp4_mixed.safetensors"),
            ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", vae, "ae.safetensors"),
        ],
        "FLUX.2-klein 4B": [
            ("black-forest-labs/FLUX.2-klein-4B", "flux-2-klein-4b.safetensors", diff, None),
            ("Comfy-Org/vae-text-encorder-for-flux-klein-4b", "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors", text, "qwen_3_4b_fp4_flux2.safetensors"),
            ("Comfy-Org/vae-text-encorder-for-flux-klein-4b", "split_files/vae/flux2-vae.safetensors", vae, "flux2-vae.safetensors"),
        ],
        "ERNIE-Image-Turbo": [
            ("unsloth/ERNIE-Image-Turbo-GGUF", "ernie-image-turbo-Q6_K.gguf", diff, None),
            ("Comfy-Org/ERNIE-Image", "text_encoders/ministral-3-3b.safetensors", text, "ministral-3-3b.safetensors"),
            ("Comfy-Org/ERNIE-Image", "vae/flux2-vae.safetensors", vae, "flux2-vae.safetensors"),
        ],
    }

    for model_name, files in specs.items():
        needed = []
        for repo, remote, dest, name in files:
            target = Path(dest) / (name or Path(remote).name)
            if REPAIR or not file_ok(target):
                needed.append((repo, remote, dest, name))
        if not needed:
            step(model_name, "Ready", "ok")
            continue
        step(model_name, f"Preparing {len(needed)} persistent file(s)")
        for repo, remote, dest, name in needed:
            hub_download(repo, remote, dest, name)
        done(model_name, "Ready")


try:
    step("Workspace", str(WS), "ok")
    start_report = write_debug_report("startup")
    if start_report:
        step("Diagnostics", f"Startup report: {start_report.name}", "ok")

    ensure_numpy()

    step("Python packages", "Checking")
    required = {
        "huggingface_hub": "huggingface_hub",
        "gradio": "gradio",
        "rembg": "rembg",
        "cv2": "opencv-python-headless",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "sentencepiece": "sentencepiece",
    }
    if _use_ngrok_route():
        required["pyngrok"] = "pyngrok"
    missing = [pkg for module, pkg in required.items() if not package_ok(module)]
    if missing:
        pip_install(*missing)
    try:
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig  # noqa: F401
    except Exception:
        pip_install("transformers>=5.13,<6", "accelerate", "bitsandbytes", "sentencepiece")
    if not package_ok("onnxruntime"):
        pip_install("onnxruntime-gpu")
    pip_install("Pillow<12")
    done("Python packages", "Ready")
    deps_report = write_debug_report("dependencies")
    if deps_report:
        step("Diagnostics", f"Dependency report: {deps_report.name}", "ok")

    step("ComfyUI", "Checking")
    state = ensure_repo(COMFYUI, "https://github.com/comfyanonymous/ComfyUI.git", COMFY_TAG, UPDATE_APP)
    done("ComfyUI", state)

    ensure_comfy_directories()

    ensure_symlink(Path("/content/ComfyUI"), COMFYUI)

    step("ComfyUI dependencies", "Reconciling this Colab session")
    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--cache-dir",
            str(CACHE / "pip"),
            "-r",
            str(COMFYUI / "requirements.txt"),
        ],
        quiet=True,
    )
    done("ComfyUI dependencies", "Installed / already satisfied")

    step("ComfyUI-GGUF", "Checking")
    gguf = COMFYUI / "custom_nodes" / "ComfyUI-GGUF"
    state = ensure_repo(gguf, "https://github.com/city96/ComfyUI-GGUF.git", None, UPDATE_APP)
    if (gguf / "requirements.txt").exists():
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "--cache-dir", str(CACHE / "pip"), "-r", str(gguf / "requirements.txt")], quiet=True)
    done("ComfyUI-GGUF", state)

    step("ComfyUI runtime", "Import smoke test")
    done("ComfyUI runtime", verify_comfy_runtime())

    step("Engine nodes", "Constructing Z-Image node set")
    done("Engine nodes", verify_engine_nodes())

    comfy_report = write_debug_report("comfy_runtime")
    if comfy_report:
        step("Diagnostics", f"ComfyUI report: {comfy_report.name}", "ok")

    ensure_models()
    ensure_custom_flux_encoder()
    step("Z-Image checkpoint", "Inspecting GGUF and FP4 headers")
    done("Z-Image checkpoint", verify_z_image_checkpoint())
    models_report = write_debug_report("models")
    if models_report:
        step("Diagnostics", f"Model report: {models_report.name}", "ok")

    step("GPU", "Checking")
    gpu_name, gpu_memory_mib = gpu_summary()
    done("GPU", f"{gpu_name} / {gpu_memory_mib / 1024:.1f} GB")

    memory = host_memory()
    total_ram = memory.get("MemTotal", 0)
    available_ram = memory.get("MemAvailable", 0)
    step("Host RAM", "Checking")
    if total_ram and total_ram < 11 * 1024**3:
        fail(f"Only {total_ram / 1024**3:.1f} GB host RAM; Z-Image requires at least 11 GB.")
        raise RuntimeError("This runtime does not have enough host RAM for Z-Image Turbo.")
    done(
        "Host RAM",
        f"{available_ram / 1024**3:.1f} GB available / {total_ram / 1024**3:.1f} GB total",
    )

    gc.collect()

    run_cmd([sys.executable, str(APP / "fixer.py")], quiet=False)

    step("FreeFakeStudio", "Launching")
    launch_report = write_debug_report("launch")
    if launch_report:
        step("Diagnostics", f"Launch report: {launch_report.name}", "ok")
    _render(final=True)
    print("\nFreeFakeStudio is starting. The HTTPS interface link will print when it is ready.\n")
    launch_app_process(timeout_seconds=180)
except Exception as exc:
    error_report = write_debug_report("error", exc)
    if error_report:
        print(f"\nFull debug report written to: {error_report}\n")
        print((DIAGNOSTICS / "latest.json").read_text(encoding="utf-8")[-4000:])
    fail(str(exc)[:240])
    raise
