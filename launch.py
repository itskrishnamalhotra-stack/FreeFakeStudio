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
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

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
NGROK_AUTHTOKEN = os.environ.get("FFS_NGROK_AUTHTOKEN", "").strip()

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
]:
    directory.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE / "huggingface")
os.environ["PIP_CACHE_DIR"] = str(CACHE / "pip")
os.environ["COMFYUI_ROOT"] = str(COMFYUI)
os.environ["FREEFAKESTUDIO_WORKSPACE"] = str(WS)
os.environ["GRADIO_TEMP_DIR"] = str(WS / "gradio_tmp")
os.environ["GRADIO_SSR_MODE"] = "false"
os.environ["FREEFAKESTUDIO_SHARE"] = "0"
os.environ["FFS_DEBUG"] = "1" if DEBUG else "0"


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
    return [
        COMFYUI / "models" / "diffusion_models" / "z_image_turbo-Q3_K_M.gguf",
        COMFYUI / "models" / "text_encoders" / "qwen_3_4b_fp4_mixed.safetensors",
        COMFYUI / "models" / "vae" / "ae.safetensors",
        COMFYUI / "models" / "diffusion_models" / "flux-2-klein-4b.safetensors",
        COMFYUI / "models" / "text_encoders" / "qwen_3_4b_fp4_flux2.safetensors",
        COMFYUI / "models" / "vae" / "flux2-vae.safetensors",
        COMFYUI / "models" / "diffusion_models" / "ernie-image-turbo-Q6_K.gguf",
        COMFYUI / "models" / "text_encoders" / "ministral-3-3b.safetensors",
    ]


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
        },
        "packages": {
            name: package_version(name)
            for name in [
                "numpy",
                "torch",
                "torchsde",
                "transformers",
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
    log_path = DIAGNOSTICS / "app_launch_latest.log"
    history = []
    url_pattern = re.compile(r"https://[^\s]+\.gradio\.live|Running on public URL:\s*(https://[^\s]+)")
    public_url = None
    public_label = None
    tunnel = None

    if NGROK_AUTHTOKEN:
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
                "Could not create the Colab HTTPS proxy URL. Add an ngrok token in "
                f"NGROK_AUTH_TOKEN and run again. Details: {exc}"
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


def hub_download(repo, filename, dest_dir, dest_name=None):
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (dest_name or Path(filename).name)
    if file_ok(target) and not REPAIR:
        return target
    cached = hf_hub_download(
        repo_id=repo,
        filename=filename,
        cache_dir=str(CACHE / "huggingface"),
        force_download=REPAIR,
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
    }
    if NGROK_AUTHTOKEN:
        required["pyngrok"] = "pyngrok"
    missing = [pkg for module, pkg in required.items() if not package_ok(module)]
    if missing:
        pip_install(*missing)
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
