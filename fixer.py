#!/usr/bin/env python3
"""
FreeFakeStudio — Runtime Fixer
Patches known issues in the Colab working copy before the app launches.
Run:  python fixer.py
Every fix is idempotent — running this script multiple times is safe.
"""

import os, sys, subprocess, shutil, textwrap
from pathlib import Path

# ── Resolve paths ──────────────────────────────────────────
# APP is the directory this script lives in (same as launch.py, app.py, etc.)
APP = Path(__file__).resolve().parent
WS = Path(os.environ.get("FFS_WORKSPACE", APP.parent))
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", "/content/ComfyUI"))

FIXED = 0
SKIPPED = 0
FAILED = 0


def log_fix(name, status, detail=""):
    global FIXED, SKIPPED, FAILED
    icon = {"ok": "✅", "skip": "⏭️", "fail": "❌"}[status]
    if status == "ok":
        FIXED += 1
    elif status == "skip":
        SKIPPED += 1
    else:
        FAILED += 1
    msg = f"  {icon} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def patch_file(filepath, old, new, label=""):
    """Replace `old` with `new` in a file.
    Handles CRLF/LF mismatch automatically.
    Returns True if patched, False if already patched or `old` not found."""
    path = Path(filepath)
    if not path.exists():
        log_fix(label or path.name, "fail", f"file not found: {path}")
        return False
    text = path.read_text(encoding="utf-8")

    # Normalize: try matching with both LF and CRLF variants
    old_lf = old.replace("\r\n", "\n")
    new_lf = new.replace("\r\n", "\n")
    old_crlf = old_lf.replace("\n", "\r\n")
    new_crlf = new_lf.replace("\n", "\r\n")

    # Check if already patched (either line ending style)
    if new_lf in text or new_crlf in text:
        log_fix(label or path.name, "skip", "already patched")
        return False

    # Try CRLF first (Windows-style, common in these files), then LF
    if old_crlf in text:
        text = text.replace(old_crlf, new_crlf, 1)
    elif old_lf in text:
        text = text.replace(old_lf, new_lf, 1)
    else:
        log_fix(label or path.name, "skip", "pattern not found (may already be fixed)")
        return False

    path.write_text(text, encoding="utf-8")
    log_fix(label or path.name, "ok", "patched")
    return True


# ════════════════════════════════════════════════════════════
#  Fix 1 — ONNX Runtime CUDA provider mismatch
#  Problem: onnxruntime-gpu is built for CUDA 13.x, Colab has 12.x
#           → libcublasLt.so.13 error, rembg falls back to CPU
#  Fix:    Replace onnxruntime-gpu with CPU-only onnxruntime.
#          rembg is fast enough on CPU for single-image operations.
#  Note:   get_available_providers() is unreliable — it lists CUDA
#          even when the shared library can't load. We check the
#          installed pip package name instead.
# ════════════════════════════════════════════════════════════
def fix_onnxruntime():
    label = "Fix 1: ONNX Runtime"
    import importlib.metadata

    # Check which package is actually installed
    has_gpu_pkg = False
    has_cpu_pkg = False
    try:
        importlib.metadata.version("onnxruntime-gpu")
        has_gpu_pkg = True
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        importlib.metadata.version("onnxruntime")
        has_cpu_pkg = True
    except importlib.metadata.PackageNotFoundError:
        pass

    # If CPU-only onnxruntime is installed and onnxruntime-gpu is not, we're good
    if has_cpu_pkg and not has_gpu_pkg:
        log_fix(label, "skip", "CPU-only onnxruntime already installed")
        return

    # onnxruntime-gpu is installed (or nothing is) — replace with CPU-only
    try:
        cache_dir = str(WS / "cache" / "pip")
        # Uninstall the GPU variant first
        if has_gpu_pkg:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu"],
                capture_output=True, text=True,
            )
        # Install CPU-only onnxruntime
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "--cache-dir", cache_dir, "onnxruntime"],
            check=True, capture_output=True, text=True,
        )
        log_fix(label, "ok", "replaced onnxruntime-gpu with CPU-only onnxruntime")
    except subprocess.CalledProcessError as exc:
        log_fix(label, "fail", f"pip install failed: {exc.stderr[-200:]}")


# ════════════════════════════════════════════════════════════
#  Fix 2 — Coroutine `init_extra_nodes()` never awaited
#  Problem: ComfyUI v0.28.0 made this async; 3 engine files call
#           it synchronously → the function body never executes
#  Fix:    Wrap with asyncio.run() when the function is async
# ════════════════════════════════════════════════════════════
def fix_coroutine_init_extra_nodes():
    label = "Fix 2: Coroutine await"
    old = "nodes.init_extra_nodes()"
    # Runtime-safe: checks if function is actually async before using asyncio.
    # Works whether init_extra_nodes is sync (older ComfyUI) or async (v0.28+).
    new = (
        "(__import__('asyncio').run(nodes.init_extra_nodes()) "
        "if __import__('inspect').iscoroutinefunction(nodes.init_extra_nodes) "
        "else nodes.init_extra_nodes())"
    )
    engine_files = [
        APP / "engine_flux_klein_4b.py",
        APP / "engine_flux_klein_9b.py",
        APP / "engine_ernie_image_turbo.py",
    ]
    for f in engine_files:
        if f.exists():
            text = f.read_text(encoding="utf-8")
            # Check if already patched
            if "__import__('asyncio').run(nodes.init_extra_nodes())" in text:
                log_fix(f"{label} [{f.name}]", "skip", "already patched")
                continue
            if old not in text:
                log_fix(f"{label} [{f.name}]", "skip", "pattern not found")
                continue
            # Replace ALL occurrences in this file (there's only 1 per file)
            text = text.replace(old, new)
            f.write_text(text, encoding="utf-8")
            log_fix(f"{label} [{f.name}]", "ok", "patched")
        else:
            log_fix(f"{label} [{f.name}]", "skip", "file not present")


# ════════════════════════════════════════════════════════════
#  Fix 3 — PyTorch CUDA version warning (cu130)
#  Problem: ComfyUI v0.28.0 wants cu130 optimized ops, Colab
#           has cu121/cu124 → noisy WARNING in logs
#  Fix:    Add a targeted logging filter in model_manager.py
# ════════════════════════════════════════════════════════════
def fix_cuda_warning():
    label = "Fix 3: CUDA warning suppression"
    target = APP / "model_manager.py"
    # Insert a logging filter right before the Environment Detection section
    old = "# ── Environment Detection ──"
    new = (
        "# [fixer.py] Suppress noisy ComfyUI cu130 warning (Colab limitation)\n"
        "import logging as _logging\n"
        "class _Cu130Filter(_logging.Filter):\n"
        "    def filter(self, record):\n"
        "        return 'cu130' not in str(getattr(record, 'msg', ''))\n"
        "_logging.getLogger().addFilter(_Cu130Filter())\n"
        "\n"
        "# ── Environment Detection ──"
    )
    patch_file(target, old, new, label)


# ════════════════════════════════════════════════════════════
#  Fix 4 — OpenCV face detector unavailable
#  Problem: Colab's pre-installed opencv-python conflicts with
#           opencv-python-headless → broken cv2 module (missing
#           CascadeClassifier, cvtColor, ellipse). Also the Haar
#           cascade XML may be missing.
#  Fix:    1) Repair cv2 by clean-installing opencv-python-headless
#          2) Download the cascade XML if needed
# ════════════════════════════════════════════════════════════
def fix_opencv_cascade():
    label = "Fix 4: OpenCV face detector"
    cv2_needs_repair = False

    # Step 1: Check if cv2 has the required functions
    try:
        import cv2
        required = ("CascadeClassifier", "cvtColor", "ellipse")
        if not all(hasattr(cv2, name) for name in required):
            cv2_needs_repair = True
    except ImportError:
        cv2_needs_repair = True

    # Step 2: Repair cv2 if broken (conflicting opencv packages)
    if cv2_needs_repair:
        try:
            cache_dir = str(WS / "cache" / "pip")
            # Remove ALL opencv variants to avoid conflicts
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y",
                 "opencv-python", "opencv-python-headless",
                 "opencv-contrib-python", "opencv-contrib-python-headless"],
                capture_output=True, text=True,
            )
            # Clean install of headless variant
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q",
                 "--cache-dir", cache_dir, "opencv-python-headless"],
                check=True, capture_output=True, text=True,
            )
            # Verify via subprocess — cv2 is a C extension and can't be
            # reliably reimported in the same process after reinstall
            verify = subprocess.run(
                [sys.executable, "-c",
                 "import cv2; print(all(hasattr(cv2, n) for n in "
                 "('CascadeClassifier', 'cvtColor', 'ellipse')))"],
                capture_output=True, text=True,
            )
            if "True" not in verify.stdout:
                log_fix(label, "fail", "reinstalled opencv but cv2 still broken")
                return
            log_fix(f"{label} [cv2 repair]", "ok", "reinstalled opencv-python-headless")
        except Exception as exc:
            log_fix(label, "fail", f"cv2 repair failed: {str(exc)[:120]}")
            return
    else:
        log_fix(f"{label} [cv2 module]", "skip", "cv2 functions OK")

    # Step 3: Ensure Haar cascade XML exists — use subprocess to get
    # cascade path from the (possibly freshly installed) cv2
    try:
        get_path = subprocess.run(
            [sys.executable, "-c",
             "import cv2, os; d = getattr(getattr(cv2, 'data', None), "
             "'haarcascades', ''); print(d if d else 'NONE')"],
            capture_output=True, text=True,
        )
        cascade_dir = get_path.stdout.strip()
        if not cascade_dir or cascade_dir == "NONE":
            log_fix(f"{label} [cascade]", "fail", "cv2.data.haarcascades not available")
            return
        cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        if os.path.isfile(cascade_path) and os.path.getsize(cascade_path) > 10000:
            log_fix(f"{label} [cascade]", "skip", "cascade file already exists")
            return
        url = (
            "https://raw.githubusercontent.com/opencv/opencv/4.x"
            "/data/haarcascades/haarcascade_frontalface_default.xml"
        )
        import urllib.request
        os.makedirs(cascade_dir, exist_ok=True)
        urllib.request.urlretrieve(url, cascade_path)
        if os.path.isfile(cascade_path) and os.path.getsize(cascade_path) > 10000:
            log_fix(f"{label} [cascade]", "ok", f"downloaded to {cascade_path}")
        else:
            log_fix(f"{label} [cascade]", "fail", "download succeeded but file seems too small")
    except Exception as exc:
        log_fix(f"{label} [cascade]", "fail", str(exc)[:150])


# ════════════════════════════════════════════════════════════
#  Fix 5 — FLUX model unnecessary unload/reload cycle
#  Problem: set_flux_encoder_mode() unloads the model even when
#           the encoder mode hasn't actually changed → wastes
#           10-15 seconds reloading weights from Drive
#  Fix:    Add early-return guard when mode is already active
# ════════════════════════════════════════════════════════════
def fix_encoder_mode_guard():
    label = "Fix 5: Encoder mode guard"
    target = APP / "model_manager.py"
    old = (
        '    mode = str(selection or "Official").strip().lower()\n'
        '    if mode not in {"official", "custom"}:\n'
        '        raise ValueError("FLUX encoder must be Official or Custom.")'
    )
    new = (
        '    mode = str(selection or "Official").strip().lower()\n'
        '    if mode not in {"official", "custom"}:\n'
        '        raise ValueError("FLUX encoder must be Official or Custom.")\n'
        '    # [fixer.py] Skip unload if the encoder mode hasn\'t actually changed\n'
        '    if os.environ.get("FFS_FLUX_ENCODER_MODE", "official").strip().lower() == mode:\n'
        '        if _current_model == FLUX_MODEL_NAME:\n'
        '            _eng = _get_engine(FLUX_MODEL_NAME)\n'
        '            if hasattr(_eng, "get_loaded_encoder") and _eng.get_loaded_encoder() == mode:\n'
        '                os.environ["FFS_FLUX_ENCODER_MODE"] = mode\n'
        '                return mode'
    )
    patch_file(target, old, new, label)


# ════════════════════════════════════════════════════════════
#  Fix 6 — Duplicate setup panel in notebook output
#  Problem: _render(final=True) skips clear_output(), so it
#           appends a second copy of the HTML status panel
#  Fix:    Always call clear_output before display
# ════════════════════════════════════════════════════════════
def fix_duplicate_panel():
    label = "Fix 6: Duplicate panel"
    target = APP / "launch.py"
    old = (
        "    if not final:\n"
        "        clear_output(wait=True)\n"
        "    display(HTML(html))"
    )
    new = (
        "    clear_output(wait=True)\n"
        "    display(HTML(html))"
    )
    patch_file(target, old, new, label)


# ════════════════════════════════════════════════════════════
#  Fix 7 — rembg u2net.onnx downloads to ephemeral storage
#  Problem: 176MB model saved to /root/.u2net/ which is wiped
#           every Colab session restart → re-downloaded on first use
#  Fix:    Symlink ~/.u2net → persistent Drive workspace cache
# ════════════════════════════════════════════════════════════
def fix_u2net_cache():
    label = "Fix 7: u2net persistent cache"
    persistent_dir = WS / "cache" / "u2net"
    home_dir = Path.home() / ".u2net"

    try:
        persistent_dir.mkdir(parents=True, exist_ok=True)

        # Already a correct symlink
        if home_dir.is_symlink():
            if home_dir.resolve() == persistent_dir.resolve():
                log_fix(label, "skip", "symlink already correct")
                return
            # Wrong symlink target — remove and recreate
            home_dir.unlink()

        # Real directory with files → move contents to persistent location
        if home_dir.is_dir():
            for item in home_dir.iterdir():
                dest = persistent_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(str(home_dir))

        # Remove if it's a regular file
        if home_dir.exists():
            home_dir.unlink()

        # Create symlink: ~/.u2net → persistent workspace cache
        os.symlink(str(persistent_dir), str(home_dir))
        log_fix(label, "ok", f"symlinked → {persistent_dir}")
    except Exception as exc:
        log_fix(label, "fail", str(exc)[:150])


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════
def main():
    print("━" * 50)
    print("🔧 FreeFakeStudio Fixer — patching known issues")
    print("━" * 50)

    fix_onnxruntime()
    fix_coroutine_init_extra_nodes()
    fix_cuda_warning()
    fix_opencv_cascade()
    fix_encoder_mode_guard()
    fix_duplicate_panel()
    fix_u2net_cache()

    print("━" * 50)
    print(f"  Done: {FIXED} fixed · {SKIPPED} already ok · {FAILED} failed")
    print("━" * 50)

    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
