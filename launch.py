# ============================================================
#  FreeFakeStudio — Launch Script
#  Executed by the notebook cell after drive mount & repo clone.
#  Handles: ComfyUI, GGUF, deps, model downloads, GPU check, launch.
# ============================================================

import os, sys, shutil, subprocess, glob, time, gc
from pathlib import Path
from IPython.display import display, HTML, clear_output

# ── Config from notebook globals / env ─────────────────────
WS = Path(os.environ.get('FFS_WORKSPACE', '/content/drive/MyDrive/FreeFakeStudio'))
REPAIR = os.environ.get('FFS_REPAIR', '') == '1'
COMFYUI = WS / 'ComfyUI'
CACHE   = WS / 'cache'
APP     = WS / 'app'
RESULTS = WS / 'results'

# Ensure directories
for _d in [COMFYUI, CACHE / 'huggingface', CACHE / 'pip', RESULTS]:
    _d.mkdir(parents=True, exist_ok=True)

# Set persistent caches so nothing re-downloads between sessions
os.environ['HF_HOME']               = str(CACHE / 'huggingface')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE / 'huggingface')
os.environ['PIP_CACHE_DIR']         = str(CACHE / 'pip')
os.environ['COMFYUI_ROOT']          = str(COMFYUI)
os.environ['FREEFAKESTUDIO_WORKSPACE'] = str(WS)


# ═══════════════════════════════════════════════════════════
#  BEAUTIFUL STATUS DISPLAY
# ═══════════════════════════════════════════════════════════
_steps = []
_t0 = time.time()

_CSS = '''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.ffs-s{font-family:'Inter',-apple-system,sans-serif;max-width:580px}
.ffs-h{background:linear-gradient(135deg,#0f0f1a,#1a1a2e,#16213e);border-radius:14px;
  padding:20px 24px;margin-bottom:14px;border:1px solid rgba(100,130,220,.12);
  box-shadow:0 4px 20px rgba(0,0,0,.25)}
.ffs-t{font-size:1.7em;font-weight:800;
  background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.ffs-sub{color:#9ca3af;font-size:.88em;margin-top:4px}
.ffs-b{background:#111118;border-radius:12px;border:1px solid rgba(255,255,255,.06);overflow:hidden}
.ffs-r{display:flex;align-items:center;gap:10px;padding:10px 16px;font-size:.9em;color:#d0d0dc;
  border-bottom:1px solid rgba(255,255,255,.04);transition:background .2s}
.ffs-r:last-child{border-bottom:none}
.ffs-r:hover{background:rgba(255,255,255,.02)}
.ffs-i{width:22px;text-align:center;font-size:1em;flex-shrink:0}
.ffs-ok .ffs-i{color:#34d399}
.ffs-err .ffs-i{color:#f87171}
.ffs-run .ffs-i{color:#60a5fa;animation:ffsp 1.2s ease-in-out infinite}
@keyframes ffsp{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
.ffs-d{color:#6b7280;font-size:.82em;margin-left:auto;white-space:nowrap}
.ffs-ft{padding:10px 16px;font-size:.78em;color:#4b5563;
  border-top:1px solid rgba(255,255,255,.04);display:flex;justify-content:space-between}
</style>'''

def _render(final=False):
    elapsed = time.time() - _t0
    h  = _CSS + '<div class="ffs-s">'
    h += '<div class="ffs-h"><div class="ffs-t">🎭 FreeFakeStudio</div>'
    h += '<div class="ffs-sub">Setting up your AI Image Studio…</div></div>'
    h += '<div class="ffs-b">'
    icons = {'ok': '✓', 'err': '✗', 'run': '◌', 'info': '•'}
    for s in _steps:
        cls  = f'ffs-{s["st"]}'
        icon = icons.get(s['st'], '•')
        det  = f'<span class="ffs-d">{s["d"]}</span>' if s.get('d') else ''
        h += f'<div class="ffs-r {cls}"><span class="ffs-i">{icon}</span><span>{s["t"]}</span>{det}</div>'
    h += '</div>'
    h += f'<div class="ffs-ft"><span>⏱ {elapsed:.0f}s</span><span>FreeFakeStudio</span></div>'
    h += '</div>'
    if not final:
        clear_output(wait=True)
    display(HTML(h))

def step(title, detail='', status='run'):
    _steps.append({'t': title, 'd': detail, 'st': status})
    _render()

def done(title=None, detail=''):
    if _steps:
        s = _steps[-1]
        s['st'] = 'ok'
        if title: s['t'] = title
        if detail: s['d'] = detail
        _render()

def fail(detail=''):
    if _steps:
        _steps[-1]['st'] = 'err'
        if detail: _steps[-1]['d'] = detail
        _render()

def _run(cmd, quiet=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet and r.returncode != 0:
        print(f'⚠️ {cmd}\n{r.stderr[-500:] if r.stderr else ""}')
    return r.returncode, r.stdout


# ═══════════════════════════════════════════════════════════
#  1. COMFYUI
# ═══════════════════════════════════════════════════════════
if not (COMFYUI / 'main.py').exists():
    step('ComfyUI', 'Cloning repository…')
    _run(f'git clone https://github.com/comfyanonymous/ComfyUI.git "{COMFYUI}"', quiet=False)
    _run(f'pip install -q -r "{COMFYUI}/requirements.txt"', quiet=False)
    done('ComfyUI', 'Installed')
else:
    step('ComfyUI', 'Cached on Drive', 'ok')

# Symlink  /content/ComfyUI  →  persistent copy
_link = Path('/content/ComfyUI')
if _link.is_symlink():
    _link.unlink()
elif _link.exists():
    shutil.rmtree(str(_link))
os.symlink(str(COMFYUI), str(_link))

# Model directories
for _d in ['diffusion_models', 'text_encoders', 'vae', 'clip']:
    (COMFYUI / 'models' / _d).mkdir(parents=True, exist_ok=True)

# Cross-link  clip/ ↔ text_encoders/  (ComfyUI compatibility)
_clip_d = COMFYUI / 'models' / 'clip'
_te_d   = COMFYUI / 'models' / 'text_encoders'
for _f in glob.glob(str(_clip_d / '*')):
    _t = _te_d / os.path.basename(_f)
    if not _t.exists():
        try: os.symlink(_f, str(_t))
        except OSError: pass
for _f in glob.glob(str(_te_d / '*')):
    _t = _clip_d / os.path.basename(_f)
    if not _t.exists():
        try: os.symlink(_f, str(_t))
        except OSError: pass


# ═══════════════════════════════════════════════════════════
#  2. COMFYUI-GGUF  (required for ERNIE-Image-Turbo)
# ═══════════════════════════════════════════════════════════
_gguf = COMFYUI / 'custom_nodes' / 'ComfyUI-GGUF'
if not (_gguf / 'nodes.py').exists():
    step('ComfyUI-GGUF', 'Installing…')
    (COMFYUI / 'custom_nodes').mkdir(exist_ok=True)
    _run(f'git clone https://github.com/city96/ComfyUI-GGUF.git "{_gguf}"', quiet=False)
    if (_gguf / 'requirements.txt').exists():
        _run(f'pip install -q -r "{_gguf}/requirements.txt"', quiet=False)
    done('ComfyUI-GGUF', 'Installed')
else:
    step('ComfyUI-GGUF', 'Ready', 'ok')


# ═══════════════════════════════════════════════════════════
#  3. PYTHON DEPENDENCIES
# ═══════════════════════════════════════════════════════════
step('Dependencies', 'Checking…')
_deps = []
for _mod, _pkg in [('rembg', 'rembg[gpu]'), ('onnxruntime', 'onnxruntime-gpu'),
                    ('gradio', 'gradio'), ('cv2', 'opencv-python-headless')]:
    try:
        __import__(_mod)
    except ImportError:
        _deps.append(_pkg)
if _deps:
    _steps[-1]['d'] = f'Installing {len(_deps)} packages…'
    _render()
    _run(f'pip install -q {" ".join(_deps)}', quiet=False)

# Always ensure Pillow is up-to-date (ComfyUI may install incompatible version)
_steps[-1]['d'] = 'Checking Pillow…'
_render()
_run('pip install -q --upgrade Pillow', quiet=False)
# Clear any cached PIL modules so the upgraded version is used
import sys as _sys
for _k in list(_sys.modules.keys()):
    if _k == 'PIL' or _k.startswith('PIL.'):
        del _sys.modules[_k]

done('Dependencies', f'{4 - len(_deps)}/4 cached' if len(_deps) < 4 else 'Installed')


# ═══════════════════════════════════════════════════════════
#  4. MODEL DOWNLOADS  (HuggingFace → Drive persistence)
# ═══════════════════════════════════════════════════════════
from huggingface_hub import hf_hub_download

DIFF = str(COMFYUI / 'models' / 'diffusion_models')
TE   = str(COMFYUI / 'models' / 'text_encoders')
CL   = str(COMFYUI / 'models' / 'clip')
VA   = str(COMFYUI / 'models' / 'vae')

def _ok(directory, filename, min_bytes=500_000):
    """Check a model file exists and isn't corrupt (truncated)."""
    p = os.path.join(directory, filename)
    return os.path.isfile(p) and os.path.getsize(p) >= min_bytes

def _dl(repo, remote_path, dest_dir, dest_name=None):
    """Download from HuggingFace cache, then copy to the expected directory."""
    cached = hf_hub_download(repo_id=repo, filename=remote_path)
    nm = dest_name or os.path.basename(remote_path)
    fp = os.path.join(dest_dir, nm)
    os.makedirs(dest_dir, exist_ok=True)
    if not os.path.isfile(fp) or os.path.getsize(fp) < 1024:
        shutil.copy2(cached, fp)
    # Also symlink into clip/ or text_encoders/ for ComfyUI compatibility
    if dest_dir == TE:
        _alt = os.path.join(CL, nm)
        if not os.path.exists(_alt):
            try: os.symlink(fp, _alt)
            except OSError: pass
    elif dest_dir == CL:
        _alt = os.path.join(TE, nm)
        if not os.path.exists(_alt):
            try: os.symlink(fp, _alt)
            except OSError: pass


# ── Z-Image Turbo ─────────────────────────────────────────
_need = not all([
    _ok(DIFF, 'z-image-turbo-fp8-e4m3fn.safetensors'),
    _ok(TE,   'qwen_3_4b.safetensors'),
    _ok(VA,   'ae.safetensors'),
])
if _need or REPAIR:
    step('Z-Image Turbo', 'Downloading…')
    if not _ok(DIFF, 'z-image-turbo-fp8-e4m3fn.safetensors') or REPAIR:
        _dl('T5B/Z-Image-Turbo-FP8',
            'z-image-turbo-fp8-e4m3fn.safetensors', DIFF)
    if not _ok(TE, 'qwen_3_4b.safetensors') or REPAIR:
        _dl('Comfy-Org/z_image_turbo',
            'split_files/text_encoders/qwen_3_4b.safetensors', TE,
            'qwen_3_4b.safetensors')
    if not _ok(VA, 'ae.safetensors') or REPAIR:
        _dl('Comfy-Org/z_image_turbo',
            'split_files/vae/ae.safetensors', VA,
            'ae.safetensors')
    done('Z-Image Turbo', 'model · encoder · VAE')
else:
    step('Z-Image Turbo', 'model · encoder · VAE', 'ok')

# ── FLUX.2-klein 4B ───────────────────────────────────────
_need = not all([
    _ok(DIFF, 'flux-2-klein-4b.safetensors'),
    _ok(TE,   'qwen_3_4b_fp4_flux2.safetensors'),
    _ok(VA,   'flux2-vae.safetensors'),
])
if _need or REPAIR:
    step('FLUX.2-klein 4B', 'Downloading…')
    if not _ok(DIFF, 'flux-2-klein-4b.safetensors') or REPAIR:
        _dl('black-forest-labs/FLUX.2-klein-4B',
            'flux-2-klein-4b.safetensors', DIFF)
    if not _ok(TE, 'qwen_3_4b_fp4_flux2.safetensors') or REPAIR:
        _dl('Comfy-Org/vae-text-encorder-for-flux-klein-4b',
            'split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors', TE,
            'qwen_3_4b_fp4_flux2.safetensors')
    if not _ok(VA, 'flux2-vae.safetensors') or REPAIR:
        _dl('Comfy-Org/vae-text-encorder-for-flux-klein-4b',
            'split_files/vae/flux2-vae.safetensors', VA,
            'flux2-vae.safetensors')
    done('FLUX.2-klein 4B', 'model · encoder · VAE')
else:
    step('FLUX.2-klein 4B', 'model · encoder · VAE', 'ok')

# ── ERNIE-Image-Turbo ─────────────────────────────────────
_need = not all([
    _ok(DIFF, 'ernie-image-turbo-Q6_K.gguf'),
    _ok(TE,   'ministral-3-3b.safetensors'),
    _ok(VA,   'flux2-vae.safetensors'),
])
if _need or REPAIR:
    step('ERNIE-Image-Turbo', 'Downloading…')
    if not _ok(DIFF, 'ernie-image-turbo-Q6_K.gguf') or REPAIR:
        _dl('unsloth/ERNIE-Image-Turbo-GGUF',
            'ernie-image-turbo-Q6_K.gguf', DIFF)
    if not _ok(TE, 'ministral-3-3b.safetensors') or REPAIR:
        _dl('Comfy-Org/ERNIE-Image',
            'text_encoders/ministral-3-3b.safetensors', TE,
            'ministral-3-3b.safetensors')
    if not _ok(VA, 'flux2-vae.safetensors') or REPAIR:
        _dl('Comfy-Org/ERNIE-Image',
            'vae/flux2-vae.safetensors', VA,
            'flux2-vae.safetensors')
    done('ERNIE-Image-Turbo', 'GGUF · Ministral · VAE')
else:
    step('ERNIE-Image-Turbo', 'GGUF · Ministral · VAE', 'ok')


# ═══════════════════════════════════════════════════════════
#  5. GPU HEALTH CHECK
# ═══════════════════════════════════════════════════════════
import torch as _torch
if _torch.cuda.is_available():
    _gn = _torch.cuda.get_device_name(0)
    _vm = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
    step('GPU', f'{_gn} · {_vm:.1f} GB VRAM', 'ok')
else:
    step('GPU', 'No CUDA device — generation will fail!', 'err')


# ═══════════════════════════════════════════════════════════
#  6. LAUNCH FREEFAKESTUDIO
# ═══════════════════════════════════════════════════════════
step('FreeFakeStudio', 'Launching…')

# Final render — don't clear output so Gradio URL appears below
_render(final=True)

os.chdir(str(APP))
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

# Execute app.py in this process
exec(open(str(APP / 'app.py')).read())
