# 🎭 FreeFakeStudio

Open-source AI image generation and editing studio with a Gemini-like conversational interface, powered by ComfyUI inference engines.

## Features

- **Conversational Image Workspace** — Gemini-like chat interface for generating and editing images
- **Text to Image** — Generate images from text prompts
- **Image to Image** — Transform existing images with text guidance  
- **Inpainting** — Edit specific regions using manual or auto-generated masks
- **Add to Prompt** — Use a generated image as input for the next edit
- **One Model at a Time** — Memory-safe model switching for T4 GPUs
- **Persistent Google Drive Setup** — Download once, use across Colab sessions

## Supported Models

| Model | Capabilities | Notes |
|-------|-------------|-------|
| **Z-Image Turbo** | Text → Image, Img2Img, Inpaint | Fast generation (8 steps) |
| **FLUX.2-klein 4B** | Text → Image, Img2Img, Inpaint | Best all-purpose model |
| **ERNIE-Image-Turbo** | Text → Image | Fast generation, GGUF format |

## Quick Start (Google Colab)

1. Open `FreeFakeStudio.ipynb` in Google Colab
2. Optionally set your preferred Drive workspace path
3. Click **Run** on the single setup cell
4. Authorize Google Drive when prompted
5. Wait for models to download (first run only)
6. Click the Gradio share link to open the interface

### First Run

The first run downloads ~25 GB of model weights to your Google Drive. This takes time but only happens once.

### Later Runs

Subsequent sessions skip all downloads. The notebook verifies existing files and launches the UI directly.

## Architecture

### Single-Model-at-a-Time VRAM Design

Designed for Google Colab T4 (~15 GB VRAM):

- App starts with **no model loaded**
- User selects a model and submits a request
- The selected model loads into VRAM
- **Same model remains loaded** for subsequent requests (fast)
- Switching models **unloads the previous** model first
- Memory cleanup: `gc.collect()` + `torch.cuda.empty_cache()` + `ipc_collect()`

### Persistent Google Drive Storage

Everything persists under your selected workspace directory:

```
Workspace/
├── app/               # FreeFakeStudio source code
├── ComfyUI/           # ComfyUI + model weights
│   ├── models/
│   │   ├── diffusion_models/
│   │   ├── text_encoders/
│   │   └── vae/
│   └── custom_nodes/ComfyUI-GGUF/
├── cache/             # HuggingFace + pip caches
└── results/           # Generated images
```

Symlinks (`/content/ComfyUI` → Drive) avoid copying multi-GB files.

## Gemini-like Interface

The UI provides a conversational workflow:

1. **Type a prompt** and click Send
2. **View results** with Download, Add to Prompt, and Regenerate buttons
3. **Add to Prompt** attaches the result to the composer for iterative editing
4. **Mask/Edit Tools** panel for manual painting or auto-mask (Background Only, Everything Except Face)
5. **Settings** accordion for aspect ratio, steps, CFG, seed, negative prompt
6. **Model selector** in the header with status indicator

## Advanced Options

### Notebook Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WORKSPACE_DIR` | `/content/drive/MyDrive/FreeFakeStudio` | Persistent storage location |
| `UPDATE_APP` | `False` | Pull latest source code |
| `REPAIR_INSTALL` | `False` | Re-download model files |
| `PROJECT_REPO_URL` | `...` | Git repository for source |

### Local Development

When run outside Colab, the app enters **Development Mode**:
- Mock engines return placeholder images
- No GPU or model downloads required  
- Full UI testing with responsive layout

```bash
python app.py
```

## Project Structure

```
FreeFakeStudio/
├── app.py                      # Main Gradio application
├── model_manager.py            # Thread-safe model switching
├── workspace.py                # Path configuration
├── engine_z_image.py           # Z-Image Turbo engine
├── engine_flux_klein_4b.py     # FLUX.2-klein 4B engine
├── engine_ernie_image_turbo.py # ERNIE-Image-Turbo engine
├── engine_flux_klein_9b.py     # (Inactive) FLUX.2-klein 9B
├── engine_qwen_edit_2511.py    # (Inactive) Qwen-Image-Edit
├── FreeFakeStudio.ipynb        # Colab notebook
└── readme.md
```

## License

See [LICENSE](LICENSE) for details.
