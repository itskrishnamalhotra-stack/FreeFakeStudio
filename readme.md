# FreeFakeStudio

FreeFakeStudio is a Google Colab-first AI image generation and editing studio built around the existing Python, Gradio, and ComfyUI engines.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itskrishnamalhotra-stack/FreeFakeStudio/blob/main/FreeFakeStudio.ipynb)

This version is designed for:

- one main Colab cell;
- persistent Google Drive storage;
- no local model downloads or local inference;
- a Gemini-like image workspace;
- memory-safe one-model-at-a-time loading on a Colab T4.

## Supported Models

Only these models are active in the normal UI:

| Model | Main use | Capabilities |
| --- | --- | --- |
| Z-Image Turbo | Fast text to image | Text to image |
| FLUX.2-klein 4B | Best all-purpose model | Text to image, img2img, inpaint |
| ERNIE-Image-Turbo | Fast text to image | Text to image |

The older FLUX.2-klein 9B and Qwen-Image-Edit engine files may remain in the repo for reference, but this setup does not download, load, or show those models.

## Colab Quick Start

1. Open `FreeFakeStudio.ipynb` in Google Colab.
2. Keep or edit `WORKSPACE_DIR`, for example:

   ```python
   /content/drive/MyDrive/FreeFakeStudio
   ```

3. Run the single cell.
4. Authorize Google Drive.
5. Wait for first-run setup and model downloads.
6. Open the large `OPEN FREEFAKESTUDIO` HTTPS link printed after the local server starts.

Later runs reuse the Drive workspace and should skip existing downloads.

### Optional FLUX Encoder

The single Colab cell includes three FLUX-only fields:

- `FLUX_ENCODER`: start with `Official` or a previously validated `Custom` encoder;
- `FLUX_CUSTOM_ENCODER_URL`: a complete Hugging Face `.gguf` or `.safetensors` file URL;
- `HUGGINGFACE_TOKEN`: optional access token for a gated repository whose terms you accepted.

Custom encoders must be single-file Qwen3-4B text encoders made specifically for
FLUX.2 Klein 4B. Generic Qwen chat models, Z-Image encoders, model folders, ZIPs,
and sharded Transformers repositories are not compatible with this loader.

Free Colab constraints:

- recommended format: GGUF Q4 (`Q4_K_M` or `Q4_0`);
- recommended download size: 2.0-2.7 GiB;
- accepted size: 500 MiB through 3.0 GiB;
- hard rejection above 3.0 GiB to protect host RAM;
- only the selected encoder is loaded, never Official and Custom together.

The launcher checks Hugging Face metadata before downloading, validates the file
header after downloading, and stores the file plus its manifest in Google Drive.
When FLUX is selected, open Settings, choose `Official` or `Custom`, and press
`Apply encoder`. Applying a change unloads FLUX; the next generation reloads it
with the selected encoder. Z-Image and ERNIE are unaffected.

For the most reliable route, paste your ngrok auth token into the notebook's
`NGROK_AUTH_TOKEN` field. Leave token fields blank to reuse saved private Drive
settings or Colab Secrets. The launcher will print:

```text
OPEN FREEFAKESTUDIO (ngrok):
https://...
```

Leave `NGROK_AUTH_TOKEN` blank to use Colab's signed-in HTTPS proxy. The launcher
passes the selected external URL to Gradio as an absolute proxy root so Gradio
does not generate blocked internal HTTP URLs for its API, theme, or assets.

## Persistent Workspace Layout

The notebook stores project-owned files under the selected Drive workspace:

```text
FreeFakeStudio/
|-- app/                         # Modified FreeFakeStudio source
|-- ComfyUI/
|   |-- models/
|   |   |-- diffusion_models/
|   |   |-- text_encoders/
|   |   |-- clip/
|   |   `-- vae/
|   `-- custom_nodes/
|       `-- ComfyUI-GGUF/
|-- cache/
|   |-- huggingface/
|   `-- pip/
|-- gradio_tmp/
`-- results/
```

`/content/ComfyUI` is only a compatibility symlink to the persistent Drive `ComfyUI` folder. Large model files are not copied from Drive to `/content` every session.

## Startup Behavior

First run:

- mounts Google Drive;
- clones the configured modified repo into `WORKSPACE_DIR/app` if missing;
- prepares persistent cache paths;
- checks/repairs NumPy consistency before ComfyUI imports;
- installs only missing required Python packages where practical;
- installs ComfyUI and ComfyUI-GGUF if missing;
- enforces the model-compatible ComfyUI `v0.28.0` backend revision;
- reconciles ComfyUI and ComfyUI-GGUF Python requirements in every fresh Colab session;
- verifies `torchsde`, `comfy.samplers`, and `comfy.sd` imports before opening the UI;
- constructs and validates the complete Z-Image engine node set before opening the UI;
- checks the Z-Image GGUF and mixed-FP4 headers before launch;
- uses the 4.12 GB Q3_K_M Z-Image GGUF diffusion model with Comfy-Org's 3.48 GB mixed-FP4 text encoder to fit free Colab host RAM;
- enables DynamicVRAM, disables execution caching and pinned-memory duplication, and logs RAM/VRAM around every Z-Image component load;
- downloads only missing or repair-requested model files;
- validates and persists an optional FLUX custom encoder without loading two encoders;
- stops a previous PID-verified FreeFakeStudio child before a cell rerun;
- creates one HTTPS route (ngrok when configured, otherwise Colab proxy);
- launches Gradio with that route set as its absolute proxy root.

Later runs:

- mount Drive;
- reuse `app`, `ComfyUI`, caches, and models;
- validate files with fast size checks;
- skip existing downloads;
- launch the UI.

`UPDATE_APP=True` fast-forwards the Drive app copy and refreshes managed backend repositories. It does not hard reset or delete your modified app folder. The required ComfyUI compatibility tag is enforced automatically even when this option is off.

`REPAIR_INSTALL=True` rechecks/redownloads suspicious or missing install files.

## Debugging

Debug mode is enabled by default with `FFS_DEBUG=1`.

Colab setup writes JSON reports here:

```text
WORKSPACE_DIR/diagnostics/latest.json
WORKSPACE_DIR/diagnostics/startup_*.json
WORKSPACE_DIR/diagnostics/dependencies_*.json
WORKSPACE_DIR/diagnostics/models_*.json
WORKSPACE_DIR/diagnostics/comfy_runtime_*.json
WORKSPACE_DIR/diagnostics/launch_*.json
WORKSPACE_DIR/diagnostics/error_*.json
```

These reports include:

- Python and platform details;
- important environment variables;
- package versions;
- ComfyUI/app path checks;
- required model file size and symlink checks;
- active FLUX encoder mode, format, size, filename, and source URL;
- `nvidia-smi` output when available;
- full traceback on setup failure.

Runtime generation/model-load errors write full tracebacks here:

```text
WORKSPACE_DIR/results/_debug/error_*.txt
```

The final Gradio app startup process streams to:

```text
WORKSPACE_DIR/diagnostics/app_launch_latest.log
```

If Colab fails, download or open `diagnostics/latest.json` and the newest `results/_debug/error_*.txt`, then paste those into the issue/chat.

## Memory Behavior

The app starts with no image model loaded.

On generation:

- selected model is loaded lazily;
- repeated requests with the same selected model reuse it;
- switching models unloads the previous engine first;
- Python garbage collection and CUDA cache cleanup run during model switches;
- generation queue concurrency is limited for a single Colab GPU.

If a load fails, the model manager unloads partial state and clears memory before reporting the error.

Changing between already installed Official and Custom encoders in the UI does
not require reconnecting Colab. To add or replace the custom URL, stop the running
cell, edit the form, and run the same cell again. Disconnect/reconnect the runtime
only after an actual Colab RAM crash or when a stale CUDA allocation survives the
normal unload path.

## UI Workflow

The Gradio app uses a conversational image workflow:

- type a prompt for text to image;
- attach an image for editing;
- use FLUX.2-klein 4B for img2img or mask/inpaint;
- choose mask modes: `Manual Paint`, `Background Only`, or `Everything Except Face`;
- view generated images in the result gallery;
- download result files;
- use `Add to Prompt` to continue editing a generated image;
- use `Regenerate` to rerun the previous request with a fresh seed;
- use `New` to clear the current conversation workspace.

## Avatar Studio

Avatar Studio is a persistent, Flux-only identity workflow stored under
`results/avatars/` in Google Drive:

1. Create or select a named avatar.
2. Generate or upload a face reference, then confirm it. SmolVLM records the
   visible identity details before the Body step unlocks.
3. Generate or upload a full-body reference, then confirm it. The Console and
   Gallery steps unlock after both references are saved.
4. Use the Console for a saved conversation. Face and body occupy two of Flux's
   four reference slots, leaving two optional user reference slots.
5. Use Auto Gallery to discover references, write prompts, generate with Flux,
   validate identity/anatomy with SmolVLM, and automatically repair failed
   prompts. Selecting an image exposes a manual failed-generation regeneration
   control. Gallery images and chat images are stored separately.

Auto Gallery reads `GEMINI_API_KEY` and `TAVILY_API_KEY` from the single Colab
cell form, Colab Secrets (the key icon in Colab's left sidebar), or saved private
Drive settings. The committed notebook keeps those fields blank. Do not place API
keys in committed notebooks or source files. Local development uses mock search,
generation, and validation and never loads an AI model.

The Colab form also exposes the practical knobs from the reference-finder and
SmolVLM notebooks:

- `GEMINI_MODEL`: optional override; blank auto-picks an available Gemini Flash model.
- `AVATAR_REFERENCE_DOMAINS`: comma-separated Tavily domain filter, defaulting to `instagram.com`.
- `AVATAR_REFERENCE_TIME_RANGE`: optional recency filter such as `month`.
- `AVATAR_SEARCH_ROUNDS`: 1-5 Tavily/Gemini planning rounds; more rounds cost more but improve fill rate.
- `AVATAR_GALLERY_RETRIES`: 0-3 prompt repair attempts after a failed validation.
- `AVATAR_MAX_CANDIDATE_DOWNLOADS`: candidate download cap per search round.
- `AVATAR_VISION_MAX_EDGE` and `AVATAR_VISION_MAX_TOKENS`: SmolVLM memory/detail controls.

Free T4 loading stays lazy on purpose. Opening the app does not preload Flux,
Z-Image, ERNIE, or SmolVLM. The first Avatar Studio generation or analysis pays
the model-load cost; later operations reuse the loaded Flux and SmolVLM models.
This avoids making every launch consume the peak RAM required by both models.

Local execution defaults to development mode. It builds the UI and uses mock engines only. It does not download or load real AI models.

```bash
python app.py
```

## Important Local Safety Rule

Do not download model checkpoints or run real inference on a normal laptop.

Local work is for:

- source edits;
- syntax checks;
- notebook JSON validation;
- mock model-manager tests;
- UI-only development.

Real generation must happen in Google Colab with a GPU runtime.

## Files

```text
FreeFakeStudio.ipynb
launch.py
app.py
model_manager.py
workspace.py
engine_z_image.py
engine_flux_klein_4b.py
engine_ernie_image_turbo.py
```

## Known Limitations

- Real Z-Image, FLUX, and ERNIE generation still require final Colab GPU verification.
- First run can still take a long time because the required checkpoints are large.
- Hugging Face gated or rate-limited files may require the user to be logged in or provide a token in Colab.
- Gradio temporary public links are session-based and expire when the Colab runtime stops.

## Colab Test Checklist

1. Start a fresh Colab runtime with GPU enabled.
2. Open `FreeFakeStudio.ipynb`.
3. Set `WORKSPACE_DIR` to your Drive folder.
4. Keep `UPDATE_APP=False` and `REPAIR_INSTALL=False` for normal testing.
5. Run the single cell.
6. Confirm setup reaches `FreeFakeStudio / Launching`.
7. Open the printed `OPEN FREEFAKESTUDIO` HTTPS link.
8. Generate with `Z-Image Turbo`.
9. Generate with `FLUX.2-klein 4B`.
10. If configured, switch FLUX to `Custom`, press `Apply encoder`, and generate again.
11. Switch FLUX back to `Official`, press `Apply encoder`, and generate again.
12. Attach an image, use FLUX img2img.
13. Test `Background Only`, `Everything Except Face`, and `Manual Paint`.
14. Test `Add to Prompt`, then edit the attached generated image.
15. Test `Regenerate`.
16. Switch models in this order: Z-Image -> FLUX -> ERNIE -> FLUX.
17. If anything fails, collect `diagnostics/latest.json` and the newest `results/_debug/error_*.txt`.
