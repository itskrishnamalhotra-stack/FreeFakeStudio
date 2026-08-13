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
| Z-Image Turbo | Fast text to image | Text to image, img2img, inpaint engine support |
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
6. Open the public `gradio.live` link printed by Gradio.

Later runs reuse the Drive workspace and should skip existing downloads.

If Gradio share is unreliable, paste your ngrok auth token into the notebook's
`NGROK_AUTH_TOKEN` field. The launcher will print:

```text
OPEN INTERFACE (ngrok): https://...
```

Leave `NGROK_AUTH_TOKEN` blank to use Gradio share and the Colab proxy fallback.

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
- downloads only missing or repair-requested model files;
- launches Gradio with `share=True`.

Later runs:

- mount Drive;
- reuse `app`, `ComfyUI`, caches, and models;
- validate files with fast size checks;
- skip existing downloads;
- launch the UI.

`UPDATE_APP=True` performs a fast-forward `git pull` only. It does not hard reset or delete your modified app folder.

`REPAIR_INSTALL=True` rechecks/redownloads suspicious or missing install files.

## Debugging

Debug mode is enabled by default with `FFS_DEBUG=1`.

Colab setup writes JSON reports here:

```text
WORKSPACE_DIR/diagnostics/latest.json
WORKSPACE_DIR/diagnostics/startup_*.json
WORKSPACE_DIR/diagnostics/dependencies_*.json
WORKSPACE_DIR/diagnostics/models_*.json
WORKSPACE_DIR/diagnostics/launch_*.json
WORKSPACE_DIR/diagnostics/error_*.json
```

These reports include:

- Python and platform details;
- important environment variables;
- package versions;
- ComfyUI/app path checks;
- required model file size and symlink checks;
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
7. Open the printed `gradio.live` link.
8. Generate with `Z-Image Turbo`.
9. Generate with `FLUX.2-klein 4B`.
10. Attach an image, use FLUX img2img.
11. Test `Background Only`, `Everything Except Face`, and `Manual Paint`.
12. Test `Add to Prompt`, then edit the attached generated image.
13. Test `Regenerate`.
14. Switch models in this order: Z-Image -> FLUX -> ERNIE -> FLUX.
15. If anything fails, collect `diagnostics/latest.json` and the newest `results/_debug/error_*.txt`.
