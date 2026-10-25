# FreeFakeStudio

FreeFakeStudio is a Colab/Kaggle AI image studio with a React/TypeScript interface,
FastAPI job server, and the existing Python/ComfyUI image engines.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itskrishnamalhotra-stack/FreeFakeStudio/blob/main/FreeFakeStudio.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https%3A%2F%2Fraw.githubusercontent.com%2Fitskrishnamalhotra-stack%2FFreeFakeStudio%2Fmain%2FFreeFakeStudio.ipynb)

**New interface and Kaggle support:** follow [CLOUD_SETUP.md](CLOUD_SETUP.md) for
the complete current setup, Google Drive OAuth instructions, recovery behavior,
UI acceptance checks and hosting limitations. Import this same notebook into
Kaggle; enable Internet/GPU and supply private credentials. Colab still uses its
normal Drive login. Kaggle uses Drive API synchronization, not Colab's mount.

**Image-only influencer workflow:** see [INFLUENCER_STUDIO.md](INFLUENCER_STUDIO.md)
for the Identity, Quality and Content tabs, reference-pack approval, CPU identity
calibration, resumable consistency tests, caption/export workflow and optional
Instagram publishing. No LoRA training, video or voice providers are required.

**Scene-aware enhancement:** [ENHANCEMENT_PROFILES.md](ENHANCEMENT_PROFILES.md)
lists the 15 editable prompt profiles, automatic subject/focus routing, saved
analysis details and review limitations. Background people no longer turn a
landscape into a portrait; artwork and documents retain their original medium.

This release also fixes identity-pack invalidation, historical policy selection,
benchmark correction and empty Drive restores. Reference views can be selected
explicitly; reconstruction has matched-size/detail-crop review; offline captions
have persona-aware styles. Optional runtime scheduling only publishes explicitly
approved future posts while that runtime remains active.

**Provider limits:** free Colab restricts bypassing its notebook UI to interact
primarily through a web UI, including content-generation interfaces. Use a
permitted plan/runtime and follow the provider's rules; this launcher cannot
bypass quotas or guarantee session duration. See the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html).

The notebook now launches `web_server.py`, not Gradio. The frontend is prebuilt:
Node is not needed in the notebook. Chat, avatar setup, console, reference
pipeline, reconstruction, settings, jobs and image viewing are native web UI.
The default model download set is FLUX-only; choose `MODEL_SET=all` for optional
Z-Image/ERNIE. Existing files are never deleted. Historical Gradio-specific
sections below describe the retained legacy UI, not the new default interface.

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

### One-Time Keys File

To avoid pasting API keys into the Colab form every run:

1. Open `FreeFakeStudio.keys.example.txt`.
2. Fill your real values and save it locally as `FreeFakeStudio.keys.txt`.
3. In Colab, set `UPLOAD_KEYS_TXT=True` once and run the cell.
4. Upload your filled txt file when Colab asks.
5. The cell saves it privately to:

   ```text
   WORKSPACE_DIR/config/freefakestudio_keys.txt
   ```

6. On later runs, keep `UPLOAD_KEYS_TXT=False`; the cell auto-loads the saved
   Drive copy. The direct token fields can stay blank.

The parser accepts `KEY=VALUE` or `KEY: VALUE` lines and ignores comments. These
keys are supported:

```text
PUBLIC_ROUTE
PRELOAD_FLUX
PRELOAD_AVATAR_VISION
FAST_RESTORE
STAGE_FLUX_TO_SSD
WARMUP_ENABLED
WARMUP_SIZE
EXPOSE_AFTER_WARMUP
FORCE_REBUILD_BUNDLES
FORCE_CACHE_REFRESH
STARTUP_VERBOSE
NGROK_AUTH_TOKEN
HUGGINGFACE_TOKEN
GEMINI_API_KEY
TAVILY_API_KEY
GEMINI_MODEL
FLUX_ENCODER
FLUX_CUSTOM_ENCODER_URL
AVATAR_REFERENCE_DOMAINS
AVATAR_REFERENCE_TIME_RANGE
AVATAR_SEARCH_ROUNDS
AVATAR_GALLERY_RETRIES
AVATAR_MAX_CANDIDATE_DOWNLOADS
AVATAR_VISION_MAX_EDGE
AVATAR_VISION_MAX_TOKENS
```

`FreeFakeStudio.keys.txt` is ignored by Git. Commit only the example template.

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
Later starts reuse a matching validated file without a Hugging Face metadata
request. The default Official mode does not download an unused custom encoder.
When FLUX is selected, open Settings, choose `Official` or `Custom`, and press
`Apply encoder`. Applying a change unloads FLUX; the next generation reloads it
with the selected encoder. Z-Image and ERNIE are unaffected.

### Public Route

Use `PUBLIC_ROUTE=Colab proxy` for the most stable Colab run. This route keeps
Gradio's queue stream on Colab's signed-in HTTPS proxy and avoids the ngrok
`/gradio_api/queue/data` reconnect loop.

Use `PUBLIC_ROUTE=ngrok` only when you need an external public link, and set
`NGROK_AUTH_TOKEN` in the notebook or keys file. The launcher will print:

```text
OPEN FREEFAKESTUDIO (ngrok):
https://...
```

`PUBLIC_ROUTE=Auto` preserves the older behavior: use ngrok when a token is
available, otherwise use Colab proxy. If the browser says the server connection
was lost and keeps reconnecting on an ngrok URL, switch back to
`PUBLIC_ROUTE=Colab proxy`, run the cell once with `UPDATE_APP=True`, then run
again normally.

Leave token fields blank to reuse saved private Drive settings or Colab Secrets.
The launcher passes the selected external URL to Gradio as an absolute proxy root
so Gradio does not generate blocked internal HTTP URLs for its API, theme, or
assets.

### Startup Priority

The recommended defaults are `PRELOAD_FLUX=True`,
`PRELOAD_AVATAR_VISION=True`, `WARMUP_ENABLED=True`, and
`EXPOSE_AFTER_WARMUP=True`. The launcher loads FLUX, runs discarded
256x256 one-step text-to-image and reference-edit requests, loads and cheaply warms SmolVLM,
starts Gradio, checks its local HTTP health, saves the warmed caches, and only
then prints the external URL.

`FAST_RESTORE=True` restores compatible prepared bundles to local Colab SSD.
`STAGE_FLUX_TO_SSD=True` also copies the active FLUX diffusion model, encoder,
and VAE to SSD when at least 4 GiB of disk headroom remains. If there is not
enough local disk, the launcher safely links that file to its persistent Drive
copy instead.

Disabling `PRELOAD_FLUX` requires `EXPOSE_AFTER_WARMUP=False`, because a strict
warmup gate cannot declare a model ready when it was not loaded.

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
|-- manifests/
|   |-- current-runtime.json
|   |-- app-source.json
|   |-- comfyui-source.json
|   |-- python-overlay.json
|   |-- runtime-cache.json
|   `-- last-ready.json
|-- bundles/
|   |-- source/
|   |-- environment/
|   |-- caches/
|   `-- wheelhouse/
|-- gradio_tmp/
`-- results/
```

Prepared source, environment, and cache bundles are restored under
`/content/freefakestudio_runtime`. The app and ComfyUI source run from that local
SSD directory. Persistent results and master model files remain on Drive. The
active FLUX files are staged locally when configured and disk space permits;
other model files are linked from the local ComfyUI model tree to Drive.

## Startup Behavior

First run:

- mounts Google Drive;
- clones the configured modified repo into `WORKSPACE_DIR/app` if missing;
- creates the organized manifests, bundles, wheelhouse, cache, and result paths;
- checks/repairs NumPy consistency before ComfyUI imports;
- installs only missing required Python packages where practical;
- installs ComfyUI and ComfyUI-GGUF if missing;
- enforces the model-compatible ComfyUI `v0.28.0` backend revision;
- fingerprints Python, PyTorch, CUDA, cuDNN, GPU capability, and NVIDIA driver;
- restores only environment and compiled-cache bundles matching that fingerprint;
- rebuilds an invalid or incompatible Python overlay from pip/wheel caches;
- verifies `torchsde`, `comfy.samplers`, and `comfy.sd` imports before opening the UI;
- constructs and validates the complete Z-Image engine node set before opening the UI;
- checks the Z-Image GGUF and mixed-FP4 headers before launch;
- uses the 4.12 GB Q3_K_M Z-Image GGUF diffusion model with Comfy-Org's 3.48 GB mixed-FP4 text encoder to fit free Colab host RAM;
- enables DynamicVRAM, disables execution caching and pinned-memory duplication, and logs RAM/VRAM around every Z-Image component load;
- downloads only missing or repair-requested model files;
- validates and persists an optional FLUX custom encoder without loading two encoders;
- creates checksum-verified app and ComfyUI source archives;
- extracts those archives atomically to local Colab SSD;
- stages the selected FLUX files to SSD when space permits;
- stops a previous PID-verified FreeFakeStudio child before a cell rerun;
- creates one HTTPS route from `PUBLIC_ROUTE` (`Colab proxy` by default, `ngrok` when selected);
- preloads FLUX when `PRELOAD_FLUX=True`;
- runs discarded FLUX and SmolVLM warmups when `WARMUP_ENABLED=True`;
- starts Gradio and verifies its local HTTP health;
- atomically saves warmed Torch, Triton, CUDA, extensions, and Hugging Face caches;
- prints the external link only after the readiness gate succeeds.

Later runs:

- mount Drive;
- compare the runtime fingerprint and source/requirements signatures;
- restore verified source, Python overlay, and warm-cache archives to local SSD;
- skip dependency installation when the restored environment passes its import probe;
- reuse persistent models without downloading them again;
- skip existing downloads;
- preload and warm FLUX;
- expose the link after the health gate passes.

`UPDATE_APP=True` fast-forwards the Drive app copy and refreshes managed backend repositories. If the Drive app checkout has local edits, the notebook saves them first with `git stash push -u` and then pulls, so the update is not blocked by files such as `model_manager.py`. It does not hard reset or delete your modified app folder. The required ComfyUI compatibility tag is enforced automatically even when this option is off.

`REPAIR_INSTALL=True` rechecks/redownloads suspicious or missing install files.

`FORCE_REBUILD_BUNDLES=True` rebuilds the source and Python environment archives.
Use it after changing backend files manually or when diagnosing a restore issue.
`FORCE_CACHE_REFRESH=True` ignores the saved compiled-cache archive and writes a
clean replacement after warmup. Both switches should return to `False` after one
successful repair run. Corrupt archives and wheels are detected and repaired
automatically without these switches.

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

- the full runtime fingerprint and active local/Drive paths;
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

With the recommended settings, FLUX and the small NF4 SmolVLM analyzer are loaded
and warmed before the URL appears. Z-Image and ERNIE remain lazy because keeping
all image generators resident would exceed the practical free-T4 memory budget.
If a particular free runtime cannot retain FLUX and SmolVLM together, set
`PRELOAD_AVATAR_VISION=False`; Avatar Studio will then load it on first analysis.

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

New avatars can start with **Guided setup** or **Custom avatar**. Existing avatars
continue to use their saved workflow. The searchable portrait library switches
between identities. A compact identity bar keeps the saved face/body references
and confirmation state visible. The last selected avatar is restored when the app reconnects.

Use **All avatars** to return to the full library. Search, filter by readiness,
sort by name, or rename an avatar without changing its saved folder or history.
The guide keeps its primary action in a floating footer, including on mobile.
The generation workspace puts the image beside its prompt and image controls.
Chat history appears above the composer; generation settings are collapsed until
opened. Selecting a gallery image opens its contextual validation/regeneration inspector.

Face and body generation use automatic reference instructions, shared by guided
and custom creation. The face brief specifies a single clear headshot, framing,
expression, lighting and background. The body brief specifies full head-to-toe
framing and a neutral pose; a confirmed face is attached as its identity reference.
Your appearance details are appended. **Generation brief** displays that exact
assembled FLUX prompt and updates while you edit. This is an image-generation
prompt prefix, not a separate chat-model system role, and cannot guarantee that
every generated image obeys the framing perfectly. Uploads bypass generation.

Confirmation requires an explicit usable-reference verdict from SmolVLM. An
unusable or uncertain result shows an error and leaves any previously confirmed
identity untouched. Prompt summaries prioritize appearance and proportions rather
than visibility checks, so hair, eyes and facial features reach avatar generation.

Vision analysis follows `imagetexttotext.ipynb`: the same NF4/FP16 model is asked
short questions sequentially, and Python assembles the saved specifications.
SmolVLM no longer has to emit twenty answers in one JSON response. Quality checks
also use short questions; an uncertain mandatory answer cannot count as a pass.
This uses more individual inference calls, not additional model copies.

An empty reply gets one shorter-prompt retry without the KV cache. Persistent
failure is reported rather than replaced with guessed details. Warmup now checks
decoded text too. Recovery/failure reports are saved as `results/_debug/vision_*.json`
with package versions, token counts, raw special-token output and retry details;
the diagnostic retry also checks numerical scores. Share the newest report and
runtime error log when reporting a vision failure. No model downloads or install
repair are required merely to get these code changes.
The legacy `FFS_AVATAR_ANALYZER_STRICT=0` flag no longer substitutes mock identity
specifications after a production failure; mocks are limited to development mode.

Guided setup uses FLUX and the local SmolVLM analyzer without requiring Gemini or
Tavily keys:

1. Name the avatar and choose a starting style. The suggested face, body, and two
   scene prompts are editable.
2. Click **Generate face**, or upload your own headshot, then confirm it with
   **Use this face & next**. The body draft uses the confirmed face as a reference.
3. Review or replace the body, then click **Use this body & create 2 images**.
4. Two starter images are generated sequentially from the locked identity,
   checked, and saved to the avatar's gallery. A failed quality check permits up
   to two further attempts; remaining mismatches are visibly marked **Needs review**.
5. Open the avatar console or gallery. The regular reference-search Auto Gallery
   still requires Gemini and Tavily credentials.

Guide options, drafts, confirmed identities, progress, and each finished image
are saved in the avatar workspace. Reopen an unfinished guided avatar to resume;
successful starter images are not regenerated. **Use custom controls** leaves the
guide and retains the confirmed references and drafts. Production generation
errors are reported and never replaced with mock images.

**Back** returns to the previous identity review without deleting saved work.
Replacing a confirmed face requires confirming the body again; earlier starter
images stay on disk but are marked as belonging to the previous identity.
Changing one starter scene regenerates that scene rather than both. Custom
face/body drafts and their prompts also survive reopening the avatar.

The console saves uploaded extra references with each user message. **Stop** is
available during guided, console, and gallery generation. A running GPU operation
may finish before cancellation takes effect; already saved work is retained.
Gallery selection, validation details, and regeneration use the same visible
100-image window so a selected thumbnail always identifies the corresponding file.
The reference section displays its attachment count. Send waits for the image
component's upload acknowledgement, preventing a just-selected reference from
being silently omitted. The conversation scrolls to each new image after it loads.

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
   control. Console images also appear in the avatar gallery, while retaining
   their original chat file and conversation entry.

The gallery prompt writer receives the selected reference image as well as its
caption. To reject a result, optionally fill **What should change?** in the image
inspector before regenerating. User rejection overrides an earlier automatic
pass, is supplied to the prompt repair, and is saved alongside the previous
assessment and image attempt, even if the repair API fails.

### Texture Reconstruction

Saved console images now appear in the avatar gallery, including older results
indexed from saved chat history. User-uploaded references are not added as results.
The index points at existing files without copying them.

Use **Enhance realism** below a console image or in a selected gallery image's
inspector. The dialog shows the original, output dimensions, reconstruction
strength, detail steps, the automatic realism prompt, and the enhanced result.
Choose 1x or 2x; both are capped at a 2048-pixel longest edge, with aspect preserved.
Defaults are **Full photo**, **1x**, strength **1.00**, and **four** distilled FLUX
steps. This performs one FLUX edit and saves one output image. The before/after
views use matching dimensions; the original file is never resized or overwritten.

This is generative texture reconstruction, not conventional sharpening or recovery
of genuine camera RAW information. Full photo edits the whole image at a working
size of at most 1024 pixels per edge, padded to multiples of 64. A larger export
in this mode is resized after generation: it is not native 2K reconstruction.
The full generated correction is retained, including material and lighting changes.
It bypasses Smart Mask and downloads no extra model weights.

Optional **HD tiles** performs that full-photo edit followed by sequential,
overlapping detail edits at the output resolution. Tiles are at most 768 pixels
per edge, with up to nine tiles (ten FLUX passes including the full-photo edit).
Complete generated tiles are feather-blended. The dialog states the pass count
before starting and reports each pass; multiple sampling bars still produce only
one saved image. Four steps means four sampler steps per pass, not four outputs.

The original is never overwritten. A new **Enhanced / Unchecked** gallery entry
records the source image, prompt, seed, strength, steps, method, native working
size, output size, pass count, and pixel-change diagnostics. An almost unchanged
result is marked **No change**, not reported as a verified enhancement. Pixel
change is not a realism, identity, or quality score. Compare
faces, clothing, text, and tiny details yourself: generative edits can invent
details or change identity, and stronger reconstruction increases this risk.
**Stop enhancement** waits for the current FLUX pass and discards incomplete output.
An interrupted enhancement can be started again from its preserved source.

To update an existing Colab runtime, stop the running launch cell, set
`UPDATE_APP = True`, and run the cell once. Existing Drive model files are reused.
For the first comparison use Full photo, 1x, strength 1.00, and four steps. Inspect
the result at matching size before trying the slower HD tiles mode. This feature
does not guarantee photorealism or preservation of every detail.

### Reference Review And Batch Recovery

The Gallery workspace shows live Finding, Cleaning, Prompting, Generating, and
Checking stages. Open **References** to inspect all findings, approved images,
rejections with reasons, and images awaiting checks. Findings without a downloaded
file still appear in the source table. **Results** contains generated images.

Local file checks reject unreadable, undersized, blank, oversized, and exact
duplicate images. Gemini then checks clarity, adult-subject confidence, and useful
outfit/pose visibility. **Require visible face** is optional and off by default:
a hidden face can still provide a useful clothing or pose reference. These checks
are estimates, so inspect their findings and generated results yourself.

Approved references are committed to a batch, and every prompt is saved before
FLUX starts generating. Each saved image is checked with SmolVLM. Prompt repairs
through Gemini happen only after all committed references have their first image.
Repair attempts remain bounded by `AVATAR_GALLERY_RETRIES`.

Temporary Gemini/Tavily connection errors, HTTP 408, 429, and 5xx responses receive
up to three retries with backoff and jitter. Authentication/configuration errors
are not retried. If retries fail, use **Resume batch**: saved prompts, completed
images, pending visual checks, and remaining references are reused. A resumed
batch retains its original theme, quantity, aspect, and face requirement. **Stop**
preserves the batch; an in-progress inference/API request may finish first.

The avatar's `gallery/references/catalog.json` retains findings and rejection
reasons; `gallery/batches/` retains checkpoints, with `gallery/active_batch.json`
pointing to the current batch. Existing `search_*.json` reports and reference files
are imported without deletion. Unused matching references are consumed before a
new search. References associated with existing generated images are not repeated.
Runs made before batch checkpoints existed can recover their saved references and
images, but not prompts or progress that the old code never saved.

If an empty search is paused after exhausting its budget, **New search** closes
that search while retaining its findings. A committed unfinished batch must be
resumed first. A changed confirmed identity cannot silently resume an older batch.

Auto Gallery reads `GEMINI_API_KEY` and `TAVILY_API_KEY` from the uploaded keys
txt, the single Colab cell form, Colab Secrets (the key icon in Colab's left
sidebar), or saved private Drive settings. The committed notebook keeps those
fields blank. Do not place API keys in committed notebooks or source files. Local
development uses mock search, generation, and validation and never loads an AI
model.

The Colab form also exposes the practical knobs from the reference-finder and
SmolVLM notebooks:

- `GEMINI_MODEL`: optional override; blank auto-picks an available Gemini Flash model.
- `AVATAR_REFERENCE_DOMAINS`: comma-separated Tavily domain filter, defaulting to `instagram.com`.
- `AVATAR_REFERENCE_TIME_RANGE`: optional recency filter such as `month`.
- `AVATAR_SEARCH_ROUNDS`: 1-5 Tavily/Gemini planning rounds; more rounds cost more but improve fill rate.
- `AVATAR_GALLERY_RETRIES`: 0-3 prompt repair attempts after a failed validation.
- `AVATAR_MAX_CANDIDATE_DOWNLOADS`: total candidate download cap per gallery batch.
- `AVATAR_VISION_MAX_EDGE` and `AVATAR_VISION_MAX_TOKENS`: SmolVLM memory/detail controls.

FLUX and SmolVLM are preloaded and warmed by default. Z-Image and ERNIE remain
lazy and replace FLUX through the one-image-model-at-a-time manager when selected.
The Avatar analyzer is much smaller and stays resident for validation unless its
preload switch is disabled.

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
startup_restore.py
app.py
model_manager.py
workspace.py
engine_z_image.py
engine_flux_klein_4b.py
engine_ernie_image_turbo.py
```

## Known Limitations

- A newly built prepared-runtime system still requires one complete first-run and
  second-run Colab T4 acceptance test before startup-time claims can be measured.
- First run can still take a long time because the required checkpoints are large.
- Warmup removes lazy pipeline initialization but cannot make a new Colab VM keep
  GPU memory from a previous VM; FLUX must be loaded into the new T4 each session.
- Hugging Face gated or rate-limited files may require the user to be logged in or provide a token in Colab.
- Gradio temporary public links are session-based and expire when the Colab runtime stops.

## Colab Test Checklist

1. Start a fresh Colab runtime with GPU enabled.
2. Open `FreeFakeStudio.ipynb`.
3. Set `WORKSPACE_DIR` to your Drive folder.
4. Keep `UPDATE_APP=False` and `REPAIR_INSTALL=False` for normal testing.
5. Run the single cell.
6. Confirm `Runtime fingerprint`, `Prepared environment`, `Local SSD runtime`,
   FLUX preload, warmup, and local health all succeed.
7. Confirm the `OPEN FREEFAKESTUDIO` link appears only after those messages.
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
18. Disconnect the runtime, reconnect to a fresh T4, run with update/repair/force
    switches off, and confirm `Prepared environment=restored`, source bundles are
    restored, downloads are skipped, warmup completes, and history/avatars return.
## Automatic Enhancement

Enhance now analyzes the source with the existing local vision model and builds
a material-aware, preservation-first prompt automatically. The default strength
is 0.65. It no longer universally requests pores, follicles and creases. Source
observations are cached; the viewer retains the actual prompt and texture review.
Unexpected new skin markings or heavy texture require review. These checks are
heuristics, not a guarantee of photographic quality. Start from the original image,
not a previous over-textured enhancement. See [the workflow](INFLUENCER_STUDIO.md).
