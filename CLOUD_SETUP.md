# Colab and Kaggle Setup

The same `FreeFakeStudio.ipynb` contains one launch cell for both providers. The
default interface is React/TypeScript, served by FastAPI on one port. Gradio is
not involved in the new interface, uploads, image viewer, or job transport.
The existing ComfyUI model engines, identity analysis and gallery pipeline remain
the backend. No model runs on your laptop.

## Before You Start

Open the notebook in [Colab](https://colab.research.google.com/github/itskrishnamalhotra-stack/FreeFakeStudio/blob/main/FreeFakeStudio.ipynb)
or [import it into Kaggle](https://www.kaggle.com/kernels/welcome?src=https%3A%2F%2Fraw.githubusercontent.com%2Fitskrishnamalhotra-stack%2FFreeFakeStudio%2Fmain%2FFreeFakeStudio.ipynb).
If Kaggle does not prefill the import, choose File > Import Notebook > URL and
use `https://raw.githubusercontent.com/itskrishnamalhotra-stack/FreeFakeStudio/main/FreeFakeStudio.ipynb`.

Google's [official FAQ](https://research.google.com/colaboratory/faq.html) restricts
primarily web-UI interaction on free Colab managed runtimes. Account quotas and
runtime termination are provider-controlled. Use a permitted plan/runtime;
the code does not bypass policies, reconnect quotas or keep sessions alive.

- Keep the existing `My Drive/FreeFakeStudio` folder. Do not delete its models,
  avatars, results, references, or config files.
- Stop the other notebook before switching providers. This is a single-user,
  single-writer workspace, not a multi-machine database.
- Keep notebooks, keys datasets, output cells and launch links private. The link
  contains a per-session access token. Sharing it grants access to your studio.
- `MODEL_SET=flux` downloads/restores only FLUX components; existing optional
  models are not deleted. Select `all` to install Z-Image and ERNIE too. They
  remain text-to-image-only; Avatar Studio always uses FLUX.
- The distilled FLUX.2 Klein 4B default is **4 steps**, not 20. Two T4s do not
  become one 30 GB GPU: this app uses one device, not distributed inference.

## Colab

1. Open the [current notebook in Colab](https://colab.research.google.com/github/itskrishnamalhotra-stack/FreeFakeStudio/blob/main/FreeFakeStudio.ipynb).
2. Select a GPU runtime. Keep the existing `WORKSPACE_DIR`.
3. For this migration, set `UPDATE_APP=True`, `REPAIR_INSTALL=False`.
4. If keys already exist in Drive, leave `UPLOAD_KEYS_TXT=False` and
   `KEYS_TXT_PATH` blank. Alternatively enter the cloud Drive path to the file,
   or upload it once. A Windows `C:\...` path cannot be used in a cloud notebook.
5. Keep FLUX preload, vision preload, SSD staging, warmup and the ready gate on.
6. Run the one cell and authorize Drive. It updates source, restores dependencies
   and model files, warms FLUX and attempts vision preload, then checks HTTP
   health. Only after that does it print the private interface link.
7. Open the **entire** link including `#access=...`. The browser exchanges this
   token for an HTTP-only cookie and removes it from the address bar.
8. For later launches use `UPDATE_APP=False` unless pulling another update.

Colab keeps its native Drive login. No OAuth client or refresh token is needed
for this path. `PUBLIC_ROUTE=Colab proxy` is supported; ngrok is also supported.
This frontend uses relative HTTPS requests, not Gradio's queue URLs.

## One-Time Drive Connection for Kaggle

Kaggle does not implement `google.colab.drive.mount`. The app uses Google's
official Drive API with your own OAuth client instead. It accesses the **same
Google account and existing folder**, without making that folder public.

1. In [Google Cloud Console](https://console.cloud.google.com/), create/select a
   project and enable **Google Drive API**.
2. Configure **Google Auth Platform / OAuth consent** for your personal app. If
   the app is External and in Testing, add your Google account as a test user.
3. Create an **OAuth client ID**, application type **Web application**. Add this
   exact authorized redirect URI:
   `https://developers.google.com/oauthplayground`.
4. Open [Google OAuth Playground](https://developers.google.com/oauthplayground/).
   In its gear/settings panel select **Use your own OAuth credentials**, enter
   your client ID and secret, select **Server-side**, **Offline**, and consent.
5. In Step 1 authorize `https://www.googleapis.com/auth/drive` using the same
   Google account as Colab. This broad scope is needed to read and update the
   pre-existing folder; `drive.file` alone does not grant access to arbitrary
   existing files. Only use a project/client you control.
6. In Step 2 select **Exchange authorization code for tokens**. Retain the
   **refresh token**, not the short-lived access token.
7. Open your existing `FreeFakeStudio` folder in Drive. Copy the folder ID from
   the end of `https://drive.google.com/drive/folders/FOLDER_ID`.
8. Add these private values to your existing keys text file:

   ```text
   GOOGLE_DRIVE_WORKSPACE_FOLDER_ID=YOUR_EXISTING_FOLDER_ID
   GOOGLE_DRIVE_CLIENT_ID=YOUR_CLIENT_ID
   GOOGLE_DRIVE_CLIENT_SECRET=YOUR_CLIENT_SECRET
   GOOGLE_DRIVE_REFRESH_TOKEN=YOUR_REFRESH_TOKEN
   NGROK_AUTH_TOKEN=YOUR_NGROK_TOKEN
   GEMINI_API_KEY=YOUR_GEMINI_KEY
   TAVILY_API_KEY=YOUR_TAVILY_KEY
   HUGGINGFACE_TOKEN=YOUR_HF_TOKEN_IF_NEEDED
   ```

Google requires account authorization by you; the application cannot grant its
own permissions. Playground's shared client revokes tokens after 24 hours: use
your own client as above. External apps in Testing usually receive refresh
tokens expiring after seven days for Drive scopes. Renew when required, or
configure the consent app appropriately under Google's rules. Tokens can also
be revoked by account/security changes. Never promise permanent login.

## Kaggle Launch

1. Sign into Kaggle and complete any account verification required for GPUs.
2. Create a **private** notebook, import the same `FreeFakeStudio.ipynb`, enable
   **Internet**, and choose **GPU T4 x2** when available. P100 availability and
   performance vary; T4 is the primary target for this setup.
3. Supply keys using either method:
   - **Add-ons > Secrets**: enable notebook access for `GEMINI_API_KEY`,
     `TAVILY_API_KEY`, `HF_TOKEN`, `FFS_NGROK_AUTHTOKEN`, and the four
     `GOOGLE_DRIVE_...` names above.
   - A **private** Kaggle keys dataset: attach it, then set `KEYS_TXT_PATH` to
     its actual `/kaggle/input/.../FreeFakeStudio.keys.txt` path. Never publish
     that dataset or notebook. `UPLOAD_KEYS_TXT` must be False on Kaggle.
4. Edit cell values directly: Kaggle does not render Colab's `@param` controls.
   Set `MODEL_SET="flux"`, `UPDATE_APP=True` for the migration, and leave
   `REPAIR_INSTALL=False`. Keep preload and warmup on. `WORKSPACE_DIR` applies
   only to Colab; Kaggle uses `/tmp/freefakestudio_workspace`.
5. Run the cell. It detects Kaggle, verifies credentials, restores existing
   results and FLUX files from Drive, installs/validates ComfyUI, starts the
   backend, preloads FLUX and warms it before printing the private ngrok link.
6. Open that link. Existing avatars, face/body specs, history and references
   appear. Old Colab absolute paths are relocated with backups of changed JSON.
7. Use **Activity** to resume an interrupted job. Use **Gallery > Pipeline**
   to inspect findings, approvals, rejected references and the committed queue.
   **Continue batch** consumes saved work before searching again. A new search
   is blocked while a committed queue is unfinished.
8. When finished, stop active jobs and wait until Activity reports completion
   without a Drive sync error. Then stop the notebook session to release quota.

Kaggle requires ngrok credentials for this implementation. Its Colab proxy route
is unavailable. The connector does not bypass platform policies or quotas.

## Persistence and Recovery

- Models persist in the existing Drive `ComfyUI/models` tree. Every new Kaggle
  machine still needs a Drive-to-local-disk transfer; persistent storage is not
  persistent RAM. Files with matching sizes/checksums are reused within a session.
- Results, reference assets, job checkpoints, avatars, chat and config sync to
  Drive. Binary assets are uploaded before JSON records that refer to them.
  API failures are retried with limits; unsuccessful sync leaves local files
  intact and reports an error. Completed local work is not called durable until
  the sync succeeds. The upload protocol supports resumable chunks and checksum
  verification, not public sharing or a fragile FUSE mount.
- Warm runtime cache archives also persist. ComfyUI source can be cloned again;
  Kaggle does not rely on a Colab binary environment archive being compatible.
- The laptop preview runs mock engines and mock vision only. Real generation,
  detailed identity fidelity and cloud memory peaks need a GPU runtime test.
- A runtime termination before a checkpoint sync finishes can lose the latest
  unsynced operation. Earlier successfully synced checkpoints survive. There is
  no honest way to guarantee persistence after an abrupt machine kill.
- Do not run Colab and Kaggle against this workspace simultaneously. Checksums
  detect conflicting overwrites, but this is not a distributed locking service.
- A stopped job resumes from saved images/checkpoints, not necessarily in the
  middle of a GPU sampling pass. A browser refresh never resubmits generation.
- Authentication expires when the server restarts. Reopen the new notebook link.
- If Google OAuth fails, renew the refresh token. If ngrok fails, verify its
  token/account. If a key or private link is exposed, revoke/rotate it.

## UI and Acceptance Test

1. **Create**: submit one prompt. Expect a user turn followed by its image turn.
   Try multiple outputs; the footer stays docked and earlier images remain.
2. **References**: add images one at a time (up to four), remove one, move it
   left/right and select a canvas. The highlighted canvas and backend order
   must match. Text-only models have no image-input control.
3. **Settings**: test steps, aspect, seed, reference mode, encoder and masks.
   Slider fill and number inputs share the same value. Paint, erase, undo and
   export a mask. Confirm the automatic-mask preview via Activity.
4. **Avatar Studio**: create Guided or Custom. Generate or upload a face, review,
   confirm analysis, then do the body. Guided mode creates two starter images.
   Prompts automatically include the existing headshot/body instructions.
5. **Console**: send a photograph prompt with zero to two extra references.
   Confirmed face/body always occupy the first two slots. The output appears
   both in chat and in that avatar's gallery.
6. **Gallery**: create a two-image batch first. Pipeline shows finding, cleaning,
   committing, prompt planning, generation and identity checking. Test stop and
   resume; completed images should not be regenerated. Expand to 20 only after
   a small real GPU test works.
7. Click a gallery image while scrolled down. The native modal opens in the
   current viewport, supports previous/next and Escape, and returns to the same
   scroll position. Download, reconstruct, compare the source, or flag a
   correctable recipe for regeneration with feedback.
8. **Reconstruction**: one-pass mode is one FLUX edit and one saved variant.
   HD tiled mode intentionally uses several declared passes for one output.
   2x dimensions alone do not prove extra detail; diagnostics mark near-no-op
   outputs. Generative reconstruction may alter details and is not a guarantee
   of factual recovery of missing information.
9. Reload the browser, then restart the cloud runtime and rerun the cell.
   Verify that saved avatars, history, gallery and references return. Check
   Activity and Diagnostics for interrupted jobs or sync errors.
10. Test a narrow mobile viewport, theme toggle and the settings drawer. No
    model needs to run locally for these UI checks.

## Gallery Sections

Inside an avatar, open Gallery and choose Generations, Enhancements, Resources
or Needs Review. Search and pagination apply to the selected category.
Enhancements are separate variants: Compare opens the preserved source beside
the result. Processing mode, output size, strength and editable reconstruction
prompt remain available in Enhance.

Resources includes face/body anchors, uploaded chat references and saved search
findings. Filter All, Pending, Approved, Rejected or Used, and filter by type.
Used is independent of approval; an uploaded reference is not automatically
declared vision-approved. Pending findings without a local image show a
placeholder, not an output thumbnail. Opening or using a resource never removes
it. Archived chat resources remain available. Pipeline retains batch diagnostics
and continue/resume controls.

Needs Review overlaps the output categories. Open an image to inspect it, then
Approve or Correct when a regeneration recipe exists. Manual approval records
the decision without deleting machine validation, the original, or the variant.

## App Link

### Page Routes

- `/chats/new`: the unsaved Image Studio composer (also the root-page default).
- `/chats/<chat-id>`: a saved Image Studio conversation.
- `/avatars`: the avatar tile library.
- `/avatars/<avatar-id>/face`, `/body`, `/guide`: the respective setup step.
- `/avatars/<avatar-id>/console/<chat-id>`: that avatar's conversation.
- `/avatars/<avatar-id>/console/new`: that avatar's unsaved composer.
- `/avatars/<avatar-id>/gallery/generations`, `/enhancements`, `/resources`,
  `/needs-review`: the respective gallery section under the same gallery prefix.

Reload and bookmarks open the same saved page. Browser Back/Forward changes
pages and preserves unsent chat drafts within the current browser session;
unsent drafts are not durable across a full reload. Missing/archived chats and
invalid avatar IDs show Page unavailable, never another avatar's conversation.
API and asset errors are not replaced with frontend HTML.

Create and New chat reuse the unsaved composer, including its current text and
attachments, without creating a sidebar entry. Only an accepted first prompt
creates a saved conversation and assigns its permanent URL. Retrying that same
request reuses its chat and job. Blank reloads do not create records. Existing
empty placeholder files are retained for recovery but hidden from chat lists.

### Private Launch URL

The notebook starts the single-port web server, verifies readiness and health,
then prints `OPEN FREEFAKESTUDIO` followed by the private HTTPS link. Kaggle
uses ngrok with the supplied auth token. Colab uses the selected ngrok route or
its native HTTPS port proxy. No Gradio share link is required.

The printed `#access=...` fragment is exchanged for a private session cookie and
removed from the address bar. Treat the full printed link as a credential. The
link only works while the cloud server and its tunnel/proxy are running; after
a restart, use the newly printed link. A localhost preview is not this cloud URL.

## Timing and Hosting Limits

Allow roughly 5-20+ minutes for a new cloud machine's setup, transfer and warmup;
large existing models still need transfer to Kaggle. Network speed, dependency
changes and GPU availability can make this longer. Same-machine reruns should
reuse local files; changing providers is not a warm-memory resume. Generation
timing must be measured on the assigned GPU; there is no fixed guarantee.

Kaggle documents sessions up to 12 hours and a quota generally around 30 GPU
hours/week, subject to availability. That can fit a 3-4 hour target, not guarantee
it. Colab free GPU limits remain variable and its FAQ warns about web-UI-driven
content generation. Switching frontend frameworks does not change provider
rules. No keepalive automation, multi-account rotation or quota bypass is used.
Check terms for your intended personal/commercial usage; this notebook is not
a permanent hosting service.

## Verified Documentation

- [Kaggle notebook resources and session limits](https://www.kaggle.com/docs/notebooks)
- [Kaggle GPU quotas](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle acceptable use](https://www.kaggle.com/aup)
- [Kaggle terms](https://www.kaggle.com/terms)
- [Colab limitations](https://research.google.com/colaboratory/faq.html)
- [Google OAuth and token expiration](https://developers.google.com/identity/protocols/oauth2/)
- [Google OAuth Playground](https://developers.google.com/oauthplayground/)
- [Drive resumable upload protocol](https://developers.google.com/workspace/drive/api/guides/manage-uploads)
- [Vite production builds](https://vite.dev/guide/build.html)
- [FastAPI static files](https://fastapi.tiangolo.com/tutorial/static-files/)

## Local Development (No Real Models)

Install the web dependencies with `pip install -r requirements-web.txt`, then
run `python web_server.py`. Local provider detection forces mock inference even
without an explicit development flag. Do not spoof cloud-provider environment
variables on the laptop. The committed frontend bundle needs no Node at runtime.

To edit the frontend: use Node 20.19+ or a supported newer LTS, run `npm ci` and
`npm run build` inside `frontend`. Commit source, lockfile and `frontend/dist`.
Tests: `python -m unittest discover`. Legacy `python app.py` remains a separate
Gradio mock preview/rollback path; it is not the default notebook entry point.
