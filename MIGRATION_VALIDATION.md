# Migration Validation

## Completed Locally

- `python -m unittest discover`: 219 tests passed. Includes legacy engine/memory
  contracts and new HTTP, authentication, traversal, upload, idempotency,
  checkpoint, recovery, guided avatar, console, gallery and reconstruction tests.
- Drive protocol tests use mocked HTTP responses: pagination, expired upload
  sessions, resumable status probes, checksum rejection, create-ID reuse,
  conflict preservation, safe auth errors and cached directory enumeration.
- Kaggle bootstrap is exercised with mocked GPU, Secrets, Drive and launch
  boundaries. It does not invoke a GPU, download a model or run cloud setup.
- TypeScript type checking and production Vite build passed. Prebuilt assets
  are included in Git, so notebook startup requires no Node build.
- `npm audit`: zero vulnerabilities after updating Vite to 6.4.5.
- Browser QA at 1440x960, 390x844 and 320x740: prompt clearing, visible chat
  results, append/remove/reorder references, canvas state, slider fill, mask
  painting/export, model-specific controls, footer placement, themes, gallery
  modal geometry, Escape and scroll-position preservation.
- Browser guided workflow: confirmed mock face/body, two starter images,
  console output in gallery, one-pass reconstruction creating one variant,
  browser reload persistence and a two-image reference-search/gallery batch.

Browser scripts are replayable through Playwright CLI `run-code --filename`:
`test_web_browser.js` and `test_web_guide_browser.js`. The first uses a local
fixture at `results/qa/supplied-preview.png` and a prepared avatar named
`Enhancement QA` with a populated gallery. These private preview fixtures are
not committed. The guided script creates its own mock avatar. Run against a
local `web_server.py` mock preview, never an inference-enabled cloud session.

## Not Verified Without Account Authorization

- Actual Kaggle GPU provisioning, internet access and ngrok connectivity for
  the user's account.
- Real Google OAuth consent/refresh, existing workspace transfer, upload speed
  and cross-provider restore against the user's Drive.
- Actual GPU generation quality, peak RAM/VRAM, vision preload co-residency and
  a 3-4 hour cloud endurance run.

The implementation and mocked integration tests do not prove those external
conditions. Follow the small-batch acceptance test in `CLOUD_SETUP.md` before
running a large gallery. No actual image/vision models were run on the laptop.

## Retained Boundaries

The original reference source folder and the primary checkout's unrelated
influencer/Instagram work were not edited or published by this migration.
`app.py` remains the legacy Gradio rollback entry point. Notebook launches use
the new `web_server.py` entry point by default.
