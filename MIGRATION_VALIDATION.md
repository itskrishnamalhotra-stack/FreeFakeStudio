# Migration Validation

## Completed Locally

- `python -m unittest discover`: 301 tests passed. Includes legacy engine/memory
  contracts and new HTTP, authentication, traversal, upload, idempotency,
  checkpoint, recovery, guided avatar, console, gallery and reconstruction tests.
- Release regressions cover anchor-change invalidation without losing snapshots,
  historical image policy pinning, benchmark correction, and empty Drive files.
  Explicit reference views, matched lossless detail crops, provisional versus
  held-out face calibration, persona-aware offline caption styles, and opt-in
  runtime scheduling are covered without live model or publishing calls.
- App-wide memory management defaults off, persists across app recreation,
  ignores legacy per-avatar flags, and gates both image/vision forced unloads,
  including reference analysis. Strict settings validation and busy-job locking
  are covered. Explicit unload and out-of-memory recovery remain available.
- `test_web_memory_browser.js` passes for the shared main/avatar switch, removal
  of the old avatar checkbox, saved On/Off state after reload, and 1366px, 390px
  and 320px layouts. Background refresh cannot overwrite a pending/saved toggle.
- Source-aware enhancement tests cover material-specific prompt construction,
  conservative skin/blur preservation, enum-only model observations, mock/real
  cache isolation, image-content invalidation, cancellation, explicit analysis
  fallback, and recovery without repeated analysis or reconstruction. New skin
  markings/roughness warnings cannot be cleared by an identity-only recheck.
- Targeted-analysis follow-up covers independent visibility probes, partial
  rather than false-complete status, verbose yes/no parsing, bounded retries,
  face-region coordinate bounds, old-cache rejection, and no Not Applicable
  bypass for uncertain portraits. Synthetic image tests cover brightness-only
  correction, explicit opt-out, retained generated detail, and tone-only review.
- `test_web_enhancement_analysis_browser.js` passes for automatic prompt planning,
  default 0.65 strength, one-pass/one-image output, saved applied prompt and
  analysis after reload, and 1366px/390px/320px viewer layouts. Images/vision are
  mocked: this does not establish real GPU output quality.
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
- Enhancement follow-up: visible actions under generated main/Avatar Console
  images and in the full-screen viewer. Editable reconstruction prompt with
  default reset, blank-prompt rejection, and persisted job/variant prompt.
  Tests verify custom prompts reach both full-photo and tiled passes, source
  images survive, and results remain in their originating conversation.
- Browser enhancement checks at 1440x960 and 390x844 verify the prompt editor,
  reset, retained processing/size/strength controls, one output for one-pass
  mode, and the viewer's Compare toggle, including after browser reload.
  All enhancement output in these checks is mocked, not GPU quality validation.
- Gallery sections: Generations excludes enhancements, Enhancements retains
  original links, Resources combines identity/upload/search records, and Needs
  Review lists unchecked/no-change/failed-check outputs. Tests cover server-side
  category pagination, pending URL-only findings, checks/rejection reasons,
  used references, archived-chat resources, isolation and manual approval.
- `test_web_gallery_browser.js`: category counts/isolation, comparison, retained
  prompt editor, reference fullscreen/filtering, approval, keyboard tabs and
  responsive layout at 1440x960, 390x844 and 320x740 passed in mock preview.
  Matched center-detail comparison crops also load at 1366px, 390px and 320px.
- Routing uses React Router 7.18.4. The URL selects the main conversation, avatar
  workspace tab, avatar-scoped conversation and gallery category. Server tests
  cover GET/HEAD deep-link fallback, reserved API/asset 404s and authentication.
- `test_web_routes_browser.js`: root canonicalization, new chat routes,
  Back/Forward with separate drafts, saved history on reload, library/category/
  body deep links, access-fragment exchange, missing records, cross-avatar chat
  rejection and mobile layout passed in mock mode. The guided end-to-end
  workflow also passed again after adding routing.
- Lazy conversations: Create/New chat perform no creation POST and reuse one
  unsaved composer per scope. The accepted first prompt creates and names the
  chat; retries reuse the original chat/job. Tests cover invalid first requests,
  failed-generation prompt preservation, avatar isolation and hidden legacy
  empty placeholders without file deletion.
- `test_web_drafts_browser.js`: repeated blank Create/New chat, zero eager chat
  POSTs, retained text/attachments, exactly one saved chat and output per first
  prompt, blank reload, avatar parity and mobile layout passed in mock mode.
- Image-only influencer integration: immutable/restorable identity versions,
  approved view packs, contact-sheet/specification export, CPU face calibration,
  per-attribute quality policy, pinned-policy batches, and resumable 12-case
  benchmarks with human labels. Inconclusive identity checks require review.
- Console and enhancement validation retries reuse generated files. Tests cover
  interrupted benchmark recovery, metadata-preserving console reindexing,
  calibration image restoration, optional caption API use, editable/exportable
  drafts, approval invalidation, mock publishing guards and uncertain network
  outcomes without blind duplicate publishing.
- `test_web_influencer_browser.js`: reference generation/approval, pack download,
  policy persistence, 12-case mock benchmark, human labels, API-free caption,
  content draft approval/export, and 1366px/390px/320px layouts passed. No real
  face, vision, image, voice or video model was run on the laptop.

Browser scripts are replayable through Playwright CLI `run-code --filename`:
`test_web_browser.js`, `test_web_guide_browser.js` and
`test_web_gallery_browser.js`, `test_web_routes_browser.js` and
`test_web_drafts_browser.js`, `test_web_influencer_browser.js` and
`test_web_memory_browser.js` and `test_web_enhancement_analysis_browser.js`. The first,
gallery, routing and draft scripts use a local
fixture at `results/qa/supplied-preview.png` and a prepared avatar named
`Enhancement QA` with a populated gallery. These private preview fixtures are
not committed. Run `python prepare_browser_fixtures.py` to create repeatable
synthetic fixtures on a fresh clone. Existing local fixture images are preserved.
The guided script creates its own mock avatar. Run against a
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

The original reference source folder and the primary checkout's paused work
were not edited. The image-only identity/quality/content modules have now been
ported into this worktree and integrated with the native API/UI. Publishing
against a real Instagram account has not been authorized or executed in testing.
`app.py` remains the legacy Gradio rollback entry point. Notebook launches use
the new `web_server.py` entry point by default.
