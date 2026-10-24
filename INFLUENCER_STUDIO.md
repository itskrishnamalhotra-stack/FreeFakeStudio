# Image-only Influencer Studio

This workflow uses the existing FLUX.2 Klein 4B distilled model (four steps),
SmolVLM image analysis and small CPU face-comparison models. There is no LoRA
training, video generation, voice/TTS service or mandatory paid generation API.

## Start and Keep Existing Work

1. Start the existing single-cell notebook on a Colab T4, with `MODEL_SET=flux`.
   Keep the same Drive workspace and private keys file. Update the application
   when this revision has been published. Existing avatars, chats and images are
   retained. Do not run two runtimes writing the same workspace.
2. Setup caches YuNet and SFace under `models/avatar_identity`, approximately
   39 MB total, with a pinned upstream revision and SHA-256 verification. They
   run on CPU, not on the GPU. Already verified files are reused.
3. Open the private HTTPS launch link and select an existing confirmed avatar,
   or use Guided setup to confirm its face/body and create two starter images.

## Identity Tab

1. Choose a reference view: front, three-quarter, profile, smile, body-front or
   body-side. Generate a candidate using FLUX, or upload a photograph.
2. Inspect it full-screen. Approve only candidates that genuinely preserve the
   avatar's identity. Generated candidates are never automatically approved.
3. Approved references create a new immutable identity version. Old versions
   retain their own images/specifications even when a candidate is removed.
   Approving a candidate replaces the previously active candidate for that view;
   other views remain approved. Old versions still retain their previous choices.
4. Generation selects a matching face/body view from that version. It still
   uses at most two identity references, leaving two additional FLUX slots.
   Console attachments have individual Outfit, Pose, Scene or Other person roles.
   Console settings also offer explicit Face reference view / Body reference
   view controls. Auto mode handles common negations and does not mistake a
   social-profile picture request for a side-profile pose.
5. Export pack downloads references, face/body specifications, version signature
   and a contact sheet. Restore version restores the corresponding confirmed
   anchors and approved pack; subsequent images use that version.

## Quality Tab

1. Upload at least four distinct, human-verified photographs of the same avatar
   and four of different people. Each must contain exactly one clear face.
   Vary angle/expression, but avoid tiny, occluded or heavily stylized faces.
2. Calibrate identity measures cosine separation against the approved identity
   references. Overlapping positive/negative examples are rejected. A similarity
   score is not an identity probability and cannot guarantee consistency.
   The last two samples from each group are held out from threshold fitting.
   Automatic acceptance requires them to pass with a separation margin. Two
   samples per group can be saved as provisional, but cannot approve identity.
3. Save per-attribute Required, Important or Flexible policies. Required and
   Important visible attributes are checked with the local vision model.
   Flexible attributes are not repeatedly queried. Uncertain baseline attributes
   are not treated as known facts.
4. Leave Sequential GPU memory mode enabled on free Colab. FLUX is released
   before vision validation and the vision model is released before avatar
   generation. This reduces concurrent memory use at the cost of reload time.
5. Run the explicit 12-image consistency test. It covers angles, lighting, body
   framing, two-person composition and a hidden-face case. This consumes GPU
   time; it is not run automatically at startup. Stop through Job activity and
   resume saved work. Images and checks already completed are reused.
6. Label cases Correct identity, Wrong identity or Uncertain. The report exposes
   automatic passes, human reviews, false acceptances and false rejections.
   Mock preview results are labeled and are not quality evidence.

Calibration is scoped to an identity version. Approving a different reference
pack requires recalibration for automatic identity acceptance. Uncalibrated,
unavailable or inconclusive face comparisons go to Needs Review, not Passed;
they do not trigger futile automatic regeneration. Actual identity mismatches
can enter the existing bounded gallery-repair workflow.

Console, guided starters, gallery batches, repairs and avatar enhancement share
the validation path. Gallery full-screen viewing includes Check quality for an
existing image. Enhancements also compare the source, retaining originals.
Low-frequency image difference is only a preservation heuristic, not proof
that pose/expression are identical. If validation fails after generation, retry
reuses the saved output rather than generating another console/enhancement image.
Compare includes matched-region Center detail and detected Face crops when
available. Both sides use the same normalized region, and cropped previews are
lossless. Detail-energy measurements can flag smoothing/no change, but noise or
sharpening can raise them: they are explicitly not a realism or quality score.

Replacing confirmed face/body anchors deactivates the old pack's approvals;
old versions remain restorable. Historical rechecks, repairs and enhancement
select the source image's policy before snapshotting it. Benchmark images can
be corrected without a search-reference dependency.

Image records retain identity version, seed, prompt, references, encoder and
runtime/package details. Batches retain their quality-policy snapshot. An old
batch whose confirmed base identity changed requires restoring that identity
before resuming; it is never silently continued against a different person.

## Content Tab

1. Save the persona's niche, tone, values and optional AI disclosure.
2. Choose a saved photograph. Compose caption uses your brief, niche, values,
   tone and disclosure through local templates with concise/editorial/conversational
   styles, without external calls. This is not a hidden local language model.
   You can edit the text directly. Use Gemini
   is an explicit optional choice, using the existing private Gemini credential.
3. Save a draft with an optional planned date/time. Publishing is manual by
   default. Optional Runtime scheduling requires configured Instagram credentials
   and explicit confirmation, and queues only approved future posts while this
   runtime is active. It waits for the worker to be idle, continues pending Meta
   containers, and does not blindly retry uncertain outcomes. A restart turns
   scheduling off; missed posts stay manual. There is no offline/24-hour guarantee.
4. Export downloads a ZIP containing the JPEG, caption and provenance metadata.
   This is the default, API-independent publishing workflow.
5. Optional Instagram publishing reads `INSTAGRAM_ACCESS_TOKEN`,
   `INSTAGRAM_USER_ID`, `INSTAGRAM_GRAPH_VERSION` and `INSTAGRAM_LOGIN_MODE` from
   the notebook controls, private keys TXT or notebook Secrets. No credentials
   are sent to the frontend. Choose the Graph version supported by your Meta app.
6. Your professional account and Meta app need the appropriate publishing
   permissions. Supply an HTTPS URL that serves the exact exported JPEG without
   the studio's private-session login. An authenticated `/api/files` URL will not
   work for Meta. The app does not make your private gallery public.
7. Use an Instagram-compatible image (JPEG, under 8 MB, aspect ratio 4:5 to
   1.91:1). Approve the exact image/caption, then explicitly click Publish now.
   Editing a draft clears that approval. Approval by itself never publishes.
8. An uncertain network outcome is saved and never blindly resent. Check the
   configured Instagram account, then reconcile its actual media ID. Publishing
   is disabled in local mock preview.

Existing reference discovery still uses Tavily/Gemini when requested. Identity
references, console generation, vision checks, consistency tests and manual
caption/export do not require those APIs. No video or voice APIs were added.

## Small Acceptance Test

Use one avatar and one reference before a large run. Confirm one candidate,
calibrate, generate one Console image, inspect its checks, enhance it once and
export one draft. Reload the page and verify all records remain. Then stop and
resume a small batch; only after that run the 12-case benchmark. On cloud, record
actual stage timings and peak system/GPU RAM; timings cannot be guaranteed from
the laptop mock preview. CPU downloads should happen only on first setup or repair.

Local regression: `python -m unittest discover`, `npm run build` in `frontend`,
and Playwright CLI `run-code --filename test_web_influencer_browser.js` against
the mocked local server. Run `python prepare_browser_fixtures.py` first to create
the synthetic `Enhancement QA` fixture on a fresh clone. Existing private preview
images are never overwritten. No real models run locally.

Actual Colab/Kaggle GPU quality, OAuth/Drive operations, Instagram authorization
and multi-hour endurance still require a real cloud acceptance run. The code
and mock tests cannot establish those external outcomes.

## Primary References

- [OpenCV face detection/recognition tutorial](https://docs.opencv.org/4.7.0/d0/dd4/tutorial_dnn_face.html)
- [Pinned YuNet asset hash/size](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
- [Pinned SFace asset hash/size](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)
- [Meta Instagram content publishing](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/content-publishing/)
