# Enhancement Prompt Profiles

Automatic image-aware enhancement uses **one standalone profile + targeted focus
rules + any user-provided additional reconstruction instructions**. All 15 profile
bodies use the supplied final AI-image repair prompts (`user-refinement-v2`). No
universal preservation prompt is prepended or appended, and no second category's
full prompt is added. Each profile balances conservative defect repair, appropriate
texture reconstruction and preservation of correct details. The final applied
prompt is saved with the output.

## The 15 Profiles

| ID | Name | Purpose |
| --- | --- | --- |
| `female_portrait` | Female-Presenting Portrait | Facial detail and hair; preserve identity and complexion. |
| `male_portrait` | Male-Presenting Portrait | Facial detail and existing facial hair; no invented ruggedness. |
| `female_body` | Female-Presenting Body / Fashion | Clothing and visible body detail; preserve proportions. |
| `male_body` | Male-Presenting Body / Fashion | Clothing and visible body detail; no added muscles or definition. |
| `group` | Couples / Groups | Keep each person's identity, position and features distinct. |
| `person_unspecified` | Unclear Presentation / Obscured Person | Visible details only; do not invent hidden features. |
| `landscape` | Landscapes / Nature | Terrain, vegetation and water; retain atmosphere and distance. |
| `architecture` | Architecture / Streets / Interiors | Structural detail; retain perspective and layout. |
| `product` | Products / Vehicles / Objects | Shape, finish, branding and construction details. |
| `animal` | Animals | Existing fur, feathers or scales; retain anatomy and markings. |
| `food` | Food / Still Life | Existing ingredients, arrangement and surface detail. |
| `macro` | Macro / Close-Up | Supported fine detail within the original plane of focus. |
| `artwork` | Illustration / Anime / Painting / CGI | Clarity within the original artistic style. |
| `document` | Documents / Screenshots / Graphics | Legibility without inventing information. |
| `general` | General-Purpose Fallback | Conservative enhancement when classification is uncertain. |

## Automatic Selection

The existing local vision model answers narrow yes/no/unclear questions. New
enhancements do not ask skin, pore, freckle or material-inventory questions.

1. Check documents/graphics and deliberately stylized artwork first. A person
   depicted in a painting does not turn it into a photographic portrait.
2. For photographic-looking images, check whether people exist and whether they
   are the main subject or incidental background figures.
3. Main groups use the group profile without gender inference. A dominant-person
   question cross-checks group answers. When available, the already-installed CPU
   face detector provides a dominant-face crop for a second count/visibility check.
   Comparable-sized faces are not cropped down to one. This is a heuristic, not
   proof: unresolved count conflicts use the neutral person profile. A clearly visible
   single subject uses framing and apparent visual presentation to select a
   portrait/body profile. Ambiguous/conflicting replies and obscured faces use
   the neutral person profile. Presentation is not a claim about gender identity.
4. Other scenes use nature, architecture, animals, food or objects. Macro framing
   gets the standalone macro profile without a second full subject prompt.
   Background people remain background elements and do not get portrait treatment.
5. Add focus rules for in-focus subjects, soft subjects, groups, background people
   and uncertainty. Preserve lighting in every category; do not relight to simulate
   improvement. Unclear scenes use a general fallback rather than invented detail.

Partial recognition is labeled partial, not perfect. Classifier replies are
recorded for debugging but are never inserted as arbitrary prompt instructions.
Complete classifications are cached by source pixels, vision configuration,
mock/real mode and planner version. Old skin-first and v1 scene caches are not reused
by the v2 foreground-aware classifier. Analysis completion does not verify its truth.
The small vision model no longer decides that a source has compression blocks or
sharpening halos and does not add automatic softening instructions. Its old
single-image artifact judgments are not used to claim that reconstruction failed.

## Where to See or Change It

Open an enhanced image, then expand **Enhancement analysis**. It shows the
profile, medium, main subject, people's role/count, framing, presentation, focus
and assessment scope. Expand questions/answers or the applied prompt for details.

The exact 15 prompt texts, names and short descriptions live in
`enhancement_profiles.py` (`PROFILES`), with targeted focus instructions in
`FOCUS_RULES`. Supply a replacement using the profile
name or ID; no model checkpoint or notebook changes are needed to edit its text.
Version the library when changing routing or classification semantics. Text-only
revisions use `PROMPT_VERSION` and are saved in each new plan as `prompt_version`;
they do not invalidate compatible source classifications or force model downloads.

In automatic mode, **Additional instructions (optional)** starts blank. The
automatic profile is marked pending, not displayed as the old generic prompt.
The actual assembled prompt is recorded after analysis. You can explicitly select
one of the 15 profiles if automatic recognition is wrong, preview its known text,
and choose **Use as custom prompt** to edit it yourself.

Switch **Automatic image-aware prompt** off to use **Custom enhancement prompt**.
It starts blank and must be filled. One-pass custom mode sends that exact text
without scene analysis, automatic templates or hidden additional instructions.
HD tile passes also receive their required crop/context prefix; it is recorded
under sampling diagnostics. Automatic and custom drafts do not mix in requests.
Opening another image resets text, profile selection and sampling defaults.

Start a new enhancement after updating: resuming an existing job keeps its saved
prompt, source and generated image rather than silently rerunning it.

## Reconstruction Sampling

New web enhancements default to **FLUX.2 reference reconstruction**: the source
is reference conditioning, output starts from `EmptyFlux2LatentImage`, and four
Euler steps follow ComfyUI's native size-dependent `Flux2Scheduler`. The denoise
strength slider is not shown because partial denoise is not used in this mode.
This is a reconstruction, not an identity/pixel-preservation guarantee. Inspect
the result; full-schedule edits can change identity or geometry despite the prompt.

**Low-noise img2img (experimental)** keeps the prior encoded-source path and
its denoise strength slider. It is an explicit alternative, not a claim that low
denoise on a four-step distilled model is a proven restoration workflow. Legacy
jobs without a sampling-mode field continue using their original img2img path.
Normal chat editing and masked editing are unchanged. Warmup exercises the native
reference path before a ready link when warmup is enabled.

One reconstruction pass still samples at up to 1024px, then resizes to the chosen
output dimensions. A 2x output is not equivalent to native 2x detail. HD tiled
reconstruction samples the additional tiles; progress reports every pass but saves
one output. Diagnostics record sampling mode, scheduler, denoise, steps and sizing.

Implementation references: [ComfyUI FLUX.2 latent and scheduler](https://github.com/Comfy-Org/ComfyUI/blob/v0.28.0/comfy_extras/nodes_flux.py),
[ComfyUI custom sampler](https://github.com/Comfy-Org/ComfyUI/blob/v0.28.0/comfy_extras/nodes_custom_sampler.py),
and [BFL's reference-conditioned inference](https://github.com/black-forest-labs/flux2/blob/main/scripts/cli.py).

## Review and Limits

New outputs get scene/focus consistency checks rather than skin-category checks.
These are advisory; consistency is not proof of identity, realism, exact text or
better quality. Unclear checks or detected mismatches require review. Pixel/detail
metrics and before/after comparisons remain available. Documents require careful
visual verification because generative editing cannot guarantee exact lettering.
For legal records, screenshots and other character-perfect work, use deterministic
restoration rather than relying solely on FLUX. Validate readable content against
OCR or original structured data and render text deterministically when exactness
matters. Never accept an unchecked generative document edit. This warning is documentation,
not an instruction embedded in the document profile's model prompt.

Legacy jobs keep their original prompts and texture-review metadata. The API
retains `texture_review` for older consumers; new plans also record the same
advisory result as `content_review`, displayed as Content review in the UI.

Local previews and automated tests use mock inference only. They verify routing,
persistence and controls, not actual FLUX output quality. No extra model, paid
API, LoRA training, voice or video dependency is introduced.
