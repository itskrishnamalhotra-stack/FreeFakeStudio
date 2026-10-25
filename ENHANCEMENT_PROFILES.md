# Enhancement Prompt Profiles

Automatic image-aware enhancement uses **shared preservation instructions + one
profile + focus rules + your reconstruction instructions**. Source preservation
and the original medium take precedence over conflicting requests for generic
texture. The final applied prompt is saved with the output.

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
3. Main groups use the group profile without gender inference. A clearly visible
   single subject uses framing and apparent visual presentation to select a
   portrait/body profile. Ambiguous/conflicting replies and obscured faces use
   the neutral person profile. Presentation is not a claim about gender identity.
4. Other scenes use nature, architecture, animals, food or objects. Macro framing
   gets the macro profile, supplemented by the detected subject's instructions.
   Background people remain background elements and do not get portrait treatment.
5. Add focus rules for in-focus subjects, soft subjects, groups, background people
   and uncertainty. Preserve lighting in every category; do not relight to simulate
   improvement. Unclear scenes use a general fallback rather than invented detail.

Partial recognition is labeled partial, not perfect. Classifier replies are
recorded for debugging but are never inserted as arbitrary prompt instructions.
Complete classifications are cached by source pixels, vision configuration,
mock/real mode and planner version. Old skin-first caches are not reused.

## Where to See or Change It

Open an enhanced image, then expand **Enhancement analysis**. It shows the
profile, medium, main subject, people's role/count, framing, presentation, focus
and visible defects. Expand questions/answers or the applied prompt for details.

The exact 15 prompt texts, names and short descriptions live in
`enhancement_profiles.py` (`PROFILES`). Shared constraints are in `PRESERVATION`
and focus instructions in `FOCUS_RULES`. Supply a replacement using the profile
name or ID; no model checkpoint or notebook changes are needed to edit its text.
Version the library when changing routing or classification semantics.

The existing editable Reconstruction prompt remains an additional instruction.
Switch **Automatic image-aware prompt** off to use that text without automatic
profiles. Start a new enhancement after updating: resuming an existing job keeps
its saved prompt, source and generated image rather than silently rerunning it.

## Review and Limits

New outputs get scene/focus/artifact checks rather than skin-category checks.
These are advisory; consistency is not proof of identity, realism, exact text or
better quality. Unclear checks or detected mismatches require review. Pixel/detail
metrics and before/after comparisons remain available. Documents require careful
visual verification because generative editing cannot guarantee exact lettering.

Legacy jobs keep their original prompts and texture-review metadata. The API
retains `texture_review` for older consumers; new plans also record the same
advisory result as `content_review`, displayed as Content review in the UI.

Local previews and automated tests use mock inference only. They verify routing,
persistence and controls, not actual FLUX output quality. No extra model, paid
API, LoRA training, voice or video dependency is introduced.
