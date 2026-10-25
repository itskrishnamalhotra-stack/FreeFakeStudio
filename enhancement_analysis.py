"""Bounded, cached source observations for conservative image enhancement."""
import hashlib
import json
from pathlib import Path

from PIL import ImageOps

from avatar_studio import _atomic_json

VERSION = "targeted-observations-v2"
CHOICES = {
    "scene": ("portrait", "people", "landscape", "product", "architecture", "animal", "food", "artwork", "document", "other", "uncertain"),
    "lighting": ("soft_daylight", "hard_daylight", "indoor", "studio", "low_light", "mixed", "uncertain"),
    "condition": ("clear", "soft", "blurred", "compressed", "oversharpened", "uncertain"),
    "skin_texture": ("smooth", "natural", "fine_texture", "strong_texture", "not_visible", "uncertain"),
    "skin_marks": ("none_visible", "sparse_visible", "many_visible", "not_visible", "uncertain"),
}
MATERIALS = ("skin", "hair", "fabric", "metal", "glass", "wood", "foliage", "water", "stone", "fur", "food", "plastic", "paper")
PROBES = {
    "person_visible": "Is a person clearly visible as the main foreground subject?",
    "face_visible": "Is the main person's face clearly visible?",
    "skin_visible": "Can you clearly see bare human skin on the main subject?",
    "hair_visible": "Can you clearly see hair on the main subject's head?",
    "fabric_visible": "Is cloth or clothing fabric clearly visible on the main subject?",
    "subject_blurred": "Are the main subject's eyes or outline visibly blurred? Ignore background blur.",
    "compression_blocks": "Are square compression blocks clearly visible on the main subject?",
    "edge_halos": "Are bright or dark sharpening halos clearly visible along the main subject's outline?",
    "daylight": "Is the main subject outdoors in natural daylight?",
    "hard_shadows": "Are there sharply edged shadows on the main subject?",
    "artificial_light": "Is the main subject indoors under artificial lighting?",
    "skin_rough": "Do the visible cheeks or forehead have coarse, strongly pronounced skin texture?",
    "skin_smooth": "Do the visible cheeks or forehead look mostly smooth at this image scale?",
    "skin_many_marks": "Are many clearly visible freckles or dark spots spread across the visible cheeks and forehead?",
    "skin_any_marks": "Can you clearly see any freckles or dark spots on the visible cheeks or forehead?",
    "landscape_visible": "Is an outdoor landscape the main subject, rather than a person or object?",
    "product_visible": "Is a manufactured object or product the main subject?",
    "architecture_visible": "Is a building the main subject?",
    "animal_visible": "Is an animal the main subject?",
    "food_visible": "Is food the main subject?",
    "document_visible": "Is a page of text or a document the main subject?",
}
PRESERVATION = (
    "Image 1 is the authoritative source. Keep identity, apparent age, facial geometry, body proportions, "
    "expression, pose, clothes, object count, composition, exposure and colors unchanged. "
    "Keep the existing skin tone, smoothness and markings exactly as shown. Do not invent freckles, "
    "spots, wrinkles, coarse pores, body hair or age lines. Preserve intentional background blur and bokeh. "
    "Visible source evidence takes precedence over requests for generic microtexture or extreme sharpness. "
)
MATERIAL_RULES = {
    "skin": "Keep skin detail gentle and at the original scale, with the source's existing smoothness and markings.",
    "hair": "Resolve the separation of existing hair strands, eyelashes and eyebrow hairs; keep the same hairline, hairstyle and density.",
    "fabric": "Restore fine existing fabric weave, fibres, seams and fold edges without changing the garment or adding patterns.",
    "metal": "Keep existing metallic reflections, finish and highlight shape physically consistent.",
    "glass": "Preserve transparency and existing reflections without inventing objects behind the glass.",
    "wood": "Clarify only existing wood grain, keeping its direction and scale.",
    "foliage": "Preserve supported leaf shapes and keep out-of-focus foliage soft.",
    "water": "Preserve existing water surfaces, reflections and wave scale.",
    "stone": "Retain existing stone texture at its original scale.",
    "fur": "Preserve the animal's markings and existing fur direction and density.",
    "food": "Keep the same food shapes, ingredients, surface finish and moisture.",
    "plastic": "Retain the original plastic finish and reflection softness.",
    "paper": "Keep existing lettering, drawings and paper surfaces unchanged; do not invent text.",
}


def choice(answer, options):
    # Only exact enum answers become instructions; free-form model prose is never spliced into a prompt.
    value = str(answer).strip().strip('"\'` .').lower().replace(" ", "_").replace("-", "_")
    return value if value in options else "uncertain"


def fingerprint(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    return image, digest


def _probe(image, key, ask, answers, check):
    import avatar_vision
    question = PROBES.get(key, "Is " + key.removesuffix("_visible") + " clearly visible on the main subject?")
    attempts = []
    verdict = None
    for attempt in range(2):
        check()
        prompt = question + (" Answer yes, no, or unclear." if not attempt else " Inspect only what is visible. Reply with one word: yes, no, or unclear.")
        answer = str(ask(image, prompt, max_tokens=24, context="enhancement/" + key))
        verdict = avatar_vision._yes_no(answer)
        attempts.append(answer[:240])
        if verdict is not None:
            break
    answers[key] = {"question": question, "result": verdict, "answers": attempts}
    return verdict


def _skin_summary(checks):
    if checks.get("skin_rough") is True:
        texture = "strong_texture"
    elif checks.get("skin_smooth") is True:
        texture = "smooth"
    elif checks.get("skin_rough") is False and checks.get("skin_smooth") is False:
        texture = "natural"
    else:
        texture = "uncertain"
    marks = ("many_visible" if checks.get("skin_many_marks") is True else
             "sparse_visible" if checks.get("skin_any_marks") is True else
             "none_visible" if checks.get("skin_many_marks") is False and checks.get("skin_any_marks") is False else "uncertain")
    return texture, marks


def _crop(image, box):
    if not box or len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        return image
    return image.crop((int(box[0] * image.width), int(box[1] * image.height),
                       max(int(box[0] * image.width) + 1, round(box[2] * image.width)),
                       max(int(box[1] * image.height) + 1, round(box[3] * image.height))))


def _face_box(image):
    import avatar_face
    faces = avatar_face.features(image)
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda row: row["box"][2] * row["box"][3])["box"]
    factor = min(1, 1024 / max(image.size))
    width, height = max(1, round(image.width * factor)), max(1, round(image.height * factor))
    return [max(0, (x - .15 * w) / width), max(0, (y - .15 * h) / height),
            min(1, (x + 1.15 * w) / width), min(1, (y + 1.15 * h) / height)]


def analyze(image, root, mock=False, check=lambda: None, emit=lambda message: None, ask=None):
    import avatar_vision
    source, digest = fingerprint(image)
    key = hashlib.sha256(f"{VERSION}|{avatar_vision.MODEL_ID}|{avatar_vision.MAX_EDGE}|{mock}|{digest}".encode()).hexdigest()
    path = Path(root) / "_enhancement_analysis" / (key + ".json")
    check()
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("source_sha256") == digest and cached.get("version") == VERSION and cached.get("status") == "complete":
            fields = {name: choice(cached.get(name), options) for name, options in CHOICES.items()}
            return {**cached, **fields, "materials": [v for v in cached.get("materials", []) if v in MATERIALS], "cache_hit": True}
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    result = {"version": VERSION, "source_sha256": digest, "model": avatar_vision.MODEL_ID,
              "status": "complete", "cache_hit": False, "mock_not_quality_evidence": mock,
              **{name: "uncertain" for name in CHOICES}, "materials": [], "checks": {}, "answers": {}}
    if mock:
        result.update(scene="other", condition="clear", skin_texture="not_visible", skin_marks="not_visible")
    else:
        if ask is None:
            try:
                result["inspection_box"] = _face_box(source)
            except Exception as exc:
                result["focus_note"] = "Face localization unavailable; full image used: " + str(exc)[:200]
        ask = ask or avatar_vision._ask_question
        photo = avatar_vision.prepare_image(source)
        focused = avatar_vision.prepare_image(_crop(source, result.get("inspection_box")))
        checks = result["checks"]
        def probe(key, skin=False):
            emit("Analyzing source: " + key.replace("_", " "))
            checks[key] = _probe(focused if skin else photo, key, ask, result["answers"], check)
            return checks[key]
        try:
            for key in ("person_visible", "face_visible", "skin_visible", "hair_visible", "fabric_visible"):
                probe(key)
            human = checks["person_visible"] is True or checks["face_visible"] is True or checks["skin_visible"] is True or bool(result.get("inspection_box"))
            result["scene"] = "portrait" if human and (checks["face_visible"] is True or result.get("inspection_box")) else "people" if human else "uncertain"
            if not human:
                for scene in ("landscape", "product", "architecture", "animal", "food", "document"):
                    if probe(scene + "_visible") is True:
                        result["scene"] = scene
                        break
                candidates = {"landscape": ("foliage", "water", "stone"), "product": ("metal", "glass", "plastic"),
                              "architecture": ("wood", "glass", "stone"), "animal": ("fur",), "food": ("food",), "document": ("paper",)}
                for material in candidates.get(result["scene"], ()):
                    probe(material + "_visible")
            result["materials"] = [v for v in MATERIALS if checks.get(v + "_visible") is True]
            if checks["skin_visible"] is not False or human:
                if probe("skin_rough", True) is not True:
                    probe("skin_smooth", True)
                if probe("skin_many_marks", True) is not True:
                    probe("skin_any_marks", True)
                result["skin_texture"], result["skin_marks"] = _skin_summary(checks)
                if human and checks["skin_visible"] is False:
                    result["skin_texture"] = result["skin_marks"] = "uncertain"
                    result["visibility_conflict"] = True
            else:
                result.update(skin_texture="not_visible", skin_marks="not_visible")
            for key in ("subject_blurred", "compression_blocks", "edge_halos"):
                probe(key)
            result["condition"] = ("oversharpened" if checks["edge_halos"] is True else "compressed" if checks["compression_blocks"] is True
                                   else "blurred" if checks["subject_blurred"] is True else "clear" if all(checks[v] is False for v in ("subject_blurred", "compression_blocks", "edge_halos")) else "uncertain")
            if probe("daylight") is True:
                shadow = probe("hard_shadows")
                result["lighting"] = "hard_daylight" if shadow is True else "soft_daylight" if shadow is False else "uncertain"
            elif probe("artificial_light") is True:
                result["lighting"] = "indoor"
        except RuntimeError as exc:
            from web_jobs import Cancelled
            if isinstance(exc, Cancelled):
                raise
            result.update(status="unavailable", error=str(exc)[:500])
            emit("Source analysis unavailable; using conservative preservation instructions")
    check()
    if not mock and result["status"] == "complete":
        result["unresolved"] = [key for key, value in result["checks"].items() if value is None]
        if result["scene"] == "uncertain" or result.get("visibility_conflict") or any(result[key] == "uncertain" for key in ("condition", "lighting", "skin_texture", "skin_marks")) or result["unresolved"]:
            result["status"] = "partial" if any(v is not None for v in result["checks"].values()) else "inconclusive"
    if result["status"] == "complete":
        _atomic_json(path, result)
    return result


def plan(analysis, instructions):
    parts = [PRESERVATION, str(instructions).strip()]
    if analysis.get("status") in {"complete", "partial"}:
        parts += [MATERIAL_RULES[v] for v in analysis.get("materials", []) if v in MATERIAL_RULES]
        if analysis.get("condition") in {"soft", "blurred"}:
            parts.append("Recover only supported subject edges, leaving depth-of-field blur unchanged. Keep uncertain details understated.")
        elif analysis.get("condition") == "compressed":
            parts.append("Reduce visible compression blocks while retaining existing fine detail and smooth tonal transitions.")
        elif analysis.get("condition") == "oversharpened":
            parts.append("Reduce edge halos and brittle texture; retain natural edge softness instead of adding sharpness.")
        if analysis.get("skin_texture") == "smooth":
            parts.append("The source skin is smooth. Keep it smooth while recovering delicate non-repeating tonal detail, without coarse pores or spot patterns.")
        if analysis.get("skin_marks") == "none_visible":
            parts.append("Keep the clear complexion shown in the source; add no freckles or speckles.")
        if analysis.get("scene") in {"artwork", "document"}:
            parts.append("Preserve the source medium, line work and lettering rather than converting it into a photograph.")
        if analysis.get("lighting") != "uncertain":
            parts.append("Retain the source lighting and its existing highlight softness; do not relight the image.")
    parts.append("Preserve the original traits and texture scale even if an earlier instruction asks for extra detail. Return only the edited image.")
    return {"version": VERSION, "analysis": analysis, "prompt": " ".join(parts), "automatic": True}


def review_texture(image, analysis, mock=False, check=lambda: None, ask=None):
    if analysis.get("status") not in {"complete", "partial"}:
        return {"status": "uncertain", "reasons": ["Source analysis was unavailable"]}
    if mock:
        return {"status": "mock", "reasons": [], "mock_not_quality_evidence": True}
    checks = analysis.get("checks", {})
    if checks.get("skin_visible") is False and checks.get("face_visible") is False and checks.get("person_visible") is False and not analysis.get("inspection_box"):
        return {"status": "not_applicable", "reasons": []}
    if "skin" not in analysis.get("materials", []) and analysis.get("scene") not in {"portrait", "people"}:
        return {"status": "uncertain", "reasons": ["Visible skin was not reliably established"]}
    import avatar_vision
    ask = ask or avatar_vision._ask_question
    result = {"status": "consistent", "reasons": [], "method": "vision-category comparison; not proof of preservation"}
    try:
        photo = avatar_vision.prepare_image(_crop(image, analysis.get("inspection_box")))
        observed = {}
        result["answers"] = {}
        for key, followup in (("skin_rough", "skin_smooth"), ("skin_many_marks", "skin_any_marks")):
            observed[key] = _probe(photo, key, ask, result["answers"], check)
            if observed[key] is not True:
                observed[followup] = _probe(photo, followup, ask, result["answers"], check)
        result["skin_texture"], result["skin_marks"] = _skin_summary(observed)
    except RuntimeError as exc:
        from web_jobs import Cancelled
        if isinstance(exc, Cancelled):
            raise
        return {"status": "uncertain", "reasons": ["Texture review unavailable"], "error": str(exc)[:500]}
    if analysis.get("skin_texture") in {"smooth", "natural", "fine_texture"} and result["skin_texture"] == "strong_texture":
        result["reasons"].append("Skin texture became much stronger than the source")
    if analysis.get("skin_marks") in {"none_visible", "sparse_visible"} and result["skin_marks"] == "many_visible":
        result["reasons"].append("Many skin markings appeared that were not present in the source analysis")
    if any(result[key] in {"uncertain", "not_visible"} or analysis.get(key) in {"uncertain", "not_visible"} for key in ("skin_texture", "skin_marks")):
        result["status"] = "uncertain"
    if result["reasons"]:
        result["status"] = "needs_review"
    return result


def needs_review(diagnostics):
    data = diagnostics or {}
    return (data.get("texture_review", {}).get("status") in {"uncertain", "needs_review"}
            or data.get("enhancement_review", {}).get("status") == "needs_review")


def output_status(diagnostics, passed=False):
    if needs_review(diagnostics):
        return "needs_review"
    if (diagnostics or {}).get("near_noop"):
        return "no_change"
    return "passed" if passed else "needs_review"
