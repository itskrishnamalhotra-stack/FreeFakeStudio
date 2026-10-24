"""Bounded, cached source observations for conservative image enhancement."""
import hashlib
import json
from pathlib import Path

from PIL import ImageOps

from avatar_studio import _atomic_json

VERSION = "source-faithful-v1"
CHOICES = {
    "scene": ("portrait", "people", "landscape", "product", "architecture", "animal", "food", "artwork", "document", "other", "uncertain"),
    "lighting": ("soft_daylight", "hard_daylight", "indoor", "studio", "low_light", "mixed", "uncertain"),
    "condition": ("clear", "soft", "blurred", "compressed", "oversharpened", "uncertain"),
    "skin_texture": ("smooth", "natural", "fine_texture", "strong_texture", "not_visible", "uncertain"),
    "skin_marks": ("none_visible", "sparse_visible", "many_visible", "not_visible", "uncertain"),
}
MATERIALS = ("skin", "hair", "fabric", "metal", "glass", "wood", "foliage", "water", "stone", "fur", "food", "plastic", "paper")
QUESTIONS = {
    "scene": "What is the main image category?",
    "lighting": "What lighting is visibly present?",
    "condition": "What is the most visible technical condition? Intentional background blur does not mean the subject is blurred.",
    "skin_texture": "For clearly visible skin, how textured is it at this image scale? Do not infer microscopic details.",
    "skin_marks": "How many freckles or dark spots are clearly visible on the main subject's skin? Do not infer hidden or microscopic marks.",
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
    "hair": "Resolve only supported hair and eyebrow edges; keep the same hairline, hairstyle and density.",
    "fabric": "Clarify existing fabric weave, seams and folds without changing the garment or adding patterns.",
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


def _question(image, key, ask):
    return choice(ask(image, QUESTIONS[key] + " Return exactly one of: " + ", ".join(CHOICES[key]) + ".",
                      max_tokens=32, context="enhancement/" + key), CHOICES[key])


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
              **{name: "uncertain" for name in CHOICES}, "materials": []}
    if mock:
        result.update(scene="other", condition="clear", skin_texture="not_visible", skin_marks="not_visible")
    else:
        ask = ask or avatar_vision._ask_question
        photo = avatar_vision.prepare_image(source)
        try:
            for name in ("scene", "lighting", "condition"):
                check()
                emit("Analyzing source: " + name.replace("_", " "))
                result[name] = _question(photo, name, ask)
            check()
            emit("Analyzing source: visible materials")
            answer = ask(photo, "Which materials are clearly visible? Return only a comma-separated selection from: " + ", ".join(MATERIALS) + ". Return uncertain if none can be identified.",
                         max_tokens=64, context="enhancement/materials")
            result["materials"] = [v for v in MATERIALS if v in {word.strip().lower() for word in str(answer).split(",")}]
            if "skin" in result["materials"]:
                for name in ("skin_texture", "skin_marks"):
                    check()
                    emit("Analyzing source: " + name.replace("_", " "))
                    result[name] = _question(photo, name, ask)
        except RuntimeError as exc:
            from web_jobs import Cancelled
            if isinstance(exc, Cancelled):
                raise
            result.update(status="unavailable", error=str(exc)[:500])
            emit("Source analysis unavailable; using conservative preservation instructions")
    check()
    if result["status"] == "complete" and result["scene"] == "uncertain" and not result["materials"]:
        result["status"] = "inconclusive"
    if result["status"] == "complete":
        _atomic_json(path, result)
    return result


def plan(analysis, instructions):
    parts = [PRESERVATION, str(instructions).strip()]
    if analysis.get("status") == "complete":
        parts += [MATERIAL_RULES[v] for v in analysis.get("materials", []) if v in MATERIAL_RULES]
        if analysis.get("condition") in {"soft", "blurred"}:
            parts.append("Recover only supported subject edges, leaving depth-of-field blur unchanged. Keep uncertain details understated.")
        elif analysis.get("condition") == "compressed":
            parts.append("Reduce visible compression blocks while retaining existing fine detail and smooth tonal transitions.")
        elif analysis.get("condition") == "oversharpened":
            parts.append("Reduce edge halos and brittle texture; retain natural edge softness instead of adding sharpness.")
        if analysis.get("skin_texture") == "smooth":
            parts.append("The source skin is smooth. Keep it smooth, without turning clarity into coarse texture.")
        if analysis.get("skin_marks") == "none_visible":
            parts.append("Keep the clear complexion shown in the source; add no freckles or speckles.")
        if analysis.get("scene") in {"artwork", "document"}:
            parts.append("Preserve the source medium, line work and lettering rather than converting it into a photograph.")
        if analysis.get("lighting") != "uncertain":
            parts.append("Retain the source lighting and its existing highlight softness; do not relight the image.")
    parts.append("Preserve the original traits and texture scale even if an earlier instruction asks for extra detail. Return only the edited image.")
    return {"version": VERSION, "analysis": analysis, "prompt": " ".join(parts), "automatic": True}


def review_texture(image, analysis, mock=False, check=lambda: None, ask=None):
    if analysis.get("status") != "complete":
        return {"status": "uncertain", "reasons": ["Source analysis was unavailable"]}
    if mock:
        return {"status": "mock", "reasons": [], "mock_not_quality_evidence": True}
    if "skin" not in analysis.get("materials", []):
        return {"status": "not_applicable", "reasons": []}
    import avatar_vision
    ask = ask or avatar_vision._ask_question
    result = {"status": "consistent", "reasons": [], "method": "vision-category comparison; not proof of preservation"}
    try:
        photo = avatar_vision.prepare_image(image)
        for key in ("skin_texture", "skin_marks"):
            check()
            result[key] = _question(photo, key, ask)
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
    return (diagnostics or {}).get("texture_review", {}).get("status") in {"uncertain", "needs_review"}


def output_status(diagnostics, passed=False):
    if needs_review(diagnostics):
        return "needs_review"
    if (diagnostics or {}).get("near_noop"):
        return "no_change"
    return "passed" if passed else "needs_review"
