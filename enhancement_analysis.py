"""Scene-first classification and source-preserving enhancement planning."""
import hashlib
import json
from pathlib import Path

from PIL import ImageOps

from avatar_studio import _atomic_json
import enhancement_profiles as profiles

VERSION = profiles.VERSION
CHOICES = {
    "scene": ("portrait", "people", "landscape", "product", "architecture", "animal", "food", "artwork", "document", "other", "uncertain"),
    "medium": ("photographic", "artwork", "document", "uncertain"),
    "people_role": ("main_subject", "background", "not_present", "not_assessed", "uncertain"),
    "people_count": ("single", "multiple", "not_present", "not_assessed", "uncertain"),
    "framing": ("portrait", "body", "obscured", "not_applicable", "uncertain"),
    "presentation": ("female_presenting", "male_presenting", "unspecified", "not_assessed", "uncertain"),
    "focus": ("in_focus", "soft_subject", "uncertain"),
    "condition": ("not_assessed", "uncertain"),
}
PROBES = {
    "document_visible": "Is this primarily a document, screenshot, interface, diagram or text-based graphic rather than a photograph or painting?",
    "artwork_visible": "Is this deliberately stylized illustration, anime, painting or non-photorealistic CGI rather than a photographic-looking image?",
    "person_visible": "Is any person visible anywhere in the image, including small people in the background?",
    "people_main_subject": "Is a person or group of people the main visual subject, rather than incidental background figures?",
    "multiple_people": "Are two or more large people posing together in the foreground?",
    "single_foreground_person": "Is one person much larger and closer to the camera than everyone else?",
    "single_face_crop": "Does this crop show exactly one person's face?",
    "face_visible": "Is the main person's face clearly visible, rather than hidden, turned away or too small to inspect?",
    "body_visible": "Does the framing show the main person's torso below the shoulders or more of their body, rather than just a headshot?",
    "female_presentation": "Does the single main person have clearly feminine visual presentation? Describe visible presentation only, not actual gender identity. If ambiguous, answer unclear.",
    "male_presentation": "Does the single main person have clearly masculine visual presentation? Describe visible presentation only, not actual gender identity. If ambiguous, answer unclear.",
    "macro_visible": "Is the main subject shown as an extreme close-up or macro detail, rather than a normal portrait or wider scene?",
    "subject_blurred": "Is the main foreground subject out of focus?",
    "compression_blocks": "Are square compression blocks clearly visible on the main subject?",
    "edge_halos": "Are bright or dark sharpening halos clearly visible along the main subject's outline?",
    "new_artifacts": "Does this image contain clear rendering errors such as duplicated body parts or broken object geometry? Do not count intentional artistic stylization.",
    "skin_rough": "Do the visible cheeks or forehead have coarse, strongly pronounced skin texture?",
    "skin_smooth": "Do the visible cheeks or forehead look mostly smooth at this image scale?",
    "skin_many_marks": "Are many clearly visible freckles or dark spots spread across the visible cheeks and forehead?",
    "skin_any_marks": "Can you clearly see any freckles or dark spots on the visible cheeks or forehead?",
    "landscape_visible": "Is outdoor natural scenery the main subject, rather than a person, animal, building or product?",
    "product_visible": "Is a manufactured object or product the main subject?",
    "architecture_visible": "Is architecture, a street or a room interior the main subject, rather than people or a product?",
    "animal_visible": "Is an animal the main subject?",
    "food_visible": "Is food the main subject?",
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


def foreground_face_box(faces):
    """A dominant-face crop is a counting aid, not a proof of the number of people."""
    valid = [row for row in faces if row.get("detector_score", 0) >= .85 and
             len(row.get("box", [])) == 4 and min(row["box"][2:]) > 0]
    if not valid:
        return None
    largest = max(valid, key=lambda row: row["box"][2] * row["box"][3])
    x, y, w, h = largest["box"]
    width, height = largest["coordinate_size"]
    area = w * h
    if area < .02 * width * height or any(row is not largest and row["box"][2] * row["box"][3] >= .4 * area for row in valid):
        return None
    box = [max(0, (x - .2 * w) / width), max(0, (y - .2 * h) / height),
           min(1, (x + 1.2 * w) / width), min(1, (y + 1.2 * h) / height)]
    return box if 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1 else None


def _classify(result, probe):
    if probe("document_visible") is True:
        result.update(scene="document", medium="document", people_role="not_assessed", people_count="not_assessed",
                      framing="not_applicable", presentation="not_assessed")
        return
    if probe("artwork_visible") is True:
        result.update(scene="artwork", medium="artwork", people_role="not_assessed", people_count="not_assessed",
                      framing="not_applicable", presentation="not_assessed")
        return
    if all(result["checks"][key] is False for key in ("document_visible", "artwork_visible")):
        result["medium"] = "photographic"
    present = probe("person_visible")
    main = probe("people_main_subject") if present is not False else False
    if main is True:
        result.update(scene="people", people_role="main_subject")
        multiple = probe("multiple_people")
        dominant = probe("single_foreground_person")
        if dominant is True and multiple is True:
            confirmed_crop = probe("single_face_crop") if result.get("subject_box") else None
            if confirmed_crop is True:
                multiple = False
                result["routing_notes"].append("Dominant foreground subject confirmed in a face crop; smaller/background people excluded from the group profile.")
            else:
                multiple = None
                result["routing_notes"].append("Foreground count is conflicting; neutral person profile used rather than assuming a group.")
        result["people_count"] = "multiple" if multiple is True else "single" if multiple is False else "uncertain"
        if multiple is True:
            result.update(presentation="unspecified", framing="not_applicable")
        else:
            face, body = probe("face_visible"), probe("body_visible")
            result["framing"] = "obscured" if face is False else "body" if body is True else "portrait" if face is True and body is False else "uncertain"
            if result["framing"] == "portrait":
                result["scene"] = "portrait"
            if face is True and multiple is False:
                female, male = probe("female_presentation"), probe("male_presentation")
                result["presentation"] = ("female_presenting" if female is True and male is False else
                                          "male_presenting" if male is True and female is False else "unspecified")
                if female is True and male is True:
                    result["presentation"] = "uncertain"
            else:
                result["presentation"] = "not_assessed"
        return
    result.update(people_role="background" if present is True and main is False else "not_present" if present is False else "uncertain",
                  people_count="not_present" if present is False else "not_assessed", framing="not_applicable", presentation="not_assessed")
    result["macro"] = probe("macro_visible") is True
    for scene in ("architecture", "landscape", "animal", "food", "product"):
        if probe(scene + "_visible") is True:
            result["scene"] = scene
            break
    else:
        result["scene"] = "other" if all(result["checks"][key + "_visible"] is False for key in ("architecture", "landscape", "animal", "food", "product")) else "uncertain"


def analyze(image, root, mock=False, check=lambda: None, emit=lambda message: None, ask=None, face_boxes=None):
    import avatar_vision
    source, digest = fingerprint(image)
    key = hashlib.sha256(f"{VERSION}|{avatar_vision.MODEL_ID}|{avatar_vision.MAX_EDGE}|{mock}|{digest}".encode()).hexdigest()
    path = Path(root) / "_enhancement_analysis" / (key + ".json")
    check()
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("source_sha256") == digest and cached.get("version") == VERSION and cached.get("status") == "complete":
            fields = {name: choice(cached.get(name), options) for name, options in CHOICES.items()}
            cached = {**cached, **fields, "macro": cached.get("macro") is True, "cache_hit": True}
            selected = profiles.select(cached)
            return {**cached, "profile_id": selected, "profile_name": profiles.PROFILES[selected]["name"]}
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    result = {"version": VERSION, "source_sha256": digest, "model": avatar_vision.MODEL_ID,
              "status": "complete", "cache_hit": False, "mock_not_quality_evidence": mock,
              **{name: "uncertain" for name in CHOICES}, "macro": False, "checks": {}, "answers": {},
              "classification_verified": False, "routing_notes": []}
    if mock:
        result.update(scene="other", medium="uncertain", condition="not_assessed", focus="uncertain",
                      people_role="uncertain", people_count="not_assessed", framing="not_applicable", presentation="not_assessed")
    else:
        if face_boxes is not None or ask is None:
            try:
                if face_boxes is None:
                    import avatar_face
                    check()
                    face_boxes = avatar_face.detect_faces(source)
                result["subject_box"] = foreground_face_box(face_boxes)
            except RuntimeError as exc:
                from web_jobs import Cancelled
                if isinstance(exc, Cancelled):
                    raise
                result["localization_unavailable"] = True
                result["routing_notes"].append("Foreground localization unavailable: " + str(exc)[:180])
            except Exception as exc:
                result["localization_unavailable"] = True
                result["routing_notes"].append("Foreground localization unavailable: " + str(exc)[:180])
        ask = ask or avatar_vision._ask_question
        photo = avatar_vision.prepare_image(source)
        focused = avatar_vision.prepare_image(_crop(source, result.get("subject_box")))
        checks = result["checks"]
        def probe(key):
            emit("Analyzing source: " + key.replace("_", " "))
            use_crop = bool(result.get("subject_box")) and key in {"single_face_crop", "face_visible", "female_presentation", "male_presentation", "subject_blurred"}
            checks[key] = _probe(focused if use_crop else photo, key, ask, result["answers"], check)
            result["answers"][key]["scope"] = "foreground_face_crop" if use_crop else "whole_image"
            return checks[key]
        try:
            _classify(result, probe)
            probe("subject_blurred")
            result["focus"] = "soft_subject" if checks["subject_blurred"] is True else "in_focus" if checks["subject_blurred"] is False else "uncertain"
            result["condition"] = "not_assessed"
        except RuntimeError as exc:
            from web_jobs import Cancelled
            if isinstance(exc, Cancelled):
                raise
            result.update(status="unavailable", error=str(exc)[:500])
            emit("Source analysis unavailable; using conservative preservation instructions")
    check()
    if not mock and result["status"] == "complete":
        result["unresolved"] = [key for key, value in result["checks"].items() if value is None]
        if any(result[key] == "uncertain" for key in CHOICES) or result["unresolved"] or (result.get("localization_unavailable") and result.get("people_role") == "main_subject"):
            result["status"] = "partial" if any(v is not None for v in result["checks"].values()) else "inconclusive"
    selected = profiles.select(result) if result["status"] in {"complete", "partial"} else "general"
    result.update(profile_id=selected, profile_name=profiles.PROFILES[selected]["name"])
    if result["status"] == "complete":
        _atomic_json(path, result)
    return result


def plan(analysis, instructions, profile_override="auto"):
    from avatar_enhance import PHOTO_PROMPT
    if profile_override != "auto" and profile_override not in profiles.PROFILES:
        raise ValueError("Choose an available enhancement prompt profile")
    selected = (profile_override if profile_override != "auto" else
                profiles.select(analysis) if analysis.get("status") in {"complete", "partial"} else "general")
    profile = profiles.PROFILES[selected]
    automatic = profile_override == "auto"
    focus = profiles.focus_rules(analysis) if automatic else []
    parts = [profile["prompt"]]
    custom = str(instructions).strip()
    if custom and custom != PHOTO_PROMPT:
        parts.append("Additional reconstruction instructions: " + custom)
    parts.extend(profiles.FOCUS_RULES[key] for key in focus)
    return {"version": VERSION, "prompt_version": profiles.PROMPT_VERSION, "analysis": analysis, "profile_id": selected, "profile_name": profile["name"],
            "selection": "automatic" if profile_override == "auto" else "manual_override",
            "user_instructions": custom if custom != PHOTO_PROMPT else "",
            "focus_rules": focus, "prompt": " ".join(parts), "automatic": automatic}


def review_texture(image, analysis, mock=False, check=lambda: None, ask=None):
    if analysis.get("version") in {"scene-profiles-v1", VERSION}:
        return review_content(image, analysis, mock, check, ask)
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


def review_content(image, analysis, mock=False, check=lambda: None, ask=None):
    """Compare scene observations, not skin categories; this is advisory, not proof."""
    if mock:
        return {"status": "mock", "reasons": [], "mock_not_quality_evidence": True}
    if analysis.get("status") not in {"complete", "partial"}:
        return {"status": "uncertain", "reasons": ["Source classification was unavailable"]}
    import avatar_vision
    photo = avatar_vision.prepare_image(image)
    focused = avatar_vision.prepare_image(_crop(image, analysis.get("subject_box")))
    ask = ask or avatar_vision._ask_question
    result = {"status": "consistent", "reasons": [], "answers": {},
              "method": "scene-category comparison; not proof of identity, text accuracy or realism"}
    expected = {}
    if analysis.get("medium") in {"artwork", "document"}:
        expected[analysis["medium"] + "_visible"] = True
    elif analysis.get("people_role") == "main_subject":
        expected["people_main_subject"] = True
        if analysis.get("people_count") in {"single", "multiple"}:
            if analysis.get("subject_box") and analysis["people_count"] == "single":
                expected["single_face_crop"] = True
            else:
                expected["multiple_people"] = analysis["people_count"] == "multiple"
    elif analysis.get("scene") in {"landscape", "architecture", "product", "animal", "food"}:
        expected[analysis["scene"] + "_visible"] = True
    if analysis.get("focus") in {"in_focus", "soft_subject"}:
        expected["subject_blurred"] = analysis["focus"] == "soft_subject"
    observed = {}
    try:
        for key in expected:
            use_crop = analysis.get("subject_box") and key in {"single_face_crop", "subject_blurred"}
            observed[key] = _probe(focused if use_crop else photo, key, ask, result["answers"], check)
        for key, value in expected.items():
            if observed[key] is not None and observed[key] != value:
                result["reasons"].append("Source observation changed: " + key.replace("_", " "))
    except RuntimeError as exc:
        from web_jobs import Cancelled
        if isinstance(exc, Cancelled):
            raise
        return {**result, "status": "uncertain", "error": str(exc)[:500], "reasons": ["Content review unavailable"]}
    if not observed or any(value is None for value in observed.values()) or analysis["status"] != "complete":
        result["status"] = "uncertain"
    if result["reasons"]:
        result["status"] = "needs_review"
    return result


def needs_review(diagnostics):
    data = diagnostics or {}
    return (data.get("texture_review", {}).get("status") in {"uncertain", "needs_review"}
            or data.get("content_review", {}).get("status") in {"uncertain", "needs_review"}
            or data.get("enhancement_review", {}).get("status") == "needs_review")


def output_status(diagnostics, passed=False):
    if needs_review(diagnostics):
        return "needs_review"
    if (diagnostics or {}).get("near_noop"):
        return "no_change"
    return "passed" if passed else "needs_review"
