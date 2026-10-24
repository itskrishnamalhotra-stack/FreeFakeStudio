"""One explainable validator for guided, console, gallery, repair and enhancement."""

import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageFilter

import avatar_face
import avatar_identity as identity
import avatar_studio as studio
import image_review


def profile(root, avatar):
    version = avatar.get("identity_version")
    return identity._read(identity._folder(root, avatar["id"]) / "profiles" / f"{version}.json", {})


def save_profile(root, avatar, changes):
    if set(changes) - {"policies", "sequential", "face", "caption_mode"}:
        raise ValueError("Unknown quality setting.")
    if "policies" in changes:
        if any(key not in studio.IDENTITY_SPEC_KEYS or value not in {"required", "important", "flexible"} for key, value in changes["policies"].items()):
            raise ValueError("Unknown appearance attribute or consistency level.")
    data = {**profile(root, avatar), **changes, "identity_version": avatar["identity_version"], "updated_at": studio._now()}
    studio._atomic_json(identity._folder(root, avatar["id"]) / "profiles" / f"{avatar['identity_version']}.json", data)
    return data


def attribute_checks(image, specs, part, ask, policies=None, visible=True):
    checks = []
    appearance = set(studio.IDENTITY_SPEC_KEYS)
    for row in (specs or {}).get("answers", []):
        if row["key"] not in appearance:
            continue
        expected = str(row.get("answer", "")).strip()
        if not expected or re.search(r"uncertain|not confidently|unknown|not visible", expected, re.I):
            continue
        strictness = (policies or {}).get(row["key"], row.get("strictness", "flexible"))
        verdict = "not_assessed"
        if visible and strictness in {"required", "important"}:
            answer = ask(image, f"Inspect only the visible {part}. Does {row['key'].replace('_', ' ')} match '{expected}'? "
                         "Answer MATCH, MISMATCH, NOT_VISIBLE, or UNCLEAR. Use NOT_VISIBLE if this detail cannot be seen.")
            normalized = str(answer).strip().upper().replace(" ", "_")
            verdict = "mismatch" if re.search(r"\bMISMATCH\b", normalized) else "not_visible" if "NOT_VISIBLE" in normalized else "match" if re.search(r"\bMATCH\b", normalized) else "unclear"
        checks.append({"part": part, "key": row["key"], "expected": expected,
                       "policy": strictness, "result": verdict})
    return checks


def source_preservation(source, result):
    first = source.convert("RGB").resize((128, 128)).filter(ImageFilter.GaussianBlur(3))
    second = result.convert("RGB").resize((128, 128)).filter(ImageFilter.GaussianBlur(3))
    difference = float(np.abs(np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)).mean() / 255)
    aspect_error = abs(source.width / source.height - result.width / result.height) / (source.width / source.height)
    return {"status": "passed" if difference <= 0.10 and aspect_error <= 0.01 else "needs_review",
            "detail_evidence": image_review.detail_evidence(source, result),
            "low_frequency_difference": difference, "aspect_error": aspect_error,
            "method": "low-frequency pixel heuristic, not proof of pose or expression preservation"}


def validate(root, image, avatar, prompt, source=None, basic=None, ask=None, face_compare=None):
    import avatar_vision
    saved_profile = avatar.get("_quality_profile", profile(root, avatar))
    if saved_profile.get("identity_version", avatar.get("identity_version")) != avatar.get("identity_version"):
        raise ValueError("Quality policy belongs to a different identity version.")
    specs = {part: studio.load_specs(avatar.get(part + "_specs")) for part in ("face", "body")}
    baseline = basic(image, specs["face"], specs["body"], prompt) if basic else avatar_vision.validate_avatar_generation(image, specs["face"], specs["body"], prompt)
    if ask is None:
        ask = lambda photo, question: avatar_vision._ask_question(photo, question, max_tokens=32, context="attribute-consistency")
    compare = face_compare or avatar_face.compare
    anchors = [avatar["face_image"]] + [row["path"] for row in avatar.get("_identity_pack", []) if not row["view"].startswith("body")]
    facial = compare(image, anchors, saved_profile.get("face"))
    attributes = []
    for part in ("face", "body"):
        attributes.extend(attribute_checks(image, specs[part], part, ask, saved_profile.get("policies"), baseline.get(part + "_test_applicable") is True))
    reasons = list(baseline.get("reasons") or [])
    face_applicable = baseline.get("face_test_applicable") is True
    if face_applicable and facial["status"] != "passed":
        reasons.append("Direct identity comparison: " + facial.get("reason", facial["status"]))
    for check in attributes:
        if check["policy"] in {"required", "important"} and check["result"] in {"mismatch", "unclear", "not_visible"}:
            reasons.append(f"{check['part']} {check['key']}: {check['result']}")
    preservation = None
    if source is not None:
        preservation = source_preservation(source, image)
        source_face = compare(image, [source], saved_profile.get("face"))
        preservation["source_face"] = source_face
        if preservation["status"] != "passed" or (face_applicable and source_face["status"] != "passed"):
            reasons.append("Enhancement source preservation needs review")
    if baseline.get("pass") is not True and not reasons:
        reasons.append("Scene or anatomy validation did not pass")
    review_only = bool(reasons) and facial["status"] in {"uncalibrated", "unavailable", "not_assessed", "uncertain"} and all(reason.startswith("Direct identity comparison:") for reason in reasons)
    actionable = [] if review_only else reasons
    return {**baseline, "pass": not reasons, "reasons": reasons, "repair_instruction": "; ".join(actionable),
            "review_only": review_only,
            "identity_version": avatar.get("identity_version"), "direct_identity": facial,
            "attribute_checks": attributes, "source_preservation": preservation,
            "score_source": "basic scene/anatomy checks only; identity and attributes are separate verdicts",
            "quality_policy": hashlib.sha256(json.dumps(saved_profile, sort_keys=True).encode()).hexdigest()[:16],
            "face_identity_assessed": face_applicable and facial["status"] == "passed"}
