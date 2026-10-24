"""Persistent, optional avatar onboarding using the existing generation engines."""

import os
import uuid

from PIL import Image

import avatar_studio as studio


PRESETS = {
    "Studio creator": {
        "face": "Adult woman, age 28, dark shoulder-length hair, brown eyes, oval face, natural skin texture, understated makeup, friendly neutral expression.",
        "body": "Natural balanced proportions, relaxed upright stance, plain fitted white T-shirt, straight-leg dark jeans, simple sneakers.",
        "scenes": [
            "Waist-up editorial portrait beside a large studio window, soft daylight, plain white T-shirt, relaxed expression, natural skin texture.",
            "Full-body lifestyle photograph on a quiet leafy city street, white T-shirt, dark jeans and sneakers, relaxed standing pose, natural daylight.",
        ],
    },
    "Fashion editorial": {
        "face": "Adult woman, age 30, short textured dark hair, defined cheekbones, brown eyes, natural skin texture, minimal makeup.",
        "body": "Tall-looking balanced proportions, neutral standing pose, fitted black top, straight-leg trousers, flat shoes, hands visible.",
        "scenes": [
            "Three-quarter fashion portrait in a white photography studio, tailored charcoal jacket over a black top, soft editorial lighting.",
            "Full-body fashion photograph on a modern pedestrian street, charcoal blazer and straight-leg trousers, natural walking pose, overcast daylight.",
        ],
    },
    "Lifestyle creator": {
        "face": "Adult man, age 29, short wavy brown hair, brown eyes, light stubble, natural skin texture, warm relaxed expression.",
        "body": "Natural medium build, relaxed standing pose, plain grey T-shirt, dark jeans and white sneakers, hands visible.",
        "scenes": [
            "Waist-up lifestyle portrait beside a cafe window, grey T-shirt, relaxed expression, soft natural light.",
            "Full-body lifestyle photograph in a green urban park, grey T-shirt and dark jeans, casual standing pose, gentle afternoon light.",
        ],
    },
}


def create(save_dir, name, preset="Studio creator", mode="Guided"):
    if preset not in PRESETS:
        raise ValueError("Choose an available starting style.")
    avatar = studio.create_avatar(save_dir, name)
    defaults = PRESETS[preset]
    avatar["creation_mode"] = "guided" if mode == "Guided" else "custom"
    avatar["guide"] = {
        "stage": "face",
        "preset": preset,
        "face_prompt": defaults["face"],
        "body_prompt": defaults["body"],
        "scenes": list(defaults["scenes"]),
        "drafts": {},
    }
    if mode == "Guided":
        avatar["current_step"] = "guide"
    return studio.save_avatar(save_dir, avatar)


def state(avatar):
    return (avatar or {}).get("guide") or {}


def active(avatar):
    return bool(avatar and avatar.get("creation_mode") == "guided")


def stage(avatar):
    return state(avatar).get("stage", "face")


def next_label(avatar):
    return {
        "face": "Generate face", "face_review": "Use this face & next",
        "body": "Generate body", "body_review": "Use this body & create 2 images",
        "starters": "Continue starter images", "complete": "Complete",
    }.get(stage(avatar), "Next")


def draft(avatar, kind):
    drafts = state(avatar).get("drafts", {})
    path = drafts.get(kind)
    if kind not in drafts and (avatar or {}).get(f"{kind}_locked"):
        path = avatar.get(f"{kind}_image")
    return path if path and os.path.isfile(path) else None


def options(avatar):
    guide = state(avatar)
    defaults = PRESETS.get(guide.get("preset"), PRESETS["Studio creator"])
    scenes = guide.get("scenes") or defaults["scenes"]
    return (
        guide.get("face_prompt", defaults["face"]),
        guide.get("body_prompt", defaults["body"]),
        scenes[0], scenes[1],
    )


def _save(save_dir, avatar, **changes):
    avatar["guide"] = {**state(avatar), **changes}
    avatar["current_step"] = "guide"
    return studio.save_avatar(save_dir, avatar)


def save_options(save_dir, avatar_id, face, body, scene_one, scene_two):
    avatar = studio.load_avatar(save_dir, avatar_id)
    defaults = options(avatar)
    values = [str(value or "").strip() or defaults[index]
              for index, value in enumerate((face, body, scene_one, scene_two))]
    return _save(save_dir, avatar, face_prompt=values[0], body_prompt=values[1], scenes=values[2:])


def save_draft(save_dir, avatar, kind, image):
    pil = studio._coerce_image(image)
    if pil is None:
        raise ValueError(f"Add or generate a {kind} image first.")
    path = studio.avatar_path(save_dir, avatar["id"]) / kind / f"{kind}_draft.png"
    temporary = path.with_name(f"{kind}_draft_{uuid.uuid4().hex}.tmp")
    pil.save(temporary, format="PNG")
    os.replace(temporary, path)
    drafts = {**state(avatar).get("drafts", {}), kind: str(path)}
    return _save(save_dir, avatar, drafts=drafts, stage=f"{kind}_review")


def use_custom(save_dir, avatar_id):
    avatar = studio.load_avatar(save_dir, avatar_id)
    avatar["creation_mode"] = "custom"
    avatar["current_step"] = (
        "console" if avatar.get("body_locked") and avatar.get("face_locked")
        else "body" if avatar.get("face_locked") else "face"
    )
    return studio.save_avatar(save_dir, avatar)


def clear_draft(save_dir, avatar_id):
    avatar = studio.load_avatar(save_dir, avatar_id)
    kind = "face" if stage(avatar).startswith("face") else "body"
    drafts = {**state(avatar).get("drafts", {}), kind: None}
    return _save(save_dir, avatar, drafts=drafts, stage=kind)


def back(save_dir, avatar_id):
    avatar = studio.load_avatar(save_dir, avatar_id)
    current = stage(avatar)
    if current in {"body", "body_review"}:
        target = "face_review"
    elif current in {"starters", "complete"}:
        target = "body_review"
    else:
        target = "face_review" if draft(avatar, "face") else "face"
    return _save(save_dir, avatar, stage=target)


def advance(save_dir, avatar_id, image, generate, validate, dev_mode=False):
    """Yield saved progress; completed images survive a failed or interrupted retry."""
    avatar = studio.load_avatar(save_dir, avatar_id)
    if not active(avatar):
        raise ValueError("Select a guided avatar first.")
    current = stage(avatar)
    if current == "complete":
        yield avatar, "Your avatar and two starter images are ready."
        return

    if current in {"face", "body"}:
        kind = current
        yield avatar, f"Preparing {kind} draft"
        candidate = studio._coerce_image(image) if image is not None else generate(
            kind, state(avatar)[f"{kind}_prompt"], avatar
        )
        avatar = save_draft(save_dir, avatar, kind, candidate)
        yield avatar, f"Review your {kind} before continuing."
        return

    if current in {"face_review", "body_review"}:
        kind = current.split("_")[0]
        candidate = image if image is not None else draft(avatar, kind)
        if candidate is None:
            raise ValueError(f"The {kind} draft is missing. Upload or generate it again.")
        yield avatar, f"Analyzing and saving {kind} identity"
        studio.lock_reference(save_dir, avatar_id, kind, candidate, dev_mode=dev_mode)
        avatar = studio.load_avatar(save_dir, avatar_id)
        avatar = _save(save_dir, avatar, stage="body" if kind == "face" else "starters")
        if kind == "face":
            if avatar.get("body_locked") and draft(avatar, "body"):
                avatar = _save(save_dir, avatar, stage="body_review")
                yield avatar, "Your saved body reference is ready for review."
                return
            yield avatar, "Creating a body draft with your confirmed face"
            candidate = generate("body", state(avatar)["body_prompt"], avatar)
            avatar = save_draft(save_dir, avatar, "body", candidate)
            yield avatar, "Review your full-body reference before continuing."
            return

    if not avatar.get("face_locked") or not avatar.get("body_locked"):
        raise ValueError("Confirm the face and body before creating starter images.")
    import avatar_identity
    avatar = avatar_identity.freeze(save_dir, avatar)
    identity = studio.identity_fingerprint(avatar)
    for index, scene in enumerate(state(avatar)["scenes"]):
        item_id = f"guided-starter-{index + 1}"
        items = studio.load_gallery_items(save_dir, avatar_id)
        item = next((row for row in items if row.get("id") == item_id), None)
        identity_matches = item and item.get("identity_fingerprint", identity) == identity and item.get("scene", scene) == scene
        if item and identity_matches and item.get("status") in {"passed", "needs_review"} and os.path.isfile(item.get("generated_path") or ""):
            continue
        if not item:
            item = studio.record_gallery_item(save_dir, avatar_id, {
                "id": item_id, "origin": "guided", "theme": "Starter collection",
                "aspect": "864x1152 (3:4)", "status": "generating", "attempt": 0,
                "identity_fingerprint": identity,
                "scene": scene,
            })
        prompt = studio.build_generation_prompt(avatar, scene, "Identity Strict")
        for retry in range(3):
            pending = identity_matches and item.get("status") == "checking" and os.path.isfile(item.get("generated_path") or "")
            if pending:
                prompt = item.get("prompt") or prompt
                with Image.open(item["generated_path"]) as saved:
                    generated = saved.convert("RGB")
            else:
                attempt = int(item.get("attempt", 0)) + 1
                yield avatar, f"Creating starter image {index + 1} of 2 (attempt {attempt})"
                generated = generate("starter", prompt, avatar)
                path = studio.save_gallery_image(save_dir, avatar_id, item_id, generated, attempt)
                item = studio.update_gallery_item(
                    save_dir, avatar_id, item_id, generated_path=path, prompt=prompt,
                    attempt=attempt, status="checking",
                    identity_fingerprint=identity,
                    identity_version=avatar["identity_version"], recipe=avatar_identity.recipe(avatar),
                    scene=scene,
                )
            yield studio.load_avatar(save_dir, avatar_id), f"Checking starter image {index + 1} of 2"
            validation = validate(generated, avatar, prompt)
            passed = validation.get("pass") is True
            item = studio.update_gallery_item(
                save_dir, avatar_id, item_id, validation=validation,
                status="passed" if passed else "needs_review" if retry == 2 or validation.get("review_only") else "retrying",
            )
            if passed or validation.get("review_only"):
                break
            correction = validation.get("repair_instruction") or "; ".join(validation.get("reasons") or [])
            prompt += f" Correct this issue while preserving the identity references: {correction}"
        avatar = studio.load_avatar(save_dir, avatar_id)
        yield avatar, f"Starter image {index + 1} saved"
    avatar = _save(save_dir, studio.load_avatar(save_dir, avatar_id), stage="complete")
    yield avatar, "Your avatar and two starter images are ready. Review the results below."
