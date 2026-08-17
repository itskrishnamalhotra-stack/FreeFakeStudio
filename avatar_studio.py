import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


AVATAR_VERSION = 1
MAX_EXTRA_REFERENCES = 2


FACE_SPEC_QUESTIONS = [
    ("usable_face_reference", "Is this a usable adult face reference?", "required"),
    ("face_visibility", "Is the face clearly visible?", "required"),
    ("adult_presenting", "Is the subject adult-presenting?", "required"),
    ("face_angle", "What is the face angle?", "important"),
    ("face_shape", "What is the face shape?", "important"),
    ("skin_tone", "What is the skin tone?", "important"),
    ("hair_color", "What is the hair color?", "required"),
    ("hair_length", "What is the hair length?", "important"),
    ("hairstyle", "What is the hairstyle?", "important"),
    ("eye_color", "What is the eye color, if visible?", "ignore_if_not_visible"),
    ("eyebrows", "What is the eyebrow thickness and shape?", "flexible"),
    ("nose_shape", "What is the nose shape?", "flexible"),
    ("lip_fullness", "What is the lip fullness?", "important"),
    ("cheek_fullness", "What is the cheek fullness?", "important"),
    ("jawline", "What is the jawline shape?", "important"),
    ("chin_shape", "What is the chin shape?", "flexible"),
    ("visible_marks", "Are freckles, moles, scars, or other marks visible?", "flexible"),
    ("makeup_level", "What is the makeup level?", "flexible"),
    ("expression", "What is the expression?", "flexible"),
    ("uncertain_details", "Which details are uncertain?", "flexible"),
]


BODY_SPEC_QUESTIONS = [
    ("usable_body_reference", "Is this a usable adult full-body reference?", "required"),
    ("body_visibility", "Is the full body visible enough?", "required"),
    ("adult_presenting", "Is the subject adult-presenting?", "required"),
    ("face_visible", "Is the face visible?", "ignore_if_not_visible"),
    ("body_framing", "What is the body framing?", "required"),
    ("body_build", "What is the body type or build?", "important"),
    ("shoulder_width", "What is the shoulder width?", "flexible"),
    ("waist_shape", "What is the waist shape?", "important"),
    ("hip_shape", "What is the hip shape?", "important"),
    ("height_impression", "What is the overall height impression?", "flexible"),
    ("pose", "What is the pose?", "flexible"),
    ("clothing", "What clothing is visible?", "flexible"),
    ("occlusion", "Is the body heavily occluded?", "required"),
    ("hands_visible", "Are hands visible?", "ignore_if_not_visible"),
    ("feet_visible", "Are feet visible?", "ignore_if_not_visible"),
    ("skin_tone_consistency", "Does skin tone appear consistent with the face reference?", "important"),
    ("uncertain_details", "Which details are uncertain?", "flexible"),
]


def avatar_root(save_dir):
    root = Path(save_dir) / "avatars"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_slug(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "avatar").strip()).strip("-")
    return (value or "avatar")[:48].lower()


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def avatar_path(save_dir, avatar_id):
    return avatar_root(save_dir) / avatar_id


def load_avatar(save_dir, avatar_id):
    if not avatar_id:
        return None
    path = avatar_path(save_dir, avatar_id) / "avatar.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_avatar(save_dir, avatar):
    avatar = dict(avatar)
    avatar["updated_at"] = _now()
    _atomic_json(avatar_path(save_dir, avatar["id"]) / "avatar.json", avatar)
    return avatar


def create_avatar(save_dir, name):
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("Give the avatar a name first.")
    avatar_id = f"{_safe_slug(clean_name)}-{uuid.uuid4().hex[:8]}"
    root = avatar_path(save_dir, avatar_id)
    for child in ("face", "body", "chat/images", "gallery/images", "gallery/references", "debug"):
        (root / child).mkdir(parents=True, exist_ok=True)
    avatar = {
        "version": AVATAR_VERSION,
        "id": avatar_id,
        "name": clean_name[:80],
        "created_at": _now(),
        "updated_at": _now(),
        "current_step": "face",
        "face_locked": False,
        "body_locked": False,
        "face_image": None,
        "body_image": None,
        "face_specs": None,
        "body_specs": None,
        "identity_sheet": None,
        "chat_history": [],
        "gallery_count": 0,
        "gallery_manifest": "gallery/gallery.json",
    }
    return save_avatar(save_dir, avatar)


def list_avatars(save_dir):
    avatars = []
    root = avatar_root(save_dir)
    for metadata_path in root.glob("*/avatar.json"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                avatar = json.load(handle)
            avatars.append(avatar)
        except Exception:
            continue
    avatars.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return avatars


def avatar_choices(save_dir):
    avatars = list_avatars(save_dir)
    return [(avatar.get("name", avatar["id"]), avatar["id"]) for avatar in avatars]


def selected_or_first(save_dir, avatar_id=None):
    if avatar_id:
        try:
            return load_avatar(save_dir, avatar_id)
        except Exception:
            pass
    avatars = list_avatars(save_dir)
    return avatars[0] if avatars else None


def _coerce_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, dict):
        value = image.get("path") or image.get("name")
        if value:
            return Image.open(value).convert("RGB")
    return Image.fromarray(image).convert("RGB")


def save_reference_image(save_dir, avatar_id, kind, image):
    if kind not in ("face", "body"):
        raise ValueError("kind must be face or body")
    pil_image = _coerce_image(image)
    if pil_image is None:
        raise ValueError(f"Add or generate a {kind} image first.")
    path = avatar_path(save_dir, avatar_id) / kind / f"{kind}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)
    return str(path)


def save_chat_image(save_dir, avatar_id, image):
    pil_image = _coerce_image(image)
    if pil_image is None:
        raise ValueError("No generated image to save.")
    folder = avatar_path(save_dir, avatar_id) / "chat" / "images"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    pil_image.save(path)
    return str(path)


def gallery_manifest_path(save_dir, avatar_id):
    return avatar_path(save_dir, avatar_id) / "gallery" / "gallery.json"


def load_gallery_items(save_dir, avatar_id):
    path = gallery_manifest_path(save_dir, avatar_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def save_gallery_items(save_dir, avatar_id, items):
    _atomic_json(gallery_manifest_path(save_dir, avatar_id), list(items))
    avatar = load_avatar(save_dir, avatar_id)
    avatar["gallery_count"] = len([item for item in items if item.get("generated_path")])
    save_avatar(save_dir, avatar)
    return list(items)


def record_gallery_item(save_dir, avatar_id, item):
    items = load_gallery_items(save_dir, avatar_id)
    payload = dict(item)
    payload.setdefault("id", uuid.uuid4().hex[:12])
    payload.setdefault("created_at", _now())
    payload["updated_at"] = _now()
    items.append(payload)
    save_gallery_items(save_dir, avatar_id, items)
    return payload


def update_gallery_item(save_dir, avatar_id, item_id, **changes):
    items = load_gallery_items(save_dir, avatar_id)
    updated = None
    for item in items:
        if item.get("id") == item_id:
            item.update(changes)
            item["updated_at"] = _now()
            updated = item
            break
    if updated is None:
        raise KeyError(f"Gallery item {item_id} was not found.")
    save_gallery_items(save_dir, avatar_id, items)
    return updated


def gallery_item_by_index(save_dir, avatar_id, index):
    items = [item for item in load_gallery_items(save_dir, avatar_id) if item.get("generated_path")]
    if index is None or not 0 <= int(index) < len(items):
        return None
    return items[int(index)]


def save_gallery_image(save_dir, avatar_id, item_id, image, attempt=1):
    pil_image = _coerce_image(image)
    if pil_image is None:
        raise ValueError("No gallery image to save.")
    folder = avatar_path(save_dir, avatar_id) / "gallery" / "images"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"gallery_{item_id}_a{int(attempt)}.png"
    pil_image.save(path)
    return str(path)


def save_gallery_reference(save_dir, avatar_id, image, label=None):
    pil_image = _coerce_image(image)
    if pil_image is None:
        raise ValueError("No gallery reference image to save.")
    folder = avatar_path(save_dir, avatar_id) / "gallery" / "references"
    folder.mkdir(parents=True, exist_ok=True)
    name = _safe_slug(label or "reference")[:24]
    path = folder / f"{name}_{uuid.uuid4().hex[:10]}.jpg"
    pil_image.save(path, "JPEG", quality=94)
    return str(path)


def reference_images(save_dir, avatar, extra_images=None):
    images = []
    for key in ("face_image", "body_image"):
        path = avatar.get(key) if avatar else None
        if path and os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
    for image in (extra_images or [])[:MAX_EXTRA_REFERENCES]:
        pil_image = _coerce_image(image)
        if pil_image is not None:
            images.append(pil_image)
    return images[:4]


def build_generation_prompt(avatar, user_prompt, mode="Identity Strict"):
    face_specs = summarize_specs(avatar.get("face_specs"))
    body_specs = summarize_specs(avatar.get("body_specs"))
    mode_note = {
        "Identity Strict": "Preserve face identity and body proportions from the locked references.",
        "Outfit Focus": "Use extra references for outfit and styling only, not identity.",
        "Pose Focus": "Use extra references for pose/composition while preserving identity.",
        "Scene Focus": "Use extra references for setting, mood, and lighting while preserving identity.",
        "Group Image": "Keep the avatar distinct from other people in extra references.",
    }.get(mode, "Preserve the avatar identity from the locked references.")
    return (
        "Create an image of the saved avatar. "
        f"{mode_note} "
        "Use the face and body references as identity anchors. "
        f"Face notes: {face_specs}. "
        f"Body notes: {body_specs}. "
        f"User request: {(user_prompt or '').strip()}"
    ).strip()


def _mock_specs(kind, image):
    width, height = image.size if image else (0, 0)
    questions = FACE_SPEC_QUESTIONS if kind == "face" else BODY_SPEC_QUESTIONS
    defaults = {
        "usable_face_reference": "yes, clear face reference",
        "face_visibility": "face is visible",
        "usable_body_reference": "yes, usable body reference",
        "body_visibility": "body is visible enough",
        "adult_presenting": "adult-presenting",
        "face_angle": "front or slight three-quarter angle",
        "face_shape": "balanced oval face shape",
        "skin_tone": "medium skin tone",
        "hair_color": "dark hair",
        "hair_length": "medium to long",
        "hairstyle": "loose styled hair",
        "eye_color": "not confidently visible",
        "body_framing": "full or mostly full body",
        "body_build": "balanced build",
        "waist_shape": "defined waist",
        "hip_shape": "balanced hips",
        "occlusion": "no major occlusion",
        "skin_tone_consistency": "appears consistent",
        "uncertain_details": "mock analysis; run in Colab for model-based answers",
    }
    answers = []
    for key, question, strictness in questions:
        answers.append({
            "key": key,
            "question": question,
            "answer": defaults.get(key, "not confidently visible"),
            "confidence": 0.82 if "uncertain" not in key else 0.35,
            "strictness": strictness,
        })
    return {
        "kind": kind,
        "analyzer": "mock-smolvlm-dev",
        "created_at": _now(),
        "image_size": [width, height],
        "answers": answers,
    }


def analyze_reference_image(kind, image, dev_mode=True):
    pil_image = _coerce_image(image)
    if pil_image is None:
        raise ValueError(f"Add or generate a {kind} image first.")
    if dev_mode:
        return _mock_specs(kind, pil_image)
    questions = FACE_SPEC_QUESTIONS if kind == "face" else BODY_SPEC_QUESTIONS
    try:
        import avatar_vision

        specs = avatar_vision.analyze_reference_image(kind, pil_image, questions)
        specs["created_at"] = _now()
        return specs
    except Exception as exc:
        if os.environ.get("FFS_AVATAR_ANALYZER_STRICT", "1").lower() not in ("0", "false", "no"):
            raise
        specs = _mock_specs(kind, pil_image)
        specs["analyzer"] = "mock-after-vision-error"
        specs["analyzer_error"] = str(exc)
        return specs


def lock_reference(save_dir, avatar_id, kind, image, dev_mode=True):
    avatar = load_avatar(save_dir, avatar_id)
    image_path = save_reference_image(save_dir, avatar_id, kind, image)
    specs = analyze_reference_image(kind, image_path, dev_mode=dev_mode)
    spec_path = avatar_path(save_dir, avatar_id) / kind / f"{kind}_specs.json"
    _atomic_json(spec_path, specs)
    avatar[f"{kind}_image"] = image_path
    avatar[f"{kind}_specs"] = str(spec_path)
    avatar[f"{kind}_locked"] = True
    avatar["current_step"] = "body" if kind == "face" else "console"
    return save_avatar(save_dir, avatar), specs


def load_specs(path):
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def summarize_specs(path_or_payload, limit=6):
    specs = path_or_payload if isinstance(path_or_payload, dict) else load_specs(path_or_payload)
    if not specs:
        return "not analyzed yet"
    parts = []
    for item in specs.get("answers", []):
        if item.get("strictness") in ("required", "important"):
            parts.append(f"{item.get('key')}: {item.get('answer')}")
        if len(parts) >= limit:
            break
    return "; ".join(parts) if parts else "analysis available"


def specs_html(path_or_payload, title):
    specs = path_or_payload if isinstance(path_or_payload, dict) else load_specs(path_or_payload)
    if not specs:
        return f'<div class="ffs-avatar-specs"><strong>{title}</strong><span>Not analyzed yet</span></div>'
    rows = []
    for item in specs.get("answers", [])[:10]:
        rows.append(
            '<div class="ffs-avatar-spec-row">'
            f'<span>{item.get("question", "")}</span>'
            f'<strong>{item.get("answer", "")}</strong>'
            f'<em>{item.get("strictness", "")} / {int(float(item.get("confidence", 0)) * 100)}%</em>'
            '</div>'
        )
    return f'<div class="ffs-avatar-specs"><strong>{title}</strong>{"".join(rows)}</div>'


def status_html(avatar):
    if not avatar:
        return (
            '<div class="ffs-avatar-empty">'
            '<strong>No avatar selected</strong>'
            '<span>Create an avatar to start locking face and body references.</span>'
            '</div>'
        )
    face = "locked" if avatar.get("face_locked") else "needed"
    body = "locked" if avatar.get("body_locked") else "needed"
    step = avatar.get("current_step", "face")
    return (
        '<div class="ffs-avatar-status">'
        f'<strong>{avatar.get("name", "Avatar")}</strong>'
        f'<span>Step: {step}</span>'
        f'<small>Face {face} / Body {body} / Gallery {avatar.get("gallery_count", 0)} images</small>'
        '</div>'
    )


def make_dev_reference(kind, details=""):
    width, height = (768, 768) if kind == "face" else (768, 1024)
    bg = (34, 39, 52) if kind == "face" else (30, 44, 48)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        mix = y / max(1, height - 1)
        draw.line([(0, y), (width, y)], fill=(int(bg[0] + 30 * mix), int(bg[1] + 22 * mix), int(bg[2] + 45 * mix)))
    if kind == "face":
        draw.ellipse((width * 0.28, height * 0.18, width * 0.72, height * 0.72), fill=(178, 132, 112), outline=(230, 200, 180), width=5)
        draw.ellipse((width * 0.39, height * 0.40, width * 0.44, height * 0.45), fill=(35, 35, 45))
        draw.ellipse((width * 0.56, height * 0.40, width * 0.61, height * 0.45), fill=(35, 35, 45))
        draw.arc((width * 0.42, height * 0.52, width * 0.58, height * 0.61), 0, 180, fill=(100, 45, 62), width=4)
    else:
        draw.ellipse((width * 0.40, height * 0.08, width * 0.60, height * 0.25), fill=(178, 132, 112), outline=(230, 200, 180), width=4)
        draw.rounded_rectangle((width * 0.32, height * 0.26, width * 0.68, height * 0.70), radius=60, fill=(82, 117, 140), outline=(180, 214, 220), width=4)
        draw.line((width * 0.38, height * 0.70, width * 0.32, height * 0.95), fill=(178, 132, 112), width=28)
        draw.line((width * 0.62, height * 0.70, width * 0.68, height * 0.95), fill=(178, 132, 112), width=28)
    label = f"DEV {kind.upper()} REFERENCE"
    draw.rectangle((24, height - 96, width - 24, height - 24), fill=(10, 12, 18))
    draw.text((42, height - 78), label, fill=(245, 247, 255))
    if details:
        draw.text((42, height - 54), details[:90], fill=(180, 190, 205))
    return image


def append_avatar_chat(save_dir, avatar_id, role, content, image_path=None, metadata=None):
    avatar = load_avatar(save_dir, avatar_id)
    history = list(avatar.get("chat_history") or [])
    history.append({
        "time": _now(),
        "role": role,
        "content": content,
        "image_path": image_path,
        "metadata": metadata or {},
    })
    avatar["chat_history"] = history[-100:]
    return save_avatar(save_dir, avatar)


def validate_generated_image(image, avatar, requested_prompt, dev_mode=True):
    pil_image = _coerce_image(image)
    if pil_image is None:
        raise ValueError("No generated image to validate.")
    if dev_mode:
        return {
            "pass": True,
            "score": 96,
            "adult_person_visible": True,
            "face_test_applicable": True,
            "face_identity_match": True,
            "body_test_applicable": True,
            "body_identity_match": True,
            "request_match": True,
            "anatomy_ok": True,
            "reasons": [],
            "repair_instruction": "",
            "validator": "mock-smolvlm-dev",
        }
    import avatar_vision

    return avatar_vision.validate_avatar_generation(
        pil_image,
        load_specs(avatar.get("face_specs")),
        load_specs(avatar.get("body_specs")),
        requested_prompt,
    )
