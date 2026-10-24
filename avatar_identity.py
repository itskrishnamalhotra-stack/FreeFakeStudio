"""Immutable identity snapshots, reviewed reference packs and generation provenance."""

import hashlib
import json
import os
import shutil
from importlib import metadata
from pathlib import Path

from PIL import Image

import avatar_studio as studio


VIEWS = ("front", "three-quarter", "profile", "smile", "body-front", "body-side")
ROLES = ("Outfit", "Pose", "Scene", "Other person")


def _read(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Identity data cannot be read: {path}. Preserve it for diagnosis.") from exc


def _folder(root, avatar_id):
    return studio.avatar_path(root, avatar_id) / "identity"


def pack(root, avatar_id):
    return _read(_folder(root, avatar_id) / "pack.json", {"candidates": []})


def invalidate_candidates(root, avatar_id):
    data = pack(root, avatar_id)
    for row in data["candidates"]:
        if row.get("approved"):
            row.update(approved=False, invalidated_at=studio._now(),
                       review_method="Confirmed anchors changed; review against the new identity")
    studio._atomic_json(_folder(root, avatar_id) / "pack.json", data)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(root, avatar):
    if not avatar or not avatar.get("face_locked") or not avatar.get("body_locked"):
        return avatar
    if avatar.get("_identity_frozen"):
        return avatar
    files = {key: avatar.get(key) for key in ("face_image", "body_image", "face_specs", "body_specs")}
    if not all(path and Path(path).is_file() for path in files.values()):
        raise ValueError("Confirmed identity files are missing; restore them before generation.")
    approved = [row for row in pack(root, avatar["id"])["candidates"] if row.get("approved")]
    signature = {"files": {key: _digest(path) for key, path in files.items()},
                 "pack": [{"id": row["id"], "view": row["view"], "sha256": _digest(row["path"])} for row in approved]}
    version = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    folder = _folder(root, avatar["id"]) / "versions" / version
    if not (folder / "identity.json").exists():
        folder.mkdir(parents=True, exist_ok=True)
        pinned = {"version": version, "created_at": studio._now(), "signature": signature, "pack": []}
        for key, path in files.items():
            target = folder / (key + Path(path).suffix)
            shutil.copyfile(path, target)
            pinned[key] = str(target)
        for row in approved:
            target = folder / (row["id"] + ".png")
            shutil.copyfile(row["path"], target)
            pinned["pack"].append({"id": row["id"], "view": row["view"], "path": str(target)})
        studio._atomic_json(folder / "identity.json", pinned)
    if avatar.get("identity_version") != version:
        avatar = studio.save_avatar(root, {**avatar, "identity_version": version})
    return pin(root, avatar, version)


def pin(root, avatar, version=None):
    version = version or avatar.get("identity_version")
    if not version:
        return freeze(root, avatar)
    if not isinstance(version, str) or len(version) != 20 or not all(char in "0123456789abcdef" for char in version):
        raise ValueError("Invalid saved identity version.")
    saved = _read(_folder(root, avatar["id"]) / "versions" / version / "identity.json")
    if not saved:
        raise RuntimeError("This image's immutable identity snapshot is missing.")
    for key, digest in saved["signature"]["files"].items():
        if not Path(saved[key]).is_file() or _digest(saved[key]) != digest:
            raise RuntimeError("An immutable identity reference was changed or lost. Restore its saved version.")
    pack_digests = {row["id"]: row["sha256"] for row in saved["signature"]["pack"]}
    if any(_digest(row["path"]) != pack_digests[row["id"]] for row in saved["pack"]):
        raise RuntimeError("An approved reference snapshot was changed. Restore its saved version.")
    return {**avatar, **{key: saved[key] for key in ("face_image", "body_image", "face_specs", "body_specs")},
            "identity_version": version, "_identity_frozen": True, "_identity_pack": saved["pack"]}


def activate(root, avatar_id, version):
    avatar = studio.load_avatar(root, avatar_id)
    saved = pin(root, avatar, version)
    for key in ("face_image", "body_image", "face_specs", "body_specs"):
        part = key.split("_")[0]
        target = studio.avatar_path(root, avatar_id) / part / (part + ("_specs.json" if key.endswith("specs") else ".png"))
        shutil.copyfile(saved[key], target)
        avatar[key] = str(target)
    data = pack(root, avatar_id)
    approved_ids = {row["id"] for row in saved["_identity_pack"]}
    restored = [{**row, "approved": True, "review_method": "restored approved snapshot"} for row in saved["_identity_pack"]]
    data["candidates"] = restored + [{**row, "approved": False} for row in data["candidates"] if row["id"] not in approved_ids]
    studio._atomic_json(_folder(root, avatar_id) / "pack.json", data)
    avatar.update(identity_version=version, face_locked=True, body_locked=True)
    return studio.save_avatar(root, avatar)


def add_candidate(root, avatar_id, view, image, candidate_id=None):
    if view not in VIEWS:
        raise ValueError("Choose an identity reference view.")
    image = studio._coerce_image(image)
    if image is None or min(image.size) < 128 or image.width * image.height > 20_000_000:
        raise ValueError("Upload a clear reference between 128 pixels per edge and 20 megapixels.")
    data = pack(root, avatar_id)
    if candidate_id and any(row["id"] == candidate_id for row in data["candidates"]):
        return candidate_id
    if len(data["candidates"]) >= 40:
        raise ValueError("Remove an unused candidate before adding more than 40 references.")
    identifier = candidate_id or __import__("uuid").uuid4().hex[:12]
    if len(identifier) != 12 or any(char not in "0123456789abcdef" for char in identifier):
        raise ValueError("Invalid identity candidate identifier.")
    path = _folder(root, avatar_id) / "candidates" / (identifier + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    data["candidates"].append({"id": identifier, "view": view, "path": str(path), "approved": False,
                               "created_at": studio._now()})
    studio._atomic_json(_folder(root, avatar_id) / "pack.json", data)
    return identifier


def review_candidate(root, avatar_id, candidate_id, approved):
    data = pack(root, avatar_id)
    row = next((row for row in data["candidates"] if row["id"] == candidate_id), None)
    if not row:
        raise ValueError("Select a reference candidate first.")
    if approved:
        for other in data["candidates"]:
            if other["view"] == row["view"] and other["id"] != candidate_id:
                other["approved"] = False
    row.update(approved=bool(approved), reviewed_at=studio._now(), review_method="explicit human review")
    studio._atomic_json(_folder(root, avatar_id) / "pack.json", data)
    freeze(root, studio.load_avatar(root, avatar_id))
    return data


def remove_candidate(root, avatar_id, candidate_id):
    data = pack(root, avatar_id)
    row = next((row for row in data["candidates"] if row["id"] == candidate_id), None)
    if not row:
        raise ValueError("Reference not found.")
    data["candidates"] = [item for item in data["candidates"] if item["id"] != candidate_id]
    studio._atomic_json(_folder(root, avatar_id) / "pack.json", data)
    # Versioned copies remain available for old jobs and reproducibility.
    freeze(root, studio.load_avatar(root, avatar_id))
    return data


def select_references(avatar, prompt="", face_view="auto", body_view="auto"):
    import re
    if face_view not in {"auto", "front", "three-quarter", "profile", "smile"} or body_view not in {"auto", "body-front", "body-side"}:
        raise ValueError("Choose a supported face/body reference view.")
    text = str(prompt).lower()
    text = re.sub(r"\b(?:no|not|without|avoid)\s+(?:\w+\s+){0,2}(?:smil\w*|profile(?:\s+view)?|side[- ]view|three[- ]quarter(?:\s+view)?)\b", "", text)
    if face_view == "auto":
        face_view = ("profile" if re.search(r"\b(?:side[- ](?:view|profile)|in profile|profile (?:view|portrait)|(?:left|right)[- ]profile)\b", text)
                     else "three-quarter" if re.search(r"\bthree[- ]quarter\b|3/4", text)
                     else "smile" if re.search(r"\bsmil(?:e|ing)\b", text) else "front")
    if body_view == "auto":
        body_view = "body-side" if face_view == "profile" else "body-front"
    refs = avatar.get("_identity_pack", [])
    face = next((row["path"] for row in refs if row["view"] == face_view), avatar.get("face_image"))
    body = next((row["path"] for row in refs if row["view"] == body_view), avatar.get("body_image"))
    return [path for path in (face, body) if path]


def recipe(avatar, references=None, roles=None):
    versions = {}
    for name in ("torch", "numpy", "transformers", "opencv-python-headless"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"identity_version": avatar.get("identity_version"), "model": "FLUX.2-klein 4B",
            "encoder_mode": os.environ.get("FFS_FLUX_ENCODER_MODE", "official"),
            "encoder_file": os.environ.get("FFS_FLUX_CUSTOM_ENCODER_FILE", "") if os.environ.get("FFS_FLUX_ENCODER_MODE") == "custom" else "qwen_3_4b_fp4_flux2.safetensors",
            "runtime_fingerprint": os.environ.get("FFS_RUNTIME_FINGERPRINT_ID"), "packages": versions,
            "references": list(references or select_references(avatar)), "extra_roles": list(roles or [])}


def role_instructions(roles):
    rules = {"Outfit": "Copy clothing/materials only; never copy this person's face or body shape.",
             "Pose": "Use pose and framing only; keep the avatar's identity, outfit and setting unless requested.",
             "Scene": "Use background and lighting only; do not copy any person's identity.",
             "Other person": "Keep this other person distinct from the avatar; never blend their faces."}
    if len(roles) > 2 or any(role not in ROLES for role in roles):
        raise ValueError("Choose at most two supported extra-reference roles.")
    return " ".join(f"Image {index + 3}: {role}. {rules[role]}" for index, role in enumerate(roles))


def export_pack(root, avatar_id):
    import zipfile
    from PIL import ImageDraw
    avatar = freeze(root, studio.load_avatar(root, avatar_id))
    pinned = _read(_folder(root, avatar_id) / "versions" / avatar["identity_version"] / "identity.json")
    rows = [{"view": "face", "path": avatar["face_image"]}, {"view": "body", "path": avatar["body_image"]}] + pinned["pack"]
    folder = _folder(root, avatar_id) / "exports"
    folder.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (600, ((len(rows) + 2) // 3) * 240), "white")
    draw = ImageDraw.Draw(sheet)
    target = folder / (avatar["identity_version"] + ".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, row in enumerate(rows):
            with Image.open(row["path"]) as opened:
                image = opened.convert("RGB")
                image.thumbnail((190, 210))
                x, y = (index % 3) * 200, (index // 3) * 240
                sheet.paste(image, (x + (200 - image.width) // 2, y))
                draw.text((x + 8, y + 216), row["view"], fill="black")
            archive.write(row["path"], f"references/{index:02d}-{row['view']}.png")
        import io
        buffer = io.BytesIO()
        sheet.save(buffer, "JPEG", quality=92)
        archive.writestr("contact-sheet.jpg", buffer.getvalue())
        archive.writestr("identity.json", json.dumps({"version": avatar["identity_version"], "name": avatar["name"], "signature": pinned["signature"]}, indent=2))
        for part in ("face", "body"):
            archive.write(avatar[part + "_specs"], part + "-specs.json")
    return target
