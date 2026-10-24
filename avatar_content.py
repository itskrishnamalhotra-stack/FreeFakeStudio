"""Persistent editorial drafts and explicit-approval Instagram publishing."""

import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

import avatar_identity as identity
import avatar_studio as studio

_publish_lock = threading.Lock()


def _path(root, avatar_id):
    return studio.avatar_path(root, avatar_id) / "content" / "calendar.json"


def load(root, avatar_id):
    return identity._read(_path(root, avatar_id), {"persona": {}, "posts": []})


def save(root, avatar_id, data):
    studio._atomic_json(_path(root, avatar_id), data)
    return data


def persona(root, avatar_id, niche, tone, values, disclosure):
    data = load(root, avatar_id)
    data["persona"] = {"niche": str(niche)[:1000], "tone": str(tone)[:1000],
                       "values": str(values)[:2000], "disclosure": str(disclosure)[:300]}
    return save(root, avatar_id, data)


def local_caption(brand, brief, context="", style="auto"):
    if style not in {"auto", "concise", "editorial", "conversational"}:
        raise ValueError("Choose a supported caption style.")
    tone = str(brand.get("tone", "")).lower()
    if style == "auto":
        style = "conversational" if any(word in tone for word in ("warm", "casual", "friendly", "playful")) else "concise" if any(word in tone for word in ("short", "concise", "minimal")) else "editorial"
    subject = str(brief).strip()
    if not subject:
        setting = next((word for word in ("beach", "studio", "city", "cafe", "garden") if re.search(r"\b" + word + r"\b", context, re.I)), "creative")
        subject = f"A {setting} portrait study"
    if subject[-1:] not in ".!?":
        subject += "."
    niche = str(brand.get("niche", "")).strip().strip(".")
    values = [value.strip() for value in re.split(r"[,;\n]", str(brand.get("values", ""))) if value.strip()][:3]
    direction = ", ".join(values)
    if style == "conversational":
        parts = [subject, f"A little inspiration for {niche.lower()}." if niche else "A different perspective, one frame at a time.", f"Keeping {direction.lower()} at the heart of the creative direction." if direction else ""]
    elif style == "editorial":
        parts = [subject, f"A visual exploration of {niche.lower()}." if niche else "A study in light, texture and composition.", f"Creative direction: {direction}." if direction else ""]
    else:
        parts = [subject, direction + "." if direction else ""]
    tags = []
    for phrase in [niche, *values]:
        words = re.findall(r"[a-zA-Z0-9]+", phrase)[:4]
        if words:
            tag = ("#" + "".join(word.capitalize() for word in words))[:40]
            if tag.lower() not in {item.lower() for item in tags}:
                tags.append(tag)
    disclosure = str(brand.get("disclosure", "")).strip()
    ending = "\n\n".join(part for part in (" ".join(tags[:4]), disclosure) if part)
    body = "\n\n".join(part for part in parts if part)
    allowance = max(0, 2200 - len(ending) - (2 if ending else 0))
    return body[:allowance].rstrip() + ("\n\n" + ending if ending else "")


def caption(root, avatar_id, image_id, brief, dev=False, use_api=False, style="auto"):
    import avatar_gallery
    item = studio.gallery_item_by_id(root, avatar_id, image_id)
    if not item:
        raise ValueError("Select a saved image.")
    brand = load(root, avatar_id)["persona"]
    if not use_api:
        return local_caption(brand, brief, item.get("prompt", ""), style)
    if dev:
        return "[MOCK caption] " + str(brief) + " " + brand.get("disclosure", "AI-generated character")
    result = avatar_gallery.gemini_json(
        f"Write one Instagram caption, at most 2000 characters. Style: {style}. Persona: {json.dumps(brand)}. "
        f"Image-generation context: {item.get('prompt', '')}. Editorial brief: {brief}. "
        "Do not claim real experiences, endorsements or events that the brief does not support. "
        "Include the supplied AI disclosure verbatim when nonempty. Return JSON caption only.",
        {"type": "object", "properties": {"caption": {"type": "string"}}, "required": ["caption"]})
    return str(result["caption"])[:2200]


def draft(root, avatar_id, image_id, text, planned_at="", public_url=""):
    item = studio.gallery_item_by_id(root, avatar_id, image_id)
    if not item:
        raise ValueError("Select a saved avatar image.")
    if len(text) > 2200 or not text.strip():
        raise ValueError("Write a caption between 1 and 2200 characters.")
    if planned_at:
        when = datetime.fromisoformat(planned_at)
        if when.tzinfo is None:
            raise ValueError("Use an ISO date/time with timezone, for example 2026-10-25T18:00:00+05:30.")
    identifier = uuid.uuid4().hex[:12]
    image_path = studio.avatar_path(root, avatar_id) / "content" / "assets" / f"{identifier}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    with Image.open(item["generated_path"]) as image:
        image.convert("RGB").save(image_path, "JPEG", quality=95)
    data = load(root, avatar_id)
    data["posts"].append({"id": identifier, "image_id": image_id, "image_path": str(image_path),
                          "image_sha256": identity._digest(image_path), "caption": text.strip(),
                          "planned_at": planned_at, "public_url": public_url.strip(), "status": "draft",
                          "identity_version": item.get("identity_version"), "created_at": studio._now()})
    save(root, avatar_id, data)
    return identifier


def edit(root, avatar_id, post_id, text, planned_at="", public_url=""):
    data = load(root, avatar_id)
    post = _post(data, post_id)
    if post["status"] not in {"draft", "approved", "publish_failed"}:
        raise ValueError("Only unpublished drafts can be edited.")
    if not text.strip() or len(text) > 2200:
        raise ValueError("Caption must contain 1 to 2200 characters.")
    if planned_at and datetime.fromisoformat(planned_at).tzinfo is None:
        raise ValueError("Planned time must include a timezone.")
    post.update(caption=text.strip(), planned_at=planned_at, public_url=public_url.strip(), status="draft")
    for key in ("container_id", "container_status", "last_error"):
        post.pop(key, None)
    post.pop("approval", None)
    return save(root, avatar_id, data)


def _post(data, post_id):
    post = next((post for post in data["posts"] if post["id"] == post_id), None)
    if not post:
        raise ValueError("Select a saved content draft.")
    return post


def approve(root, avatar_id, post_id, confirmed):
    if not confirmed:
        raise ValueError("Explicitly approve this exact image, public URL and caption before publishing.")
    data = load(root, avatar_id)
    post = _post(data, post_id)
    if post["status"] not in {"draft", "approved"}:
        raise ValueError("This post cannot be approved again; reconcile any uncertain publish first.")
    post["approval"] = _approval_digest(post)
    post["approved_at"] = studio._now()
    post["status"] = "approved"
    return save(root, avatar_id, data)


def _approval_digest(post):
    fields = {key: post.get(key) for key in ("image_sha256", "caption", "public_url", "identity_version")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def due_posts(root, avatar_id, enabled_since, now=None):
    now = now or datetime.now(timezone.utc)
    since = datetime.fromtimestamp(enabled_since, timezone.utc)
    result = []
    for post in load(root, avatar_id)["posts"]:
        if post.get("status") not in {"approved", "container_created"} or not post.get("planned_at") or not post.get("public_url"):
            continue
        when = datetime.fromisoformat(post["planned_at"])
        if when.tzinfo and since <= when <= now and post.get("approval") == _approval_digest(post):
            result.append(post)
    return sorted(result, key=lambda post: datetime.fromisoformat(post["planned_at"]))


def credentials():
    token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "").strip()
    account = os.environ.get("INSTAGRAM_USER_ID", "").strip()
    version = os.environ.get("INSTAGRAM_GRAPH_VERSION", "").strip()
    mode = os.environ.get("INSTAGRAM_LOGIN_MODE", "instagram").strip()
    if not token or not account.isdigit() or not re.fullmatch(r"v\d+\.\d+", version):
        raise ValueError("Configure INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_USER_ID and INSTAGRAM_GRAPH_VERSION in the private keys file.")
    if mode not in {"instagram", "facebook"}:
        raise ValueError("INSTAGRAM_LOGIN_MODE must be instagram or facebook.")
    return token, account, "https://graph.instagram.com/" + version if mode == "instagram" else "https://graph.facebook.com/" + version


def _api(method, path, fields=None, session=requests):
    token, _, base = credentials()
    try:
        response = session.request(method, base + "/" + path, headers={"Authorization": "Bearer " + token},
                                   **({"params": fields} if method == "GET" else {"data": fields}), timeout=45)
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError("Instagram response was interrupted or invalid; inspect the saved publishing state before retrying.") from None
    if not response.ok or "error" in body:
        error = body.get("error") or {}
        message = str(error.get("message", "Request rejected")).replace(token, "[redacted]")
        raise RuntimeError(f"Instagram API rejected the request ({error.get('code', response.status_code)}): {message[:500]}")
    return body


def publish(root, avatar_id, post_id, dev=False, session=requests):
    if dev:
        raise RuntimeError("Instagram publishing is disabled in local development. Export a draft instead.")
    with _publish_lock:
        data = load(root, avatar_id)
        post = _post(data, post_id)
        if post["status"] == "published":
            return post
        if post["status"] in {"publishing", "publishing_unknown", "creating_unknown"}:
            raise RuntimeError("A previous request has an uncertain outcome. Check Instagram before creating or publishing another post.")
        if not post.get("approval") or post.get("approval") != _approval_digest(post) or identity._digest(post["image_path"]) != post["image_sha256"]:
            raise ValueError("This exact draft needs approval; its image or caption changed.")
        parsed = urlparse(post["public_url"])
        host = parsed.hostname or ""
        if parsed.scheme != "https" or parsed.username or parsed.password or host in {"localhost", ""} or host.endswith((".local", ".internal")):
            raise ValueError("Instagram needs a publicly accessible HTTPS image URL that shows the approved export image.")
        try:
            if not ipaddress.ip_address(host).is_global:
                raise ValueError("Private image addresses cannot be used for Instagram.")
        except ValueError as exc:
            if "Private" in str(exc):
                raise
        _, account, _ = credentials()
        from PIL import Image
        with Image.open(post["image_path"]) as photo:
            if not 0.8 <= photo.width / photo.height <= 1.91 or Path(post["image_path"]).stat().st_size > 8_000_000:
                raise ValueError("Instagram export must be JPEG under 8 MB, with aspect ratio 4:5 to 1.91:1. Choose a compatible image.")
        if not post.get("container_id"):
            post["status"] = "creating_unknown"
            save(root, avatar_id, data)
            container = _api("POST", account + "/media", {"image_url": post["public_url"], "caption": post["caption"]}, session)
            post.update(container_id=container["id"], status="container_created")
            save(root, avatar_id, data)
        status = _api("GET", post["container_id"], {"fields": "status_code,status"}, session)
        post["container_status"] = status
        if status.get("status_code") in {"ERROR", "EXPIRED"}:
            post.update(status="publish_failed", last_error="Instagram container " + status["status_code"])
            save(root, avatar_id, data)
            raise RuntimeError("Instagram rejected or expired the image container. Edit the draft, then approve it again.")
        save(root, avatar_id, data)
        if status.get("status_code") != "FINISHED":
            return post
        post["status"] = "publishing_unknown"
        save(root, avatar_id, data)
        result = _api("POST", account + "/media_publish", {"creation_id": post["container_id"]}, session)
        post.update(status="published", media_id=result["id"], published_at=studio._now())
        save(root, avatar_id, data)
        return post


def reconcile(root, avatar_id, post_id, media_id, session=requests):
    if not str(media_id).isdigit():
        raise ValueError("Enter the actual published Instagram media ID.")
    _, account, _ = credentials()
    recent = _api("GET", account + "/media", {"fields": "id,caption,permalink", "limit": 100}, session)
    data = load(root, avatar_id)
    post = _post(data, post_id)
    match = next((row for row in recent.get("data", []) if row.get("id") == media_id and row.get("caption", "").strip() == post["caption"]), None)
    if not match:
        raise ValueError("That media ID and caption were not found on the configured account. No retry was sent.")
    post.update(status="published", media_id=media_id, permalink=match.get("permalink"), reconciled_at=studio._now())
    save(root, avatar_id, data)
    return post


def export(root, avatar_id, post_id):
    post = _post(load(root, avatar_id), post_id)
    path = studio.avatar_path(root, avatar_id) / "content" / (post_id + ".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(post["image_path"], "image.jpg")
        archive.writestr("caption.txt", post["caption"])
        archive.writestr("post.json", json.dumps({key: post.get(key) for key in ("id", "planned_at", "identity_version", "image_sha256")}, indent=2))
    return str(path)
