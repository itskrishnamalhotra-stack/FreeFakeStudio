"""Presentation helpers for the Avatar Studio library and guided workspace."""

import base64
import html
import io
import os
from pathlib import Path

from PIL import Image, ImageDraw

import avatar_guide as guide
import avatar_studio as studio


def thumbnail(path, name="Avatar"):
    if path and os.path.isfile(path):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((220, 220))
    else:
        image = Image.new("RGB", (180, 180), (40, 65, 60))
        draw = ImageDraw.Draw(image)
        draw.text((72, 68), (name or "A")[:2].upper(), fill=(220, 240, 231), font_size=32)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=84)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def icon(name):
    return str(Path(__file__).parent / "assets" / "avatar-icons" / f"{name}.svg")


def library(save_dir, query="", status="All avatars", sort="Recent"):
    search = str(query or "").strip().casefold()
    values = [avatar for avatar in studio.list_avatars(save_dir)
              if search in avatar.get("name", "").casefold()]
    if status != "All avatars":
        ready = status == "Ready"
        values = [avatar for avatar in values if bool(avatar.get("face_locked") and avatar.get("body_locked")) == ready]
    if sort == "Name":
        values.sort(key=lambda avatar: avatar.get("name", "").casefold())
    return values


def library_values(save_dir, query="", status="All avatars", sort="Recent"):
    values = []
    for avatar in library(save_dir, query, status, sort):
        path = avatar.get("face_image") or guide.draft(avatar, "face")
        if path and os.path.isfile(path):
            image = path
        else:
            image = Image.new("RGB", (180, 180), (40, 65, 60))
            ImageDraw.Draw(image).text((70, 65), avatar["name"][:2].upper(), fill="white", font_size=32)
        ready = avatar.get("face_locked") and avatar.get("body_locked")
        values.append((image, f"{avatar['name']}\n{'Ready to create' if ready else 'Continue setup'}"))
    return values


def library_empty(save_dir, query="", status="All avatars"):
    if studio.list_avatars(save_dir):
        return '<div class="av-library-empty"><strong>No matching avatars</strong><p>Try another name or filter.</p></div>'
    return '<div class="av-library-empty"><strong>Your next character starts here</strong><p>Create an avatar to begin.</p></div>'


def dashboard(avatar):
    if not avatar:
        return '<div class="av-overview av-empty"><div><span class="av-eyebrow">YOUR CHARACTERS</span><h1>Avatar Studio</h1><p>A new identity starts here.</p></div></div>'
    name = html.escape(avatar.get("name", "Avatar"))
    ready = bool(avatar.get("face_locked") and avatar.get("body_locked"))
    face = thumbnail(avatar.get("face_image") or guide.draft(avatar, "face"), name)
    body = avatar.get("body_image")
    body_html = f'<img class="av-body-thumb" src="{thumbnail(body)}" alt="Body reference">' if body else ""
    return (
        '<div class="av-overview"><div class="av-identity-images">'
        f'<img class="av-face-thumb" src="{face}" alt="Face reference">{body_html}</div>'
        f'<div class="av-overview-title"><h1>{name}</h1>'
        f'<span class="av-badge {"ready" if ready else "pending"}">{"Ready to create" if ready else "Identity in progress"}</span></div>'
        '</div>'
    )


def guide_header(avatar):
    current = guide.stage(avatar)
    index = 0 if current.startswith("face") else 1 if current.startswith("body") else 2 if current == "starters" else 3
    labels = ["Face", "Body", "Collection"]
    steps = "".join(
        f'<li class="{"done" if number < index else "current" if number == index else ""}"><b>{number + 1}</b><span>{label}</span></li>'
        for number, label in enumerate(labels)
    )
    titles = {
        "face": ("Face reference", "01 / IDENTITY"),
        "face_review": ("Face reference", "01 / REVIEW"),
        "body": ("Body reference", "02 / IDENTITY"),
        "body_review": ("Body reference", "02 / REVIEW"),
        "starters": ("First collection", "03 / CREATING"),
        "complete": ("Your avatar is ready", "03 / SAVED"),
    }
    title, subtitle = titles.get(current, titles["face"])
    return f'<div class="av-guide-heading"><div><span class="av-eyebrow">{subtitle}</span><h2>{title}</h2></div><ol class="av-progress">{steps}</ol></div>'


def reference_context(avatar, kind):
    locked = bool(kind == "body" and avatar and avatar.get("face_locked"))
    anchor = (
        f'<img src="{thumbnail(avatar.get("face_image"))}" alt="Locked face reference">'
        '<span>Face identity attached</span>'
    ) if locked else '<span>Text to image</span>'
    return (
        '<div class="av-generation-context"><span class="av-model-tag">FLUX.2 Klein 4B</span>'
        '<span class="av-step-tag">4 steps</span>'
        f'<div class="av-anchor">{anchor}</div></div>'
    )


CSS = Path(__file__).with_name("avatar_ui.css").read_text(encoding="utf-8")
