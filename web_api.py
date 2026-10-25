"""Authenticated same-origin HTTP API for the TypeScript studio."""
from contextlib import asynccontextmanager
import asyncio
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Literal
from urllib.parse import quote, urlparse

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from PIL import Image, ImageOps, UnidentifiedImageError

import avatar_enhance
import enhancement_profiles
import app_settings
import avatar_resources
import avatar_gallery
import avatar_gallery_jobs
import avatar_resource_tabs
import avatar_guide
import avatar_identity as identity
import avatar_quality as quality
import avatar_benchmark as benchmark
import avatar_content as content
import avatar_studio as studio
import generation_service as generation
import model_manager
import workspace
from portable_state import relocate_results
from runtime_environment import provider
from web_jobs import JobStore
from web_service import StudioService

SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AvatarCreate(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    mode: Literal["Guided", "Custom"] = "Guided"
    preset: str = "Studio creator"


class Rename(StrictModel):
    name: str = Field(min_length=1, max_length=80)


class ChatCreate(StrictModel):
    avatar_id: str | None = None
    title: str = Field(default="New chat", min_length=1, max_length=100)


class IdentityUpdate(StrictModel):
    action: Literal["upload", "approve", "remove", "activate", "label"]
    image: str | None = None
    view: Literal["front", "three-quarter", "profile", "smile", "body-front", "body-side"] = "front"
    candidate_id: str | None = None
    approved: bool = False
    version: str | None = None
    index: int = Field(default=0, ge=0, le=11)
    label: Literal["correct", "wrong", "uncertain"] = "uncertain"


class QualitySettings(StrictModel):
    policies: dict[str, Literal["required", "important", "flexible"]] = Field(default_factory=dict)


class AppSettings(StrictModel):
    gpu_memory_management: StrictBool


class BatchAction(StrictModel):
    action: Literal["rename", "archive", "restore", "finish"]
    name: str = Field(default="", max_length=120)


class Persona(StrictModel):
    niche: str = Field(default="", max_length=1000)
    tone: str = Field(default="", max_length=1000)
    values: str = Field(default="", max_length=2000)
    disclosure: str = Field(default="AI-generated character", max_length=300)


class ContentDraft(StrictModel):
    image_id: str
    caption: str = Field(min_length=1, max_length=2200)
    planned_at: str = Field(default="", max_length=60)
    public_url: str = Field(default="", max_length=2000)


class Approval(StrictModel):
    confirmed: bool


class JobRequest(StrictModel):
    kind: Literal["generate", "reference", "confirm", "guide", "console", "gallery", "resource_search", "resource_generate", "regenerate", "enhance", "mask_preview", "encoder", "unload", "identity_view", "calibrate", "benchmark", "quality_check", "caption", "publish", "reconcile"]
    avatar_id: str | None = None
    batch_id: str | None = None
    new_batch: bool = False
    batch_name: str = Field(default="", max_length=120)
    avatar_gender: Literal["female", "male", "any"] = "any"
    resource_tab_id: str | None = None
    resource_tab_name: str = Field(default="", max_length=120)
    chat_id: str | None = None
    new_chat: bool = False
    item_id: str | None = None
    reference_kind: Literal["face", "body"] = "face"
    details: str = Field(default="", max_length=12000)
    prompt: str = Field(default="", max_length=16000)
    negative: str = Field(default=generation.DEFAULT_NEGATIVE, max_length=4000)
    model: str = model_manager.FLUX_MODEL_NAME
    aspect: str = generation.ASPECTS[0]
    seed: int = Field(default=0, ge=0, lt=2**63)
    cfg: float = Field(default=1, ge=0, le=20)
    denoise: float = Field(default=1, ge=.05, le=1)
    quantity: int = Field(default=1, ge=1, le=20)
    steps: int = Field(default=4, ge=1, le=50)
    images: list[str] = Field(default_factory=list, max_length=4)
    image: str | None = None
    canvas: int | None = Field(default=None, ge=0, le=3)
    mask_mode: Literal["None", "Smart", "Manual Paint", "Background Only", "Everything Except Face"] = "None"
    mask: str | None = None
    mode: Literal["Identity Strict", "Outfit Focus", "Pose Focus", "Scene Focus", "Group Image"] = "Identity Strict"
    options: dict[str, str] | None = None
    theme: str = Field(default="", max_length=4000)
    require_face: bool = False
    reason: str = Field(default="", max_length=4000)
    method: Literal["photo", "hd_tiles"] = "photo"
    sampling_mode: Literal["reference", "img2img"] = "reference"
    enhancement_profile: str = Field(default="auto", max_length=40)
    additional_instructions: str = Field(default="", max_length=16000)
    scale: Literal[1, 2] = 1
    strength: float = Field(default=avatar_enhance.DEFAULT_STRENGTH, ge=.35, le=1)
    adaptive_prompt: StrictBool = True
    preserve_tone: StrictBool = True
    reconstruction_prompt: str = Field(default=avatar_enhance.PHOTO_PROMPT, min_length=1, max_length=16000)
    encoder: Literal["Official", "Custom"] = "Official"
    roles: list[Literal["Outfit", "Pose", "Scene", "Other person"]] = Field(default_factory=list, max_length=2)
    face_view: Literal["auto", "front", "three-quarter", "profile", "smile"] = "auto"
    body_view: Literal["auto", "body-front", "body-side"] = "auto"
    caption_style: Literal["auto", "concise", "editorial", "conversational"] = "auto"
    view: Literal["front", "three-quarter", "profile", "smile", "body-front", "body-side"] = "front"
    positives: list[str] = Field(default_factory=list, max_length=12)
    negatives: list[str] = Field(default_factory=list, max_length=12)
    use_api: bool = False
    post_id: str | None = None
    media_id: str = Field(default="", max_length=100)


def redact(value):
    text = str(value)
    for key, secret in os.environ.items():
        if any(word in key.upper() for word in ("TOKEN", "SECRET", "API_KEY", "PASSWORD")) and len(secret) >= 8:
            text = text.replace(secret, "[redacted]")
    return re.sub(r"([?&](?:key|token|access_token)=)[^\s&]+", r"\1[redacted]", text, flags=re.I)


def create_app(root=None, persist=None, token=None):
    root = Path(root or workspace.get_save_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    relocate_results(root)
    service = StudioService(root)
    access_token = token if token is not None else os.environ.get("FFS_ACCESS_TOKEN", "")
    public = os.environ.get("FREEFAKESTUDIO_PUBLIC_URL", "").rstrip("/")
    secure_cookie = public.startswith("https://")
    cookie_value = secrets.token_urlsafe(48)
    mutation_lock = threading.RLock()
    auth_attempts = {}
    scheduling = {}

    def handle(kind, payload, context):
        if kind == "publish" and payload.get("scheduled") and payload.get("avatar_id") not in scheduling:
            from web_jobs import Cancelled
            raise Cancelled("Runtime scheduling is off. Review and publish manually or enable scheduling again.")
        if kind == "encoder":
            context.emit("Switching FLUX text encoder")
            return {"encoder": model_manager.set_flux_encoder_mode(payload["encoder"])}
        if kind == "unload":
            model_manager.unload_current(status_callback=context.emit)
            return {"unloaded": True}
        return service.handle(kind, payload, context)

    jobs = JobStore(root / "_jobs", handle, persist)

    def schedule_tick():
        if model_manager.DEV_MODE or jobs.stopping.is_set():
            return
        with mutation_lock:
            if any(row["status"] in {"queued", "running"} for row in jobs.list()):
                return
            for avatar_id, enabled_since in list(scheduling.items()):
                for post in content.due_posts(str(root), avatar_id, enabled_since):
                    key = f"scheduled-{avatar_id}-{post['id']}-{post['approval'][:12]}"
                    if post.get("status") == "container_created":
                        key += f"-poll-{int(time.time() // 30)}"
                    if any(row.get("request_key") == key for row in jobs.list()):
                        continue
                    jobs.submit("publish", {"avatar_id": avatar_id, "post_id": post["id"], "scheduled": True}, key)
                    return

    async def scheduler_loop():
        while True:
            await asyncio.sleep(30)
            try:
                await asyncio.to_thread(schedule_tick)
            except Exception as exc:
                studio._atomic_json(root / "_debug" / "scheduler.json", {"error": redact(exc), "time": time.time()})

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(scheduler_loop())
        try:
            yield
        finally:
            jobs.request_shutdown()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            report = getattr(app.state, "report_shutdown", lambda stage: None)
            report("saving_jobs")
            await asyncio.to_thread(jobs.shutdown, on_progress=report)
            report("saved")

    app = FastAPI(title="FreeFakeStudio", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.jobs, app.state.service = jobs, service
    app.state.scheduling, app.state.schedule_tick = scheduling, schedule_tick

    def check_id(value):
        if not value or not SAFE_ID.fullmatch(value):
            raise ValueError("Invalid avatar or image identifier")
        return value

    def ensure_idle(avatar_id=None):
        pending = [row for row in jobs.list() if row["status"] in {"queued", "running"}]
        if any(avatar_id is None or row["payload"].get("avatar_id") == avatar_id for row in pending):
            raise HTTPException(409, "Wait for the current operation or stop it first")

    @app.middleware("http")
    async def protect(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path not in {"/api/health", "/api/session"}:
            if access_token and not secrets.compare_digest(request.cookies.get("ffs_session", ""), cookie_value):
                return JSONResponse({"detail": "Open the private launch link from your notebook"}, status_code=401)
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                origin = request.headers.get("origin")
                allowed = {public, str(request.base_url).rstrip("/")}
                if origin and origin not in allowed:
                    return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
            if jobs.stopping.is_set() and request.method not in {"GET", "HEAD", "OPTIONS"}:
                return JSONResponse({"detail": "Server is saving and restarting. Use the new notebook link when ready."}, status_code=503)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": redact(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Saved item not found"}, status_code=404)

    @app.exception_handler(RuntimeError)
    async def operational_error(request, exc):
        return JSONResponse({"detail": redact(exc)}, status_code=503)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    def info(path):
        if not path:
            return None
        try:
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.suffix.lower() not in IMAGE_EXTENSIONS:
                return None
            relative = resolved.relative_to(root).as_posix()
            with Image.open(resolved) as source:
                width, height = source.size
            return {"path": relative, "url": "/api/files/" + quote(relative), "thumb": "/api/thumbs/" + quote(relative), "width": width, "height": height}
        except (ValueError, OSError):
            return None

    def avatar_value(avatar):
        if not avatar:
            return None
        guide = avatar_guide.state(avatar)
        chat = []
        for index, turn in enumerate(avatar.get("chat_history", [])):
            metadata = turn.get("metadata") or {}
            images = [info(turn.get("image_path"))] if turn.get("image_path") else []
            if turn.get("role") == "user":
                images += [info(path) for path in metadata.get("extra_references", [])]
            chat.append({"id": metadata.get("job_id", str(index)) + "-" + turn.get("role", "assistant"),
                         "role": turn.get("role", "assistant"), "text": turn.get("content", ""),
                         "images": [image for image in images if image], "seed": metadata.get("seed")})
        return {"id": avatar["id"], "name": avatar["name"], "created_at": avatar.get("created_at"), "updated_at": avatar.get("updated_at"),
            "face_locked": bool(avatar.get("face_locked")), "body_locked": bool(avatar.get("body_locked")),
            "face": info(avatar.get("face_image")), "body": info(avatar.get("body_image")),
            "face_candidate": info(avatar.get("face_candidate")), "body_candidate": info(avatar.get("body_candidate")),
            "face_prompt": avatar.get("face_prompt", ""), "body_prompt": avatar.get("body_prompt", ""),
            "face_specs": studio.load_specs(avatar.get("face_specs")) if avatar.get("face_specs") else {},
            "body_specs": studio.load_specs(avatar.get("body_specs")) if avatar.get("body_specs") else {},
            "current_step": avatar.get("current_step"), "creation_mode": avatar.get("creation_mode", "custom"), "identity_version": avatar.get("identity_version"),
            "guide": {"stage": avatar_guide.stage(avatar), "next_label": avatar_guide.next_label(avatar),
                      "active": avatar_guide.active(avatar) and avatar_guide.stage(avatar) != "complete", "options": dict(zip(["face", "body", "scene_one", "scene_two"], avatar_guide.options(avatar))),
                      "face_draft": info(avatar_guide.draft(avatar, "face")), "body_draft": info(avatar_guide.draft(avatar, "body"))},
            "chat": chat, "gallery_count": avatar.get("gallery_count", 0)}

    def gallery_value(row):
        return {key: row.get(key) for key in ("id", "status", "origin", "prompt", "seed", "attempt", "validation", "source_id", "diagnostics",
                                             "working_size", "output_size", "processing_passes", "method", "strength", "steps", "manual_review", "identity_version", "recipe", "batch_id")} | {
            "image": info(row.get("generated_path")), "source": info(row.get("source_path")),
            "reference": info(row.get("reference_path")), "regenerable": row.get("origin") not in {"enhancement", "legacy"} and bool(row.get("prompt"))}

    def chat_summary(row):
        return {key: row.get(key) for key in ("id", "avatar_id", "title", "created_at", "updated_at", "archived")} | {"turn_count": len(row["turns"])}

    def chat_value(row):
        return {**chat_summary(row), "turns": [{**turn, "source": info(root / turn["source"]) if turn.get("source") else None,
                "images": [picture for path in turn.get("images", []) if (picture := info(root / path))]} for turn in row["turns"]]}

    @app.get("/api/chats")
    def conversations(avatar_id: str | None = None):
        return [chat_summary(row) for row in service.chats.list(avatar_id) if row.get("turns")]

    @app.post("/api/chats")
    def new_conversation(body: ChatCreate):
        with mutation_lock:
            service.chats.ensure(body.avatar_id)
            row = service.chats.create(body.avatar_id, body.title)
            jobs.flush()
            return chat_value(row)

    @app.get("/api/chats/{chat_id}")
    def conversation(chat_id: str, avatar_id: str | None = None):
        return chat_value(service.chats.get(chat_id, avatar_id))

    @app.patch("/api/chats/{chat_id}")
    def rename_conversation(chat_id: str, body: Rename, avatar_id: str | None = None):
        with mutation_lock:
            row = service.chats.rename(chat_id, avatar_id, body.name)
            jobs.flush()
            return chat_summary(row)

    @app.delete("/api/chats/{chat_id}")
    def archive_conversation(chat_id: str, avatar_id: str | None = None):
        if any(row["status"] in {"queued", "running"} and row["payload"].get("chat_id") == chat_id
               and row["payload"].get("avatar_id") == avatar_id for row in jobs.list()):
            raise HTTPException(409, "Stop the conversation's active job before archiving it")
        with mutation_lock:
            service.chats.archive(chat_id, avatar_id)
            jobs.flush()
            return {"archived": True}

    @app.post("/api/session")
    async def session(request: Request):
        address = request.client.host if request.client else "unknown"
        now = time.monotonic()
        attempts = [stamp for stamp in auth_attempts.get(address, []) if now - stamp < 60]
        if len(attempts) >= 20:
            raise HTTPException(429, "Too many authentication attempts. Wait a minute.")
        auth_attempts[address] = [*attempts, now]
        if int(request.headers.get("content-length", "0")) > 2048:
            raise HTTPException(413, "Authentication request too large")
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("Invalid authentication request")
        if access_token and not secrets.compare_digest(str(body.get("token", "")), access_token):
            raise HTTPException(401, "The private launch token is invalid")
        response = JSONResponse({"authenticated": True})
        response.set_cookie("ffs_session", cookie_value, httponly=True, secure=secure_cookie, samesite="strict", max_age=43200)
        return response

    @app.get("/api/health")
    def health():
        return {"ready": True, "mock": model_manager.DEV_MODE, "provider": provider()}

    @app.get("/api/config")
    def config():
        return {"models": [{"name": name, "capabilities": row["capabilities"], "defaults": model_manager.get_defaults(name),
                             "status": model_manager.get_model_status().get(name)} for name, row in model_manager.MODEL_REGISTRY.items()],
                "aspects": generation.ASPECTS, "modes": generation.MODES, "mask_modes": generation.MASK_MODES,
                "presets": list(avatar_guide.PRESETS), "encoder_choices": model_manager.get_flux_encoder_choices(),
                "encoder": model_manager.get_flux_encoder_config()["mode"].title(), "reconstruction_prompt": avatar_enhance.PHOTO_PROMPT,
                "enhancement_profiles": [{"id": key, "name": value["name"], "prompt": value["prompt"]} for key, value in enhancement_profiles.PROFILES.items()],
                "mock": model_manager.DEV_MODE, "provider": provider(), "gallery_api": avatar_gallery.configuration_status() if not model_manager.DEV_MODE else {"ready": True},
                "active_model": model_manager.get_current_model(), "max_references": 4, "max_avatar_extras": 2,
                "app_settings": app_settings.load(root)}

    @app.get("/api/settings")
    def settings():
        return app_settings.load(root)

    @app.patch("/api/settings")
    def update_settings(body: AppSettings):
        with mutation_lock:
            ensure_idle()
            result = app_settings.save(root, body.gpu_memory_management)
            jobs.flush()
            return result

    @app.get("/api/avatars")
    def avatars():
        return [avatar_value(row) for row in studio.list_avatars(str(root))]

    @app.post("/api/avatars")
    def new_avatar(body: AvatarCreate):
        with mutation_lock:
            result = avatar_guide.create(str(root), body.name, body.preset, body.mode)
            jobs.flush()
            return avatar_value(result)

    @app.get("/api/avatars/{avatar_id}")
    def avatar(avatar_id: str):
        return avatar_value(studio.load_avatar(str(root), check_id(avatar_id)))

    @app.patch("/api/avatars/{avatar_id}")
    def rename(avatar_id: str, body: Rename):
        ensure_idle(check_id(avatar_id))
        with mutation_lock:
            result = studio.rename_avatar(str(root), avatar_id, body.name)
            jobs.flush()
            return avatar_value(result)

    @app.post("/api/avatars/{avatar_id}/guide/{action}")
    def guide_action(avatar_id: str, action: Literal["back", "regenerate", "custom"]):
        ensure_idle(check_id(avatar_id))
        function = {"back": avatar_guide.back, "regenerate": avatar_guide.clear_draft, "custom": avatar_guide.use_custom}[action]
        function(str(root), avatar_id)
        jobs.flush()
        return avatar_value(studio.load_avatar(str(root), avatar_id))

    @app.get("/api/avatars/{avatar_id}/identity")
    def get_identity(avatar_id: str):
        avatar = service.ready_avatar(check_id(avatar_id))
        folder = identity._folder(str(root), avatar_id)
        versions = [identity._read(path) for path in (folder / "versions").glob("*/identity.json")]
        data = benchmark.load(str(root), avatar_id)
        profile = quality.profile(str(root), avatar)
        return {"version": avatar["identity_version"], "views": identity.VIEWS,
                "versions": sorted([{"version": row["version"], "created_at": row["created_at"]} for row in versions], key=lambda row: row["created_at"], reverse=True),
                "candidates": [{key: value for key, value in row.items() if key != "path"} | {"image": info(row["path"])} for row in identity.pack(str(root), avatar_id)["candidates"]],
                "profile": profile, "attribute_keys": studio.IDENTITY_SPEC_KEYS,
                "calibration": {name: [info(str(root / path)) for path in (profile.get("face") or {}).get(key, [])] for name, key in (("positive", "positive_files"), ("negative", "negative_files"))},
                "benchmark": benchmark.report(data),
                "cases": [{key: value for key, value in case.items() if key != "path"} | {"image": info(case.get("path")), "index": index} for index, case in enumerate(data["cases"] if data else [])]}

    @app.get("/api/avatars/{avatar_id}/identity/export")
    def export_identity(avatar_id: str):
        with mutation_lock:
            ensure_idle(avatar_id)
            target = identity.export_pack(str(root), check_id(avatar_id))
            return FileResponse(target, filename=target.name, media_type="application/zip")

    @app.post("/api/avatars/{avatar_id}/identity")
    def update_identity(avatar_id: str, body: IdentityUpdate):
        with mutation_lock:
            ensure_idle(avatar_id)
            service.ready_avatar(check_id(avatar_id))
            if body.action == "upload":
                image_file(body.image or "")
                with Image.open(service.file(body.image)) as image:
                    identity.add_candidate(str(root), avatar_id, body.view, image.convert("RGB"))
            elif body.action == "approve":
                identity.review_candidate(str(root), avatar_id, body.candidate_id, body.approved)
            elif body.action == "remove":
                identity.remove_candidate(str(root), avatar_id, body.candidate_id)
            elif body.action == "activate":
                identity.activate(str(root), avatar_id, body.version)
            else:
                benchmark.label(str(root), avatar_id, body.index, body.label)
            jobs.flush()
            return get_identity(avatar_id)

    @app.patch("/api/avatars/{avatar_id}/quality")
    def update_quality(avatar_id: str, body: QualitySettings):
        with mutation_lock:
            ensure_idle(avatar_id)
            avatar = service.ready_avatar(check_id(avatar_id))
            result = quality.save_profile(str(root), avatar, body.model_dump())
            jobs.flush()
            return result

    @app.get("/api/avatars/{avatar_id}/content")
    def get_content(avatar_id: str):
        studio.load_avatar(str(root), check_id(avatar_id))
        data = content.load(str(root), avatar_id)
        data["posts"] = [{key: value for key, value in row.items() if key != "image_path"} | {"image": info(row["image_path"])} for row in data["posts"]]
        try:
            content.credentials()
            data["instagram_configured"] = not model_manager.DEV_MODE
        except ValueError:
            data["instagram_configured"] = False
        data["scheduling_enabled"] = avatar_id in scheduling
        return data

    @app.post("/api/avatars/{avatar_id}/scheduling")
    def schedule_content(avatar_id: str, body: Approval):
        with mutation_lock:
            studio.load_avatar(str(root), check_id(avatar_id))
            if body.confirmed:
                if model_manager.DEV_MODE:
                    raise ValueError("Runtime publishing is disabled in mock preview.")
                content.credentials()
                scheduling[avatar_id] = time.time()
            else:
                scheduling.pop(avatar_id, None)
                for job in jobs.list():
                    if job["status"] == "queued" and job["payload"].get("scheduled") and job["payload"].get("avatar_id") == avatar_id:
                        jobs.cancel(job["id"])
            return get_content(avatar_id)

    @app.patch("/api/avatars/{avatar_id}/persona")
    def save_persona(avatar_id: str, body: Persona):
        with mutation_lock:
            ensure_idle(avatar_id)
            studio.load_avatar(str(root), check_id(avatar_id))
            content.persona(str(root), avatar_id, **body.model_dump())
            jobs.flush()
            return get_content(avatar_id)

    @app.post("/api/avatars/{avatar_id}/content")
    def create_content(avatar_id: str, body: ContentDraft):
        with mutation_lock:
            ensure_idle(avatar_id)
            service.ready_avatar(check_id(avatar_id))
            identifier = content.draft(str(root), avatar_id, body.image_id, body.caption, body.planned_at, body.public_url)
            jobs.flush()
            return {"id": identifier}

    @app.patch("/api/avatars/{avatar_id}/content/{post_id}")
    def edit_content(avatar_id: str, post_id: str, body: ContentDraft):
        with mutation_lock:
            ensure_idle(avatar_id)
            check_id(avatar_id)
            content.edit(str(root), avatar_id, check_id(post_id), body.caption, body.planned_at, body.public_url)
            jobs.flush()
            return get_content(avatar_id)

    @app.post("/api/avatars/{avatar_id}/content/{post_id}/approve")
    def approve_content(avatar_id: str, post_id: str, body: Approval):
        with mutation_lock:
            ensure_idle(avatar_id)
            content.approve(str(root), check_id(avatar_id), check_id(post_id), body.confirmed)
            jobs.flush()
            return get_content(avatar_id)

    @app.get("/api/avatars/{avatar_id}/content/{post_id}/export")
    def export_content(avatar_id: str, post_id: str):
        path = content.export(str(root), check_id(avatar_id), check_id(post_id))
        return FileResponse(path, filename=Path(path).name, media_type="application/zip")

    @app.get("/api/avatars/{avatar_id}/gallery")
    def gallery(avatar_id: str, offset: int = 0, limit: int = 60, status: str = "all", origin: str = "all", query: str = "",
                category: Literal["all", "generations", "enhancements", "review"] = "all", batch_id: str | None = None):
        check_id(avatar_id)
        rows = studio.visible_gallery_items(str(root), avatar_id, limit=None)
        by_id = {row["id"]: row for row in rows}
        by_path = {row.get("generated_path"): row for row in rows if row.get("generated_path")}
        for row in rows:
            if row.get("origin") == "enhancement" and not row.get("batch_id"):
                parent = by_id.get(row.get("source_id")) or by_path.get(row.get("source_path")) or {}
                row["batch_id"] = parent.get("batch_id")
        if batch_id:
            if not avatar_gallery_jobs.load_batch(str(root), avatar_id, batch_id):
                raise ValueError("Gallery batch not found")
            rows = [row for row in rows if row.get("batch_id") == batch_id]
        review_states = {"needs_review", "unchecked", "no_change", "failed", "stopped"}
        counts = {"generations": sum(row.get("origin") != "enhancement" for row in rows),
                  "enhancements": sum(row.get("origin") == "enhancement" for row in rows),
                  "review": sum(row.get("status") in review_states for row in rows),
                  "resources": len(avatar_resources.collect(root, avatar_id, service.chats, batch_id))}
        rows = [row for row in rows if category == "all"
                or (category == "generations" and row.get("origin") != "enhancement")
                or (category == "enhancements" and row.get("origin") == "enhancement")
                or (category == "review" and row.get("status") in review_states)]
        rows = [row for row in reversed(rows) if (status == "all" or row.get("status") == status)
                and (origin == "all" or row.get("origin") == origin) and query.casefold() in str(row.get("prompt", "")).casefold()]
        offset, limit = max(0, offset), max(1, min(100, limit))
        return {"total": len(rows), "items": [gallery_value(row) for row in rows[offset:offset + limit]], "offset": offset, "counts": counts}

    @app.get("/api/avatars/{avatar_id}/resources")
    def resources(avatar_id: str, status: Literal["all", "pending", "approved", "rejected", "used", "unused", "reserved"] = "all",
                  kind: Literal["all", "identity", "uploaded", "search"] = "all", query: str = "", offset: int = 0, limit: int = 48,
                  batch_id: str | None = None, tab_id: str | None = None):
        check_id(avatar_id)
        if batch_id and not avatar_gallery_jobs.load_batch(str(root), avatar_id, batch_id):
            raise ValueError("Gallery batch not found")
        if tab_id:
            saved = avatar_gallery_jobs.load_batch(str(root), avatar_id, tab_id)
            if saved and saved.get("resource_deleted") and saved.get("role") in {None, "", "resource_tab"}:
                if batch_id and saved.get("owner_batch_id", saved["id"]) != batch_id:
                    raise ValueError("Resource tab does not belong to the selected batch")
                return {"total": 0, "items": [], "deleted": True,
                        "counts": {key: 0 for key in ("all", "pending", "approved", "rejected", "used", "unused", "reserved")}}
        rows = avatar_resources.collect(root, avatar_id, service.chats, batch_id, tab_id)
        counts = {"all": len(rows), **{state: sum(row["state"] == state for row in rows) for state in ("pending", "approved", "rejected")},
                  "used": sum(row["used"] for row in rows), "reserved": sum(row.get("reserved", False) for row in rows),
                  "unused": sum(not row["used"] and not row.get("reserved") and row["state"] != "rejected" for row in rows)}
        def status_matches(row):
            if status == "all":
                return True
            if status == "used":
                return row["used"]
            if status == "reserved":
                return row.get("reserved", False)
            if status == "unused":
                return not row["used"] and not row.get("reserved") and row["state"] != "rejected"
            return row["state"] == status
        rows = [row for row in rows if status_matches(row)
                and (kind == "all" or row["kind"] == kind)
                and query.casefold() in " ".join(str(row.get(key, "")) for key in ("name", "reason", "check_state", "source_url")).casefold()]
        offset, limit = max(0, offset), max(1, min(100, limit))
        return {"total": len(rows), "counts": counts, "items": [{key: value for key, value in row.items() if key != "path"} | {"image": info(row.get("path"))} for row in rows[offset:offset + limit]]}

    @app.get("/api/avatars/{avatar_id}/resource-tabs")
    def resource_tabs(avatar_id: str, batch_id: str | None = None):
        check_id(avatar_id)
        if batch_id and not avatar_gallery_jobs.load_batch(str(root), avatar_id, batch_id):
            raise ValueError("Gallery batch not found")
        return {"items": avatar_resource_tabs.list_tabs(str(root), avatar_id, batch_id)}

    @app.delete("/api/avatars/{avatar_id}/resource-tabs/{tab_id}")
    def delete_resource_tab(avatar_id: str, tab_id: str):
        with mutation_lock:
            ensure_idle(check_id(avatar_id))
            result = avatar_resource_tabs.delete(str(root), avatar_id, tab_id)
            jobs.flush()
            return result

    @app.post("/api/avatars/{avatar_id}/gallery/{item_id}/approve")
    def approve_gallery_image(avatar_id: str, item_id: str):
        with mutation_lock:
            ensure_idle(check_id(avatar_id))
            item = studio.gallery_item_by_id(str(root), avatar_id, check_id(item_id))
            if not item or not item.get("generated_path"):
                raise ValueError("Select a generated image first")
            image_file(service.relative(item["generated_path"]))
            updated = studio.update_gallery_item(str(root), avatar_id, item_id, status="passed",
                manual_review={"decision": "approved", "time": time.time()})
            jobs.flush()
            return gallery_value(updated)

    @app.get("/api/avatars/{avatar_id}/references")
    def references(avatar_id: str, status: str = "All findings", batch_id: str | None = None):
        check_id(avatar_id)
        if status not in {"All findings", "Approved", "Rejected", "Awaiting checks"}:
            raise ValueError("Unknown reference filter")
        if batch_id and not avatar_gallery_jobs.load_batch(str(root), avatar_id, batch_id):
            raise ValueError("Gallery batch not found")
        snapshot = avatar_gallery_jobs.safe_snapshot(str(root), avatar_id, status, batch_id)
        catalog = (avatar_resource_tabs.family_references(str(root), avatar_id, snapshot["job"]["id"])
            if snapshot["job"] and not snapshot["job"].get("role")
            else avatar_gallery_jobs.batch_references(str(root), avatar_id, snapshot["job"]))
        def matching(row):
            return status == "All findings" or (status == "Approved" and row["state"] == "approved") or (status == "Rejected" and row["state"] == "rejected") or (status == "Awaiting checks" and row["state"] in {"found", "downloaded"})
        return {"counts": snapshot["counts"], "job": snapshot["job"], "events": snapshot["events"], "error": snapshot.get("error"),
                "items": [{"id": row["id"], "state": row["state"], "reason": row.get("reason", ""), "image": info(row.get("path")),
                           "source_url": row.get("source_url") or row.get("image_url"), "validation": row.get("validation")} for row in catalog if matching(row)]}

    @app.get("/api/avatars/{avatar_id}/gallery/batches")
    def gallery_batches(avatar_id: str):
        return {"items": avatar_gallery_jobs.list_batches(str(root), check_id(avatar_id))}

    @app.post("/api/avatars/{avatar_id}/gallery/batches/{batch_id}")
    def manage_gallery_batch(avatar_id: str, batch_id: str, body: BatchAction):
        with mutation_lock:
            ensure_idle(check_id(avatar_id))
            result = avatar_gallery_jobs.manage_batch(str(root), avatar_id, batch_id, body.action, body.name)
            jobs.flush()
            return result

    @app.post("/api/avatars/{avatar_id}/gallery/new-search")
    def new_search(avatar_id: str):
        ensure_idle(check_id(avatar_id))
        avatar_gallery_jobs.new_search(str(root), avatar_id)
        jobs.flush()
        return {"saved_references_retained": True}

    @app.get("/api/chat")
    def chat():
        data = service.main_history()
        return {"turns": [{**row, "images": [picture for path in row.get("images", []) if (picture := info(root / path))]} for row in data["turns"]]}

    @app.delete("/api/chat")
    def clear_chat():
        ensure_idle()
        path = root / "_sessions" / "web_chat.json"
        if path.exists():
            path.replace(path.with_name(f"chat-archive-{int(time.time())}.json"))
        service.chats.archive(service.chats.get()["id"])
        service.save_main(service.chats.create())
        jobs.flush()
        return {"archived": True}

    @app.post("/api/uploads")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "Images must be under 20 MB")
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > 25_000_000:
                    raise ValueError("Image dimensions exceed 25 megapixels")
                image = ImageOps.exif_transpose(source).convert("RGBA" if source.mode == "RGBA" else "RGB")
                path = root / "_uploads" / f"{secrets.token_hex(16)}.png"
                path.parent.mkdir(exist_ok=True)
                image.save(path)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Upload a readable PNG, JPEG or WebP image") from exc
        return {**info(path), "width": image.width, "height": image.height, "name": Path(file.filename or "Image").name}

    def image_file(path):
        result = service.file(path)
        if result.suffix.lower() not in IMAGE_EXTENSIONS or any(part.startswith(".") for part in Path(path).parts):
            raise HTTPException(404, "Image not found")
        return result

    @app.get("/api/files/{path:path}")
    def file(path: str, download: bool = False):
        target = image_file(path)
        return FileResponse(target, filename=target.name if download else None, content_disposition_type="attachment" if download else "inline")

    @app.get("/api/thumbs/{path:path}")
    def thumbnail(path: str, size: int = 420, region: str | None = None):
        source = image_file(path)
        size = min(800, max(80, size))
        box = None
        if region:
            if len(region) > 100:
                raise ValueError("Invalid detail crop")
            box = [float(value) for value in region.split(",")]
            if len(box) != 4 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in box) or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("Invalid detail crop")
        stamp = f"{source}:{source.stat().st_mtime_ns}:{size}:{box}"
        target = root / "_thumbs" / (hashlib.sha256(stamp.encode()).hexdigest() + ".webp")
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                if box:
                    pixels = (int(box[0] * image.width), int(box[1] * image.height), max(1, math.ceil(box[2] * image.width)), max(1, math.ceil(box[3] * image.height)))
                    image = image.crop(pixels)
                image.thumbnail((size, size))
                buffer = io.BytesIO()
                image.save(buffer, format="WEBP", quality=100 if box else 86, lossless=bool(box))
            temporary = target.with_suffix("." + secrets.token_hex(4) + ".tmp")
            temporary.write_bytes(buffer.getvalue())
            temporary.replace(target)
        return FileResponse(target, media_type="image/webp")

    @app.get("/api/jobs")
    def all_jobs():
        return [{**row, "error": redact(row.get("error", "")), "message": redact(row["message"])} for row in jobs.list()[:100]]

    @app.get("/api/activity/events")
    async def activity_events(request: Request):
        async def events():
            import asyncio
            signature = None
            heartbeat = time.monotonic()
            while not jobs.stopping.is_set() and not await request.is_disconnected():
                rows = await asyncio.to_thread(all_jobs)
                current = tuple((row["id"], row["revision"]) for row in rows)
                if current != signature:
                    signature = current
                    yield f"data: {json.dumps(rows)}\n\n"
                    heartbeat = time.monotonic()
                elif time.monotonic() - heartbeat >= 15:
                    yield "event: heartbeat\ndata: {}\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(1)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    @app.post("/api/jobs")
    def submit(body: JobRequest, idempotency_key: str = Header(..., min_length=8, max_length=128)):
        args = body.model_dump(exclude={"kind"})
        if body.avatar_id:
            check_id(body.avatar_id)
            studio.load_avatar(str(root), body.avatar_id)
        if body.item_id:
            check_id(body.item_id)
        if body.kind not in {"generate", "enhance", "mask_preview", "encoder", "unload"} and not body.avatar_id:
            raise ValueError("Select an avatar first")
        if body.kind in {"generate", "console"}:
            scope = body.avatar_id if body.kind == "console" else None
            if body.kind == "generate" and body.avatar_id:
                raise ValueError("Use the avatar console for avatar conversations")
            if body.new_chat and body.chat_id:
                raise ValueError("A new conversation cannot also select an existing chat")
            if body.chat_id:
                service.chats.get(body.chat_id, scope)
        for path in [*body.images, *body.positives, *body.negatives, *[value for value in (body.image, body.mask) if value]]:
            image_file(path)
        if body.kind in {"identity_view", "calibrate", "benchmark", "quality_check", "caption", "publish", "reconcile"}:
            service.ready_avatar(body.avatar_id)
        if body.kind == "console" and body.roles and len(body.roles) != len(body.images):
            raise ValueError("Choose one role for each additional reference.")
        if body.kind in {"quality_check", "caption"} and not body.item_id:
            raise ValueError("Select a saved avatar image.")
        if body.kind in {"publish", "reconcile"} and not body.post_id:
            raise ValueError("Select a saved content draft.")
        if body.kind in {"generate", "console"}:
            if not body.prompt.strip():
                raise ValueError("Write a prompt first")
            if body.kind == "generate" and body.quantity > 8:
                raise ValueError("Generate at most eight images per chat request")
            if body.kind == "console" and len(body.images) > 2:
                raise ValueError("Avatar Console has two extra reference slots")
            if body.model not in model_manager.MODEL_REGISTRY:
                raise ValueError("Unknown model")
            generation.dimensions(body.aspect)
            if body.canvas is not None and body.canvas >= len(body.images):
                raise ValueError("The selected canvas no longer exists")
        if body.kind in {"confirm", "mask_preview"} and not body.image:
            raise ValueError("Choose an input image first")
        if body.kind == "enhance":
            if not body.adaptive_prompt and (body.enhancement_profile != "auto" or body.additional_instructions.strip()):
                raise ValueError("Custom prompt mode uses only the supplied reconstruction prompt; clear automatic profile instructions")
            if body.enhancement_profile != "auto" and body.enhancement_profile not in enhancement_profiles.PROFILES:
                raise ValueError("Choose an available enhancement prompt profile")
            if body.sampling_mode == "reference" and body.steps != 4:
                raise ValueError("FLUX Klein reference reconstruction uses four distilled steps")
            if not body.reconstruction_prompt.strip():
                raise ValueError("Write a reconstruction prompt first")
            if body.chat_id or not body.avatar_id:
                conversation = service.chats.get(body.chat_id, body.avatar_id)
                args["chat_id"] = conversation["id"]
                if not body.image or not any(body.image in turn.get("images", []) for turn in conversation["turns"] if turn.get("role") == "assistant"):
                    raise ValueError("Select a generated image from this conversation")
            elif not body.item_id:
                raise ValueError("Select a saved image first")
        if body.kind == "regenerate" and not body.item_id:
            raise ValueError("Select a saved image first")
        with mutation_lock:
            saved_jobs = jobs.list()
            previous = next((row for row in saved_jobs if row["request_key"] == idempotency_key), None)
            if body.kind in {"resource_search", "resource_generate"}:
                if body.kind == "resource_generate":
                    generation.dimensions(body.aspect)
                if body.resource_tab_id:
                    source, owner = avatar_resource_tabs.get(str(root), body.avatar_id, body.resource_tab_id)
                    if body.batch_id and owner["id"] != body.batch_id:
                        raise ValueError("Resource tab does not belong to the selected batch")
                    if owner.get("archived"):
                        raise ValueError("Restore this batch before using its resource tabs")
                    if body.kind == "resource_search" and body.theme.strip() and body.theme.strip() != source["theme"]:
                        raise ValueError("Create a new resource tab for a different search description")
                    args["batch_id"] = owner["id"]
                elif body.kind == "resource_generate":
                    raise ValueError("Select a resource tab before generating")
                elif not body.batch_id or not body.theme.strip():
                    raise ValueError("Choose a batch and write a resource search description")
            if body.kind == "gallery" and body.batch_id and previous is None:
                if body.new_batch:
                    raise ValueError("Choose either a new batch or an existing batch")
                batch = avatar_gallery_jobs.load_batch(str(root), check_id(body.avatar_id), body.batch_id)
                if not batch or batch.get("role") or batch.get("archived") or batch["status"] in {"complete", "exhausted"}:
                    raise ValueError("Select an unfinished, unarchived batch to resume")
            for pending in saved_jobs:
                if previous is None and pending["request_key"] != idempotency_key and pending["status"] in {"queued", "running"}:
                    if pending["payload"].get("avatar_id") == body.avatar_id:
                        raise HTTPException(409, "An operation is already active for this workspace")
            if body.kind in {"generate", "console"}:
                if body.new_chat:
                    if previous:
                        if previous["kind"] != body.kind or not previous["payload"].get("new_chat") or previous["payload"].get("avatar_id") != body.avatar_id:
                            raise ValueError("This request key was already used for a different operation")
                        args["chat_id"] = previous["payload"]["chat_id"]
                    else:
                        if body.kind == "console":
                            service.ready_avatar(body.avatar_id)
                        service.chats.ensure(scope)
                        args["chat_id"] = service.chats.create(scope)["id"]
                else:
                    args["chat_id"] = service.chats.get(body.chat_id, scope)["id"]
            if body.kind == "gallery" and body.new_batch:
                if previous:
                    args["batch_id"] = previous["payload"]["batch_id"]
                else:
                    service.ready_avatar(body.avatar_id)
                    batch_id = hashlib.sha256((body.avatar_id + idempotency_key).encode()).hexdigest()[:12]
                    batch = avatar_gallery_jobs._create_batch(str(root), body.avatar_id, body.quantity, body.aspect,
                        body.theme, body.require_face, body.batch_name, batch_id, body.avatar_gender)
                    avatar_gallery_jobs.update_batch(str(root), body.avatar_id, batch_id=batch_id, status="pending", phase="pending")
                    args["batch_id"] = batch["id"]
            if body.kind == "resource_search" and not body.resource_tab_id:
                if previous:
                    args["resource_tab_id"] = previous["payload"]["resource_tab_id"]
                else:
                    service.ready_avatar(body.avatar_id)
                    identifier = hashlib.sha256((body.avatar_id + idempotency_key + ":resource").encode()).hexdigest()[:12]
                    tab = avatar_resource_tabs.create_tab(str(root), body.avatar_id, body.batch_id, identifier, body.theme, body.resource_tab_name)
                    args["resource_tab_id"] = tab["id"]
            job = jobs.submit(body.kind, args, idempotency_key)
            if body.kind in {"generate", "console"} and body.new_chat:
                service.chats.append(args["chat_id"], scope, {"id": job["id"] + "-user", "role": "user", "text": body.prompt, "images": body.images})
                jobs.flush()
            return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        row = jobs.get(job_id)
        return {**row, "message": redact(row["message"]), "error": redact(row.get("error", ""))}

    @app.post("/api/jobs/{job_id}/{action}")
    def job_action(job_id: str, action: Literal["cancel", "resume"]):
        row = jobs.get(job_id)
        if action == "resume":
            ensure_idle(row["payload"].get("avatar_id"))
        elif row["kind"] == "gallery" and row["payload"].get("avatar_id"):
            avatar_gallery_jobs.stop_batch(str(root), row["payload"]["avatar_id"], row.get("checkpoint", {}).get("batch_id") or row["payload"].get("batch_id"))
        return jobs.cancel(job_id) if action == "cancel" else jobs.resume(job_id)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        jobs.get(job_id)
        async def events():
            import asyncio
            revision = -1
            while not jobs.stopping.is_set() and not await request.is_disconnected():
                row = await asyncio.to_thread(get_job, job_id)
                if row["revision"] != revision:
                    revision = row["revision"]
                    yield f"id: {revision}\ndata: {json.dumps(row)}\n\n"
                else:
                    yield ": heartbeat\n\n"
                if row["status"] in {"completed", "failed", "interrupted", "cancelled"}:
                    return
                await asyncio.sleep(1)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get("/api/diagnostics")
    def diagnostics():
        return {"provider": provider(), "mock": model_manager.DEV_MODE, "model": model_manager.get_current_model(),
                "models": model_manager.get_model_status(), "sync": os.environ.get("FFS_STORAGE_MODE", "direct"),
                "errors": [{"name": path.name, "text": redact(path.read_text(encoding="utf-8", errors="replace")[-20000:])}
                           for path in sorted((root / "_debug").glob("job_*.txt"), reverse=True)[:5]]}

    frontend = Path(__file__).parent / "frontend" / "dist"
    if frontend.is_dir():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")
        @app.api_route("/", methods=["GET", "HEAD"])
        @app.api_route("/{path:path}", methods=["GET", "HEAD"])
        def index(path: str = ""):
            if path.split("/", 1)[0] in {"api", "assets"} or Path(path).suffix:
                return JSONResponse({"detail": "Not found"}, status_code=404)
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/")
        def index():
            return JSONResponse({"error": "Frontend bundle missing. Run npm ci and npm run build in frontend/."}, status_code=503)
    return app
