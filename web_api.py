"""Authenticated same-origin HTTP API for the TypeScript studio."""
from contextlib import asynccontextmanager
import hashlib
import io
import json
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
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageOps, UnidentifiedImageError

import avatar_enhance
import avatar_gallery
import avatar_gallery_jobs
import avatar_guide
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


class JobRequest(StrictModel):
    kind: Literal["generate", "reference", "confirm", "guide", "console", "gallery", "regenerate", "enhance", "mask_preview", "encoder", "unload"]
    avatar_id: str | None = None
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
    scale: Literal[1, 2] = 1
    strength: float = Field(default=1, ge=.35, le=1)
    encoder: Literal["Official", "Custom"] = "Official"


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

    def handle(kind, payload, context):
        if kind == "encoder":
            context.emit("Switching FLUX text encoder")
            return {"encoder": model_manager.set_flux_encoder_mode(payload["encoder"])}
        if kind == "unload":
            model_manager.unload_current(status_callback=context.emit)
            return {"unloaded": True}
        return service.handle(kind, payload, context)

    jobs = JobStore(root / "_jobs", handle, persist)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.shutdown()

    app = FastAPI(title="FreeFakeStudio", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.jobs, app.state.service = jobs, service

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
            "current_step": avatar.get("current_step"), "creation_mode": avatar.get("creation_mode", "custom"),
            "guide": {"stage": avatar_guide.stage(avatar), "next_label": avatar_guide.next_label(avatar),
                      "active": avatar_guide.active(avatar) and avatar_guide.stage(avatar) != "complete", "options": dict(zip(["face", "body", "scene_one", "scene_two"], avatar_guide.options(avatar))),
                      "face_draft": info(avatar_guide.draft(avatar, "face")), "body_draft": info(avatar_guide.draft(avatar, "body"))},
            "chat": chat, "gallery_count": avatar.get("gallery_count", 0)}

    def gallery_value(row):
        return {key: row.get(key) for key in ("id", "status", "origin", "prompt", "seed", "attempt", "validation", "source_id", "diagnostics",
                                             "working_size", "output_size", "processing_passes", "method", "strength", "steps")} | {
            "image": info(row.get("generated_path")), "source": info(row.get("source_path")),
            "reference": info(row.get("reference_path")), "regenerable": row.get("origin") not in {"enhancement", "legacy"} and bool(row.get("prompt"))}

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
                "mock": model_manager.DEV_MODE, "provider": provider(), "gallery_api": avatar_gallery.configuration_status() if not model_manager.DEV_MODE else {"ready": True},
                "active_model": model_manager.get_current_model(), "max_references": 4, "max_avatar_extras": 2}

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

    @app.get("/api/avatars/{avatar_id}/gallery")
    def gallery(avatar_id: str, offset: int = 0, limit: int = 60, status: str = "all", origin: str = "all", query: str = ""):
        check_id(avatar_id)
        rows = studio.visible_gallery_items(str(root), avatar_id, limit=None)
        rows = [row for row in reversed(rows) if (status == "all" or row.get("status") == status)
                and (origin == "all" or row.get("origin") == origin) and query.casefold() in str(row.get("prompt", "")).casefold()]
        offset, limit = max(0, offset), max(1, min(100, limit))
        return {"total": len(rows), "items": [gallery_value(row) for row in rows[offset:offset + limit]], "offset": offset}

    @app.get("/api/avatars/{avatar_id}/references")
    def references(avatar_id: str, status: str = "All findings"):
        check_id(avatar_id)
        if status not in {"All findings", "Approved", "Rejected", "Awaiting checks"}:
            raise ValueError("Unknown reference filter")
        snapshot = avatar_gallery_jobs.safe_snapshot(str(root), avatar_id, status)
        catalog = avatar_gallery_jobs.load_catalog(str(root), avatar_id)["references"]
        def matching(row):
            return status == "All findings" or (status == "Approved" and row["state"] == "approved") or (status == "Rejected" and row["state"] == "rejected") or (status == "Awaiting checks" and row["state"] in {"found", "downloaded"})
        return {"counts": snapshot["counts"], "job": snapshot["job"], "events": snapshot["events"], "error": snapshot.get("error"),
                "items": [{"id": row["id"], "state": row["state"], "reason": row.get("reason", ""), "image": info(row.get("path")),
                           "source_url": row.get("source_url") or row.get("image_url"), "validation": row.get("validation")} for row in catalog if matching(row)]}

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
        service.save_main({"version": 1, "turns": []})
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
    def thumbnail(path: str, size: int = 420):
        source = image_file(path)
        size = min(800, max(80, size))
        stamp = f"{source}:{source.stat().st_mtime_ns}:{size}"
        target = root / "_thumbs" / (hashlib.sha256(stamp.encode()).hexdigest() + ".webp")
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((size, size))
                buffer = io.BytesIO()
                image.save(buffer, format="WEBP", quality=86)
            temporary = target.with_suffix("." + secrets.token_hex(4) + ".tmp")
            temporary.write_bytes(buffer.getvalue())
            temporary.replace(target)
        return FileResponse(target, media_type="image/webp")

    @app.get("/api/jobs")
    def all_jobs():
        return [{**row, "error": redact(row.get("error", "")), "message": redact(row["message"])} for row in jobs.list()[:100]]

    @app.post("/api/jobs")
    def submit(body: JobRequest, idempotency_key: str = Header(..., min_length=8, max_length=128)):
        args = body.model_dump(exclude={"kind"})
        if body.avatar_id:
            check_id(body.avatar_id)
            studio.load_avatar(str(root), body.avatar_id)
        if body.item_id:
            check_id(body.item_id)
        if body.kind not in {"generate", "mask_preview", "encoder", "unload"} and not body.avatar_id:
            raise ValueError("Select an avatar first")
        for path in [*body.images, *[value for value in (body.image, body.mask) if value]]:
            image_file(path)
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
        if body.kind in {"enhance", "regenerate"} and not body.item_id:
            raise ValueError("Select a saved image first")
        with mutation_lock:
            for pending in jobs.list():
                if pending["request_key"] != idempotency_key and pending["status"] in {"queued", "running"}:
                    if pending["payload"].get("avatar_id") == body.avatar_id:
                        raise HTTPException(409, "An operation is already active for this workspace")
            return jobs.submit(body.kind, args, idempotency_key)

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
            avatar_gallery_jobs.stop_batch(str(root), row["payload"]["avatar_id"])
        return jobs.cancel(job_id) if action == "cancel" else jobs.resume(job_id)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        jobs.get(job_id)
        async def events():
            import asyncio
            revision = -1
            while not await request.is_disconnected():
                row = get_job(job_id)
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
        @app.get("/")
        def index():
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/")
        def index():
            return JSONResponse({"error": "Frontend bundle missing. Run npm ci and npm run build in frontend/."}, status_code=503)
    return app
