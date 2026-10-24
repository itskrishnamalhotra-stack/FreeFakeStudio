"""Durable reference catalog and resumable Avatar Studio gallery batches."""

import hashlib
import json
import os
import threading
import uuid
from contextvars import copy_context
from pathlib import Path

from PIL import Image

import avatar_gallery as gallery
import avatar_studio as studio


_run_lock = threading.Lock()
_state_lock = threading.RLock()
TERMINAL = {"passed", "needs_review"}


def _read(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Saved gallery state cannot be read: {path}. Preserve this file for diagnosis.") from exc


def _root(save_dir, avatar_id):
    return studio.avatar_path(save_dir, avatar_id) / "gallery"


def load_catalog(save_dir, avatar_id):
    return _read(_root(save_dir, avatar_id) / "references" / "catalog.json",
                 {"version": 1, "references": [], "imported_reports": []})


def _save_catalog(save_dir, avatar_id, catalog):
    studio._atomic_json(_root(save_dir, avatar_id) / "references" / "catalog.json", catalog)


def _reference_id(row):
    return hashlib.sha256(str(row.get("image_url") or row.get("path")).encode("utf-8")).hexdigest()[:20]


def update_reference(save_dir, avatar_id, reference_id, **changes):
    catalog = load_catalog(save_dir, avatar_id)
    row = next(row for row in catalog["references"] if row["id"] == reference_id)
    row.update(changes)
    row["updated_at"] = studio._now()
    _save_catalog(save_dir, avatar_id, catalog)
    return row


def add_references(save_dir, avatar_id, references, theme="", legacy=False):
    catalog = load_catalog(save_dir, avatar_id)
    known = {row["id"] for row in catalog["references"]}
    for reference in references:
        reference_id = _reference_id(reference)
        if reference_id in known:
            continue
        row = {key: value for key, value in reference.items() if key != "validation_bytes"}
        row.update(id=reference_id, theme=theme, state="downloaded" if row.get("path") else "found",
                   reason="Previously gathered reference" if legacy else "Awaiting download",
                   created_at=studio._now())
        catalog["references"].append(row)
        known.add(reference_id)
    _save_catalog(save_dir, avatar_id, catalog)


def import_legacy_references(save_dir, avatar_id):
    folder = _root(save_dir, avatar_id) / "references"
    catalog = load_catalog(save_dir, avatar_id)
    imported = set(catalog.get("imported_reports", []))
    for path in sorted(folder.glob("search_*.json")):
        if path.name in imported:
            continue
        report = _read(path, {})
        add_references(save_dir, avatar_id, report.get("references", []), report.get("theme", ""), legacy=True)
        imported.add(path.name)
    catalog = load_catalog(save_dir, avatar_id)
    catalog["imported_reports"] = sorted(imported)
    _save_catalog(save_dir, avatar_id, catalog)
    known_paths = {os.path.abspath(row["path"]) for row in catalog["references"] if row.get("path")}
    orphans = [{"path": str(path), "description": "Previously saved reference"}
               for path in folder.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
               and os.path.abspath(path) not in known_paths]
    if orphans:
        add_references(save_dir, avatar_id, orphans, legacy=True)


def _batch_path(save_dir, avatar_id, batch_id):
    return _root(save_dir, avatar_id) / "batches" / f"{batch_id}.json"


def load_batch(save_dir, avatar_id):
    if not avatar_id:
        return None
    active = _read(_root(save_dir, avatar_id) / "active_batch.json")
    if not active:
        return None
    job = _read(_batch_path(save_dir, avatar_id, active["id"]))
    if job is None:
        raise RuntimeError("The active gallery batch file is missing. Preserve the gallery folder for diagnosis.")
    return job


def update_batch(save_dir, avatar_id, **changes):
    with _state_lock:
        job = load_batch(save_dir, avatar_id)
        job.update(changes)
        job["updated_at"] = studio._now()
        studio._atomic_json(_batch_path(save_dir, avatar_id, job["id"]), job)
        return job


def note(save_dir, avatar_id, message, phase=None):
    job = load_batch(save_dir, avatar_id)
    events = list(job.get("events", []))
    events.append({"time": studio._now(), "message": str(message)[:1500]})
    return update_batch(save_dir, avatar_id, message=str(message), events=events[-100:],
                        **({"phase": phase} if phase else {}))


def stop_batch(save_dir, avatar_id):
    job = load_batch(save_dir, avatar_id)
    if job and job["status"] != "complete":
        return update_batch(save_dir, avatar_id, stop_requested=True, message="Stopping after the current operation")
    return job


def new_search(save_dir, avatar_id):
    if _run_lock.locked():
        raise RuntimeError("Wait for the current gallery operation to finish stopping.")
    job = load_batch(save_dir, avatar_id)
    if job and job.get("item_ids") and job["status"] != "complete":
        raise ValueError("Resume the committed batch before starting another search.")
    if job and job["status"] != "complete":
        update_batch(save_dir, avatar_id, status="complete", phase="closed_search")
        note(save_dir, avatar_id, "Search closed; reference findings retained")


def _used_references(save_dir, avatar_id):
    items = studio.load_gallery_items(save_dir, avatar_id)
    return {str(item.get(key)) for item in items if item.get("generated_path")
            for key in ("reference_id", "reference_url", "reference_path") if item.get(key)}


def _available(save_dir, avatar_id, job):
    used = _used_references(save_dir, avatar_id)
    return [row for row in load_catalog(save_dir, avatar_id)["references"]
            if row.get("theme", "") in ("", job["theme"])
            and not any(str(row.get(key)) in used for key in ("id", "image_url", "path") if row.get(key))]


def _approved(save_dir, avatar_id, job):
    return [row for row in _available(save_dir, avatar_id, job)
            if row.get("state") == "approved" and os.path.isfile(row.get("path") or "")
            and (not job["require_face"] or (row.get("validation") or {}).get("face_visible") is True)]


def _create_batch(save_dir, avatar_id, quantity, aspect, theme, require_face):
    avatar = studio.load_avatar(save_dir, avatar_id)
    job = {"version": 1, "id": uuid.uuid4().hex[:12], "avatar_id": avatar_id,
           "status": "running", "phase": "finding", "quantity": max(1, min(20, int(quantity))),
           "aspect": aspect, "theme": str(theme or "").strip(), "require_face": bool(require_face),
           "identity_fingerprint": studio.identity_fingerprint(avatar), "item_ids": [],
           "reference_ids": [], "queries": [], "round": 0, "downloads": 0, "events": [],
           "stop_requested": False, "created_at": studio._now(), "message": "Gathering references"}
    studio._atomic_json(_batch_path(save_dir, avatar_id, job["id"]), job)
    studio._atomic_json(_root(save_dir, avatar_id) / "active_batch.json", {"id": job["id"]})
    return job


def _discover(save_dir, avatar_id, job, mock_discover=None):
    import_legacy_references(save_dir, avatar_id)
    folder = _root(save_dir, avatar_id) / "references"
    folder.mkdir(parents=True, exist_ok=True)
    if mock_discover:
        references, _ = mock_discover(avatar_id, job["quantity"], job["theme"])
        add_references(save_dir, avatar_id, references, job["theme"])
        for row in references:
            update_reference(save_dir, avatar_id, _reference_id(row), state="approved", reason="Mock visual check",
                             validation={"accept": True, "score": 100, "face_visible": True, "image_clear": True})
        return
    rounds = gallery._env_int("FFS_AVATAR_SEARCH_ROUNDS", 3, 1, 5)
    maximum = gallery._env_int("FFS_AVATAR_MAX_CANDIDATE_DOWNLOADS", 60, 10, 120)
    while True:
        gallery.check_cancelled()
        job = load_batch(save_dir, avatar_id)
        if len(_approved(save_dir, avatar_id, job)) >= job["quantity"]:
            return
        rows = _available(save_dir, avatar_id, job)
        pending = [row for row in rows if row.get("state") == "found"]
        for row in pending:
            if job["downloads"] >= maximum:
                break
            gallery.check_cancelled()
            note(save_dir, avatar_id, "Downloading and checking reference files", "cleaning")
            job = update_batch(save_dir, avatar_id, downloads=job["downloads"] + 1)
            try:
                if not str(row.get("image_url") or "").startswith(("http://", "https://")):
                    raise ValueError("Saved reference is missing and has no downloadable URL")
                downloaded = gallery._download_candidate(row, folder)
                clean = {key: value for key, value in downloaded.items() if key != "validation_bytes"}
                duplicates = {other.get("pixel_hash") for other in load_catalog(save_dir, avatar_id)["references"]
                              if other["id"] != row["id"] and other.get("state") != "rejected"}
                reason = ("Blank or nearly uniform image" if clean.get("local_quality", {}).get("blank") else
                          "Exact duplicate pixels" if clean.get("pixel_hash") and clean["pixel_hash"] in duplicates else "")
                clean.update(state="rejected" if reason else "downloaded", reason=reason or "Awaiting visual check")
                update_reference(save_dir, avatar_id, row["id"], **clean)
            except (gallery.requests.RequestException, OSError, ValueError, Image.DecompressionBombError) as exc:
                update_reference(save_dir, avatar_id, row["id"], state="rejected", reason=f"Download/file check: {type(exc).__name__}: {str(exc)[:300]}")
            yield "Reference file checked"
        rows = _available(save_dir, avatar_id, job)
        visual = [row for row in rows if row.get("state") == "downloaded" or (
            row.get("state") == "approved" and job["require_face"] and
            (row.get("validation") or {}).get("face_visible") is not True)]
        for start in range(0, len(visual), 8):
            if len(_approved(save_dir, avatar_id, job)) >= job["quantity"]:
                return
            gallery.check_cancelled()
            note(save_dir, avatar_id, "Gemini is checking clarity, subject and reference suitability", "cleaning")
            batch = []
            for row in visual[start:start + 8]:
                if not os.path.isfile(row.get("path") or ""):
                    update_reference(save_dir, avatar_id, row["id"], state="found", reason="Saved image missing; download pending")
                    continue
                try:
                    with Image.open(row["path"]) as image:
                        if image.width * image.height > 20_000_000 or min(image.size) < gallery.MIN_REFERENCE_EDGE:
                            raise ValueError("Image dimensions are unsuitable")
                        thumbnail = image.convert("RGB")
                    deviation = gallery.ImageStat.Stat(thumbnail.convert("L").resize((64, 64))).stddev[0]
                    stored_hash = hashlib.sha256(str(thumbnail.size).encode("ascii") + thumbnail.tobytes()).hexdigest()
                    other_hashes = {other.get("stored_pixel_hash") for other in load_catalog(save_dir, avatar_id)["references"]
                                    if other["id"] != row["id"] and other.get("state") != "rejected"}
                    if deviation < 2.0 or stored_hash in other_hashes:
                        reason = "Blank or nearly uniform image" if deviation < 2.0 else "Exact duplicate pixels"
                        update_reference(save_dir, avatar_id, row["id"], state="rejected", reason=reason)
                        continue
                    update_reference(save_dir, avatar_id, row["id"], stored_pixel_hash=stored_hash)
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    update_reference(save_dir, avatar_id, row["id"], state="rejected", reason=f"Local image check: {exc}")
                    continue
                thumbnail.thumbnail((768, 768), Image.Resampling.LANCZOS)
                buffer = gallery.io.BytesIO()
                thumbnail.save(buffer, "JPEG", quality=80)
                batch.append({**row, "validation_bytes": buffer.getvalue()})

            def checked(row, validation, accepted):
                reason = ("Visible face required" if job["require_face"] and validation.get("face_visible") is not True
                          else validation.get("reason") or ("Approved" if accepted else "Visual check failed"))
                update_reference(save_dir, avatar_id, row["id"], state="approved" if accepted else "rejected",
                                 validation=validation, reason=reason)

            if batch:
                gallery._validate_references(batch, on_result=checked, require_face=job["require_face"])
            yield "Reference visual checks saved"
        if len(_approved(save_dir, avatar_id, job)) >= job["quantity"]:
            return
        if job["round"] >= rounds or job["downloads"] >= maximum:
            return
        note(save_dir, avatar_id, f"Searching reference round {job['round'] + 1}/{rounds}", "finding")
        queries = job["queries"]
        if len(queries) <= job["round"]:
            variation = job["theme"] if job["round"] == 0 else f"{job['theme']}; different outfit, pose and setting"
            query = gallery._search_query(variation)
            queries = queries + [query]
            job = update_batch(save_dir, avatar_id, queries=queries)
        payload = gallery._tavily_search(queries[job["round"]], max_results=20)
        add_references(save_dir, avatar_id, gallery._candidate_images(payload), job["theme"])
        job = update_batch(save_dir, avatar_id, round=job["round"] + 1)
        yield "Search findings saved"


def _item(save_dir, avatar_id, item_id):
    return next(row for row in studio.load_gallery_items(save_dir, avatar_id) if row["id"] == item_id)


def run(*args, **kwargs):
    # Gradio can advance a sync generator in different worker contexts.
    context = copy_context()
    iterator = _run(*args, **kwargs)
    try:
        while True:
            try:
                value = context.run(next, iterator)
            except StopIteration:
                return
            yield value
    finally:
        context.run(iterator.close)


def _run(save_dir, avatar_id, quantity, aspect, theme, require_face, make_prompt, generate, validate, repair,
         mock_discover=None):
    if not _run_lock.acquire(blocking=False):
        raise RuntimeError("A gallery operation is still finishing. Wait before resuming.")
    job = None
    try:
        avatar = studio.load_avatar(save_dir, avatar_id)
        if not avatar.get("face_locked") or not avatar.get("body_locked"):
            raise ValueError("Confirm the face and body before generating a gallery.")
        job = load_batch(save_dir, avatar_id)
        if not job or job["status"] == "complete":
            job = _create_batch(save_dir, avatar_id, quantity, aspect, theme, require_face)
        elif job["identity_fingerprint"] != studio.identity_fingerprint(avatar):
            raise ValueError("This batch belongs to a different confirmed identity. Restore its references before resuming.")
        import avatar_identity as identity
        if job.get("identity_version"):
            avatar = identity.pin(save_dir, avatar, job["identity_version"])
        else:
            avatar = identity.freeze(save_dir, avatar)
            job = update_batch(save_dir, avatar_id, identity_version=avatar["identity_version"])
        import avatar_quality
        if "quality_profile" not in job:
            job = update_batch(save_dir, avatar_id, quality_profile=avatar_quality.profile(save_dir, avatar))
        avatar = {**avatar, "_quality_profile": job["quality_profile"]}
        job = update_batch(save_dir, avatar_id, status="running", stop_requested=False, last_error=None)

        def cancelled():
            return bool(load_batch(save_dir, avatar_id).get("stop_requested"))

        with gallery.api_context(lambda message: note(save_dir, avatar_id, message), cancelled):
            yield note(save_dir, avatar_id, "Resuming saved batch" if job["item_ids"] else "Gathering references")
            if not job["item_ids"]:
                for message in _discover(save_dir, avatar_id, job, mock_discover):
                    yield note(save_dir, avatar_id, message)
                job = load_batch(save_dir, avatar_id)
                references = _approved(save_dir, avatar_id, job)[:job["quantity"]]
                if not references:
                    raise RuntimeError("No references passed the saved cleaning checks. Inspect the reference findings.")
                item_ids = []
                for reference in references:
                    item_id = f"{job['id']}-{reference['id']}"
                    existing = {row["id"] for row in studio.load_gallery_items(save_dir, avatar_id)}
                    if item_id not in existing:
                        studio.record_gallery_item(save_dir, avatar_id, {
                            "id": item_id, "batch_id": job["id"], "origin": "auto", "status": "awaiting_prompt",
                            "reference_id": reference["id"], "reference_path": reference["path"],
                            "reference_url": reference.get("image_url"), "reference_source": reference.get("source_url"),
                            "reference_validation": reference.get("validation"), "theme": job["theme"],
                            "aspect": job["aspect"], "attempt": 0, "repairs": 0,
                        })
                    item_ids.append(item_id)
                job = update_batch(save_dir, avatar_id, item_ids=item_ids,
                                   reference_ids=[row["id"] for row in references], phase="prompting")
                yield note(save_dir, avatar_id, f"Committed {len(item_ids)} approved references to the batch")
            # All prompts are durably prepared before any new image is generated.
            for index, item_id in enumerate(job["item_ids"], 1):
                gallery.check_cancelled()
                item = _item(save_dir, avatar_id, item_id)
                if item.get("prompt"):
                    continue
                yield note(save_dir, avatar_id, f"Preparing prompt {index}/{len(job['item_ids'])}", "prompting")
                reference = next(row for row in load_catalog(save_dir, avatar_id)["references"] if row["id"] == item["reference_id"])
                prompt = make_prompt(avatar, job["theme"], reference)
                if not str(prompt or "").strip():
                    raise RuntimeError("The prompt writer returned an empty prompt.")
                studio.update_gallery_item(save_dir, avatar_id, item_id, prompt=prompt, status="queued")
                yield note(save_dir, avatar_id, f"Prompt {index} saved")
            max_repairs = gallery._env_int("FFS_AVATAR_GALLERY_RETRIES", 2, 0, 3)
            def generate_and_check(item_id, index):
                gallery.check_cancelled()
                item = _item(save_dir, avatar_id, item_id)
                reference = next(row for row in load_catalog(save_dir, avatar_id)["references"] if row["id"] == item["reference_id"])
                if item["status"] != "checking" or not os.path.isfile(item.get("generated_path") or ""):
                    yield note(save_dir, avatar_id, f"Generating image {index}/{len(job['item_ids'])}", "generating")
                    attempt = int(item.get("attempt", 0)) + 1
                    studio.update_gallery_item(save_dir, avatar_id, item_id, attempt=attempt, status="generating")
                    path, _, seed = generate(avatar, reference, item["prompt"], job["aspect"], item_id, attempt)
                    item = studio.update_gallery_item(save_dir, avatar_id, item_id, generated_path=path,
                                                      attempt=attempt, seed=seed, status="checking")
                gallery.check_cancelled()
                yield note(save_dir, avatar_id, f"Checking image {index}/{len(job['item_ids'])} with SmolVLM", "checking")
                with Image.open(item["generated_path"]) as image:
                    validation = validate(image.convert("RGB"), avatar, item["prompt"])
                status = "passed" if validation.get("pass") is True else "needs_review" if validation.get("review_only") else (
                    "needs_review" if item.get("repairs", 0) >= max_repairs else "repair_pending")
                studio.update_gallery_item(save_dir, avatar_id, item_id, validation=validation, status=status)
                yield note(save_dir, avatar_id, f"Image {index}: {status.replace('_', ' ')}")

            # Complete first attempts for the committed queue before asking Gemini for repairs.
            for index, item_id in enumerate(job["item_ids"], 1):
                gallery.check_cancelled()
                item = _item(save_dir, avatar_id, item_id)
                if item["status"] == "repair_pending":
                    continue
                if item["status"] in TERMINAL and os.path.isfile(item.get("generated_path") or ""):
                    continue
                yield from generate_and_check(item_id, index)
            for index, item_id in enumerate(job["item_ids"], 1):
                while _item(save_dir, avatar_id, item_id)["status"] == "repair_pending":
                    gallery.check_cancelled()
                    item = _item(save_dir, avatar_id, item_id)
                    yield note(save_dir, avatar_id, f"Repairing prompt for image {index}; saved image retained", "repairing")
                    prompt = repair(item["prompt"], item["validation"])
                    if not str(prompt or "").strip():
                        raise RuntimeError("Prompt repair returned an empty prompt.")
                    studio.update_gallery_item(save_dir, avatar_id, item_id, prompt=prompt,
                                               repairs=item.get("repairs", 0) + 1, status="queued")
                    yield from generate_and_check(item_id, index)
            items = [_item(save_dir, avatar_id, item_id) for item_id in job["item_ids"]]
            passed = sum(item["status"] == "passed" for item in items)
            message = f"Saved {len(items)} image(s): {passed} passed, {len(items) - passed} need review."
            if len(items) < job["quantity"]:
                message += f" {job['quantity'] - len(items)} fewer references passed the configured search budget."
            update_batch(save_dir, avatar_id, status="complete", phase="complete")
            yield note(save_dir, avatar_id, message)
    except gallery.GalleryCancelled as exc:
        if job:
            update_batch(save_dir, avatar_id, status="stopped")
            yield note(save_dir, avatar_id, str(exc))
    except Exception as exc:
        if job:
            update_batch(save_dir, avatar_id, status="paused", last_error=str(exc))
            note(save_dir, avatar_id, str(exc))
        raise
    finally:
        try:
            if job:
                current = load_batch(save_dir, avatar_id)
                if current and current["status"] == "running":
                    update_batch(save_dir, avatar_id, status="stopped", message="Batch stopped; pending work saved")
        finally:
            _run_lock.release()


def snapshot(save_dir, avatar_id, filter_value="All findings"):
    """Read-only UI projection; images and their reasons share the same records."""
    if not avatar_id:
        return {"job": None, "counts": {}, "images": [], "rows": [], "events": []}
    path = _root(save_dir, avatar_id) / "references" / "catalog.json"
    if not path.exists():
        import_legacy_references(save_dir, avatar_id)
    catalog = load_catalog(save_dir, avatar_id)["references"]
    job = load_batch(save_dir, avatar_id)
    items = studio.load_gallery_items(save_dir, avatar_id)
    batch_items = [row for row in items if job and row.get("batch_id") == job["id"]]
    counts = {"found": len(catalog), "approved": sum(row["state"] == "approved" for row in catalog),
              "rejected": sum(row["state"] == "rejected" for row in catalog),
              "queued": len(job.get("item_ids", [])) if job else 0,
              "prompts": sum(bool(row.get("prompt")) for row in batch_items),
              "generated": sum(bool(row.get("generated_path")) for row in batch_items),
              "passed": sum(row.get("status") == "passed" for row in batch_items),
              "review": sum(row.get("status") == "needs_review" for row in batch_items)}
    used = _used_references(save_dir, avatar_id)
    images, rows = [], []
    for row in catalog:
        state = row["state"]
        if filter_value == "Approved" and state != "approved":
            continue
        if filter_value == "Rejected" and state != "rejected":
            continue
        if filter_value == "Awaiting checks" and state not in {"found", "downloaded"}:
            continue
        validation = row.get("validation") or {}
        face = str(validation.get("face_visible", "not checked"))
        consumed = any(str(row.get(key)) in used for key in ("id", "path", "image_url") if row.get(key))
        label = state.replace("_", " ").title() + (" / Used" if consumed else "")
        reason = row.get("reason", "")
        rows.append([label, reason, face, row.get("image_url", "")])
        if os.path.isfile(row.get("path") or ""):
            images.append((row["path"], f"{label}: {reason}"))
    return {"job": job, "counts": counts, "images": images, "rows": rows,
            "events": (job or {}).get("events", [])[-6:]}


def safe_snapshot(save_dir, avatar_id, filter_value="All findings"):
    try:
        return snapshot(save_dir, avatar_id, filter_value)
    except (RuntimeError, OSError, KeyError, TypeError) as exc:
        return {"job": None, "counts": {}, "images": [], "rows": [], "events": [], "error": str(exc)}
