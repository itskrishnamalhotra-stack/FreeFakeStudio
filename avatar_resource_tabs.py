"""Saved search tabs and additional generation runs within an existing batch."""
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path

import avatar_gallery as gallery
import avatar_gallery_jobs as jobs
import avatar_studio as studio


resource_mode = ContextVar("resource_unused_only", default=False)


def reference_keys(row):
    return {str(row[key]) for key in ("id", "source_reference_id", "image_url", "path", "pixel_hash", "stored_pixel_hash") if row.get(key)}


def reference_usage(root, avatar_id):
    catalog = {row["id"]: row for row in jobs.load_catalog(root, avatar_id)["references"]}
    consumed, reserved = set(), set()
    states = {}
    def state(identifier):
        if identifier not in states:
            states[identifier] = jobs.load_batch(root, avatar_id, identifier)
        return states[identifier]
    # Reserve the selected resources even if checkpointing interrupted before
    # the generation queue had its item records written.
    for row in catalog.values():
        if row.get("generation_copy"):
            owner = state(row["batch_id"])
            if owner and owner["status"] not in {"complete", "exhausted"} and owner.get("completion_reason") != "finished_by_user":
                reserved |= reference_keys(row)
    for item in studio.load_gallery_items(root, avatar_id):
        keys = {str(item[key]) for key in ("reference_id", "source_reference_id", "reference_url", "reference_path",
                "reference_pixel_hash", "reference_stored_pixel_hash") if item.get(key)}
        keys |= reference_keys(catalog.get(item.get("reference_id"), {}))
        if item.get("generated_path"):
            consumed |= keys
        elif item.get("reference_id") and item.get("status") not in jobs.TERMINAL:
            owner = state(item.get("generation_run_id") or item.get("batch_id"))
            if owner and owner.get("completion_reason") != "finished_by_user":
                reserved |= keys
    return consumed, reserved


def get(root, avatar_id, tab_id):
    tab = jobs.load_batch(root, avatar_id, tab_id)
    if not tab or tab.get("role") not in {None, "", "resource_tab"} or tab.get("resource_deleted"):
        raise ValueError("Resource tab not found")
    owner = jobs.load_batch(root, avatar_id, tab.get("owner_batch_id", tab["id"]))
    if not owner or owner.get("role"):
        raise ValueError("The resource tab's batch is unavailable")
    return tab, owner


def tab_rows(root, avatar_id, tab):
    if tab.get("resource_deleted"):
        return []
    return [row for row in jobs.batch_references(root, avatar_id, tab) if not row.get("generation_copy")]


def family_references(root, avatar_id, batch_id=None, states=None, catalog=None):
    if states is None:
        states = {path.stem: jobs.load_batch(root, avatar_id, path.stem)
                  for path in (jobs._root(root, avatar_id) / "batches").glob("*.json")}
    if catalog is None:
        catalog = jobs.load_catalog(root, avatar_id)["references"]
    legacy = states.get(batch_id, {})
    legacy_ids = {row["id"] for row in jobs.batch_references(root, avatar_id, legacy)} if legacy and legacy.get("version", 1) < 2 and not legacy.get("resource_deleted") else set()
    return [row for row in catalog
            if not row.get("generation_copy")
            and not states.get(row.get("batch_id"), {}).get("resource_deleted")
            and (not batch_id or row["id"] in legacy_ids or states.get(row.get("batch_id"), {}).get("owner_batch_id", row.get("batch_id")) == batch_id)]


def list_tabs(root, avatar_id, batch_id=None):
    consumed, reserved = reference_usage(root, avatar_id)
    result = []
    for path in sorted((jobs._root(root, avatar_id) / "batches").glob("*.json")):
        tab = jobs.load_batch(root, avatar_id, path.stem)
        if tab.get("role") not in {None, "", "resource_tab"} or tab.get("resource_deleted"):
            continue
        owner_id = tab.get("owner_batch_id", tab["id"])
        if batch_id and owner_id != batch_id:
            continue
        owner = jobs.load_batch(root, avatar_id, owner_id)
        rows = jobs.project_references(root, avatar_id, tab_rows(root, avatar_id, tab), {tab["id"]: tab})
        counts = {"found": len(rows), "used": 0, "reserved": 0, "available": 0, "pending": 0, "rejected": 0}
        for row in rows:
            keys = reference_keys(row)
            counts["used"] += bool(keys & consumed)
            counts["reserved"] += bool(keys & reserved and not keys & consumed)
            valid = jobs.reference_ready(row, tab) and os.path.isfile(row.get("path") or "")
            counts["available"] += bool(valid and not keys & (consumed | reserved))
            counts["pending"] += bool(row.get("state") in {"found", "downloaded"} and not keys & (consumed | reserved))
            counts["rejected"] += row.get("state") == "rejected"
        result.append({"id": tab["id"], "batch_id": owner_id, "batch_name": owner.get("name") or owner.get("theme") or "Untitled batch",
            "name": tab.get("name") or tab.get("theme") or "Original search", "description": tab.get("theme", ""),
            "avatar_gender": tab.get("avatar_gender", "any"), "require_face": tab.get("require_face", False),
            "search_provider": tab.get("search_provider", "tavily"), "search_attempts": tab.get("search_attempts", [])[-10:],
            "archived": owner.get("archived", False), "counts": counts,
            "status": tab.get("resource_status", tab.get("status", "saved")),
            "queries": tab.get("queries", []),
            "error": tab.get("resource_error"), "created_at": tab["created_at"]})
    return sorted(result, key=lambda row: (row["created_at"], row["id"]))


def create_tab(root, avatar_id, batch_id, identifier, description, name="", search_provider=None):
    from reference_search import normalize_provider
    owner = jobs.load_batch(root, avatar_id, batch_id)
    if not owner or owner.get("role") or owner.get("archived"):
        raise ValueError("Choose an unarchived batch for the resource search")
    if not description.strip():
        raise ValueError("Write a resource search description")
    if jobs.load_batch(root, avatar_id, identifier):
        raise ValueError("Resource tab identifier already exists")
    tab = {**owner, "version": 2, "id": jobs._check_batch_id(identifier), "role": "resource_tab", "owner_batch_id": owner["id"],
        "name": (name.strip() or description.strip())[:120], "theme": description.strip(), "quantity": 20,
        "status": "pending", "phase": "finding", "item_ids": [], "reference_ids": [], "queries": [],
        "round": 0, "downloads": 0, "events": [], "stop_requested": False, "resource_deleted": False,
        "resource_status": "pending", "resource_error": None, "created_at": studio._now()}
    tab.update(search_provider=normalize_provider(search_provider or owner.get("search_provider")), search_attempts=[])
    studio._atomic_json(jobs._batch_path(root, avatar_id, identifier), tab)
    return tab


def _write_run(root, avatar_id, source, owner, run_id, references, quantity, aspect):
    run = {**owner, "version": 2, "id": run_id, "role": "generation_run", "owner_batch_id": owner["id"],
        "source_tab_id": source["id"], "name": source.get("name") or source["theme"], "theme": source["theme"], "quantity": quantity,
        "aspect": aspect, "avatar_gender": source.get("avatar_gender", "any"), "require_face": source.get("require_face", False),
        "search_provider": source.get("search_provider", "tavily"),
        "status": "pending", "phase": "prompting", "item_ids": [], "reference_ids": [], "events": [],
        "created_at": studio._now(), "stop_requested": False}
    studio._atomic_json(jobs._batch_path(root, avatar_id, run_id), run)
    token = jobs._batch_scope.set((str(Path(root).resolve()), avatar_id, run_id))
    try:
        jobs.add_references(root, avatar_id, references, source["theme"])
        for reference in references:
            jobs.update_reference(root, avatar_id, jobs._reference_id(reference), state="approved",
                generation_copy=True, source_reference_id=reference["id"], resource_tab_id=source["id"],
                validation=reference.get("validation", {}), reason="Selected unused resource")
    finally:
        jobs._batch_scope.reset(token)
    return run


def run(root, avatar_id, tab_id, quantity, aspect, context, make_prompt, generate, validate, repair,
        search=False, mock_discover=None, review_only=False, search_provider=None):
    source, owner = get(root, avatar_id, tab_id)
    if owner.get("archived"):
        raise ValueError("Restore this batch before using its resources")
    if source["identity_fingerprint"] != studio.identity_fingerprint(studio.load_avatar(root, avatar_id)):
        raise ValueError("This resource tab belongs to a different confirmed avatar identity")
    run_id = context.saved.get("generation_batch_id")
    if not run_id:
        if not jobs._run_lock.acquire(blocking=False):
            raise RuntimeError("Another gallery operation is still finishing")
        token = jobs._batch_scope.set((str(Path(root).resolve()), avatar_id, tab_id))
        mode = resource_mode.set(True)
        try:
            source = jobs._isolate_legacy(root, avatar_id, source)
            operation = context.id
            if source.get("resource_operation") != operation:
                changes = {"resource_operation": operation, "resource_status": "searching" if search else "checking",
                           "resource_error": None, "downloads": 0,
                           "resource_download_total": source.get("resource_download_total", 0) + source.get("downloads", 0)}
                if search:
                    from reference_search import normalize_provider
                    history = list(source.get("resource_search_history", []))
                    if source.get("queries"):
                        history.append({"queries": source["queries"], "time": studio._now()})
                    changes.update(round=0, downloads=0, queries=[], resource_search_history=history,
                                   search_provider=normalize_provider(search_provider or source.get("search_provider")),
                                   resource_target=len(jobs._approved(root, avatar_id, source)) + quantity)
                else:
                    changes.update(resource_target=1_000_000 if review_only else quantity)
                source = jobs.update_batch(root, avatar_id, **changes)
            context.checkpoint(resource_tab_id=tab_id, owner_batch_id=owner["id"])
            with gallery.api_context(context.emit, lambda: context.store.stopping.is_set() or context.store.get(context.id).get("cancel_requested", False)):
                for message in jobs._discover(root, avatar_id, source, mock_discover if search else None,
                        target_quantity=source["resource_target"], allow_search=search, mock_validate=mock_discover is not None):
                    context.emit(message)
                    context.checkpoint(resource_tab_id=tab_id)
            jobs.update_batch(root, avatar_id, resource_status="ready", resource_error=None)
            if search or review_only:
                return {"resource_tab_id": tab_id, "batch_id": owner["id"], "search_only": search, "review_only": review_only}
            source = jobs.load_batch(root, avatar_id, tab_id)
            references = jobs._approved(root, avatar_id, source)[:quantity]
            if not references:
                raise ValueError("No unused approved resources remain in this tab. Search more or choose another tab.")
            run_id = hashlib.sha256((context.id + ":generation").encode()).hexdigest()[:12]
            if not jobs.load_batch(root, avatar_id, run_id):
                _write_run(root, avatar_id, source, owner, run_id, references, quantity, aspect)
            context.checkpoint(generation_batch_id=run_id, resource_tab_id=tab_id, owner_batch_id=owner["id"])
        except Exception as exc:
            jobs.update_batch(root, avatar_id, resource_status="paused", resource_error=str(exc))
            raise
        finally:
            resource_mode.reset(mode)
            jobs._batch_scope.reset(token)
            jobs._run_lock.release()
    context.check()
    saved_run = jobs.load_batch(root, avatar_id, run_id)
    if saved_run and saved_run["status"] in {"complete", "exhausted"}:
        return {"batch_id": owner["id"], "resource_tab_id": tab_id, "generation_batch_id": run_id}
    iterator = jobs.run(root, avatar_id, quantity, aspect, source["theme"], source.get("require_face", False),
                        make_prompt, generate, validate, repair, batch_id=run_id, mock_discover=mock_discover)
    try:
        for state in iterator:
            context.emit(state["message"], phase=state["phase"], resource_tab_id=tab_id)
            context.checkpoint(generation_batch_id=run_id, resource_tab_id=tab_id)
    finally:
        iterator.close()
    return {"batch_id": owner["id"], "resource_tab_id": tab_id, "generation_batch_id": run_id}


def delete(root, avatar_id, tab_id):
    tab, _ = get(root, avatar_id, tab_id)
    tab = jobs._isolate_legacy(root, avatar_id, tab)
    rows = tab_rows(root, avatar_id, tab)
    _, reserved = reference_usage(root, avatar_id)
    if any(reference_keys(row) & reserved for row in rows):
        raise ValueError("This tab has resources reserved by unfinished generations. Resume or finish those jobs before deleting it.")
    removed_ids = {row["id"] for row in rows}
    catalog = jobs.load_catalog(root, avatar_id)
    items = studio.load_gallery_items(root, avatar_id)
    needed = {row["reference_id"] for row in items if row.get("reference_id") and row.get("generated_path")}
    remaining = [row for row in catalog["references"] if row["id"] not in removed_ids
        and not (row.get("generation_copy") and row.get("resource_tab_id") == tab_id and row["id"] not in needed)]
    lookup = {row["id"]: row for row in rows}
    for item in items:
        original = lookup.get(item.get("reference_id"))
        if original:
            item.setdefault("reference_pixel_hash", original.get("pixel_hash"))
            item.setdefault("reference_stored_pixel_hash", original.get("stored_pixel_hash"))
    studio.save_gallery_items(root, avatar_id, items)
    # Persist the tombstone first so interruption cannot resurrect a deleted tab.
    jobs.update_batch(root, avatar_id, batch_id=tab_id, resource_deleted=True)
    catalog["references"] = remaining
    jobs._save_catalog(root, avatar_id, catalog)
    retained = {str(Path(row["path"]).resolve()) for row in remaining if row.get("path")}
    def paths(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from paths(child)
        elif isinstance(value, list):
            for child in value:
                yield from paths(child)
        elif isinstance(value, str):
            yield value
    for value in paths(items):
        try:
            path = Path(value)
            if path.is_absolute():
                retained.add(str(path.resolve()))
        except (OSError, ValueError):
            continue
    folder = (jobs._root(root, avatar_id) / "references").resolve()
    deleted = 0
    for path in {Path(row["path"]) for row in rows if row.get("path")}:
        resolved = path.resolve()
        if resolved.is_relative_to(folder) and str(resolved) not in retained and path.is_file():
            path.unlink()
            deleted += 1
    return {"deleted_tab": tab_id, "removed_resources": len(rows), "deleted_files": deleted, "generated_images_preserved": True}
