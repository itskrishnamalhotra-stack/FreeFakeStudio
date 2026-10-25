"""Read-only inventory of an avatar's identity anchors and saved references."""
import hashlib
from pathlib import Path

import avatar_gallery_jobs as batches
import avatar_studio as studio
import avatar_identity as identity
import avatar_resource_tabs as tabs


def collect(root, avatar_id, chats, batch_id=None, tab_id=None):
    root = Path(root).resolve()
    avatar = studio.load_avatar(str(root), avatar_id)
    catalog_path = studio.avatar_path(str(root), avatar_id) / "gallery/references/catalog.json"
    if not catalog_path.exists():
        batches.import_legacy_references(str(root), avatar_id)
    catalog = tabs.family_references(str(root), avatar_id, batch_id)
    outputs = studio.load_gallery_items(str(root), avatar_id)
    used, reserved = tabs.reference_usage(str(root), avatar_id)
    if tab_id:
        selected, owner = tabs.get(str(root), avatar_id, tab_id)
        if batch_id and owner["id"] != batch_id:
            raise ValueError("Resource tab does not belong to the selected batch")
        catalog = tabs.tab_rows(str(root), avatar_id, selected)
    if batch_id:
        outputs = [row for row in outputs if row.get("batch_id") == batch_id]
    committed = {str(row[key]) for row in outputs for key in ("reference_id", "source_reference_id") if row.get(key)}
    rows, by_path = [], {}

    def add(identifier, path, **values):
        if tab_id and values.get("resource_tab_id") != tab_id:
            return
        if batch_id and values.get("batch_id") != batch_id:
            return
        resolved = (root / path).resolve() if path else None
        if resolved is not None and not resolved.is_relative_to(root):
            return
        key = str(resolved) if resolved else None
        if key and key in by_path:
            by_path[key]["used"] |= values.get("used", False)
            by_path[key]["committed"] |= values.get("committed", False)
            return
        row = {"id": identifier, "path": str(resolved) if resolved else None,
               "used": False, "committed": False, **values}
        rows.append(row)
        if key:
            by_path[key] = row

    for kind in ("face", "body"):
        for field, title in ((kind + "_image", kind.title() + " anchor"), (kind + "_candidate", kind.title() + " candidate")):
            if avatar.get(field):
                locked = field.endswith("_image") and avatar.get(kind + "_locked")
                add(field, avatar[field], name=title, kind="identity", state="approved" if locked else "pending",
                    check_state="confirmed" if locked else "unconfirmed", reason="Confirmed identity anchor" if locked else "Awaiting confirmation",
                    used=bool(locked and any(row.get("generated_path") for row in outputs)))
    for candidate in identity.pack(str(root), avatar_id)["candidates"]:
        approved = candidate.get("approved", False)
        consumed = any(Path(path).stem == candidate["id"] for output in outputs for path in (output.get("recipe") or {}).get("references", []))
        add("identity-" + candidate["id"], candidate["path"], name=candidate["view"].replace("-", " ") + " reference", kind="identity",
            state="approved" if approved else "pending", check_state="human_approved" if approved else "awaiting_approval",
            reason="Approved identity-pack reference" if approved else "Identity candidate; awaiting human approval", used=consumed)
    sources = {}
    for row in batches.project_references(str(root), avatar_id, catalog):
        state = row.get("state", "found")
        source_id = row.get("batch_id") or tab_id
        if source_id not in sources:
            sources[source_id] = batches.load_batch(str(root), avatar_id, source_id) if source_id else {}
        source = sources[source_id]
        owner_id = (source or {}).get("owner_batch_id", source_id)
        keys = tabs.reference_keys(row)
        consumed = bool(keys & used)
        held = bool(keys & reserved and not consumed)
        add("search-" + row["id"], row.get("path"), name=row.get("title") or "Search reference", kind="search",
            batch_id=owner_id or batch_id, resource_tab_id=source_id, reserved=held,
            state=state if state in {"approved", "rejected"} else "pending", check_state=state,
            reason=row.get("reason", ""), source_url=row.get("source_url") or row.get("image_url"),
            validation=row.get("validation"), committed=row["id"] in committed,
            used=consumed)

    def uploaded(path, consumed):
        if path:
            identifier = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]
            add("upload-" + identifier, path, name="Uploaded reference", kind="uploaded", state="pending", check_state="not_checked",
                reason="Uploaded reference; not checked by the reference-search pipeline.", used=consumed)

    history = avatar.get("chat_history", [])
    conversations = chats.list(avatar_id, include_archived=True)
    scoped_users = {turn["id"].removesuffix("-user") for chat in conversations for turn in chat["turns"] if turn.get("role") == "user"}
    completed = {(turn.get("metadata") or {}).get("job_id") for turn in history if turn.get("role") == "assistant" and turn.get("image_path")}
    for turn in history:
        metadata = turn.get("metadata") or {}
        if turn.get("role") == "user" and metadata.get("job_id") not in scoped_users:
            for path in metadata.get("extra_references", []):
                uploaded(path, bool(metadata.get("job_id") and metadata["job_id"] in completed))
    for chat in conversations:
        completed = {turn["id"].removesuffix("-assistant") for turn in chat["turns"] if turn.get("role") == "assistant" and turn.get("images")}
        for turn in chat["turns"]:
            if turn.get("role") == "user":
                for path in turn.get("images", []):
                    uploaded(path, turn["id"].removesuffix("-user") in completed)
    return rows
