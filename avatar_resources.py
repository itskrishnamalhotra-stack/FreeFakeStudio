"""Read-only inventory of an avatar's identity anchors and saved references."""
import hashlib
from pathlib import Path

import avatar_gallery_jobs as batches
import avatar_studio as studio
import avatar_identity as identity


def collect(root, avatar_id, chats):
    root = Path(root).resolve()
    avatar = studio.load_avatar(str(root), avatar_id)
    catalog_path = studio.avatar_path(str(root), avatar_id) / "gallery/references/catalog.json"
    if not catalog_path.exists():
        batches.import_legacy_references(str(root), avatar_id)
    catalog = batches.load_catalog(str(root), avatar_id)["references"]
    outputs = studio.load_gallery_items(str(root), avatar_id)
    used = batches._used_references(str(root), avatar_id)
    committed = {str(row.get("reference_id")) for row in outputs if row.get("reference_id")}
    rows, by_path = [], {}

    def add(identifier, path, **values):
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
    for row in catalog:
        state = row.get("state", "found")
        add("search-" + row["id"], row.get("path"), name=row.get("title") or "Search reference", kind="search",
            state=state if state in {"approved", "rejected"} else "pending", check_state=state,
            reason=row.get("reason", ""), source_url=row.get("source_url") or row.get("image_url"),
            validation=row.get("validation"), committed=row["id"] in committed,
            used=any(str(row.get(key)) in used for key in ("id", "path", "image_url") if row.get(key)))

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
