"""Durable, avatar-scoped conversations with idempotent legacy migration."""
import json
from pathlib import Path
import re
import threading
import time
import uuid

import avatar_studio as studio

ID = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")


class ChatStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()

    def folder(self, avatar_id=None):
        if avatar_id:
            if not ID.fullmatch(avatar_id):
                raise ValueError("Invalid avatar identifier")
            studio.load_avatar(str(self.root), avatar_id)
        return self.root / "_sessions" / "chats" / ("avatar-" + avatar_id if avatar_id else "main")

    def path(self, chat_id, avatar_id=None):
        if not ID.fullmatch(str(chat_id)) or str(chat_id).startswith("_"):
            raise ValueError("Invalid conversation identifier")
        return self.folder(avatar_id) / (chat_id + ".json")

    def create(self, avatar_id=None, title="New chat", turns=None, chat_id=None):
        with self.lock:
            now = time.time()
            row = {"version": 1, "id": chat_id or uuid.uuid4().hex, "avatar_id": avatar_id,
                   "title": str(title).strip()[:100] or "New chat", "created_at": now, "updated_at": now,
                   "archived": False, "turns": turns or []}
            studio._atomic_json(self.path(row["id"], avatar_id), row)
            return row

    def ensure(self, avatar_id=None):
        with self.lock:
            folder = self.folder(avatar_id)
            marker = folder / "_migrated.json"
            if marker.exists():
                return
            if avatar_id:
                avatar = studio.load_avatar(str(self.root), avatar_id)
                turns = []
                for index, turn in enumerate(avatar.get("chat_history", [])):
                    metadata = turn.get("metadata") or {}
                    paths = ([turn["image_path"]] if turn.get("image_path") else [])
                    if turn.get("role") == "user":
                        paths += metadata.get("extra_references", [])
                    images = [str(Path(path).resolve().relative_to(self.root)).replace("\\", "/")
                              for path in paths if Path(path).resolve().is_relative_to(self.root) and Path(path).is_file()]
                    turns.append({"id": metadata.get("job_id", f"legacy-{index}") + "-" + turn.get("role", "assistant"),
                                  "role": turn.get("role", "assistant"), "text": turn.get("content", ""),
                                  "images": images, "seed": metadata.get("seed")})
                if turns and not self.path("legacy", avatar_id).exists():
                    self.create(avatar_id, "Previous conversation", turns, "legacy")
            else:
                current = self.root / "_sessions" / "web_chat.json"
                if current.exists():
                    history = json.loads(current.read_text(encoding="utf-8"))
                else:
                    from legacy_history import migrate
                    history = migrate(self.root)
                if history.get("turns") and not self.path("legacy").exists():
                    self.create(title="Previous conversation", turns=history["turns"], chat_id="legacy")
                for path in sorted((self.root / "_sessions").glob("chat-archive-*.json")):
                    history = json.loads(path.read_text(encoding="utf-8"))
                    chat_id = path.stem
                    if history.get("turns") and not self.path(chat_id).exists():
                        self.create(title="Saved conversation", turns=history["turns"], chat_id=chat_id)
            studio._atomic_json(marker, {"version": 1, "completed_at": time.time()})

    def list(self, avatar_id=None, include_archived=False):
        with self.lock:
            self.ensure(avatar_id)
            rows = [json.loads(path.read_text(encoding="utf-8")) for path in self.folder(avatar_id).glob("*.json") if not path.name.startswith("_")]
            return sorted([row for row in rows if include_archived or not row.get("archived")], key=lambda row: row["updated_at"], reverse=True)

    def get(self, chat_id=None, avatar_id=None):
        with self.lock:
            self.ensure(avatar_id)
            if chat_id is None:
                rows = self.list(avatar_id)
                return rows[0] if rows else self.create(avatar_id)
            row = json.loads(self.path(chat_id, avatar_id).read_text(encoding="utf-8"))
            if row.get("archived"):
                raise ValueError("This conversation is archived")
            return row

    def save(self, row):
        with self.lock:
            row = {**row, "updated_at": time.time()}
            studio._atomic_json(self.path(row["id"], row.get("avatar_id")), row)
            return row

    def append(self, chat_id, avatar_id, turn):
        with self.lock:
            row = self.get(chat_id, avatar_id)
            if any(item["id"] == turn["id"] for item in row["turns"]):
                return row
            if turn["role"] == "user" and not any(item.get("role") == "user" for item in row["turns"]):
                row["title"] = " ".join(turn.get("text", "").split())[:64] or "Image conversation"
            row["turns"].append(turn)
            return self.save(row)

    def rename(self, chat_id, avatar_id, title):
        row = self.get(chat_id, avatar_id)
        row["title"] = str(title).strip()[:100]
        if not row["title"]:
            raise ValueError("Give this conversation a name")
        return self.save(row)

    def archive(self, chat_id, avatar_id=None):
        row = self.get(chat_id, avatar_id)
        row["archived"] = True
        return self.save(row)
