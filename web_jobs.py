"""Single-worker, durable jobs. Reconnection never resubmits a model operation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback
import uuid

from avatar_studio import _atomic_json


class Cancelled(RuntimeError):
    pass


class JobContext:
    def __init__(self, store, job_id):
        self.store, self.id = store, job_id

    def check(self):
        if self.store.stopping.is_set() or self.store.get(self.id).get("cancel_requested"):
            raise Cancelled("Stopped after the current operation. Saved work is preserved.")

    def emit(self, message, **detail):
        self.check()
        self.store.update(self.id, message=str(message), detail=detail)

    def checkpoint(self, **changes):
        row = self.store.get(self.id)
        saved = {**row.get("checkpoint", {}), **changes}
        self.store.update(self.id, checkpoint=saved)
        self.store.flush()

    @property
    def saved(self):
        return self.store.get(self.id).get("checkpoint", {})


class JobStore:
    def __init__(self, root, handler, persist=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.handler = handler
        self.persist = persist or (lambda: None)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-gpu")
        self.closed = False
        self.stopping = threading.Event()
        for row in self.list():
            if row["status"] in {"queued", "running"}:
                self.update(row["id"], status="interrupted", message="Runtime restarted. Resume saved work.")

    def path(self, job_id):
        if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Invalid job identifier")
        return self.root / f"{job_id}.json"

    def get(self, job_id):
        with self.lock:
            return json.loads(self.path(job_id).read_text(encoding="utf-8"))

    def update(self, job_id, **changes):
        with self.lock:
            row = self.get(job_id)
            row.update(changes, updated_at=time.time(), revision=row.get("revision", 0) + 1)
            _atomic_json(self.path(job_id), row)
            return row

    def list(self):
        with self.lock:
            rows = [json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("*.json")]
        return sorted(rows, key=lambda row: row.get("created_at", 0), reverse=True)

    def submit(self, kind, payload, key):
        signature = hashlib.sha256(json.dumps({"kind": kind, "payload": payload}, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if self.closed or self.stopping.is_set():
                raise RuntimeError("The job worker is shutting down")
            previous = next((row for row in self.list() if row.get("request_key") == key), None)
            if previous:
                if previous["signature"] != signature:
                    raise ValueError("This request key was already used for a different operation")
                return previous
            job_id = uuid.uuid4().hex
            row = {"id": job_id, "kind": kind, "payload": payload, "request_key": key,
                   "signature": signature, "status": "queued", "message": "Queued",
                   "created_at": time.time(), "updated_at": time.time(), "revision": 0,
                   "checkpoint": {}, "cancel_requested": False}
            _atomic_json(self.path(job_id), row)
            try:
                self.flush()
            except Exception as exc:
                self.update(job_id, status="failed", error=str(exc), message="Drive sync failed before execution; resume when connected")
                raise
            self.executor.submit(self._run, job_id)
            return row

    def resume(self, job_id):
        with self.lock:
            if self.closed or self.stopping.is_set():
                raise RuntimeError("The job worker is shutting down")
            row = self.get(job_id)
            if row["status"] not in {"failed", "interrupted", "cancelled"}:
                raise ValueError("Only a stopped, interrupted or failed job can be resumed")
            self.update(job_id, status="queued", cancel_requested=False, message="Resuming saved work", error=None, sync_error=None)
            self.executor.submit(self._run, job_id)
        return self.get(job_id)

    def cancel(self, job_id):
        if self.get(job_id)["status"] not in {"queued", "running"}:
            return self.get(job_id)
        return self.update(job_id, cancel_requested=True, message="Stopping after the current operation")

    def flush(self):
        self.persist()

    def _run(self, job_id):
        try:
            context = JobContext(self, job_id)
            context.check()
            row = self.update(job_id, status="running", message="Starting")
            result = self.handler(row["kind"], row["payload"], context)
            context.check()
            self.update(job_id, status="completed", result=result, message="Complete")
            self.flush()
        except Cancelled as exc:
            self.update(job_id, status="cancelled", message=str(exc))
        except Exception as exc:
            debug = self.root.parent / "_debug" / f"job_{job_id}.txt"
            debug.parent.mkdir(exist_ok=True)
            debug.write_text(traceback.format_exc(), encoding="utf-8")
            self.update(job_id, status="failed", error=str(exc), message=str(exc), debug_path=str(debug))
        finally:
            try:
                self.flush()
            except Exception as exc:
                self.update(job_id, sync_error=str(exc))

    def request_shutdown(self):
        self.stopping.set()

    def shutdown(self, on_progress=None):
        self.request_shutdown()
        with self.lock:
            self.closed = True
            for row in self.list():
                if row["status"] in {"queued", "running"}:
                    self.cancel(row["id"])
        self.executor.shutdown(wait=True, cancel_futures=True)
        # Futures cancelled before starting never enter _run's saving/finally path.
        for row in self.list():
            if row["status"] in {"queued", "running"}:
                self.update(row["id"], status="cancelled", message="Server stopped. Resume saved work.")
        if on_progress:
            on_progress("syncing_storage")
        self.flush()
