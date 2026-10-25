"""Stop only a verified previous server belonging to this notebook workspace."""
import json
import os
from pathlib import Path
import signal
import time

PROC_ROOT = Path("/proc")


def _process_identity(pid):
    try:
        # Start time distinguishes the recorded server from a subsequently reused PID.
        fields = (PROC_ROOT / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        arguments = (PROC_ROOT / str(pid) / "cmdline").read_bytes().decode(errors="replace").split("\x00")
        return fields[19], tuple(arguments)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, IndexError) as exc:
        raise RuntimeError(f"Cannot verify previous server PID {pid}; refusing to restore files over it.") from exc


def _shutdown_stage(workspace, pid, started):
    labels = {
        "closing_connections": "closing live progress connections",
        "saving_jobs": "waiting for current operation / saving job checkpoints",
        "syncing_storage": "syncing saved state to storage",
        "saved": "storage saved; waiting for process exit",
        "stopped": "shutdown complete; waiting for process exit",
    }
    try:
        row = json.loads((Path(workspace) / "config" / "app.shutdown.json").read_text())
        if row.get("pid") == pid and isinstance(row.get("time"), (int, float)) and row["time"] >= started - 1:
            return labels.get(row.get("stage"), "waiting for shutdown")
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return "waiting for the previous server (close old app tabs if this is an older release)"


def stop_previous(workspace, runtime, timeout=300):
    path = Path(workspace) / "config" / "app.pid"
    if not path.is_file():
        return
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        return
    if pid <= 1 or pid == os.getpid():
        return
    identity = _process_identity(pid)
    if identity is None:
        return
    _, arguments = identity
    allowed = {str(Path(root) / "app" / name) for root in (workspace, runtime) for name in ("web_server.py", "app.py")}
    if not any(value in allowed for value in arguments):
        return
    print(f"Stopping the previous FreeFakeStudio server (PID {pid}); allowing current work to save...", flush=True)
    started = time.time()
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    begin = time.monotonic()
    deadline = begin + timeout
    heartbeat = begin + 10
    while _process_identity(pid) == identity:
        now = time.monotonic()
        if now >= deadline:
            stage = _shutdown_stage(workspace, pid, started)
            raise RuntimeError(
                f"Previous server PID {pid} has not exited after {timeout}s: {stage}. "
                "No files were restored over the live server and it was not force-killed. "
                "Close ALL old FreeFakeStudio app tabs (keep this notebook open), wait for any "
                "generation/Drive sync to finish, then rerun the cell once. "
                "Refreshing the notebook page does not stop the server. "
                "If it remains stuck, preserve results and inspect config/app.shutdown.json "
                "and diagnostics/app_launch_latest.log before restarting the runtime."
            )
        if now >= heartbeat:
            print(f"[shutdown] {int(now - begin)}s: {_shutdown_stage(workspace, pid, started)}", flush=True)
            heartbeat = now + 10
        time.sleep(.5)
    try:
        if path.read_text().strip() == str(pid):
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    print("Previous FreeFakeStudio server stopped; continuing setup.", flush=True)
