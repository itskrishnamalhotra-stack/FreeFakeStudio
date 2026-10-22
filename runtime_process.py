"""Stop only a verified previous server belonging to this notebook workspace."""
import os
from pathlib import Path
import signal
import time


def stop_previous(workspace, runtime, timeout=60):
    path = Path(workspace) / "config" / "app.pid"
    if not path.is_file():
        return
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        return
    if pid <= 1 or pid == os.getpid():
        return
    command = Path(f"/proc/{pid}/cmdline")
    if not command.exists():
        return
    arguments = command.read_bytes().decode(errors="replace").split("\x00")
    allowed = {str(Path(root) / "app" / name) for root in (workspace, runtime) for name in ("web_server.py", "app.py")}
    if not any(value in allowed for value in arguments):
        return
    print("Stopping the previous FreeFakeStudio server and saving its checkpoints...", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while command.exists() and time.monotonic() < deadline:
        # A stopped child may remain as a zombie until its notebook parent reaps it.
        status = Path(f"/proc/{pid}/status")
        if status.exists() and "State:\tZ" in status.read_text():
            break
        time.sleep(.5)
    else:
        if command.exists():
            raise RuntimeError("The previous server is still saving/stopping. Wait for it to finish before rerunning; no Drive files were restored over it.")
