"""Close progress streams before Uvicorn waits for HTTP connections to drain."""
import os
from pathlib import Path
import time

import uvicorn

from startup_restore import atomic_write_json


class StudioServer(uvicorn.Server):
    def __init__(self, config):
        super().__init__(config)
        self.studio = config.app
        self.studio.state.report_shutdown = self.report_shutdown

    def report_shutdown(self, stage):
        labels = {
            "closing_connections": "Closing live progress connections",
            "saving_jobs": "Waiting for the current operation and saving job checkpoints",
            "syncing_storage": "Saving final state to storage (including Drive sync when enabled)",
            "saved": "Job checkpoints and storage saved",
            "stopped": "Previous server stopped",
        }
        print(f"[shutdown] {labels[stage]}", flush=True)
        path = os.environ.get("FFS_SHUTDOWN_STATE_PATH")
        if path:
            try:
                atomic_write_json(Path(path), {"pid": os.getpid(), "stage": stage, "time": time.time()})
            except OSError as exc:
                print(f"[shutdown] Could not write progress report: {type(exc).__name__}", flush=True)

    async def shutdown(self, sockets=None):
        # Lifespan shutdown alone is too late: Uvicorn drains SSE responses first.
        self.studio.state.jobs.request_shutdown()
        self.report_shutdown("closing_connections")
        await super().shutdown(sockets=sockets)
        self.report_shutdown("stopped")
