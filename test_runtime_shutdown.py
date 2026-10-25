"""Shutdown regressions: real HTTP streams, mocked jobs, no AI inference."""
import asyncio
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import runtime_process
from web_jobs import JobStore
from web_lifecycle import StudioServer
import uvicorn


class ProcessStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.pid_file = self.workspace / "config" / "app.pid"
        self.pid_file.parent.mkdir()
        self.pid_file.write_text("987654")
        self.identity = ("123", ("python", str(self.workspace / "app" / "web_server.py"), ""))

    def tearDown(self):
        self.temp.cleanup()

    def test_waits_for_same_process_then_clears_pid(self):
        with patch.object(runtime_process, "_process_identity", side_effect=[self.identity, self.identity, None]), \
             patch.object(runtime_process.os, "kill") as kill, patch.object(runtime_process.time, "sleep"):
            runtime_process.stop_previous(self.workspace, self.workspace / "runtime")
        kill.assert_called_once_with(987654, signal.SIGTERM)
        self.assertFalse(self.pid_file.exists())

    def test_timeout_never_force_kills_or_removes_live_pid(self):
        with patch.object(runtime_process, "_process_identity", return_value=self.identity), \
             patch.object(runtime_process.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "Close ALL old FreeFakeStudio app tabs"):
                runtime_process.stop_previous(self.workspace, self.workspace / "runtime", timeout=0)
        kill.assert_called_once_with(987654, signal.SIGTERM)
        self.assertTrue(self.pid_file.exists())

    def test_never_signals_unrelated_or_substring_command(self):
        identity = ("123", ("python", str(self.workspace / "app" / "web_server.py") + ".other"))
        with patch.object(runtime_process, "_process_identity", return_value=identity), \
             patch.object(runtime_process.os, "kill") as kill:
            runtime_process.stop_previous(self.workspace, self.workspace / "runtime")
        kill.assert_not_called()

    def test_pid_reuse_stops_waiting_without_signalling_new_process(self):
        with patch.object(runtime_process, "_process_identity", side_effect=[self.identity, ("456", self.identity[1])]), \
             patch.object(runtime_process.os, "kill") as kill:
            runtime_process.stop_previous(self.workspace, self.workspace / "runtime")
        kill.assert_called_once_with(987654, signal.SIGTERM)

    def test_identity_handles_zombie_and_process_name_with_spaces(self):
        proc = self.workspace / "proc" / "987654"
        proc.mkdir(parents=True)
        (proc / "cmdline").write_bytes(b"python\0/app/web_server.py\0")
        stat = "987654 (server (worker)) " + " ".join(["S"] + ["0"] * 18 + ["123"])
        (proc / "stat").write_text(stat)
        with patch.object(runtime_process, "PROC_ROOT", proc.parent):
            self.assertEqual(runtime_process._process_identity(987654), ("123", ("python", "/app/web_server.py", "")))
            (proc / "stat").write_text(stat.replace(") S ", ") Z "))
            self.assertIsNone(runtime_process._process_identity(987654))
            (proc / "stat").unlink()
            self.assertIsNone(runtime_process._process_identity(987654))

    def test_progress_ignores_stale_or_foreign_report(self):
        report = self.pid_file.with_name("app.shutdown.json")
        report.write_text(json.dumps({"pid": 987654, "time": 100, "stage": "syncing_storage"}))
        self.assertIn("syncing saved state", runtime_process._shutdown_stage(self.workspace, 987654, 100))
        self.assertIn("older release", runtime_process._shutdown_stage(self.workspace, 1, 100))
        self.assertIn("older release", runtime_process._shutdown_stage(self.workspace, 987654, 200))

    def test_unreadable_process_fails_closed(self):
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(RuntimeError, "Cannot verify previous server"):
                runtime_process._process_identity(987654)


class JobShutdownTests(unittest.TestCase):
    def test_shutdown_waits_for_checkpoint_cancels_queue_and_syncs_final_state(self):
        with tempfile.TemporaryDirectory() as directory:
            entered, release = threading.Event(), threading.Event()
            snapshots = []

            def handler(kind, payload, context):
                entered.set()
                release.wait(5)
                context.checkpoint(completed_operation=1)
                context.check()

            store = JobStore(directory, handler)
            store.persist = lambda: snapshots.append(store.list())
            try:
                first = store.submit("test", {}, "first-request")
                self.assertTrue(entered.wait(2))
                second = store.submit("test", {}, "second-request")
                store.request_shutdown()
                with self.assertRaisesRegex(RuntimeError, "shutting down"):
                    store.submit("test", {}, "third-request")
                store.update(second["id"], status="interrupted")
                with self.assertRaisesRegex(RuntimeError, "shutting down"):
                    store.resume(second["id"])
                store.update(second["id"], status="queued")
                worker = threading.Thread(target=store.shutdown)
                worker.start()
                self.assertTrue(worker.is_alive())
                release.set()
                worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(store.get(first["id"])["checkpoint"], {"completed_operation": 1})
                self.assertEqual({row["status"] for row in snapshots[-1]}, {"cancelled"})
            finally:
                release.set()
                store.shutdown()


class StreamingShutdownTests(unittest.TestCase):
    def test_shutdown_rejects_writes_but_keeps_readable_saved_jobs(self):
        os.environ.setdefault("FREEFAKESTUDIO_DEV", "1")
        from fastapi.testclient import TestClient
        from web_api import create_app
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory, token="")
            try:
                app.state.jobs.request_shutdown()
                client = TestClient(app)
                self.assertEqual(client.post("/api/avatars", json={"name": "Do not create"}).status_code, 503)
                self.assertEqual(client.get("/api/jobs").status_code, 200)
                self.assertEqual(client.get("/api/avatars").json(), [])
            finally:
                app.state.jobs.shutdown()

    def test_open_activity_and_job_streams_do_not_block_shutdown_or_saving(self):
        os.environ.setdefault("FREEFAKESTUDIO_DEV", "1")
        from web_api import create_app
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "app.shutdown.json"
            with patch.dict(os.environ, {"FFS_SHUTDOWN_STATE_PATH": str(report)}):
                app = create_app(Path(directory) / "results", token="")
                store = app.state.jobs
                entered, release = threading.Event(), threading.Event()
                persisted = []

                def handler(kind, payload, context):
                    entered.set()
                    release.wait(8)
                    context.checkpoint(operation_saved=True)
                    context.check()

                store.handler = handler
                store.persist = lambda: persisted.append(store.list())
                row = store.submit("test", {}, "stream-shutdown-test")
                self.assertTrue(entered.wait(2))
                server = StudioServer(uvicorn.Config(app, log_level="warning", access_log=False))
                listener = socket.socket()
                listener.bind(("127.0.0.1", 0))
                listener.listen(10)
                port = listener.getsockname()[1]
                errors = []

                def run():
                    try:
                        asyncio.run(server.serve(sockets=[listener]))
                    except BaseException as exc:
                        errors.append(exc)

                thread = threading.Thread(target=run, daemon=True)
                connections = []
                thread.start()
                try:
                    deadline = time.monotonic() + 5
                    while not server.started and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(server.started, errors)
                    for path in ("/api/activity/events", f"/api/jobs/{row['id']}/events"):
                        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                        connections.append(connection)
                        connection.request("GET", path)
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertIn("text/event-stream", response.getheader("Content-Type"))
                    server.should_exit = True
                    self.assertTrue(store.stopping.wait(2))
                    self.assertTrue(thread.is_alive(), "Must not kill an in-flight operation")
                    release.set()
                    thread.join(6)
                    self.assertFalse(thread.is_alive(), "Open browser progress streams blocked shutdown")
                    self.assertEqual(errors, [])
                    self.assertEqual(json.loads(report.read_text())["stage"], "stopped")
                    self.assertEqual(persisted[-1][0]["status"], "cancelled")
                    self.assertEqual(persisted[-1][0]["checkpoint"], {"operation_saved": True})
                finally:
                    release.set()
                    server.should_exit = True
                    for connection in connections:
                        connection.close()
                    thread.join(5)
                    listener.close()
                    store.shutdown()


if __name__ == "__main__":
    unittest.main()
