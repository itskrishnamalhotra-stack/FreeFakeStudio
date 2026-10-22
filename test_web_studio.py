"""HTTP/workflow regression tests. All inference is mocked, including vision."""
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["FREEFAKESTUDIO_DEV"] = "1"
from fastapi.testclient import TestClient
from PIL import Image
import model_manager
import avatar_studio
from web_api import create_app
from web_jobs import JobStore
from notebook_bootstrap import parse_keys
from portable_state import relocate_results
from runtime_environment import provider, mock_mode
from legacy_history import migrate
from drive_store import DriveStore, restore_selection, safe_name, digest


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "results"
        self.app = create_app(self.root)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.state.jobs.shutdown()
        self.temp.cleanup()

    def image(self, color="green", size=(128, 160)):
        buffer = io.BytesIO()
        Image.new("RGB", size, color).save(buffer, format="PNG")
        response = self.client.post("/api/uploads", files={"file": ("reference.png", buffer.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def job(self, kind, expect="completed", **values):
        import uuid
        response = self.client.post("/api/jobs", json={"kind": kind, **values}, headers={"Idempotency-Key": uuid.uuid4().hex})
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()["id"]
        for _ in range(400):
            job = self.client.get("/api/jobs/" + job_id).json()
            if job["status"] not in {"queued", "running"}:
                self.assertEqual(job["status"], expect, job)
                return job
            time.sleep(.02)
        self.fail("Mock job timed out")

    def avatar(self):
        response = self.client.post("/api/avatars", json={"name": "Test persona", "mode": "Custom"})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    def ready(self):
        avatar_id = self.avatar()
        for kind in ("face", "body"):
            job = self.job("reference", avatar_id=avatar_id, reference_kind=kind, details="Studio portrait")
            self.job("confirm", avatar_id=avatar_id, reference_kind=kind, image=job["result"]["candidate"])
        return avatar_id

    def test_models_and_no_gradio(self):
        config = self.client.get("/api/config").json()
        self.assertTrue(config["mock"])
        self.assertEqual(next(m for m in config["models"] if m["name"] == model_manager.FLUX_MODEL_NAME)["defaults"]["steps"], 4)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_main_generation_and_idempotency(self):
        header = {"Idempotency-Key": "same-request-123"}
        body = {"kind": "generate", "prompt": "A red studio chair", "quantity": 2}
        first = self.client.post("/api/jobs", json=body, headers=header).json()
        second = self.client.post("/api/jobs", json=body, headers=header).json()
        self.assertEqual(first["id"], second["id"])
        for _ in range(200):
            if self.app.state.jobs.get(first["id"])["status"] == "completed":
                break
            time.sleep(.02)
        self.assertEqual(len(self.client.get("/api/chat").json()["turns"]), 2)
        changed = self.client.post("/api/jobs", json={**body, "prompt": "different"}, headers=header)
        self.assertEqual(changed.status_code, 400)

    def test_multi_reference_and_canvas(self):
        a, b = self.image(), self.image("red")
        job = self.job("generate", prompt="Keep identity and change pose", images=[a["path"], b["path"]], canvas=1)
        self.assertEqual(len(job["result"]["images"]), 1)
        self.assertEqual(self.client.get(a["url"]).status_code, 200)
        self.assertEqual(self.client.get(a["thumb"]).status_code, 200)

    def test_mask_preview_and_generation(self):
        image = self.image()
        preview = self.job("mask_preview", image=image["path"], mask_mode="Background Only")
        self.assertTrue((self.root / preview["result"]["image"]).is_file())
        self.job("generate", prompt="Change background", images=[image["path"]], canvas=0, mask_mode="Background Only")

    def test_no_edit_on_text_only_model(self):
        image = self.image()
        job = self.job("generate", expect="failed", model="Z-Image Turbo", prompt="Change pose", images=[image["path"]])
        self.assertIn("does not support image input", job["error"])

    def test_manual_mask_rejects_empty_and_accepts_painted_alpha(self):
        from generation_service import resolve_mask
        import numpy as np
        image = Image.new("RGB", (80, 100))
        mask = Image.new("RGBA", (80, 100))
        with self.assertRaisesRegex(ValueError, "empty"):
            resolve_mask(image, "Manual Paint", mask)
        mask.putpixel((20, 30), (255, 255, 255, 255))
        pixels = resolve_mask(image, "Manual Paint", mask)
        self.assertEqual(pixels.shape, (100, 80))
        self.assertEqual(np.count_nonzero(pixels), 1)

    def test_body_requires_confirmed_face(self):
        avatar_id = self.avatar()
        job = self.job("reference", expect="failed", avatar_id=avatar_id, reference_kind="body")
        self.assertIn("Confirm the face", job["error"])

    def test_avatar_console_gallery_reconstruction(self):
        avatar_id = self.ready()
        self.job("console", avatar_id=avatar_id, prompt="Standing in a studio", images=[self.image()["path"]])
        avatar = self.client.get("/api/avatars/" + avatar_id).json()
        self.assertEqual(len(avatar["chat"]), 2)
        gallery = self.client.get(f"/api/avatars/{avatar_id}/gallery").json()
        self.assertEqual(gallery["total"], 1)
        item = gallery["items"][0]
        self.job("enhance", avatar_id=avatar_id, item_id=item["id"], method="photo", scale=1)
        gallery = self.client.get(f"/api/avatars/{avatar_id}/gallery").json()
        self.assertEqual(gallery["total"], 2)
        enhanced = next(row for row in gallery["items"] if row["origin"] == "enhancement")
        self.assertEqual(enhanced["processing_passes"], 1)
        self.job("regenerate", avatar_id=avatar_id, item_id=item["id"], reason="Fix expression")

    def test_guided_workflow(self):
        avatar = self.client.post("/api/avatars", json={"name": "Guided persona", "mode": "Guided"}).json()
        for _ in range(8):
            if not avatar["guide"]["active"]:
                break
            self.job("guide", avatar_id=avatar["id"])
            avatar = self.client.get("/api/avatars/" + avatar["id"]).json()
        self.assertTrue(avatar["face_locked"] and avatar["body_locked"])
        self.assertFalse(avatar["guide"]["active"])
        self.assertGreaterEqual(self.client.get(f"/api/avatars/{avatar['id']}/gallery").json()["total"], 2)

    def test_gallery_batch_resume_data(self):
        avatar_id = self.ready()
        self.job("gallery", avatar_id=avatar_id, quantity=2, theme="Casual studio")
        snapshot = self.client.get(f"/api/avatars/{avatar_id}/references").json()
        self.assertGreaterEqual(snapshot["counts"]["generated"], 2)
        self.assertEqual(self.client.get(f"/api/avatars/{avatar_id}/gallery").json()["total"], 2)

    def test_invalid_requests_and_traversal(self):
        self.assertEqual(self.client.post("/api/uploads", files={"file": ("bad.png", b"not an image")}).status_code, 400)
        response = self.client.post("/api/jobs", json={"kind": "generate", "prompt": "x", "images": ["../outside.png"]}, headers={"Idempotency-Key": "invalid-path-test"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/files/%2e%2e%2fweb_api.py").status_code, 400)
        self.assertEqual(self.client.post("/api/avatars", json={"name": "x"}, headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/jobs", json={"kind": "generate", "prompt": "x", "quantity": 999}, headers={"Idempotency-Key": "invalid-value-test"}).status_code, 422)

    def test_archive_history_and_restart(self):
        self.job("generate", prompt="Studio test")
        self.assertEqual(self.client.delete("/api/chat").status_code, 200)
        self.assertEqual(self.client.get("/api/chat").json()["turns"], [])
        self.assertTrue(list((self.root / "_sessions").glob("chat-archive-*.json")))

    def test_authentication(self):
        app = create_app(self.root / "private", token="private-test-token")
        try:
            client = TestClient(app)
            self.assertEqual(client.get("/api/config").status_code, 401)
            self.assertEqual(client.post("/api/session", json={"token": "wrong"}).status_code, 401)
            self.assertEqual(client.post("/api/session", json={"token": "private-test-token"}).status_code, 200)
            self.assertEqual(client.get("/api/config").status_code, 200)
        finally:
            app.state.jobs.shutdown()


class RuntimeTests(unittest.TestCase):
    def test_notebook_is_one_reproducible_cell(self):
        notebook = json.loads(Path("FreeFakeStudio.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 1)
        self.assertEqual("".join(notebook["cells"][0]["source"]), Path("notebook_cell.py").read_text(encoding="utf-8"))
        self.assertEqual(notebook["cells"][0]["outputs"], [])

    def test_kaggle_bootstrap_without_real_cloud_operations(self):
        import notebook_bootstrap
        class Secrets:
            def get_secret(self, name):
                return "mock-secret-value"
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), patch.dict(__import__('sys').modules, {"kaggle_secrets": SimpleNamespace(UserSecretsClient=Secrets)}):
            root = Path(directory)
            app = root / "app"
            app.mkdir()
            (app / "launch.py").write_text("pass\n")
            (root / "ComfyUI/.git").mkdir(parents=True)
            with patch.object(notebook_bootstrap.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="16384\n")), patch("drive_store.DriveStore.from_environment") as drive, patch.object(notebook_bootstrap, "exec", create=True) as execute:
                notebook_bootstrap.start({"MODEL_SET": "flux", "PRELOAD_FLUX": True}, root, app, True)
                self.assertEqual(os.environ["FFS_PROVIDER"], "kaggle")
                self.assertEqual(os.environ["FFS_PUBLIC_ROUTE"], "ngrok")
                self.assertEqual(os.environ["FFS_STORAGE_MODE"], "drive_api")
                self.assertEqual(os.environ["FFS_PRELOAD_FLUX"], "1")
                self.assertNotIn("FREEFAKESTUDIO_DEV", os.environ)
                drive.return_value.restore.assert_called_once()
                execute.assert_called_once()

    def test_key_parser_allowlist(self):
        data = parse_keys('# comment\nNGROK_AUTH_TOKEN=private\nGOOGLE_DRIVE_REFRESH_TOKEN: token\nPYTHONPATH=bad\nHF_TOKEN="example"')
        self.assertEqual(data["FFS_NGROK_AUTHTOKEN"], "private")
        self.assertNotIn("PYTHONPATH", data)
        self.assertEqual(data["GOOGLE_DRIVE_REFRESH_TOKEN"], "token")

    def test_kaggle_not_mock(self):
        with patch.dict(os.environ, {"FFS_PROVIDER": "kaggle", "FREEFAKESTUDIO_DEV": "0"}):
            self.assertEqual(provider(), "kaggle")
            self.assertFalse(mock_mode())
        with patch.dict(os.environ, {"FFS_PROVIDER": "local", "FREEFAKESTUDIO_DEV": "0"}):
            self.assertTrue(mock_mode())

    def test_relocation_and_legacy_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            root.mkdir()
            Image.new("RGB", (10, 10)).save(root / "test.png")
            (root / "record.json").write_text(json.dumps({"image": "/content/drive/MyDrive/FreeFakeStudio/results/test.png"}))
            self.assertEqual(relocate_results(root), ["record.json"])
            self.assertEqual(json.loads((root / "record.json").read_text())["image"], str(root / "test.png"))
            (root / "_sessions").mkdir()
            (root / "_sessions/current_chat.json").write_text(json.dumps({"history": ['<div class="ffs-turn-user"><div class="ffs-bubble">Hello &amp; world</div></div>','<a href="/gradio_api/file=/content/drive/MyDrive/FreeFakeStudio/results/test.png">image</a>']}))
            turns = migrate(root)["turns"]
            self.assertEqual(turns[0]["text"], "Hello & world")
            self.assertEqual(turns[1]["images"], ["test.png"])

    def test_job_restart_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            def handler(kind, payload, context):
                context.checkpoint(count=1)
                return {"saved": True}
            store = JobStore(directory, handler)
            row = store.submit("test", {}, "key")
            store.shutdown()
            store.update(row["id"], status="running")
            restored = JobStore(directory, handler)
            self.assertEqual(restored.get(row["id"])["status"], "interrupted")
            restored.shutdown()

    def test_drive_selection(self):
        self.assertFalse(safe_name("../file"))
        self.assertFalse(restore_selection("app/.git", True))
        self.assertTrue(restore_selection("results/avatar/example.png", False))
        self.assertTrue(restore_selection("ComfyUI/models/diffusion_models/flux-2-klein-4b.safetensors", False))
        self.assertFalse(restore_selection("ComfyUI/models/diffusion_models/ernie-image-turbo-Q6_K.gguf", False))
        self.assertTrue(restore_selection("bundles/caches/runtime.tar", False))

    def test_drive_conflict_preserves_local(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DriveStore("folder123456789", directory, "id", "secret", "refresh")
            path = Path(directory) / "results/test.json"
            path.parent.mkdir()
            path.write_text("local")
            store.index["results/test.json"] = {"id": "file1", "md5Checksum": "changed-remotely"}
            store.state["results/test.json"] = {"remote_md5": "original"}
            with self.assertRaisesRegex(RuntimeError, "conflict"):
                store.upload("results/test.json", path)
            self.assertEqual(path.read_text(), "local")


class DriveProtocolTests(unittest.TestCase):
    class Response:
        def __init__(self, status=200, data=None, headers=None, body=b""):
            self.status_code, self.data, self.headers, self.body = status, data or {}, headers or {}, body
        def json(self):
            return self.data
        def iter_content(self, size):
            yield self.body
        def close(self):
            pass

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = DriveStore("folder123456789", self.temp.name, "id", "secret", "refresh", sleep=lambda _: None)
        self.path = Path(self.temp.name) / "results/image.png"
        self.path.parent.mkdir()
        self.path.write_bytes(b"a" * 1048576)
        self.relative = "results/image.png"
        self.store.index["results"] = {"id": "results-folder", "mimeType": "application/vnd.google-apps.folder"}
        self.complete = self.Response(200, {"id": "new-file", "md5Checksum": digest(self.path)})
        self.start = self.Response(200, headers={"Location": "https://www.googleapis.com/upload/session"})

    def tearDown(self):
        self.temp.cleanup()

    def test_resumable_upload_probes_after_network_failure(self):
        with patch.object(self.store, "request", side_effect=[self.Response(data={"ids": ["new-file"]}), self.start,
                RuntimeError("network"), self.Response(308, headers={"Range": "bytes=0-524287"}), self.complete]) as request:
            self.store.upload(self.relative, self.path)
        self.assertEqual(request.call_args_list[3].kwargs["headers"]["Content-Range"], "bytes */1048576")
        self.assertEqual(request.call_args_list[4].kwargs["headers"]["Content-Range"], "bytes 524288-1048575/1048576")
        self.assertEqual(self.store.state[self.relative]["remote_md5"], digest(self.path))

    def test_expired_upload_restarts_with_same_id(self):
        with patch.object(self.store, "request", side_effect=[self.Response(data={"ids": ["new-file"]}), self.start,
                self.Response(404), self.start, self.complete]) as request:
            self.store.upload(self.relative, self.path)
        self.assertEqual(request.call_args_list[1].kwargs["json"]["id"], request.call_args_list[3].kwargs["json"]["id"])

    def test_create_conflict_reuses_verified_file(self):
        with patch.object(self.store, "request", side_effect=[self.Response(data={"ids": ["new-file"]}), self.Response(409), self.complete]):
            self.store.upload(self.relative, self.path)
        self.assertEqual(self.store.index[self.relative]["id"], "new-file")

    def test_pagination(self):
        with patch.object(self.store, "request", side_effect=[self.Response(data={"files": [{"id": "one"}], "nextPageToken": "next"}),
                self.Response(data={"files": [{"id": "two"}]})]) as request:
            self.assertEqual([row["id"] for row in self.store.children("folder")], ["one", "two"])
            self.assertEqual(request.call_args.kwargs["params"]["pageToken"], "next")

    def test_bad_download_preserves_original(self):
        row = {"id": "remote", "size": "3", "md5Checksum": "invalid"}
        with patch.object(self.store, "request", return_value=self.Response(body=b"bad")):
            with self.assertRaisesRegex(RuntimeError, "verification"):
                self.store.download(self.relative, row)
        self.assertEqual(self.path.stat().st_size, 1048576)

    def test_auth_error_does_not_expose_credentials(self):
        with patch.object(self.store.session, "post", return_value=self.Response(400, {"secret": "secret-must-never-appear-123"})):
            with self.assertRaisesRegex(RuntimeError, "authorization failed") as error:
                self.store.refresh()
        self.assertNotIn("secret-must-never-appear-123", str(error.exception))

    def test_repeated_checkpoints_do_not_rescan_entire_drive_tree(self):
        row = {"id": "existing", "md5Checksum": digest(self.path)}
        self.store.index[self.relative] = row
        self.store.remember(self.relative, row, self.path)
        with patch.object(self.store, "scan") as scan, patch.object(self.store, "upload") as upload:
            self.store.sync(("results",))
            self.store.sync(("results",))
        self.assertEqual(scan.call_count, 1)
        upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
