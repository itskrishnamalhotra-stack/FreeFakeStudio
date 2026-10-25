"""Identity/content integration tests. No model inference or external requests."""
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import zipfile

os.environ["FREEFAKESTUDIO_DEV"] = "1"
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import avatar_studio as studio
import avatar_identity as identity
import avatar_quality as quality
import avatar_face as face
import avatar_content as content
import app_settings
from web_api import create_app
from notebook_bootstrap import parse_keys


class InfluencerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "results"
        self.app = create_app(self.root)
        self.client = TestClient(self.app)
        self.avatar_id = self.client.post("/api/avatars", json={"name": "Identity QA", "mode": "Custom"}).json()["id"]
        for part, color in (("face", "green"), ("body", "blue")):
            image = self.upload(color)
            self.job("confirm", reference_kind=part, image=image["path"])
        self.base = "/api/avatars/" + self.avatar_id

    def tearDown(self):
        self.app.state.jobs.shutdown()
        self.temp.cleanup()

    def upload(self, color):
        buffer = io.BytesIO()
        Image.new("RGB", (128, 160), color).save(buffer, "PNG")
        response = self.client.post("/api/uploads", files={"file": ("image.png", buffer.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def job(self, kind, expect="completed", **args):
        import uuid
        response = self.client.post("/api/jobs", json={"kind": kind, "avatar_id": self.avatar_id, **args}, headers={"Idempotency-Key": uuid.uuid4().hex})
        self.assertEqual(response.status_code, 200, response.text)
        for _ in range(600):
            row = self.app.state.jobs.get(response.json()["id"])
            if row["status"] not in {"queued", "running"}:
                self.assertEqual(row["status"], expect, row)
                return row
            time.sleep(.01)
        self.fail("Mock job timed out")

    def saved_item(self):
        self.job("console", prompt="Studio portrait", new_chat=True)
        return self.client.get(self.base + "/gallery").json()["items"][0]

    def post(self):
        item = self.saved_item()
        response = self.client.post(self.base + "/content", json={"image_id": item["id"], "caption": "A new portrait", "public_url": "https://example.com/photo.jpg"})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    def test_version_snapshot_is_immutable_and_repeatable(self):
        first = self.client.get(self.base + "/identity").json()
        second = self.client.get(self.base + "/identity").json()
        self.assertEqual(first["version"], second["version"])
        avatar = self.app.state.service.ready_avatar(self.avatar_id)
        self.assertNotEqual(avatar["face_image"], studio.load_avatar(self.root, self.avatar_id)["face_image"])
        original = identity._digest(avatar["face_image"])
        studio.lock_reference(self.root, self.avatar_id, "face", Image.new("RGB", (128, 160), "red"))
        studio.lock_reference(self.root, self.avatar_id, "body", Image.new("RGB", (128, 160), "blue"))
        self.assertEqual(identity._digest(avatar["face_image"]), original)
        self.assertNotEqual(self.client.get(self.base + "/identity").json()["version"], first["version"])

    def test_upload_review_remove_and_restore_pack(self):
        photo = self.upload("purple")
        result = self.client.post(self.base + "/identity", json={"action": "upload", "image": photo["path"], "view": "profile"})
        self.assertEqual(result.status_code, 200, result.text)
        candidate = result.json()["candidates"][0]
        self.assertFalse(candidate["approved"])
        self.assertTrue(candidate["image"]["url"].startswith("/api/files/"))
        approved = self.client.post(self.base + "/identity", json={"action": "approve", "candidate_id": candidate["id"], "approved": True}).json()
        version = approved["version"]
        pinned = self.app.state.service.ready_avatar(self.avatar_id)
        self.assertEqual(identity.select_references(pinned, "profile portrait")[0], pinned["_identity_pack"][0]["path"])
        self.assertEqual(len(pinned["_identity_pack"]), 1)
        self.client.post(self.base + "/identity", json={"action": "remove", "candidate_id": candidate["id"]})
        restored = self.client.post(self.base + "/identity", json={"action": "activate", "version": version})
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertEqual(restored.json()["version"], version)
        self.assertEqual(len(self.app.state.service.ready_avatar(self.avatar_id)["_identity_pack"]), 1)

    def test_pack_export_contains_contact_sheet_specs_and_refs(self):
        response = self.client.get(self.base + "/identity/export")
        self.assertEqual(response.status_code, 200, response.text[:100])
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertIn("contact-sheet.jpg", archive.namelist())
            self.assertIn("face-specs.json", archive.namelist())
            self.assertIn("body-specs.json", archive.namelist())

    def test_approving_a_view_replaces_its_previous_active_reference(self):
        first = identity.add_candidate(self.root, self.avatar_id, "profile", Image.new("RGB", (128, 160), "red"))
        second = identity.add_candidate(self.root, self.avatar_id, "profile", Image.new("RGB", (128, 160), "blue"))
        identity.review_candidate(self.root, self.avatar_id, first, True)
        identity.review_candidate(self.root, self.avatar_id, second, True)
        approved = [row for row in identity.pack(self.root, self.avatar_id)["candidates"] if row["approved"]]
        self.assertEqual([row["id"] for row in approved], [second])
        resources = self.client.get(self.base + "/resources?kind=identity").json()["items"]
        active = next(row for row in resources if row["id"] == "identity-" + second)
        self.assertEqual(active["state"], "approved")

    def test_snapshot_tamper_is_detected(self):
        avatar = self.app.state.service.ready_avatar(self.avatar_id)
        Path(avatar["face_image"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "changed or lost"):
            identity.pin(self.root, studio.load_avatar(self.root, self.avatar_id), avatar["identity_version"])

    def test_policy_validation_and_persistence(self):
        response = self.client.patch(self.base + "/quality", json={"policies": {"hair_color": "required"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get(self.base + "/identity").json()["profile"]["policies"]["hair_color"], "required")
        self.assertEqual(self.client.patch(self.base + "/quality", json={"policies": {"random_key": "required"}}).status_code, 400)

    def test_app_memory_default_off_persists_independent_of_avatars(self):
        self.assertEqual(self.client.get("/api/settings").json(), {"gpu_memory_management": False})
        self.assertFalse(self.client.get("/api/config").json()["app_settings"]["gpu_memory_management"])
        response = self.client.patch("/api/settings", json={"gpu_memory_management": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.client.post("/api/avatars", json={"name": "Second avatar"})
        self.assertTrue(self.client.get("/api/settings").json()["gpu_memory_management"])
        restarted = create_app(self.root)
        try:
            self.assertTrue(TestClient(restarted).get("/api/config").json()["app_settings"]["gpu_memory_management"])
        finally:
            restarted.state.jobs.shutdown()
        self.client.patch("/api/settings", json={"gpu_memory_management": False})
        self.assertFalse(app_settings.memory_management_enabled(self.root))

    def test_memory_switch_controls_both_models_and_ignores_legacy_avatar_setting(self):
        service = self.app.state.service
        avatar = {**service.ready_avatar(self.avatar_id), "_quality_profile": {"sequential": True}}
        with patch("model_manager.DEV_MODE", False), patch("model_manager.unload_current") as unload_flux, patch("avatar_vision.unload") as unload_vision, patch("avatar_quality.validate", return_value={"pass": True}):
            service.prepare_generation()
            service.validate(Image.new("RGB", (128, 160)), avatar, "Portrait")
            unload_flux.assert_not_called()
            unload_vision.assert_not_called()
            app_settings.save(self.root, True)
            service.prepare_generation()
            avatar["_quality_profile"]["sequential"] = False
            service.validate(Image.new("RGB", (128, 160)), avatar, "Portrait")
            unload_flux.assert_called_once()
            unload_vision.assert_called_once()

    def test_reference_analysis_obeys_app_memory_switch(self):
        image = Image.new("RGB", (128, 160))
        with patch("model_manager.unload_current") as unload, patch("avatar_vision.analyze_reference_image", return_value={}):
            studio.analyze_reference_image("face", image, dev_mode=False, save_dir=self.root)
            unload.assert_not_called()
            app_settings.save(self.root, True)
            studio.analyze_reference_image("body", image, dev_mode=False, save_dir=self.root)
            unload.assert_called_once()

    def test_memory_settings_reject_invalid_values_and_busy_changes(self):
        for body in ({"gpu_memory_management": "false"}, {"gpu_memory_management": 1}, {"gpu_memory_management": False, "unknown": 1}):
            self.assertEqual(self.client.patch("/api/settings", json=body).status_code, 422)
        self.assertEqual(self.client.patch(self.base + "/quality", json={"sequential": True}).status_code, 422)
        with patch.object(self.app.state.jobs, "list", return_value=[{"status": "running"}]):
            self.assertEqual(self.client.patch("/api/settings", json={"gpu_memory_management": True}).status_code, 409)
        self.assertFalse(app_settings.memory_management_enabled(self.root))

    def test_legacy_image_model_path_respects_app_memory_setting(self):
        import model_manager
        engine = Mock()
        engine.is_loaded.return_value = True
        with patch("model_manager.DEV_MODE", False), patch("model_manager._get_engine", return_value=engine), patch("model_manager._current_model", "Z-Image Turbo"), patch("workspace.get_save_dir", return_value=str(self.root)), patch("avatar_vision.unload") as unload:
            model_manager._ensure_model_locked("Z-Image Turbo")
            unload.assert_not_called()
            app_settings.save(self.root, True)
            model_manager._ensure_model_locked("Z-Image Turbo")
            unload.assert_called_once()

    def test_mock_reference_generation_never_loads_face_or_vision_model(self):
        with patch.object(face, "_load", side_effect=AssertionError("No inference")):
            result = self.job("identity_view", view="three-quarter")
            self.assertTrue(result["result"]["candidate_id"])
            self.assertFalse(self.client.get(self.base + "/identity").json()["candidates"][0]["approved"])

    def test_calibration_distinct_samples_and_mock_marker(self):
        positives = [self.upload(c)["path"] for c in ("red", "orange")]
        negatives = [self.upload(c)["path"] for c in ("black", "white")]
        with patch.object(face, "features", side_effect=AssertionError("No model")):
            self.job("calibrate", positives=positives, negatives=negatives)
        result = self.client.get(self.base + "/identity").json()["profile"]["face"]
        self.assertTrue(result["mock_not_quality_evidence"])
        self.job("calibrate", expect="failed", positives=positives, negatives=positives)

    def test_console_roles_recipe_and_check_saved(self):
        ref = self.upload("orange")
        self.job("console", new_chat=True, prompt="Portrait in a new pose", images=[ref["path"]], roles=["Pose"])
        item = self.client.get(self.base + "/gallery").json()["items"][0]
        self.assertIn("Use pose and framing only", item["prompt"])
        self.assertEqual(item["recipe"]["extra_roles"], ["Pose"])
        self.assertTrue(item["validation"]["mock_not_quality_evidence"])
        self.assertTrue(item["identity_version"])
        self.assertEqual(len(item["recipe"]["references"]), 3)

    def test_validation_retry_reuses_generated_console_image(self):
        service = self.app.state.service
        real = service.validate
        with patch.object(service, "validate", side_effect=RuntimeError("Vision temporary failure")):
            row = self.job("console", expect="failed", new_chat=True, prompt="Preserve saved work")
        path = row["checkpoint"]["generated_path"]
        with patch("generation_service.generate", side_effect=AssertionError("Must not regenerate")):
            self.app.state.jobs.resume(row["id"])
            for _ in range(500):
                result = self.app.state.jobs.get(row["id"])
                if result["status"] not in {"queued", "running"}:
                    break
                time.sleep(.01)
            self.assertEqual(result["status"], "completed", result)
        self.assertTrue(Path(path).is_file())
        self.assertEqual(len(studio.load_gallery_items(self.root, self.avatar_id)), 1)

    def test_benchmark_results_human_labels_and_gallery(self):
        self.job("benchmark")
        data = self.client.get(self.base + "/identity").json()
        self.assertEqual(data["benchmark"]["completed"], 12)
        self.assertTrue(data["benchmark"]["mock_not_quality_evidence"])
        self.assertEqual(len(data["cases"]), 12)
        response = self.client.post(self.base + "/identity", json={"action": "label", "index": 0, "label": "wrong"})
        self.assertEqual(response.json()["benchmark"]["false_acceptances"], 1)
        self.assertEqual(self.client.get(self.base + "/gallery").json()["total"], 12)

    def test_quality_recheck_uses_saved_image(self):
        item = self.saved_item()
        with patch("generation_service.generate", side_effect=AssertionError("No generation")):
            self.job("quality_check", item_id=item["id"])

    def test_enhancement_also_receives_quality_validation(self):
        item = self.saved_item()
        self.job("enhance", item_id=item["id"])
        row = self.client.get(self.base + "/gallery?category=enhancements").json()["items"][0]
        self.assertTrue(row["validation"]["mock_not_quality_evidence"])
        self.assertTrue(row["identity_version"])

    def test_caption_local_default_no_api(self):
        item = self.saved_item()
        self.client.patch(self.base + "/persona", json={"niche": "Portraits", "disclosure": "AI-created persona"})
        with patch("avatar_gallery.gemini_json", side_effect=AssertionError("No API")):
            result = self.job("caption", item_id=item["id"], details="A summer portrait")
        self.assertIn("AI-created persona", result["result"]["caption"])
        self.assertIn("summer", result["result"]["caption"])

    def test_content_edit_clears_approval_and_export_has_jpeg(self):
        post_id = self.post()
        path = self.base + "/content/" + post_id
        self.assertEqual(self.client.post(path + "/approve", json={"confirmed": False}).status_code, 400)
        self.assertEqual(self.client.post(path + "/approve", json={"confirmed": True}).status_code, 200)
        item = self.client.get(self.base + "/content").json()["posts"][0]
        self.assertEqual(item["status"], "approved")
        changed = self.client.patch(path, json={"image_id": item["image_id"], "caption": "Updated caption"})
        self.assertEqual(changed.json()["posts"][0]["status"], "draft")
        self.assertNotIn("approval", changed.json()["posts"][0])
        exported = self.client.get(path + "/export")
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            self.assertEqual(archive.read("caption.txt").decode(), "Updated caption")
            self.assertIn("image.jpg", archive.namelist())

    def test_planned_date_requires_timezone(self):
        item = self.saved_item()
        response = self.client.post(self.base + "/content", json={"image_id": item["id"], "caption": "Draft", "planned_at": "2026-10-25T18:00:00"})
        self.assertEqual(response.status_code, 400)

    def test_instagram_not_enabled_and_never_called_in_preview(self):
        post_id = self.post()
        with patch("requests.sessions.Session.request", side_effect=AssertionError("No HTTP")):
            self.job("publish", expect="failed", post_id=post_id)
        self.assertFalse(self.client.get(self.base + "/content").json()["instagram_configured"])

    def test_instagram_publish_idempotence_and_unknown_outcome(self):
        post_id = self.post()
        content.approve(self.root, self.avatar_id, post_id, True)
        # Export compatible square image for mocked remote acceptance.
        data = content.load(self.root, self.avatar_id)
        post = data["posts"][0]
        Image.new("RGB", (256, 256), "blue").save(post["image_path"], "JPEG")
        post["image_sha256"] = identity._digest(post["image_path"])
        post["approval"] = content._approval_digest(post)
        content.save(self.root, self.avatar_id, data)
        session = Mock()
        session.request.side_effect = [Mock(ok=True, json=lambda: {"id": "container1"}), Mock(ok=True, json=lambda: {"status_code": "FINISHED"}), Mock(ok=True, json=lambda: {"id": "media1"})]
        env = {"INSTAGRAM_ACCESS_TOKEN": "unit-test-token", "INSTAGRAM_USER_ID": "123", "INSTAGRAM_GRAPH_VERSION": "v25.0"}
        with patch.dict(os.environ, env):
            self.assertEqual(content.publish(self.root, self.avatar_id, post_id, session=session)["status"], "published")
            content.publish(self.root, self.avatar_id, post_id, session=session)
        self.assertEqual(session.request.call_count, 3)

    def test_identity_and_content_endpoints_require_auth(self):
        secured = create_app(self.root / "secure", token="test-private-token")
        try:
            client = TestClient(secured)
            for path in ("identity", "content", "identity/export", "content/a/export"):
                self.assertEqual(client.get("/api/avatars/a/" + path).status_code, 401)
        finally:
            secured.state.jobs.shutdown()

    def test_publish_timeout_is_never_blindly_retried(self):
        import requests
        post_id = self.post()
        content.approve(self.root, self.avatar_id, post_id, True)
        data = content.load(self.root, self.avatar_id)
        post = data["posts"][0]
        Image.new("RGB", (256, 256), "blue").save(post["image_path"], "JPEG")
        post["image_sha256"] = identity._digest(post["image_path"])
        post["approval"] = content._approval_digest(post)
        content.save(self.root, self.avatar_id, data)
        session = Mock()
        session.request.side_effect = [Mock(ok=True, json=lambda: {"id": "c1"}), Mock(ok=True, json=lambda: {"status_code": "FINISHED"}), requests.Timeout("network timeout")]
        env = {"INSTAGRAM_ACCESS_TOKEN": "unit-test-token", "INSTAGRAM_USER_ID": "123", "INSTAGRAM_GRAPH_VERSION": "v25.0"}
        with patch.dict(os.environ, env):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                content.publish(self.root, self.avatar_id, post_id, session=session)
            self.assertEqual(content.load(self.root, self.avatar_id)["posts"][0]["status"], "publishing_unknown")
            with self.assertRaisesRegex(RuntimeError, "uncertain outcome"):
                content.publish(self.root, self.avatar_id, post_id, session=session)
        self.assertEqual(session.request.call_count, 3)

    def test_calibration_samples_are_restored_as_workspace_images(self):
        positive = [self.upload(c)["path"] for c in ("red", "orange")]
        negative = [self.upload(c)["path"] for c in ("black", "white")]
        self.job("calibrate", positives=positive, negatives=negative)
        data = self.client.get(self.base + "/identity").json()
        self.assertEqual(len(data["calibration"]["positive"]), 2)
        for picture in data["calibration"]["positive"] + data["calibration"]["negative"]:
            self.assertEqual(self.client.get(picture["url"]).status_code, 200)

    def test_reindex_console_keeps_validation_and_recipe(self):
        item = self.saved_item()
        studio.save_gallery_items(self.root, self.avatar_id, [])
        restored = studio.sync_console_gallery(self.root, self.avatar_id)[0]
        self.assertEqual(restored["identity_version"], item["identity_version"])
        self.assertTrue(restored["validation"]["mock_not_quality_evidence"])

    def test_benchmark_resume_does_not_regenerate_completed_cases(self):
        from web_jobs import Cancelled
        service = self.app.state.service
        actual = service.validate
        calls = []
        def stop_after_one(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise Cancelled("Intentional test stop")
            return actual(*args, **kwargs)
        with patch.object(service, "validate", side_effect=stop_after_one):
            job = self.job("benchmark", expect="cancelled")
        before = self.client.get(self.base + "/identity").json()
        self.assertEqual(before["benchmark"]["completed"], 1)
        generated = Mock(wraps=__import__("generation_service").generate)
        with patch("generation_service.generate", generated):
            self.app.state.jobs.resume(job["id"])
            for _ in range(600):
                row = self.app.state.jobs.get(job["id"])
                if row["status"] not in {"queued", "running"}:
                    break
                time.sleep(.01)
            self.assertEqual(row["status"], "completed", row)
        # Case 2 had already been generated before the validation stop.
        self.assertEqual(generated.call_count, 10)

    def test_enhancement_validation_retry_reuses_generated_variant(self):
        item = self.saved_item()
        service = self.app.state.service
        with patch.object(service, "validate", side_effect=RuntimeError("Vision temporarily unavailable")):
            job = self.job("enhance", expect="failed", item_id=item["id"])
        self.assertTrue(job["checkpoint"]["image"])
        self.assertTrue(job["checkpoint"]["enhancement_plan"]["automatic"])
        with patch("avatar_enhance.reconstruct", side_effect=AssertionError("Do not reconstruct again")), patch("enhancement_analysis.analyze", side_effect=AssertionError("Do not analyze again")):
            self.app.state.jobs.resume(job["id"])
            for _ in range(400):
                row = self.app.state.jobs.get(job["id"])
                if row["status"] not in {"queued", "running"}:
                    break
                time.sleep(.01)
            self.assertEqual(row["status"], "completed", row)
        self.assertEqual(self.client.get(self.base + "/gallery?category=enhancements").json()["total"], 1)

    def test_automatic_enhancement_records_features_prompt_and_reuses_source_cache(self):
        item = self.saved_item()
        first = self.job("enhance", item_id=item["id"])
        plan = first["checkpoint"]["enhancement_plan"]
        self.assertTrue(plan["automatic"])
        self.assertEqual(plan["version"], "scene-profiles-v2")
        self.assertEqual(plan["prompt_version"], "user-refinement-v2")
        self.assertEqual(plan["profile_id"], "general")
        self.assertEqual(plan["analysis"]["profile_name"], "General-Purpose Fallback")
        self.assertEqual(first["checkpoint"]["diagnostics"]["content_review"]["status"], "mock")
        self.assertIn("Preserve genuinely correct details and avoid speculative reconstruction", plan["prompt"])
        self.assertEqual(first["payload"]["strength"], .65)
        saved = self.client.get(self.base + "/gallery?category=enhancements").json()["items"][0]
        self.assertEqual(saved["prompt"], plan["prompt"])
        self.assertEqual(saved["diagnostics"]["enhancement_plan"], plan)
        second = self.job("enhance", item_id=item["id"])
        self.assertTrue(second["checkpoint"]["enhancement_plan"]["analysis"]["cache_hit"])
        self.assertEqual(self.client.get(self.base + "/gallery?category=enhancements").json()["total"], 2)

    def test_enhancement_skin_warning_survives_identity_recheck(self):
        item = self.saved_item()
        review = {"status": "needs_review", "reasons": ["New dense skin markings"]}
        with patch("enhancement_analysis.review_texture", return_value=review), patch.object(self.app.state.service, "validate", return_value={"pass": True}):
            self.job("enhance", item_id=item["id"])
            enhanced = self.client.get(self.base + "/gallery?category=enhancements").json()["items"][0]
            self.assertEqual(enhanced["status"], "needs_review")
            self.job("quality_check", item_id=enhanced["id"])
        self.assertEqual(studio.gallery_item_by_id(self.root, self.avatar_id, enhanced["id"])["status"], "needs_review")

    def test_unavailable_enhancement_analysis_stays_explicit_and_requires_review(self):
        item = self.saved_item()
        with patch("enhancement_analysis.analyze", return_value={"status": "unavailable", "materials": []}), patch.object(self.app.state.service, "validate", return_value={"pass": True}):
            job = self.job("enhance", item_id=item["id"])
        self.assertEqual(job["checkpoint"]["diagnostics"]["texture_review"]["status"], "uncertain")
        enhanced = self.client.get(self.base + "/gallery?category=enhancements").json()["items"][0]
        self.assertEqual(enhanced["status"], "needs_review")

    def test_new_keys_are_allowed_but_arbitrary_env_is_not(self):
        values = parse_keys("INSTAGRAM_ACCESS_TOKEN=sample\nINSTAGRAM_USER_ID=123\nINSTAGRAM_GRAPH_VERSION=v25.0\nPYTHONPATH=bad")
        self.assertEqual(values["INSTAGRAM_USER_ID"], "123")
        self.assertNotIn("PYTHONPATH", values)

    def test_anchor_change_deactivates_old_pack_without_losing_snapshot(self):
        candidate = identity.add_candidate(self.root, self.avatar_id, "profile", Image.new("RGB", (128, 160), "red"))
        identity.review_candidate(self.root, self.avatar_id, candidate, True)
        previous = self.app.state.service.ready_avatar(self.avatar_id)
        for part in ("face", "body"):
            studio.lock_reference(self.root, self.avatar_id, part, Image.new("RGB", (128, 160), "green"), dev_mode=True)
        current = self.app.state.service.ready_avatar(self.avatar_id)
        self.assertEqual(current["_identity_pack"], [])
        self.assertFalse(identity.pack(self.root, self.avatar_id)["candidates"][0]["approved"])
        with Image.open(identity.select_references(current, "profile portrait")[0]) as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 128, 0))
        restored = identity.pin(self.root, current, previous["identity_version"])
        self.assertEqual(len(restored["_identity_pack"]), 1)

    def test_historical_image_operations_use_its_own_policy(self):
        avatar = self.app.state.service.ready_avatar(self.avatar_id)
        old_version = avatar["identity_version"]
        quality.save_profile(self.root, avatar, {"policies": {"hair_color": "required"}})
        item = self.saved_item()
        for part in ("face", "body"):
            studio.lock_reference(self.root, self.avatar_id, part, Image.new("RGB", (128, 160), "red"), dev_mode=True)
        current = self.app.state.service.ready_avatar(self.avatar_id)
        quality.save_profile(self.root, current, {"policies": {"hair_color": "flexible"}})
        observed = []
        def validate(image, pinned, prompt, *args):
            observed.append((pinned["identity_version"], pinned["_quality_profile"]["policies"]["hair_color"]))
            return {"pass": True, "reasons": []}
        with patch.object(self.app.state.service, "validate", side_effect=validate):
            self.job("quality_check", item_id=item["id"])
            self.job("regenerate", item_id=item["id"], reason="Correct face")
            self.job("enhance", item_id=item["id"])
        self.assertEqual(observed, [(old_version, "required")] * 3)

    def test_benchmark_image_correction_has_no_search_reference_dependency(self):
        avatar = self.app.state.service.ready_avatar(self.avatar_id)
        path = studio.save_gallery_image(self.root, self.avatar_id, "bench-correct", Image.new("RGB", (128, 160), "blue"))
        studio.record_gallery_item(self.root, self.avatar_id, {"id": "bench-correct", "origin": "benchmark", "prompt": "Profile portrait", "status": "passed", "generated_path": path, "identity_version": avatar["identity_version"]})
        self.job("regenerate", item_id="bench-correct", reason="Correct identity")
        item = studio.gallery_item_by_id(self.root, self.avatar_id, "bench-correct")
        self.assertTrue(Path(item["generated_path"]).is_file())
        self.assertEqual(item["status"], "passed")

    def test_explicit_reference_view_and_negation(self):
        avatar = {"face_image": "front", "body_image": "body", "_identity_pack": [
            {"view": "profile", "path": "profile"}, {"view": "smile", "path": "smile"}, {"view": "body-side", "path": "side"}]}
        self.assertEqual(identity.select_references(avatar, "No smile, straight-on portrait"), ["front", "body"])
        self.assertEqual(identity.select_references(avatar, "Social profile picture"), ["front", "body"])
        self.assertEqual(identity.select_references(avatar, "side view"), ["profile", "side"])
        self.assertEqual(identity.select_references(avatar, "smiling portrait", "front", "body-side"), ["front", "side"])
        with self.assertRaises(ValueError):
            identity.select_references(avatar, "", "invalid")

    def test_empty_drive_file_restores_without_a_download_request(self):
        from drive_store import DriveStore
        import hashlib
        store = DriveStore.__new__(DriveStore)
        store.root, store.state = self.root, {}
        store.remember = Mock()
        store.request = Mock(side_effect=AssertionError("Empty files need no HTTP body"))
        row = {"id": "empty", "size": "0", "md5Checksum": hashlib.md5(b"").hexdigest()}
        store.download("empty.txt", row)
        self.assertEqual((self.root / "empty.txt").read_bytes(), b"")
        store.request.assert_not_called()
        store.remember.assert_called_once()

    def test_detail_crops_are_authenticated_bounded_and_lossless(self):
        picture = self.upload("red")
        response = self.client.get(picture["thumb"] + "?region=0.25,0.25,0.75,0.75")
        self.assertEqual(response.status_code, 200, response.text[:100] if response.status_code != 200 else "")
        with Image.open(io.BytesIO(response.content)) as image:
            self.assertEqual(image.size, (64, 80))
            self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))
        for region in ("nan,0,1,1", "0,0,2,1", "0.9,0,0.1,1", "x,0,1,1"):
            self.assertEqual(self.client.get(picture["thumb"] + "?region=" + region).status_code, 400)

    def test_local_caption_respects_persona_style_and_limit(self):
        brand = {"niche": "Travel portraits", "tone": "warm", "values": "curiosity, thoughtful design", "disclosure": "AI-created persona"}
        text = content.local_caption(brand, "A summer portrait")
        self.assertIn("curiosity", text)
        self.assertIn("#TravelPortraits", text)
        self.assertTrue(text.endswith("AI-created persona"))
        self.assertNotEqual(text, content.local_caption(brand, "A summer portrait", style="concise"))
        self.assertLessEqual(len(content.local_caption(brand, "x" * 12000)), 2200)

    def test_runtime_scheduler_is_opt_in_and_never_publishes_in_preview(self):
        post_id = self.post()
        self.assertFalse(self.client.get(self.base + "/content").json()["scheduling_enabled"])
        self.assertEqual(self.client.post(self.base + "/scheduling", json={"confirmed": True}).status_code, 400)
        with patch.object(content, "publish", side_effect=AssertionError("No publish")):
            self.app.state.schedule_tick()

    def test_scheduler_submits_once_and_ignores_missed_posts(self):
        from datetime import datetime, timezone, timedelta
        post_id = self.post()
        data = content.load(self.root, self.avatar_id)
        now = datetime.now(timezone.utc)
        data["posts"][0]["planned_at"] = (now - timedelta(seconds=5)).isoformat()
        content.save(self.root, self.avatar_id, data)
        content.approve(self.root, self.avatar_id, post_id, True)
        self.assertEqual(content.due_posts(self.root, self.avatar_id, now.timestamp()), [])
        self.app.state.scheduling[self.avatar_id] = now.timestamp() - 30
        with patch("model_manager.DEV_MODE", False), patch.object(content, "publish", return_value={"status": "published"}) as publish:
            self.app.state.schedule_tick()
            for _ in range(200):
                rows = [row for row in self.app.state.jobs.list() if row["kind"] == "publish"]
                if rows and rows[0]["status"] not in {"queued", "running"}:
                    break
                time.sleep(.01)
            self.app.state.schedule_tick()
        self.assertEqual(len([row for row in self.app.state.jobs.list() if row["kind"] == "publish"]), 1)
        self.assertEqual(publish.call_count, 1)

    def test_scheduler_finishes_an_asynchronous_instagram_container(self):
        from datetime import datetime, timezone, timedelta
        post_id = self.post()
        now = datetime.now(timezone.utc)
        data = content.load(self.root, self.avatar_id)
        data["posts"][0]["planned_at"] = (now - timedelta(seconds=2)).isoformat()
        content.save(self.root, self.avatar_id, data)
        content.approve(self.root, self.avatar_id, post_id, True)
        self.app.state.scheduling[self.avatar_id] = now.timestamp() - 10
        calls = []
        def publish(*args):
            data = content.load(self.root, self.avatar_id)
            data["posts"][0]["status"] = "container_created" if not calls else "published"
            content.save(self.root, self.avatar_id, data)
            calls.append(1)
            return data["posts"][0]
        with patch("model_manager.DEV_MODE", False), patch.object(content, "publish", side_effect=publish):
            for expected in (1, 2):
                self.app.state.schedule_tick()
                for _ in range(200):
                    rows = [row for row in self.app.state.jobs.list() if row["kind"] == "publish"]
                    if len(rows) == expected and all(row["status"] == "completed" for row in rows):
                        break
                    time.sleep(.01)
            self.app.state.schedule_tick()
        self.assertEqual(len(calls), 2)
        self.assertEqual(content.load(self.root, self.avatar_id)["posts"][0]["status"], "published")

    def test_detail_evidence_never_calls_pixel_change_proof_of_quality(self):
        import image_review
        original = Image.new("RGB", (128, 128), "red")
        result = image_review.detail_evidence(original, original.resize((256, 256)))
        self.assertFalse(result["quality_verified"])
        self.assertEqual(result["advisory"], "little_detail_change")
        self.assertEqual(result["comparison_size"], [128, 128])

    def test_small_calibration_does_not_allow_automatic_identity_acceptance(self):
        def extractor(value):
            vector = np.array([1., 0.] if value.startswith("p") or value == "anchor" else [0., 1.])
            return [{"embedding": vector}]
        result = face.calibrate(["anchor"], ["p1", "p2"], ["n1", "n2"], extractor)
        self.assertTrue(result["provisional"])
        self.assertFalse(result["calibrated"])
        self.assertEqual(face.compare("p1", ["anchor"], result, extractor)["status"], "uncalibrated")


class FaceMathTests(unittest.TestCase):
    def test_identity_model_downloads_are_disabled_locally(self):
        with tempfile.TemporaryDirectory() as root, patch("runtime_environment.provider", return_value="local"):
            with self.assertRaisesRegex(RuntimeError, "cloud runtime"):
                face.install_models(root)
    def test_calibration_and_comparison_use_injected_embeddings(self):
        embeddings = {"anchor": [1., 0.], "p1": [.99, .1], "p2": [.95, .2], "p3": [.98, .1], "p4": [.96, .1], "n1": [0., 1.], "n2": [-1., 0.], "n3": [.1, 1.], "n4": [.05, 1.], "result": [.99, .1]}
        def extract(key):
            arr = np.array(embeddings[key], dtype=np.float32)
            return [{"embedding": arr / np.linalg.norm(arr)}]
        profile = face.calibrate(["anchor"], ["p1", "p2", "p3", "p4"], ["n1", "n2", "n3", "n4"], extract)
        self.assertTrue(face.compare("result", ["anchor"], profile, extract)["status"] == "passed")
        self.assertEqual(face.compare("n1", ["anchor"], profile, extract)["status"], "failed")
        self.assertEqual(face.compare("result", ["anchor"], None, extract)["status"], "uncalibrated")
        provisional = face.calibrate(["anchor"], ["p1", "p2"], ["n1", "n2"], extract)
        self.assertFalse(provisional["calibrated"])
        self.assertEqual(face.compare("result", ["anchor"], provisional, extract)["status"], "uncalibrated")

    def test_features_in_preview_are_blocked_before_model_load(self):
        with patch.object(face, "_load", side_effect=AssertionError("No actual model")):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                face.features(Image.new("RGB", (128, 128)))

    def test_inconclusive_identity_never_passes_or_wastes_repairs(self):
        with tempfile.TemporaryDirectory() as root:
            avatar = {"id": "quality", "identity_version": "test", "face_image": "face"}
            result = quality.validate(root, Image.new("RGB", (128, 128)), avatar, "portrait",
                basic=lambda *a: {"pass": True, "reasons": [], "face_test_applicable": True},
                ask=lambda *a: "MATCH", face_compare=lambda *a: {"status": "uncalibrated", "reason": "Calibrate first"})
            self.assertFalse(result["pass"])
            self.assertTrue(result["review_only"])

    def test_required_attribute_mismatch_fails(self):
        specs = {"answers": [{"key": "hair_color", "answer": "black", "strictness": "required"}]}
        checks = quality.attribute_checks(None, specs, "face", lambda *a: "MISMATCH")
        self.assertEqual(checks[0]["result"], "mismatch")
        flexible = quality.attribute_checks(None, specs, "face", lambda *a: self.fail("No flexible attribute call"), {"hair_color": "flexible"})
        self.assertEqual(flexible[0]["result"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
