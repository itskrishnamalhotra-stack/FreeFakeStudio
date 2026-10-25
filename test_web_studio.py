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

    def test_deep_routes_serve_application_but_never_mask_api_or_assets(self):
        for path in ("/chats/example", "/avatars", "/avatars/example/face", "/avatars/example/console/chat1",
                     "/avatars/example/gallery/generations", "/avatars/example/gallery/enhancements",
                     "/avatars/example/gallery/resources", "/avatars/example/gallery/needs-review", "/unknown-page"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn("text/html", response.headers["content-type"])
            self.assertIn('id="root"', response.text)
            self.assertEqual(self.client.head(path).status_code, 200, path)
        for path in ("/api/missing", "/assets/missing.js", "/missing.css", "/api", "/assets"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404, path)
            self.assertNotIn('id="root"', response.text)

    def test_private_deep_links_retain_api_authentication(self):
        app = create_app(self.root / "private-route", token="private-route-test-token")
        try:
            client = TestClient(app)
            self.assertEqual(client.get("/avatars/example/gallery/resources").status_code, 200)
            self.assertEqual(client.get("/api/avatars").status_code, 401)
            self.assertEqual(client.post("/api/session", json={"token": "private-route-test-token"}).status_code, 200)
            self.assertEqual(client.get("/api/avatars").status_code, 200)
        finally:
            app.state.jobs.shutdown()

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

    def test_first_prompt_creates_one_chat_and_retries_reuse_it(self):
        self.assertEqual(self.client.get("/api/chats").json(), [])
        body = {"kind": "generate", "new_chat": True, "prompt": "First submitted prompt"}
        headers = {"Idempotency-Key": "new-conversation-request"}
        first = self.client.post("/api/jobs", json=body, headers=headers)
        self.assertEqual(first.status_code, 200, first.text)
        row = first.json()
        chat_id = row["payload"]["chat_id"]
        second = self.client.post("/api/jobs", json=body, headers=headers)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(row["id"], second.json()["id"])
        self.assertEqual(chat_id, second.json()["payload"]["chat_id"])
        listed = self.client.get("/api/chats").json()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["title"], body["prompt"])
        turns = self.client.get("/api/chats/" + chat_id).json()["turns"]
        self.assertEqual(sum(turn["role"] == "user" for turn in turns), 1)
        changed = self.client.post("/api/jobs", json={**body, "prompt": "Different request"}, headers=headers)
        self.assertEqual(changed.status_code, 400)
        self.assertEqual(len(self.app.state.service.chats.list()), 1)

    def test_invalid_first_prompt_does_not_create_a_chat(self):
        for extra in ({"prompt": "   "}, {"prompt": "x", "model": "missing"}, {"prompt": "x", "images": ["missing.png"]}):
            response = self.client.post("/api/jobs", json={"kind": "generate", "new_chat": True, **extra},
                                        headers={"Idempotency-Key": "invalid-first-prompt"})
            self.assertIn(response.status_code, (400, 404), response.text)
            self.assertEqual(self.app.state.service.chats.list(), [])

    def test_first_prompt_avatar_scope_and_failed_generation_are_saved(self):
        avatar_id = self.ready()
        self.assertEqual(self.client.get(f"/api/chats?avatar_id={avatar_id}").json(), [])
        job = self.job("console", avatar_id=avatar_id, new_chat=True, prompt="First avatar conversation")
        chat_id = job["payload"]["chat_id"]
        chats = self.client.get(f"/api/chats?avatar_id={avatar_id}").json()
        self.assertEqual([row["id"] for row in chats], [chat_id])
        self.assertEqual(self.client.get("/api/chats").json(), [])
        with patch("web_service.generation.generate", side_effect=RuntimeError("Mock generation failure")):
            failed = self.job("generate", expect="failed", new_chat=True, prompt="Keep my failed prompt")
        chat = self.client.get("/api/chats/" + failed["payload"]["chat_id"]).json()
        self.assertEqual(chat["title"], "Keep my failed prompt")
        self.assertEqual(chat["turns"][0]["text"], "Keep my failed prompt")

    def test_legacy_empty_placeholders_are_hidden_without_deletion(self):
        row = self.client.post("/api/chats", json={}).json()
        path = self.app.state.service.chats.path(row["id"])
        self.assertTrue(path.exists())
        self.assertEqual(self.client.get("/api/chats").json(), [])
        self.job("generate", chat_id=row["id"], prompt="Now this is a real conversation")
        self.assertEqual([chat["id"] for chat in self.client.get("/api/chats").json()], [row["id"]])

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

    def test_enhance_chat_image_stays_in_original_conversation(self):
        chat = self.client.post("/api/chats", json={}).json()
        generated = self.job("generate", chat_id=chat["id"], prompt="A studio portrait")
        other = self.client.post("/api/chats", json={}).json()
        source = generated["result"]["images"][0]
        import avatar_enhance
        custom_prompt = "Restore realistic photographic texture without changing the subject."
        with patch.object(avatar_enhance, "reconstruct", wraps=avatar_enhance.reconstruct) as reconstruct:
            enhanced = self.job("enhance", chat_id=chat["id"], image=source, scale=1, reconstruction_prompt=custom_prompt, adaptive_prompt=False)
            self.assertEqual(reconstruct.call_args.args[-1], custom_prompt)
        turns = self.client.get("/api/chats/" + chat["id"]).json()["turns"]
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[-1]["text"], "Enhanced image")
        self.assertEqual(turns[-1]["processing_passes"], 1)
        self.assertEqual(turns[-1]["source"]["path"], source)
        self.assertEqual(turns[-1]["reconstruction_prompt"], custom_prompt)
        self.assertEqual(turns[-1]["images"][0]["path"], enhanced["result"]["image"])
        self.assertEqual(self.client.get("/api/chats/" + other["id"]).json()["turns"], [])
        self.assertTrue((self.root / source).is_file())

    def test_default_reconstruction_uses_native_sampling_not_partial_img2img(self):
        generated = self.job("generate", new_chat=True, prompt="Reconstruction source")
        engine = model_manager.ensure_model(model_manager.FLUX_MODEL_NAME)
        with patch.object(engine, "reconstruct_reference", wraps=engine.reconstruct_reference) as native, \
             patch.object(engine, "img2img", side_effect=AssertionError("Not partial img2img")):
            enhanced = self.job("enhance", chat_id=generated["payload"]["chat_id"], image=generated["result"]["images"][0])
        self.assertEqual(native.call_count, 1)
        self.assertEqual(enhanced["checkpoint"]["diagnostics"]["sampling"]["scheduler"], "Flux2Scheduler")
        self.assertEqual(enhanced["checkpoint"]["diagnostics"]["sampling"]["denoise"], 1.)

    def test_custom_enhancement_prompt_bypasses_analysis_and_reaches_model_verbatim(self):
        generated = self.job("generate", new_chat=True, prompt="Custom prompt source")
        engine = model_manager.ensure_model(model_manager.FLUX_MODEL_NAME)
        prompt = "Refine only the existing black fabric stitching. Keep the face and background unchanged."
        with patch("enhancement_analysis.analyze", side_effect=AssertionError("Custom must bypass analysis")), \
             patch("enhancement_analysis.review_texture", side_effect=AssertionError("Custom must bypass scene review")), \
             patch.object(engine, "reconstruct_reference", wraps=engine.reconstruct_reference) as native:
            enhanced = self.job("enhance", chat_id=generated["payload"]["chat_id"], image=generated["result"]["images"][0],
                                adaptive_prompt=False, reconstruction_prompt=prompt)
        self.assertEqual(native.call_args.args[1], prompt)
        self.assertEqual(enhanced["checkpoint"]["enhancement_plan"]["prompt"], prompt)
        self.assertEqual(enhanced["checkpoint"]["enhancement_plan"]["selection"], "custom")

    def test_experimental_img2img_explicitly_uses_requested_denoise(self):
        generated = self.job("generate", new_chat=True, prompt="Low-noise source")
        engine = model_manager.ensure_model(model_manager.FLUX_MODEL_NAME)
        with patch.object(engine, "img2img", wraps=engine.img2img) as img2img, \
             patch.object(engine, "reconstruct_reference", side_effect=AssertionError("Explicit legacy mode must not use native reconstruction")):
            enhanced = self.job("enhance", chat_id=generated["payload"]["chat_id"], image=generated["result"]["images"][0],
                                sampling_mode="img2img", strength=.45)
        self.assertEqual(img2img.call_args.args[5], .45)
        self.assertEqual(enhanced["checkpoint"]["diagnostics"]["sampling"]["mode"], "img2img")
        self.assertEqual(enhanced["checkpoint"]["diagnostics"]["sampling"]["denoise"], .45)

    def test_failed_enhancement_keeps_zero_seed_on_resume(self):
        generated = self.job("generate", new_chat=True, prompt="Zero-seed recovery source")
        engine = model_manager.ensure_model(model_manager.FLUX_MODEL_NAME)
        with patch("web_service.secrets.randbelow", return_value=0), \
             patch.object(engine, "reconstruct_reference", side_effect=RuntimeError("Temporary sampler failure")):
            failed = self.job("enhance", expect="failed", chat_id=generated["payload"]["chat_id"], image=generated["result"]["images"][0])
        self.assertEqual(failed["checkpoint"]["seed"], 0)
        with patch("web_service.secrets.randbelow", side_effect=AssertionError("Must reuse saved zero seed")), \
             patch.object(engine, "reconstruct_reference", wraps=engine.reconstruct_reference) as native:
            self.app.state.jobs.resume(failed["id"])
            for _ in range(300):
                row = self.app.state.jobs.get(failed["id"])
                if row["status"] not in {"queued", "running"}:
                    break
                time.sleep(.02)
            self.assertEqual(row["status"], "completed", row)
        self.assertEqual(native.call_args.args[2], 0)

    def test_all_manual_profiles_skip_analysis_and_append_extra_instructions(self):
        import enhancement_profiles
        generated = self.job("generate", new_chat=True, prompt="Manual profile source")
        for profile_id, profile in enhancement_profiles.PROFILES.items():
            with self.subTest(profile=profile_id), \
                 patch("enhancement_analysis.analyze", side_effect=AssertionError("Manual selection must not analyze")), \
                 patch("enhancement_analysis.review_texture", side_effect=AssertionError("Manual selection must not run scene review")):
                enhanced = self.job("enhance", chat_id=generated["payload"]["chat_id"], image=generated["result"]["images"][0],
                                    enhancement_profile=profile_id, additional_instructions="Keep the cafe background soft.")
            plan = enhanced["checkpoint"]["enhancement_plan"]
            self.assertEqual(plan["profile_id"], profile_id)
            self.assertEqual(plan["selection"], "manual_override")
            self.assertFalse(plan["automatic"])
            self.assertEqual(plan["analysis"]["status"], "disabled")
            self.assertEqual(plan["focus_rules"], [])
            self.assertEqual(plan["prompt"], profile["prompt"] + " Additional reconstruction instructions: Keep the cafe background soft.")
            self.assertEqual(plan["requested_prompt"], "Keep the cafe background soft.")
            self.assertEqual(enhanced["checkpoint"]["diagnostics"]["texture_review"]["status"], "disabled")

    def test_manual_profile_without_extra_instructions_is_verbatim(self):
        import enhancement_profiles
        generated = self.job("generate", new_chat=True, prompt="Manual exact source")
        enhanced = self.job("enhance", chat_id=generated["payload"]["chat_id"], image=generated["result"]["images"][0],
                            enhancement_profile="female_body")
        self.assertEqual(enhanced["checkpoint"]["enhancement_plan"]["prompt"], enhancement_profiles.PROFILES["female_body"]["prompt"])

    def test_invalid_enhancement_modes_fail_before_job_creation(self):
        for extra in ({"enhancement_profile": "bogus"}, {"steps": 20},
                      {"adaptive_prompt": False, "additional_instructions": "Hidden conflicting text"},
                      {"adaptive_prompt": False, "enhancement_profile": "female_body"}):
            response = self.client.post("/api/jobs", json={"kind": "enhance", **extra},
                                        headers={"Idempotency-Key": "invalid-mode-request"})
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.app.state.jobs.list(), [])

    def test_enhance_avatar_console_image_also_enters_gallery(self):
        avatar_id = self.ready()
        chat = self.client.post("/api/chats", json={"avatar_id": avatar_id}).json()
        generated = self.job("console", avatar_id=avatar_id, chat_id=chat["id"], prompt="Outdoor portrait")
        enhanced = self.job("enhance", avatar_id=avatar_id, chat_id=chat["id"], image=generated["result"]["image"])
        turns = self.client.get(f"/api/chats/{chat['id']}?avatar_id={avatar_id}").json()["turns"]
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[-1]["images"][0]["path"], enhanced["result"]["image"])
        self.assertEqual(self.client.get(f"/api/avatars/{avatar_id}/gallery").json()["total"], 2)

    def test_enhance_rejects_uploads_and_other_chat_images(self):
        generated = self.job("generate", prompt="Studio portrait")
        other = self.client.post("/api/chats", json={}).json()
        for source in (self.image()["path"], generated["result"]["images"][0]):
            response = self.client.post("/api/jobs", json={"kind": "enhance", "chat_id": other["id"], "image": source},
                                        headers={"Idempotency-Key": "reject-enhance-" + str(len(source))})
            self.assertEqual(response.status_code, 400, response.text)

    def test_enhancement_prompt_reaches_full_and_tiled_passes(self):
        import avatar_enhance
        prompts = []
        def edit(tile, whole, prompt, seed, strength):
            prompts.append(prompt)
            return tile.copy()
        list(avatar_enhance.reconstruct(Image.new("RGB", (1024, 1024)), edit, method="hd_tiles", prompt="Custom photograph prompt"))
        self.assertEqual(prompts[0], "Custom photograph prompt")
        self.assertGreater(len(prompts), 1)
        self.assertTrue(all(prompt == avatar_enhance.TILE_PROMPT + "Custom photograph prompt" for prompt in prompts[1:]))
        response = self.client.post("/api/jobs", json={"kind": "enhance", "reconstruction_prompt": "   "},
                                    headers={"Idempotency-Key": "empty-enhancement-prompt"})
        self.assertEqual(response.status_code, 400, response.text)

    def test_gallery_batch_resume_data(self):
        avatar_id = self.ready()
        self.job("gallery", avatar_id=avatar_id, quantity=2, theme="Casual studio")
        snapshot = self.client.get(f"/api/avatars/{avatar_id}/references").json()
        self.assertGreaterEqual(snapshot["counts"]["generated"], 2)
        self.assertEqual(self.client.get(f"/api/avatars/{avatar_id}/gallery").json()["total"], 2)

    def test_resource_search_tabs_and_more_generation_stay_in_the_existing_batch(self):
        avatar_id = self.ready()
        original = self.job("gallery", avatar_id=avatar_id, new_batch=True, quantity=1, theme="Studio")
        parent = original["payload"]["batch_id"]
        base = f"/api/avatars/{avatar_id}"
        found = self.job("resource_search", avatar_id=avatar_id, batch_id=parent, theme="Beach sunsets", resource_tab_name="Beach", quantity=6)
        tab_id = found["payload"]["resource_tab_id"]
        self.assertEqual(self.client.get(base + "/gallery").json()["total"], 1)
        tabs = self.client.get(base + "/resource-tabs").json()["items"]
        self.assertEqual(len(tabs), 2)
        self.assertEqual(next(tab for tab in tabs if tab["id"] == tab_id)["description"], "Beach sunsets")
        available_before = next(tab for tab in tabs if tab["id"] == tab_id)["counts"]["available"]
        generated = self.job("resource_generate", avatar_id=avatar_id, batch_id=parent, resource_tab_id=tab_id, quantity=6)
        self.assertEqual(generated["result"]["batch_id"], parent)
        self.assertEqual(self.client.get(base + f"/gallery?batch_id={parent}").json()["total"], 7)
        resources = self.client.get(base + f"/resources?batch_id={parent}&tab_id={tab_id}").json()
        self.assertEqual(resources["total"], available_before)
        self.assertEqual(resources["counts"]["used"], 6)
        unused = self.client.get(base + f"/resources?batch_id={parent}&tab_id={tab_id}&status=unused").json()
        self.assertEqual(unused["total"], available_before - 6)
        self.assertTrue(all(not row["used"] and not row.get("reserved") for row in unused["items"]))
        self.job("resource_generate", avatar_id=avatar_id, batch_id=parent, resource_tab_id=tab_id, quantity=6)
        items = avatar_studio.load_gallery_items(str(self.root), avatar_id)
        references = [row["reference_path"] for row in items]
        self.assertEqual(len(set(references)), len(references))
        batches = self.client.get(base + "/gallery/batches").json()["items"]
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["counts"]["generated"], 13)
        deleted = self.client.delete(base + f"/resource-tabs/{tab_id}")
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(self.client.get(base + "/gallery").json()["total"], 13)
        self.assertTrue(all(Path(row["generated_path"]).is_file() and Path(row["reference_path"]).is_file() for row in items))
        self.assertEqual([tab["id"] for tab in self.client.get(base + "/resource-tabs").json()["items"]], [parent])
        stale = self.client.get(base + f"/resources?tab_id={tab_id}")
        self.assertEqual(stale.status_code, 200)
        self.assertTrue(stale.json()["deleted"])
        self.assertEqual(stale.json()["total"], 0)

    def test_resource_search_idempotency_and_cross_batch_validation(self):
        avatar_id = self.ready()
        original = self.job("gallery", avatar_id=avatar_id, new_batch=True, quantity=1)
        parent = original["payload"]["batch_id"]
        body = {"kind": "resource_search", "avatar_id": avatar_id, "batch_id": parent, "theme": "Cafe", "quantity": 1}
        headers = {"Idempotency-Key": "resource-search-repeated"}
        first = self.client.post("/api/jobs", json=body, headers=headers)
        second = self.client.post("/api/jobs", json=body, headers=headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["id"], second.json()["id"])
        for _ in range(200):
            if self.client.get("/api/jobs/" + first.json()["id"]).json()["status"] not in {"running", "queued"}:
                break
            time.sleep(.01)
        tab_id = first.json()["payload"]["resource_tab_id"]
        other = self.job("gallery", avatar_id=avatar_id, new_batch=True, quantity=1)
        invalid = self.client.post("/api/jobs", json={"kind": "resource_generate", "avatar_id": avatar_id,
            "batch_id": other["payload"]["batch_id"], "resource_tab_id": tab_id}, headers={"Idempotency-Key": "resource-wrong-owner"})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(len(self.client.get(f"/api/avatars/{avatar_id}/resource-tabs").json()["items"]), 3)

    def test_batch_gender_round_trips_through_api_and_persists_independently(self):
        avatar_id = self.ready()
        first = self.job("gallery", avatar_id=avatar_id, new_batch=True, quantity=1, avatar_gender="female")
        second = self.job("gallery", avatar_id=avatar_id, new_batch=True, quantity=1, avatar_gender="male")
        self.assertEqual(first["payload"]["avatar_gender"], "female")
        self.assertEqual(second["payload"]["avatar_gender"], "male")
        batches = {row["id"]: row for row in self.client.get(f"/api/avatars/{avatar_id}/gallery/batches").json()["items"]}
        self.assertEqual(batches[first["payload"]["batch_id"]]["avatar_gender"], "female")
        self.assertEqual(batches[second["payload"]["batch_id"]]["avatar_gender"], "male")
        response = self.client.post("/api/jobs", json={"kind": "gallery", "avatar_id": avatar_id, "new_batch": True,
            "avatar_gender": "invalid"}, headers={"Idempotency-Key": "invalid-batch-gender"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(self.client.get(f"/api/avatars/{avatar_id}/gallery/batches").json()["items"]), 2)

    def test_batch_management_and_scoped_gallery_resources(self):
        avatar_id = self.ready()
        base = f"/api/avatars/{avatar_id}"
        first = self.job("gallery", avatar_id=avatar_id, new_batch=True, batch_name="Batch A", quantity=2, theme="Cafe")
        second = self.job("gallery", avatar_id=avatar_id, new_batch=True, batch_name="Batch B", quantity=1, theme="Cafe")
        a, b = first["checkpoint"]["batch_id"], second["checkpoint"]["batch_id"]
        self.assertNotEqual(a, b)
        self.assertEqual(first["payload"]["batch_id"], a)
        batches = self.client.get(base + "/gallery/batches").json()["items"]
        self.assertEqual({row["name"] for row in batches}, {"Batch A", "Batch B"})
        self.assertEqual(self.client.get(base + f"/gallery?batch_id={a}").json()["total"], 2)
        self.assertEqual(self.client.get(base + f"/gallery?batch_id={b}").json()["total"], 1)
        self.assertEqual(self.client.get(base + f"/resources?batch_id={a}").json()["total"], 2)
        self.assertEqual(self.client.get(base + f"/resources?batch_id={b}").json()["total"], 1)
        for batch_id, count in ((a, 2), (b, 1)):
            findings = self.client.get(base + f"/references?batch_id={batch_id}").json()
            self.assertEqual(findings["job"]["id"], batch_id)
            self.assertEqual(findings["counts"]["found"], count)
            self.assertEqual(findings["counts"]["generated"], count)
        self.assertEqual(self.client.post(base + f"/gallery/batches/{a}", json={"action": "rename", "name": "Renamed A"}).status_code, 200)
        self.assertEqual(self.client.post(base + f"/gallery/batches/{a}", json={"action": "archive"}).status_code, 200)
        archived = next(row for row in self.client.get(base + "/gallery/batches").json()["items"] if row["id"] == a)
        self.assertTrue(archived["archived"])
        self.assertEqual(archived["name"], "Renamed A")
        self.assertEqual(self.client.get(base + f"/gallery?batch_id={a}").json()["total"], 2)
        self.assertEqual(self.client.post(base + f"/gallery/batches/{a}", json={"action": "restore"}).status_code, 200)
        source = self.client.get(base + f"/gallery?batch_id={b}").json()["items"][0]
        self.job("enhance", avatar_id=avatar_id, item_id=source["id"], enhancement_profile="general")
        enhanced = self.client.get(base + f"/gallery?batch_id={b}&category=enhancements").json()
        self.assertEqual(enhanced["total"], 1)
        self.assertEqual(enhanced["items"][0]["batch_id"], b)
        self.assertEqual(self.client.get(base + f"/references?batch_id={b}").json()["counts"]["generated"], 1)
        avatar_studio.update_gallery_item(str(self.root), avatar_id, enhanced["items"][0]["id"], batch_id=None)
        self.assertEqual(self.client.get(base + f"/gallery?batch_id={b}&category=enhancements").json()["total"], 1)

    def test_fresh_batch_can_start_after_unfinished_batch_and_resume_explicitly(self):
        from web_service import StudioService
        avatar_id = self.ready()
        with patch.object(StudioService, "gallery_prompt", side_effect=RuntimeError("Prompt provider unavailable")):
            first = self.job("gallery", expect="failed", avatar_id=avatar_id, new_batch=True, quantity=2, batch_name="Paused A")
        a = first["checkpoint"]["batch_id"]
        second = self.job("gallery", avatar_id=avatar_id, new_batch=True, quantity=1, batch_name="Independent B")
        b = second["checkpoint"]["batch_id"]
        resumed = self.job("gallery", avatar_id=avatar_id, batch_id=a)
        self.assertEqual(resumed["checkpoint"]["batch_id"], a)
        rows = {row["id"]: row for row in self.client.get(f"/api/avatars/{avatar_id}/gallery/batches").json()["items"]}
        self.assertEqual(rows[a]["counts"]["generated"], 2)
        self.assertEqual(rows[b]["counts"]["generated"], 1)
        self.assertEqual(rows[b]["status"], "complete")

    def test_new_batch_submission_is_idempotent_and_invalid_ids_do_not_escape(self):
        avatar_id = self.ready()
        body = {"kind": "gallery", "avatar_id": avatar_id, "new_batch": True, "quantity": 1}
        headers = {"Idempotency-Key": "fresh-batch-repeat"}
        first = self.client.post("/api/jobs", json=body, headers=headers)
        second = self.client.post("/api/jobs", json=body, headers=headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(len(self.client.get(f"/api/avatars/{avatar_id}/gallery/batches").json()["items"]), 1)
        for _ in range(300):
            if self.app.state.jobs.get(first.json()["id"])["status"] not in {"queued", "running"}:
                break
            time.sleep(.02)
        other = self.ready()
        batch_id = first.json()["payload"]["batch_id"]
        self.assertEqual(self.client.get(f"/api/avatars/{other}/references?batch_id={batch_id}").status_code, 400)
        self.assertEqual(self.client.get(f"/api/avatars/{avatar_id}/references?batch_id=..%2Foutside").status_code, 400)
        self.assertEqual(self.client.post(f"/api/avatars/{avatar_id}/gallery/batches/{batch_id}", json={"action": "delete"}).status_code, 422)

    def test_gallery_sections_and_manual_review_preserve_sources(self):
        avatar_id = self.ready()
        self.job("console", avatar_id=avatar_id, prompt="Gallery separation test")
        source = self.client.get(f"/api/avatars/{avatar_id}/gallery").json()["items"][0]
        self.job("enhance", avatar_id=avatar_id, item_id=source["id"])
        base = f"/api/avatars/{avatar_id}/gallery"
        generations = self.client.get(base + "?category=generations").json()
        enhancements = self.client.get(base + "?category=enhancements").json()
        self.assertEqual(generations["total"], 1)
        self.assertEqual(enhancements["total"], 1)
        self.assertEqual(generations["counts"]["enhancements"], 1)
        enhanced = enhancements["items"][0]
        self.assertEqual(enhanced["source"]["path"], source["image"]["path"])
        review = self.client.get(base + "?category=review").json()
        self.assertIn(enhanced["id"], [row["id"] for row in review["items"]])
        response = self.client.post(base + "/" + enhanced["id"] + "/approve")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "passed")
        self.assertNotIn(enhanced["id"], [row["id"] for row in self.client.get(base + "?category=review").json()["items"]])
        self.assertEqual(self.client.get(base + "?category=enhancements").json()["total"], 1)
        self.assertTrue((self.root / source["image"]["path"]).is_file())

    def test_resources_include_checks_pending_urls_and_used_references(self):
        import avatar_gallery_jobs as batches
        avatar_id = self.ready()
        a, b = self.image("red"), self.image("blue")
        batches.add_references(str(self.root), avatar_id, [{"path": str(self.root / a["path"]), "image_url": "https://example.com/a.png"},
            {"path": str(self.root / b["path"]), "image_url": "https://example.com/b.png"}, {"image_url": "https://example.com/pending.png"}])
        catalog = batches.load_catalog(str(self.root), avatar_id)["references"]
        batches.update_reference(str(self.root), avatar_id, catalog[0]["id"], state="approved", reason="Clear reference")
        batches.update_reference(str(self.root), avatar_id, catalog[1]["id"], state="rejected", reason="Blurred face")
        avatar_studio.record_gallery_item(str(self.root), avatar_id, {"id": "resource-use-test", "origin": "auto", "status": "passed",
            "generated_path": str(self.root / a["path"]), "reference_id": catalog[0]["id"], "reference_path": str(self.root / a["path"])})
        base = f"/api/avatars/{avatar_id}/resources"
        all_rows = self.client.get(base).json()
        self.assertEqual(all_rows["counts"]["all"], len(all_rows["items"]))
        search = self.client.get(base + "?kind=search").json()
        self.assertEqual(search["total"], 3)
        pending = self.client.get(base + "?kind=search&status=pending").json()
        self.assertEqual(pending["total"], 1)
        self.assertIsNone(pending["items"][0]["image"])
        rejected = self.client.get(base + "?status=rejected&query=blurred").json()
        self.assertEqual(rejected["items"][0]["reason"], "Blurred face")
        used = self.client.get(base + "?status=used&kind=search").json()
        self.assertEqual(used["total"], 1)
        self.assertTrue(used["items"][0]["committed"])
        self.assertEqual(self.client.get(base + "?status=approved&kind=search").json()["total"], 1)
        other = self.avatar()
        self.assertEqual(self.client.get(f"/api/avatars/{other}/resources").json()["total"], 0)

    def test_resources_survive_archived_chat_without_duplicate_copies(self):
        avatar_id = self.ready()
        chat = self.client.post("/api/chats", json={"avatar_id": avatar_id}).json()
        uploaded = self.image()["path"]
        self.job("console", avatar_id=avatar_id, chat_id=chat["id"], prompt="Reference inventory test", images=[uploaded])
        base = f"/api/avatars/{avatar_id}/resources?kind=uploaded"
        rows = self.client.get(base).json()
        self.assertEqual(rows["total"], 1)
        self.assertEqual(rows["items"][0]["image"]["path"], uploaded)
        self.assertEqual(rows["items"][0]["state"], "pending")
        self.assertTrue(rows["items"][0]["used"])
        self.assertEqual(self.client.delete(f"/api/chats/{chat['id']}?avatar_id={avatar_id}").status_code, 200)
        self.assertEqual(self.client.get(base).json()["total"], 1)

    def test_gallery_category_pagination_and_resource_pagination(self):
        import avatar_gallery_jobs as batches
        avatar_id = self.avatar()
        source = self.image()["path"]
        avatar_studio.save_gallery_items(str(self.root), avatar_id, [{"id": "image-" + str(index), "origin": "enhancement" if index % 2 else "auto",
            "status": "unchecked", "generated_path": str(self.root / source)} for index in range(60)])
        base = f"/api/avatars/{avatar_id}/gallery?category=generations&limit=10"
        first, second = self.client.get(base).json(), self.client.get(base + "&offset=10").json()
        self.assertEqual(first["total"], 30)
        self.assertFalse({row["id"] for row in first["items"]} & {row["id"] for row in second["items"]})
        batches.add_references(str(self.root), avatar_id, [{"image_url": f"https://example.com/{index}.jpg"} for index in range(60)])
        resources = self.client.get(f"/api/avatars/{avatar_id}/resources?limit=10&offset=50").json()
        self.assertEqual(resources["total"], 60)
        self.assertEqual(len(resources["items"]), 10)
        self.assertEqual(self.client.get(base + "&category=invalid").status_code, 422)

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
