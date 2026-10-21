import os
import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from PIL import Image

import avatar_design
import avatar_guide as guide
import avatar_studio as studio


class AvatarGuideTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = self.temporary.name
        self.avatar = guide.create(self.root, "Maya", "Studio creator", "Guided")
        self.calls = []

    def generate(self, kind, prompt, avatar):
        self.calls.append((kind, prompt, avatar.get("face_locked"), avatar.get("body_locked")))
        return Image.new("RGB", (64, 96), "teal")

    def advance(self, image=None, validate=None, generate=None):
        return list(guide.advance(
            self.root, self.avatar["id"], image, generate or self.generate,
            validate or (lambda *args: {"pass": True, "score": 95}), dev_mode=True,
        ))

    def prepare_body(self):
        self.advance()
        self.advance()

    def test_guided_flow_produces_two_saved_images_without_external_apis(self):
        self.advance()
        self.assertEqual(guide.stage(studio.load_avatar(self.root, self.avatar["id"])), "face_review")
        self.advance()
        self.assertEqual(guide.stage(studio.load_avatar(self.root, self.avatar["id"])), "body_review")
        self.advance()
        restored = studio.load_avatar(self.root, self.avatar["id"])
        self.assertEqual(guide.stage(restored), "complete")
        self.assertEqual(restored["gallery_count"], 2)
        self.assertEqual([call[0] for call in self.calls], ["face", "body", "starter", "starter"])
        self.assertTrue(self.calls[1][2])
        self.assertTrue(all(call[2] and call[3] for call in self.calls[2:]))
        self.assertTrue(all(Path(item["generated_path"]).is_file() for item in studio.load_gallery_items(self.root, self.avatar["id"])))

    def test_upload_skips_face_generation(self):
        self.advance(image=Image.new("RGB", (40, 40), "red"))
        self.assertEqual(self.calls, [])
        with Image.open(guide.draft(studio.load_avatar(self.root, self.avatar["id"]), "face")) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_failed_generation_never_creates_mock_reference(self):
        with self.assertRaisesRegex(RuntimeError, "offline"):
            self.advance(generate=mock.Mock(side_effect=RuntimeError("offline")))
        restored = studio.load_avatar(self.root, self.avatar["id"])
        self.assertEqual(guide.stage(restored), "face")
        self.assertIsNone(guide.draft(restored, "face"))

    def test_confirmed_face_survives_body_generation_failure(self):
        self.advance()
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            self.advance(generate=mock.Mock(side_effect=RuntimeError("body failed")))
        restored = studio.load_avatar(self.root, self.avatar["id"])
        self.assertTrue(restored["face_locked"])
        self.assertEqual(guide.stage(restored), "body")
        self.advance()
        self.assertEqual(self.calls[-1][0], "body")

    def test_starter_retry_preserves_first_completed_image(self):
        self.prepare_body()
        counter = 0

        def interrupted(kind, prompt, avatar):
            nonlocal counter
            counter += 1
            if counter == 2:
                raise RuntimeError("interrupted")
            return self.generate(kind, prompt, avatar)

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.advance(generate=interrupted)
        first = studio.load_gallery_items(self.root, self.avatar["id"])[0]
        first_mtime = Path(first["generated_path"]).stat().st_mtime_ns
        previous_calls = len(self.calls)
        self.advance()
        self.assertEqual(len(self.calls), previous_calls + 1)
        self.assertEqual(Path(first["generated_path"]).stat().st_mtime_ns, first_mtime)
        self.assertEqual(studio.load_avatar(self.root, self.avatar["id"])["gallery_count"], 2)

    def test_validation_retry_uses_correction_and_has_finite_limit(self):
        self.prepare_body()
        self.advance(validate=lambda *args: {"pass": False, "score": 45, "repair_instruction": "Preserve the nose shape"})
        starters = [call for call in self.calls if call[0] == "starter"]
        self.assertEqual(len(starters), 6)
        self.assertIn("Preserve the nose shape", starters[1][1])
        self.assertTrue(all(item["status"] == "needs_review" for item in studio.load_gallery_items(self.root, self.avatar["id"])))

    def test_completed_guide_is_idempotent(self):
        self.prepare_body()
        self.advance()
        calls = len(self.calls)
        self.advance()
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(len(studio.load_gallery_items(self.root, self.avatar["id"])), 2)

    def test_custom_mode_is_independent_and_old_avatars_remain_custom(self):
        avatar = guide.create(self.root, "Custom", mode="Custom")
        self.assertFalse(guide.active(avatar))
        self.assertEqual(avatar["current_step"], "face")
        legacy = studio.create_avatar(self.root, "Old avatar")
        self.assertFalse(guide.active(legacy))
        self.assertEqual(len(avatar_design.library(self.root, "CUSTOM")), 1)

    def test_custom_handoff_retains_identity_and_drafts(self):
        self.prepare_body()
        custom = guide.use_custom(self.root, self.avatar["id"])
        self.assertFalse(guide.active(custom))
        self.assertEqual(custom["current_step"], "body")
        self.assertTrue(custom["face_locked"])
        self.assertTrue(os.path.isfile(guide.draft(custom, "body")))

    def test_prompt_edits_survive_reload(self):
        guide.save_options(self.root, self.avatar["id"], "New face", "New body", "Scene A", "Scene B")
        self.assertEqual(guide.options(studio.load_avatar(self.root, self.avatar["id"])), ("New face", "New body", "Scene A", "Scene B"))

    def test_selected_avatar_survives_newer_unrelated_updates(self):
        other = studio.create_avatar(self.root, "Other avatar")
        studio.select_avatar(self.root, self.avatar["id"])
        studio.save_avatar(self.root, other)
        self.assertEqual(studio.selected_or_first(self.root)["id"], self.avatar["id"])

    def test_dashboard_escapes_names(self):
        self.avatar["name"] = '<script>alert("x")</script>'
        rendered = avatar_design.dashboard(self.avatar)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_failed_analysis_does_not_overwrite_confirmed_reference(self):
        self.advance()
        self.advance()
        avatar = studio.load_avatar(self.root, self.avatar["id"])
        original = Path(avatar["face_image"]).read_bytes()
        with mock.patch.object(studio, "analyze_reference_image", side_effect=RuntimeError("analysis failed")):
            with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                studio.lock_reference(self.root, avatar["id"], "face", Image.new("RGB", (64, 64), "red"), dev_mode=False)
        self.assertEqual(Path(avatar["face_image"]).read_bytes(), original)

    def test_validation_failure_preserves_image_and_can_resume(self):
        self.prepare_body()
        with self.assertRaisesRegex(RuntimeError, "vision unavailable"):
            self.advance(validate=mock.Mock(side_effect=RuntimeError("vision unavailable")))
        item = studio.load_gallery_items(self.root, self.avatar["id"])[0]
        self.assertTrue(Path(item["generated_path"]).is_file())
        self.assertEqual(item["status"], "checking")
        self.advance()
        self.assertEqual(studio.load_avatar(self.root, self.avatar["id"])["gallery_count"], 2)

    def test_back_preserves_identity_and_reuses_completed_starters(self):
        self.prepare_body()
        self.advance()
        calls = len(self.calls)
        guide.back(self.root, self.avatar["id"])
        self.assertEqual(guide.stage(studio.load_avatar(self.root, self.avatar["id"])), "body_review")
        self.advance()
        self.assertEqual(len(self.calls), calls)

    def test_replacing_face_invalidates_body_and_old_starters(self):
        self.prepare_body()
        self.advance()
        before = studio.load_gallery_items(self.root, self.avatar["id"])
        studio.lock_reference(self.root, self.avatar["id"], "face", Image.new("RGB", (64, 96), "purple"), dev_mode=True)
        avatar = studio.load_avatar(self.root, self.avatar["id"])
        self.assertFalse(avatar["body_locked"])
        self.assertIsNone(guide.draft(avatar, "body"))
        tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_avatar_image_value")
        namespace = {"os": os, "avatar_guide": guide}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
        self.assertIsNone(namespace["_avatar_image_value"](avatar, "body_image"))
        self.assertTrue(all(row["status"] == "identity_changed" for row in studio.load_gallery_items(self.root, avatar["id"])))
        self.assertTrue(all(Path(row["generated_path"]).exists() for row in before))

    def test_custom_candidate_and_prompt_survive_reload(self):
        avatar = studio.save_candidate(self.root, self.avatar["id"], "face", Image.new("RGB", (32, 40), "red"), "Distinctive face")
        restored = studio.load_avatar(self.root, avatar["id"])
        self.assertEqual(restored["face_prompt"], "Distinctive face")
        self.assertTrue(Path(restored["face_candidate"]).is_file())
        self.assertFalse(restored["face_locked"])

    def test_library_filter_sort_and_rename(self):
        self.prepare_body()
        self.advance()
        second = studio.create_avatar(self.root, "Zoe")
        studio.rename_avatar(self.root, second["id"], "Ada")
        self.assertEqual([a["name"] for a in avatar_design.library(self.root, status="Ready")], ["Maya"])
        self.assertEqual([a["name"] for a in avatar_design.library(self.root, status="In progress")], ["Ada"])
        self.assertEqual([a["name"] for a in avatar_design.library(self.root, sort="Name")], ["Ada", "Maya"])
        with self.assertRaises(ValueError):
            studio.rename_avatar(self.root, second["id"], "  ")

    def test_gallery_display_and_selection_share_the_same_window(self):
        path = studio.save_gallery_image(self.root, self.avatar["id"], "test", Image.new("RGB", (20, 20)), 1)
        for index in range(105):
            studio.record_gallery_item(self.root, self.avatar["id"], {"id": str(index), "generated_path": path})
        studio.record_gallery_item(self.root, self.avatar["id"], {"id": "missing", "generated_path": "missing.png"})
        self.assertEqual(len(studio.visible_gallery_items(self.root, self.avatar["id"])), 100)
        self.assertEqual(studio.gallery_item_by_index(self.root, self.avatar["id"], 0)["id"], "5")
        self.assertEqual(studio.gallery_item_by_index(self.root, self.avatar["id"], 99)["id"], "104")
        self.assertIsNone(studio.gallery_item_by_index(self.root, self.avatar["id"], 100))

    def test_extra_chat_references_are_copied_and_limited(self):
        refs = studio.save_console_references(self.root, self.avatar["id"], [Image.new("RGB", (20, 20))] * 3)
        self.assertEqual(len(refs), 2)
        self.assertTrue(all(Path(path).is_file() for path in refs))

    def test_clear_draft_does_not_delete_confirmed_identity(self):
        self.prepare_body()
        guide.back(self.root, self.avatar["id"])
        avatar = guide.clear_draft(self.root, self.avatar["id"])
        self.assertEqual(guide.stage(avatar), "face")
        self.assertIsNone(guide.draft(avatar, "face"))
        self.assertTrue(avatar["face_locked"])
        self.assertTrue(Path(avatar["face_image"]).is_file())

    def test_changing_one_starter_scene_only_regenerates_that_scene(self):
        self.prepare_body()
        self.advance()
        before = len(self.calls)
        guide.back(self.root, self.avatar["id"])
        values = list(guide.options(studio.load_avatar(self.root, self.avatar["id"])))
        values[2] = "A new studio portrait with red lighting"
        guide.save_options(self.root, self.avatar["id"], *values)
        self.advance()
        self.assertEqual(len(self.calls), before + 1)

    def test_gallery_candidate_is_saved_before_failed_vision_check(self):
        self.prepare_body()
        self.advance()
        avatar = studio.load_avatar(self.root, self.avatar["id"])
        studio.record_gallery_item(self.root, avatar["id"], {"id": "batch-item"})
        tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run_gallery_generation")
        namespace = {
            "avatar_studio": studio, "SAVE_DIR": self.root, "DEV_MODE": True,
            "model_manager": SimpleNamespace(FLUX_MODEL_NAME="mock"), "DEFAULT_NEG": "",
            "Image": Image, "os": os,
            "do_generate": lambda *args, **kwargs: iter([("done", [Image.new("RGB", (20, 20))], None, "123")]),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
        with mock.patch.object(studio, "validate_generated_image", side_effect=RuntimeError("vision unavailable")):
            with self.assertRaisesRegex(RuntimeError, "vision unavailable"):
                namespace["_run_gallery_generation"](avatar, {}, "scene", "1:1", "batch-item", 1, identity_only=True)
        item = next(row for row in studio.load_gallery_items(self.root, avatar["id"]) if row["id"] == "batch-item")
        self.assertEqual(item["status"], "checking")
        self.assertTrue(Path(item["generated_path"]).is_file())

    def test_face_prompt_adds_framing_to_user_details(self):
        prompt = studio.build_reference_prompt("face", "green eyes, short silver hair")
        self.assertIn("front-facing headshot", prompt)
        self.assertIn("crown to the base of the neck", prompt)
        self.assertIn("No torso, hands", prompt)
        self.assertIn("Appearance details: green eyes, short silver hair", prompt)
        self.assertNotIn("attached face reference", prompt)

    def test_body_prompt_only_claims_a_face_reference_when_confirmed(self):
        self.assertNotIn("attached face reference", studio.build_reference_prompt("body", "blue shirt", self.avatar))
        self.prepare_body()
        avatar = studio.load_avatar(self.root, self.avatar["id"])
        prompt = studio.build_reference_prompt("body", "blue shirt", avatar)
        self.assertIn("both feet inside the frame", prompt)
        self.assertIn("attached face reference defines", prompt)
        self.assertIn("do not invent a different face", prompt)
        self.assertIn("Appearance details: blue shirt", prompt)

    def test_reference_prompt_rejects_unknown_reference_kind(self):
        with self.assertRaises(ValueError):
            studio.build_reference_prompt("unknown", "test")

    def test_reference_generator_uses_the_displayed_prompt_and_face_anchor(self):
        self.prepare_body()
        avatar = studio.load_avatar(self.root, self.avatar["id"])
        tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "make_avatar_reference")
        generate = mock.Mock(return_value=iter([("done", [Image.new("RGB", (20, 20))], None, "123")]))
        namespace = {"avatar_studio": studio, "DEV_MODE": False, "SAVE_DIR": self.root, "model_manager": SimpleNamespace(FLUX_MODEL_NAME="mock"), "DEFAULT_NEG": "", "do_generate": generate}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
        namespace["make_avatar_reference"]("body", "A white shirt", avatar)
        args, kwargs = generate.call_args
        self.assertEqual(args[1], studio.build_reference_prompt("body", "A white shirt", avatar))
        self.assertEqual(args[8], 4)
        self.assertEqual(len(kwargs["input_image"]), 1)
        self.assertIsNone(kwargs["canvas_index"])

    def test_reference_ready_tokens_are_unique_and_count_both_slots(self):
        tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "avatar_reference_ready")
        namespace = {"uuid": __import__("uuid"), "gr": SimpleNamespace(update=lambda **kwargs: kwargs)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
        ready = namespace["avatar_reference_ready"]
        first, state = ready(Image.new("RGB", (10, 10)), None)
        self.assertEqual(state["label"], "Additional references (1/2)")
        second, state = ready(Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10)))
        self.assertEqual(state["label"], "Additional references (2/2)")
        self.assertNotEqual(first, second)
        self.assertEqual(ready(None, None)[1]["label"], "Additional references (optional)")


if __name__ == "__main__":
    unittest.main()
