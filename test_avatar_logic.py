"""Offline regression tests: no model imports, downloads, or live API requests."""

import ast
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import avatar_gallery
import avatar_studio as studio
import avatar_vision


def app_function(name, namespace):
    tree = ast.parse(Path(__file__).with_name("app.py").read_text(encoding="utf-8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace[name]


class AvatarLogicTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = self.temporary.name
        self.avatar = studio.create_avatar(self.root, "Logic review")

    def generation_core(self):
        engine = SimpleNamespace(
            img2img=mock.Mock(return_value=Image.new("RGB", (32, 32))),
            inpaint=mock.Mock(return_value=Image.new("RGB", (32, 32))),
            generate_with_references=mock.Mock(return_value=Image.new("RGB", (32, 32))),
        )
        manager = SimpleNamespace(
            get_capabilities=lambda name: ["generate", "img2img", "inpaint"],
            get_current_model=lambda: "FLUX.2-klein 4B", ensure_model=mock.Mock(return_value=engine),
        )
        namespace = {
            "make_seed": lambda seed: 1, "_parse_aspect": lambda aspect: (32, 32),
            "_as_image_list": lambda images: list(images), "model_manager": manager,
            "_status_html": lambda kind, message: (kind, message),
            "_select_mask_for_prompt": mock.Mock(side_effect=lambda prompt, image: (None, prompt, 1.0)),
            "_resolve_mask": mock.Mock(return_value="mask"),
            "get_save_path": lambda kind: str(Path(self.root) / "result.png"),
            "_write_runtime_error": mock.Mock(return_value=None),
        }
        return app_function("do_generate", namespace), namespace, engine

    def run_generation(self, generate, images, canvas_index, **kwargs):
        return list(generate("FLUX.2-klein 4B", "edit", "", "square", 1, 1.0, 1.0, 1, 4,
                             input_image=images, canvas_index=canvas_index, **kwargs))

    def test_selected_canvas_becomes_engine_base_without_mutating_attachments(self):
        generate, namespace, engine = self.generation_core()
        images = [Image.new("RGB", (32, 32), color) for color in ("red", "blue", "green")]
        self.run_generation(generate, images, 1)
        received = engine.img2img.call_args.args[0]
        self.assertEqual(received, [images[1], images[0], images[2]])
        self.assertIs(namespace["_select_mask_for_prompt"].call_args.args[1], images[1])
        self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))

    def test_explicit_mask_and_canvas_use_same_image(self):
        generate, namespace, engine = self.generation_core()
        images = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
        self.run_generation(generate, images, 1, mask_mode="Manual Paint")
        self.assertIs(namespace["_resolve_mask"].call_args.args[0], images[1])
        self.assertIs(engine.inpaint.call_args.args[0], images[1])
        self.assertEqual(engine.inpaint.call_args.kwargs["references"], [images[1], images[0]])

    def test_reference_only_generation_preserves_reference_order(self):
        generate, _, engine = self.generation_core()
        images = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
        self.run_generation(generate, images, None)
        self.assertEqual(engine.generate_with_references.call_args.args[0], images)
        engine.img2img.assert_not_called()

    def test_stale_canvas_fails_before_loading_model(self):
        generate, namespace, engine = self.generation_core()
        rows = self.run_generation(generate, [Image.new("RGB", (32, 32))], 2)
        self.assertIn("no longer available", rows[-1][0][1])
        namespace["model_manager"].ensure_model.assert_not_called()

    def test_rejected_or_uncertain_analysis_cannot_lock_or_overwrite_identity(self):
        studio.lock_reference(self.root, self.avatar["id"], "face", Image.new("RGB", (32, 32), "red"))
        folder = studio.avatar_path(self.root, self.avatar["id"])
        before = {str(path): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
        for verdict in (False, None, "true"):
            with self.subTest(verdict=verdict), mock.patch.object(studio, "analyze_reference_image", return_value={
                "usable_reference": verdict, "rejection_reasons": ["Face is obscured"],
            }):
                with self.assertRaisesRegex(ValueError, "Face is obscured"):
                    studio.lock_reference(self.root, self.avatar["id"], "face", Image.new("RGB", (32, 32), "blue"), dev_mode=False)
            self.assertEqual(before, {str(path): path.read_bytes() for path in folder.rglob("*") if path.is_file()})

    def test_unusable_first_reference_stays_unlocked(self):
        with mock.patch.object(studio, "analyze_reference_image", return_value={"usable_reference": False}):
            with self.assertRaisesRegex(ValueError, "could not confirm"):
                studio.lock_reference(self.root, self.avatar["id"], "body", Image.new("RGB", (32, 64)), dev_mode=False)
        avatar = studio.load_avatar(self.root, self.avatar["id"])
        self.assertFalse(avatar["body_locked"])
        self.assertIsNone(avatar["body_image"])

    def test_structured_vision_verdict_is_required(self):
        questions = studio.FACE_SPEC_QUESTIONS
        for payload, expected in (({}, False), ({"usable_reference": "true"}, False), ({"usable_reference": True}, True)):
            normalized = avatar_vision._normalise_answers("face", payload, questions, (32, 32))
            self.assertIs(normalized["usable_reference"], expected)

    def test_face_summary_includes_identity_not_eligibility_checks(self):
        specs = studio._mock_specs("face", Image.new("RGB", (32, 32)))
        text = studio.summarize_specs(specs)
        for key in ("hair_color", "hair_length", "hairstyle", "eye_color", "lip_fullness", "cheek_fullness", "jawline"):
            self.assertIn(key + ":", text)
        for key in ("adult_presenting", "usable_face_reference", "face_visibility", "face_angle", "expression"):
            self.assertNotIn(key + ":", text)
        self.assertIn("eye_color:", studio.summarize_specs(specs, limit=10))

    def test_body_summary_prioritizes_proportions_not_outfit(self):
        specs = studio._mock_specs("body", Image.new("RGB", (32, 64)))
        text = studio.summarize_specs(specs)
        for key in ("body_build", "waist_shape", "hip_shape", "shoulder_width", "height_impression"):
            self.assertIn(key + ":", text)
        self.assertNotIn("clothing:", text)
        self.assertNotIn("body_visibility:", text)

    def make_gallery_item(self):
        path = studio.save_gallery_image(self.root, self.avatar["id"], "test", Image.new("RGB", (32, 32)), 1)
        return studio.record_gallery_item(self.root, self.avatar["id"], {
            "id": "test", "generated_path": path, "attempt": 1, "status": "passed",
            "prompt": "A portrait", "validation": {"pass": True, "score": 96, "reasons": []},
        })

    def test_user_rejection_overrides_pass_and_preserves_previous_assessment(self):
        self.make_gallery_item()
        rejected = studio.reject_gallery_item(self.root, self.avatar["id"], "test", "Wrong hair color")
        self.assertFalse(rejected["validation"]["pass"])
        self.assertTrue(rejected["validation"]["user_rejected"])
        self.assertEqual(rejected["validation"]["repair_instruction"], "Wrong hair color")
        self.assertTrue(rejected["rejections"][0]["validation"]["pass"])
        self.assertEqual(rejected["rejections"][0]["attempt"], 1)
        self.assertTrue(Path(rejected["rejections"][0]["generated_path"]).is_file())

    def test_empty_rejection_still_explicitly_reports_user_failure(self):
        self.make_gallery_item()
        rejected = studio.reject_gallery_item(self.root, self.avatar["id"], "test")
        self.assertIn("user rejected", rejected["validation"]["repair_instruction"])
        self.assertFalse(rejected["validation"]["pass"])

    def regeneration_handler(self):
        self.make_gallery_item()
        namespace = {
            "avatar_studio": studio, "avatar_gallery": avatar_gallery, "SAVE_DIR": self.root,
            "DEV_MODE": False, "gr": SimpleNamespace(Error=ValueError),
            "_status_html": lambda *args: str(args), "_avatar_gallery_value": lambda avatar: [],
            "_avatar_gallery_details": lambda *args: "details",
        }
        return app_function("_regenerate_avatar_gallery_item", namespace)

    def test_retry_sends_user_rejection_to_gemini_before_generation(self):
        regenerate = self.regeneration_handler()
        with mock.patch.object(avatar_gallery, "repair_generation_prompt", return_value="Repaired prompt") as repair:
            iterator = regenerate(self.avatar["id"], 0, "Keep the original jacket")
            next(iterator)
            iterator.close()
        validation = repair.call_args.args[1]
        self.assertFalse(validation["pass"])
        self.assertEqual(validation["repair_instruction"], "Keep the original jacket")
        saved = studio.load_gallery_items(self.root, self.avatar["id"])[0]
        self.assertEqual(saved["prompt"], "Repaired prompt")

    def test_api_failure_does_not_lose_rejection(self):
        regenerate = self.regeneration_handler()
        with mock.patch.object(avatar_gallery, "repair_generation_prompt", side_effect=RuntimeError("API unavailable")):
            with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                next(regenerate(self.avatar["id"], 0, "Wrong pose"))
        saved = studio.load_gallery_items(self.root, self.avatar["id"])[0]
        self.assertEqual(saved["status"], "rejected")
        self.assertEqual(saved["rejections"][0]["reason"], "Wrong pose")

    def test_retry_control_is_reenabled_only_for_valid_selected_image(self):
        self.make_gallery_item()
        for kind in ("face", "body"):
            studio.lock_reference(self.root, self.avatar["id"], kind, Image.new("RGB", (32, 32)))
        controls = ("avatar_generate_btn", "avatar_stop_btn", "gallery_plan_btn", "gallery_stop_btn",
                    "regenerate_gallery_btn", "gallery_rejection_reason", "avatar_extra_1", "avatar_extra_2",
                    "avatar_selector", "avatar_step", "avatar_back_library", "avatar_rename_btn",
                    "gallery_require_face", "gallery_quantity", "gallery_aspect", "gallery_theme", "gallery_new_search",
                    "enhance_gallery_btn", "enhance_start", "enhance_stop", "enhance_close", "enhance_scale",
                    "enhance_strength", "enhance_steps", "enhance_action_run", "enhance_method")
        namespace = {name: name for name in controls}
        namespace.update({"avatar_studio": studio, "SAVE_DIR": self.root, "DEV_MODE": True,
                          "gr": SimpleNamespace(update=lambda **kwargs: kwargs)})
        state = app_function("avatar_task_state", namespace)
        self.assertTrue(state(self.avatar["id"], False, "gallery", 0)["regenerate_gallery_btn"]["interactive"])
        self.assertFalse(state(self.avatar["id"], True, "gallery", 0)["regenerate_gallery_btn"]["interactive"])
        self.assertFalse(state(self.avatar["id"], False, "gallery", 99)["regenerate_gallery_btn"]["interactive"])
        self.assertFalse(state(self.avatar["id"], False, "gallery")["regenerate_gallery_btn"]["interactive"])

    def test_regeneration_restores_controls_for_same_selected_image(self):
        run = mock.Mock(return_value=iter([]))
        namespace = {"run_avatar_task": run, "_regenerate_avatar_gallery_item": mock.Mock(return_value=iter([])),
                     "regenerate_columns": []}
        handler = app_function("regenerate_avatar_gallery_item", namespace)
        list(handler(self.avatar["id"], 3, "Wrong pose"))
        self.assertEqual(run.call_args.args[-1], 3)

    def test_prompt_writer_receives_selected_reference_pixels(self):
        path = Path(self.root) / "reference.png"
        Image.new("RGB", (1600, 1000), "blue").save(path)
        reference = {"path": str(path), "description": "Caption", "source_url": "https://example.test/source"}
        with mock.patch.object(avatar_gallery, "gemini_json", return_value={"prompt": "Visual prompt"}) as gemini:
            result = avatar_gallery.create_generation_prompt("Avatar", "Editorial", reference, "face", "body")
        self.assertEqual(result, "Visual prompt")
        attached = gemini.call_args.kwargs["images"][0]
        self.assertEqual(attached["source_url"], reference["source_url"])
        with Image.open(io.BytesIO(attached["validation_bytes"])) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertLessEqual(max(image.size), 768)
            self.assertGreater(image.getpixel((0, 0))[2], 240)
        self.assertNotIn("validation_bytes", reference)

    def test_prompt_writer_reuses_search_thumbnail(self):
        reference = {"validation_bytes": b"existing-thumbnail", "path": "not-needed.png"}
        with mock.patch.object(avatar_gallery, "gemini_json", return_value={"prompt": "Visual prompt"}) as gemini:
            avatar_gallery.create_generation_prompt("Avatar", "Theme", reference, "face", "body")
        self.assertEqual(gemini.call_args.kwargs["images"][0]["validation_bytes"], b"existing-thumbnail")

    def test_missing_reference_cannot_silently_use_caption_only(self):
        with mock.patch.object(avatar_gallery, "gemini_json") as gemini:
            with self.assertRaises(FileNotFoundError):
                avatar_gallery.create_generation_prompt("Avatar", "Theme", {"path": "missing.png"}, "face", "body")
        gemini.assert_not_called()


if __name__ == "__main__":
    unittest.main()
