"""Offline tests for unified avatar images and generative texture reconstruction."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

import avatar_enhance as enhance
import avatar_studio as studio
from test_avatar_logic import app_function


def fixture(size=(180, 240)):
    image = Image.new("RGB", size, (50, 90, 140))
    ImageDraw.Draw(image).rectangle((15, 20, size[0] - 20, size[1] - 30), fill=(190, 110, 60))
    return image


class ReconstructionTests(unittest.TestCase):
    def test_reconstruction_uses_conservative_source_preserving_prompt(self):
        expected = enhance.PHOTO_PROMPT
        self.assertEqual(enhance.REALISM_PROMPT, expected)
        self.assertIn("degree of skin smoothness", expected)
        self.assertIn("background blur", expected)
        self.assertNotIn("fine irregular pores", expected)
        editor = mock.Mock(side_effect=lambda tile, *args: tile.copy())
        list(enhance.reconstruct(fixture(), editor, method="hd_tiles"))
        self.assertEqual(editor.call_args_list[0].args[2], expected)
        self.assertEqual(editor.call_args_list[1].args[2], enhance.TILE_PROMPT + expected)

    def test_target_cap_preserves_aspect_and_bounds_work(self):
        self.assertEqual(enhance.target_size((512, 768)), (1024, 1536))
        self.assertEqual(enhance.target_size((1024, 1536)), (1365, 2048))
        self.assertEqual(enhance.target_size((320, 240), 1), (320, 240))
        self.assertLessEqual(len(enhance.tile_boxes((2048, 2048))), 9)
        with self.assertRaises(ValueError):
            enhance.target_size((512, 512), 4)

    def test_tiles_cover_every_pixel_and_never_exceed_768(self):
        for size in ((1, 1), (511, 777), (1365, 2048), (2048, 2048)):
            coverage = np.zeros((size[1], size[0]), dtype=np.float32)
            for box in enhance.tile_boxes(size):
                left, top, right, bottom = box
                self.assertLessEqual(max(right - left, bottom - top), 768)
                coverage[top:bottom, left:right] += enhance._weights(box, size)
            self.assertTrue(np.all(coverage > 0), size)

    def test_identity_editor_has_no_seams_and_does_not_mutate_source(self):
        source = fixture((513, 401))
        before = source.tobytes()
        calls = []

        def editor(tile, context, prompt, seed, strength):
            calls.append((tile.size, context.size if context is not None else None, prompt, seed, strength))
            return tile.copy()

        rows = list(enhance.reconstruct(source, editor, scale=2, seed=123, method="hd_tiles"))
        result = rows[-1]["image"]
        expected = source.resize((1026, 802), Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
        self.assertEqual(source.tobytes(), before)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(w % 64 == 0 and h % 64 == 0 for (w, h), *_ in calls))
        self.assertEqual([row[3] for row in calls], [123, 124, 125, 126, 127])
        self.assertIn("Preserve the exact subjects", calls[0][2])
        self.assertIn("context only", calls[1][2])

    def test_reconstruction_retains_full_model_output_instead_of_only_high_frequencies(self):
        source = Image.new("RGB", (64, 64), (100, 100, 100))

        def editor(tile, *args):
            data = np.full((tile.height, tile.width, 3), 140, dtype=np.uint8)
            data[::2, ::2] = 160
            return Image.fromarray(data)

        result = list(enhance.reconstruct(source, editor, scale=1, preserve_tone=False))[-1]["image"]
        data = np.asarray(result)
        self.assertGreater(float(data.std()), 2)
        self.assertGreater(float(data.mean()), 140)

    def test_invalid_result_never_becomes_a_success(self):
        for result in (None, Image.new("RGB", (32, 32))):
            with self.assertRaisesRegex(RuntimeError, "invalid full-photo reconstruction"):
                list(enhance.reconstruct(fixture(), lambda *args: result))

    def test_hd_tiles_retain_material_and_lighting_corrections(self):
        source = Image.new("RGB", (128, 128), (100, 100, 100))
        def editor(tile, context, *args):
            return tile.copy() if context is None else Image.new("RGB", tile.size, (180, 160, 140))
        result = list(enhance.reconstruct(source, editor, method="hd_tiles", preserve_tone=False))[-1]
        self.assertEqual(result["image"].getpixel((64, 64)), (180, 160, 140))
        self.assertEqual(result["total"], 2)
        self.assertFalse(result["diagnostics"]["near_noop"])

    def test_brightness_only_edit_is_anchored_back_and_not_called_an_improvement(self):
        source = Image.new("RGB", (128, 128), (100, 100, 100))
        result = list(enhance.reconstruct(source, lambda tile, *args: Image.new("RGB", tile.size, (110, 110, 110))))[-1]
        self.assertEqual(result["image"].getpixel((64, 64)), (100, 100, 100))
        self.assertTrue(result["diagnostics"]["near_noop"])
        self.assertTrue(result["diagnostics"]["tone_anchor"]["enabled"])

    def test_tone_only_change_is_flagged_when_preservation_is_disabled(self):
        source = fixture((128, 128))
        def editor(tile, *args):
            return Image.fromarray(np.clip(np.asarray(tile, dtype=np.int16) + 10, 0, 255).astype(np.uint8))
        result = list(enhance.reconstruct(source, editor, preserve_tone=False))[-1]
        self.assertEqual(result["diagnostics"]["enhancement_review"]["status"], "needs_review")
        self.assertFalse(result["diagnostics"]["enhancement_review"]["quality_verified"])

    def test_tone_anchor_preserves_generated_fine_detail(self):
        source = Image.new("RGB", (128, 128), (100, 100, 100))
        pixels = np.full((128, 128, 3), 110, dtype=np.uint8)
        pixels[::2, ::2] = 120
        result, _ = enhance.anchor_tone(source, Image.fromarray(pixels))
        self.assertGreater(float(np.asarray(result).std()), 2)
        self.assertLess(float(np.asarray(result).mean()), 104)

    def test_stop_before_or_after_tile_preserves_source(self):
        source = fixture()
        before = source.tobytes()
        callback = mock.Mock(return_value=source)
        with self.assertRaises(enhance.EnhancementCancelled):
            list(enhance.reconstruct(source, callback, cancelled=lambda: True))
        callback.assert_not_called()
        stopped = False

        def editor(tile, *args):
            nonlocal stopped
            stopped = True
            return tile

        with self.assertRaises(enhance.EnhancementCancelled):
            list(enhance.reconstruct(source, editor, cancelled=lambda: stopped))
        self.assertEqual(before, source.tobytes())

    def test_strength_bounds_and_editor_failure(self):
        callback = mock.Mock(side_effect=RuntimeError("CUDA out of memory (mock)"))
        with self.assertRaises(ValueError):
            list(enhance.reconstruct(fixture(), callback, strength=1.1))
        callback.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            list(enhance.reconstruct(fixture(), callback))

    def test_default_is_one_native_edit_not_nine_generations(self):
        callback = mock.Mock(side_effect=lambda tile, *args: tile.copy())
        rows = list(enhance.reconstruct(fixture(), callback))
        callback.assert_called_once()
        self.assertIsNone(callback.call_args.args[1])
        self.assertEqual(callback.call_args.args[4], enhance.DEFAULT_STRENGTH)
        self.assertEqual(rows[-1]["total"], 1)
        self.assertEqual(rows[-1]["output_images"], 1)
        self.assertTrue(rows[-1]["diagnostics"]["near_noop"])

    def test_hd_context_anchors_to_original_not_first_reconstruction(self):
        contexts = []
        source = Image.new("RGB", (128, 128), "blue")
        def editor(tile, whole, *args):
            if whole is not None:
                contexts.append(whole.getpixel((0, 0)))
            return Image.new("RGB", tile.size, "red")
        list(enhance.reconstruct(source, editor, method="hd_tiles"))
        self.assertEqual(contexts, [(0, 0, 255)])

    def test_photo_2x_is_explicitly_one_edit_then_resize_not_extra_sampling(self):
        callback = mock.Mock(side_effect=lambda tile, *args: tile.copy())
        rows = list(enhance.reconstruct(fixture((513, 401)), callback, scale=2))
        callback.assert_called_once()
        self.assertEqual(rows[-1]["image"].size, (1026, 802))
        self.assertTrue(rows[-1]["diagnostics"]["near_noop"])

    def test_high_resolution_full_photo_working_image_stays_bounded(self):
        callback = mock.Mock(side_effect=lambda tile, *args: tile.copy())
        result = list(enhance.reconstruct(fixture((1600, 1200)), callback))[-1]
        self.assertEqual(callback.call_args.args[0].size, (1024, 768))
        self.assertTrue(result["diagnostics"]["near_noop"])

    def test_hd_2048_declares_ten_passes_but_still_one_output(self):
        stopped = False
        def cancelled():
            return stopped
        iterator = enhance.reconstruct(fixture((1024, 1024)), lambda tile, *args: tile, scale=2,
                                       method="hd_tiles", cancelled=cancelled)
        first = next(iterator)
        self.assertEqual(first["total"], 10)
        self.assertEqual(first["output_images"], 1)
        stopped = True
        with self.assertRaises(enhance.EnhancementCancelled):
            next(iterator)


class AvatarImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.avatar = studio.create_avatar(self.root, "Image QA")
        self.id = self.avatar["id"]

    def console_image(self):
        path = studio.save_chat_image(self.root, self.id, fixture())
        studio.append_avatar_chat(self.root, self.id, "assistant", "Created", image_path=path,
                                  metadata={"prompt": "Original prompt", "seed": "123", "mode": "Scene",
                                            "aspect": "768x1024", "extra_references": []})
        return path

    def test_console_images_are_backfilled_once_without_copying(self):
        path = self.console_image()
        before = Path(path).read_bytes()
        first = studio.visible_gallery_items(self.root, self.id)
        second = studio.visible_gallery_items(self.root, self.id)
        self.assertEqual(len(first), 1)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["origin"], "console")
        self.assertEqual(first[0]["prompt"], "Original prompt")
        self.assertEqual(first[0]["generated_path"], path)
        self.assertEqual(Path(path).read_bytes(), before)
        self.assertEqual(studio.load_avatar(self.root, self.id)["gallery_count"], 1)
        self.assertFalse(list((studio.avatar_path(self.root, self.id) / "gallery/images").glob("*.png")))

    def test_user_uploads_and_missing_results_are_not_gallery_images(self):
        path = studio.save_chat_image(self.root, self.id, fixture())
        studio.append_avatar_chat(self.root, self.id, "user", "Upload", image_path=path)
        studio.append_avatar_chat(self.root, self.id, "assistant", "Missing", image_path="missing.png")
        self.assertEqual(studio.visible_gallery_items(self.root, self.id), [])

    def test_old_chat_source_is_addressable_outside_gallery_preview_window(self):
        path = self.console_image()
        source = studio.sync_console_gallery(self.root, self.id)[0]
        items = [source] + [{"id": f"later-{index}", "generated_path": path} for index in range(101)]
        studio.save_gallery_items(self.root, self.id, items)
        self.assertNotIn(source["id"], {row["id"] for row in studio.visible_gallery_items(self.root, self.id)})
        self.assertEqual(studio.gallery_item_by_id(self.root, self.id, source["id"])["generated_path"], path)

    def test_console_regeneration_keeps_original_index_without_duplicate_ids(self):
        path = self.console_image()
        source = studio.sync_console_gallery(self.root, self.id)[0]
        regenerated = studio.save_gallery_image(self.root, self.id, source["id"], fixture(), 2)
        studio.update_gallery_item(self.root, self.id, source["id"], generated_path=regenerated)
        items = studio.sync_console_gallery(self.root, self.id)
        self.assertEqual(len(items), 2)
        self.assertEqual(len({item["id"] for item in items}), 2)
        self.assertEqual({item["generated_path"] for item in items}, {path, regenerated})
        self.assertEqual(items, studio.sync_console_gallery(self.root, self.id))

    def handler(self, fail=False):
        path = self.console_image()
        source_id = studio.visible_gallery_items(self.root, self.id)[0]["id"]
        editor = mock.Mock(side_effect=RuntimeError("Mock inference failure")) if fail else mock.Mock(side_effect=lambda images, *args: images[0].copy())
        controls = {name: name for name in ("enhance_job_id", "enhance_after", "enhance_before", "enhance_status", "enhance_stop",
                    "avatar_gallery_display", "selected_avatar_gallery_index", "gallery_selected_panel")}
        ns = {**controls, "avatar_studio": studio, "avatar_enhance": enhance, "SAVE_DIR": self.root,
              "gr": SimpleNamespace(Error=ValueError, update=lambda **kwargs: kwargs),
              "Image": Image, "uuid": __import__("uuid"), "make_seed": lambda seed: 123,
              "_status_html": lambda *args: args, "_avatar_gallery_value": lambda avatar: studio.visible_gallery_items(self.root, self.id),
              "model_manager": SimpleNamespace(FLUX_MODEL_NAME="FLUX.2-klein 4B", ensure_model=lambda name: SimpleNamespace(img2img=editor))}
        return app_function("_enhance_avatar_image", ns), source_id, path, editor

    def test_unchanged_enhancement_saves_separate_no_change_variant_and_recipe(self):
        handler, source_id, path, editor = self.handler()
        before = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        rows = list(handler(self.id, source_id, 2, 1.0, 4))
        self.assertIn("Original preserved", rows[-1]["enhance_status"][1])
        items = studio.load_gallery_items(self.root, self.id)
        self.assertEqual(len(items), 2)
        result = items[-1]
        self.assertEqual(result["origin"], "enhancement")
        self.assertEqual(result["status"], "no_change")
        self.assertEqual(result["source_id"], source_id)
        self.assertEqual(result["prompt"], enhance.PHOTO_PROMPT)
        self.assertTrue(result["diagnostics"]["near_noop"])
        self.assertEqual(result["processing_passes"], 1)
        self.assertNotEqual(result["generated_path"], path)
        self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), before)
        with Image.open(result["generated_path"]) as image:
            self.assertEqual(image.size, (360, 480))
        self.assertEqual(editor.call_args.args[2], "")
        self.assertEqual(editor.call_args.args[5], 1.0)
        self.assertEqual(len(editor.call_args.args[0]), 1)
        self.assertEqual(editor.call_args.args[6], 4)

    def test_hd_backend_uses_context_only_for_tiles_and_saves_one_variant(self):
        handler, source_id, path, editor = self.handler()
        rows = list(handler(self.id, source_id, 2, 1.0, 4, "hd_tiles"))
        self.assertEqual(editor.call_count, 2)
        self.assertEqual([len(call.args[0]) for call in editor.call_args_list], [1, 2])
        items = studio.load_gallery_items(self.root, self.id)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[-1]["method"], "hd_tiles")
        self.assertEqual(items[-1]["processing_passes"], 2)
        self.assertIn("enhance_after", rows[-1])

    def test_inference_failure_keeps_original_and_records_failure(self):
        handler, source_id, path, editor = self.handler(fail=True)
        with self.assertRaisesRegex(RuntimeError, "Mock inference failure"):
            list(handler(self.id, source_id, 2, 1.0, 4))
        self.assertEqual(studio.load_gallery_items(self.root, self.id)[-1]["status"], "failed")
        self.assertTrue(Path(path).is_file())
        self.assertEqual(len(studio.visible_gallery_items(self.root, self.id)), 1)

    def test_stop_marks_only_pending_variant_stopped(self):
        handler, source_id, path, editor = self.handler()
        iterator = handler(self.id, source_id, 2, 1.0, 4)
        job_id = next(iterator)["enhance_job_id"]
        studio.update_gallery_item(self.root, self.id, job_id, stop_requested=True)
        list(iterator)
        editor.assert_not_called()
        self.assertEqual(studio.load_gallery_items(self.root, self.id)[-1]["status"], "stopped")
        self.assertTrue(Path(path).is_file())

    def test_stale_source_fails_before_model_access(self):
        handler, _, _, editor = self.handler()
        with self.assertRaisesRegex(ValueError, "no longer available"):
            list(handler(self.id, "missing", 2, 1.0, 4))
        editor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
