"""Offline gallery durability, filtering, retry and recovery tests."""

import io
import json
import tempfile
import unittest
from contextvars import Context
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, ImageDraw

import avatar_gallery as gallery
import avatar_gallery_jobs as jobs
import avatar_studio as studio


def image_fixture(index=0):
    image = Image.new("RGB", (360, 480), (20 + index * 10, 70, 130))
    ImageDraw.Draw(image).rectangle((30 + index, 40, 240, 350), fill=(190, 110, 60 + index))
    return image


def approval(**changes):
    return {"index": 1, "accept": True, "score": 90, "adult_confidence": 95,
            "adult_person_visible": True, "outfit_or_pose_visible": True,
            "person_large_enough": True, "major_obstruction": False,
            "image_clear": True, "face_visible": True, "reason": "Clear outfit reference", **changes}


def response(status, payload=None, headers=None):
    return SimpleNamespace(ok=status < 400, status_code=status, headers=headers or {},
                           json=lambda: payload or {"error": {"message": "Temporarily unavailable"}})


class APIRetryTests(unittest.TestCase):
    def test_atomic_save_retries_a_transient_windows_reader_lock(self):
        error = PermissionError("Temporarily locked")
        error.winerror = 5
        original = studio.os.replace
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "saved.json"
            with mock.patch.object(studio.os, "replace", wraps=original, side_effect=[error, mock.DEFAULT]) as replace, \
                    mock.patch.object(studio.time, "sleep") as sleep:
                studio._atomic_json(path, {"saved": True})
            self.assertEqual(replace.call_count, 2)
            sleep.assert_called_once_with(0.02)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"saved": True})
            self.assertFalse(list(Path(root).glob("*.tmp")))

    def test_corrupt_gallery_is_not_silently_replaced_with_an_empty_list(self):
        with tempfile.TemporaryDirectory() as root:
            avatar = studio.create_avatar(root, "Corrupt gallery QA")
            path = studio.gallery_manifest_path(root, avatar["id"])
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Preserve this file"):
                studio.load_gallery_items(root, avatar["id"])
            self.assertEqual(path.read_text(encoding="utf-8"), "{broken")

    def test_503_retries_with_backoff_then_returns_success(self):
        call = mock.Mock(side_effect=[response(503), response(503), response(200)])
        events = []
        with mock.patch.object(gallery.time, "sleep") as sleep, mock.patch.object(gallery.random, "uniform", return_value=0), \
                gallery.api_context(events.append):
            self.assertTrue(gallery._request_with_retry(call, "Gemini").ok)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(sum(row.args[0] for row in sleep.call_args_list), 3)
        self.assertEqual(len(events), 2)

    def test_persistent_503_is_bounded_and_remains_retryable(self):
        call = mock.Mock(return_value=response(503))
        with mock.patch.object(gallery.time, "sleep"), mock.patch.object(gallery.random, "uniform", return_value=0):
            with self.assertRaises(gallery.GalleryAPIError) as caught:
                gallery._request_with_retry(call, "Gemini")
        self.assertEqual(call.call_count, 4)
        self.assertTrue(caught.exception.retryable)

    def test_auth_and_invalid_requests_are_not_retried(self):
        for status in (400, 401, 402, 403):
            with self.subTest(status=status):
                call = mock.Mock(return_value=response(status))
                with self.assertRaises(gallery.GalleryAPIError) as caught:
                    gallery._request_with_retry(call, "Gemini")
                self.assertFalse(caught.exception.retryable)
                call.assert_called_once()

    def test_retry_after_header_is_honored_within_wait_budget(self):
        call = mock.Mock(side_effect=[response(429, headers={"Retry-After": "5"}), response(200)])
        with mock.patch.object(gallery.time, "sleep") as sleep, mock.patch.object(gallery.random, "uniform", return_value=0):
            gallery._request_with_retry(call, "Gemini")
        self.assertEqual(sum(row.args[0] for row in sleep.call_args_list), 5)

    def test_stop_interrupts_backoff_without_another_request(self):
        stopped = False

        def retry(message):
            nonlocal stopped
            stopped = True

        call = mock.Mock(return_value=response(503))
        with gallery.api_context(retry, lambda: stopped), mock.patch.object(gallery.time, "sleep"):
            with self.assertRaises(gallery.GalleryCancelled):
                gallery._request_with_retry(call, "Gemini")
        call.assert_called_once()

    def test_timeout_is_retried_without_leaking_request_url(self):
        call = mock.Mock(side_effect=[gallery.requests.Timeout("URL contains key=private"), response(200)])
        with mock.patch.object(gallery.time, "sleep"):
            self.assertTrue(gallery._request_with_retry(call, "Gemini").ok)

    def test_validation_503_does_not_recursively_split_into_more_api_calls(self):
        error = gallery.GalleryAPIError("Gemini", 503, "Busy", retryable=True)
        with mock.patch.object(gallery, "gemini_json", side_effect=error) as call:
            with self.assertRaises(gallery.GalleryAPIError):
                gallery._validate_references([{}] * 8)
        call.assert_called_once()

    def test_clear_image_and_optional_face_rules_are_distinct(self):
        self.assertTrue(gallery.reference_passes(approval(face_visible=False)))
        self.assertFalse(gallery.reference_passes(approval(face_visible=False), require_face=True))
        self.assertFalse(gallery.reference_passes(approval(image_clear=False)))

    def test_validation_reports_rejected_references_instead_of_discarding_findings(self):
        checks = [approval(index=1), approval(index=2, accept=False, image_clear=False, reason="Too blurry")]
        events = []
        with mock.patch.object(gallery, "gemini_json", return_value={"results": checks}):
            selected = gallery._validate_references([{"id": "one"}, {"id": "two"}],
                                                    on_result=lambda row, check, accepted: events.append((row["id"], accepted)))
        self.assertEqual(events, [("one", True), ("two", False)])
        self.assertEqual(len(selected), 1)

    def test_download_records_blank_check_and_pixel_fingerprint(self):
        image = Image.new("RGB", (360, 480), "white")
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        result = SimpleNamespace(raise_for_status=lambda: None, iter_content=lambda size: [buffer.getvalue()])
        with tempfile.TemporaryDirectory() as root, mock.patch.object(gallery.requests, "get", return_value=result):
            downloaded = gallery._download_candidate({"image_url": "https://example.test/blank.png"}, root)
            self.assertTrue(downloaded["local_quality"]["blank"])
            self.assertEqual(len(downloaded["pixel_hash"]), 64)
            self.assertTrue(Path(downloaded["path"]).is_file())


class GalleryJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = self.temporary.name
        avatar = studio.create_avatar(self.root, "Queue QA")
        self.avatar_id = avatar["id"]
        for kind in ("face", "body"):
            studio.lock_reference(self.root, self.avatar_id, kind, image_fixture())
        self.calls = []
        self.discover = mock.Mock(side_effect=self.references)

    def references(self, avatar_id, count, theme):
        result = []
        for index in range(count):
            path = studio.save_gallery_reference(self.root, avatar_id, image_fixture(index), f"ref-{index}")
            result.append({"image_url": f"https://example.test/{theme}/{index}.jpg", "path": path,
                           "description": f"Reference {index}"})
        return result, {}

    def prompt(self, avatar, theme, reference):
        self.calls.append(("prompt", reference["id"]))
        return "Prompt for " + reference["id"]

    def generate(self, avatar, reference, prompt, aspect, item_id, attempt):
        self.calls.append(("generate", reference["id"], attempt))
        path = studio.save_gallery_image(self.root, self.avatar_id, item_id, image_fixture(), attempt)
        return path, None, "123"

    def validate(self, image, avatar, prompt):
        self.calls.append(("validate", prompt))
        return {"pass": True, "score": 100}

    def repair(self, prompt, validation):
        self.calls.append(("repair", prompt))
        return prompt + " repaired"

    def execute(self, **changes):
        options = dict(save_dir=self.root, avatar_id=self.avatar_id, quantity=3,
                       aspect="864x1152 (3:4)", theme="editorial", require_face=False,
                       make_prompt=self.prompt, generate=self.generate, validate=self.validate,
                       repair=self.repair, mock_discover=self.discover)
        options.update(changes)
        return list(jobs.run(**options))

    def test_commits_entire_queue_and_prepares_all_prompts_before_generation(self):
        events = self.execute()
        self.assertEqual([call[0] for call in self.calls[:3]], ["prompt"] * 3)
        self.assertEqual(events[-1]["status"], "complete")
        items = studio.load_gallery_items(self.root, self.avatar_id)
        self.assertEqual(len(items), 3)
        self.assertTrue(all(item["status"] == "passed" for item in items))
        self.assertEqual(len({item["batch_id"] for item in items}), 1)

    def test_generator_can_advance_in_separate_gradio_worker_contexts(self):
        iterator = jobs.run(self.root, self.avatar_id, 2, "864x1152 (3:4)", "editorial", False,
                            self.prompt, self.generate, self.validate, self.repair, self.discover)
        events = []
        while True:
            try:
                events.append(Context().run(next, iterator))
            except StopIteration:
                break
        self.assertEqual(events[-1]["status"], "complete")
        self.assertFalse(jobs._run_lock.locked())

    def test_503_during_prompt_preparation_resumes_pending_prompts_without_search(self):
        calls = 0

        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise gallery.GalleryAPIError("Gemini", 503, "Busy", True)
            return self.prompt(*args)

        with self.assertRaises(gallery.GalleryAPIError):
            self.execute(make_prompt=interrupted)
        original = jobs.load_batch(self.root, self.avatar_id)
        self.assertEqual(original["status"], "paused")
        self.assertEqual(len(original["item_ids"]), 3)
        self.assertFalse(any(call[0] == "generate" for call in self.calls))
        events = self.execute(quantity=20, theme="different input ignored on resume")
        self.assertEqual(events[-1]["id"], original["id"])
        self.assertEqual(events[-1]["quantity"], 3)
        self.assertEqual(sum(call[0] == "prompt" for call in self.calls), 3)
        self.discover.assert_called_once()

    def test_generation_failure_preserves_completed_image_and_resumes_other_items(self):
        calls = 0

        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("GPU interrupted")
            return self.generate(*args)

        with self.assertRaisesRegex(RuntimeError, "GPU interrupted"):
            self.execute(generate=interrupted)
        first = studio.load_gallery_items(self.root, self.avatar_id)[0]
        before = Path(first["generated_path"]).read_bytes()
        self.execute()
        self.assertEqual(sum(call[0] == "generate" for call in self.calls), 3)
        self.assertEqual(Path(first["generated_path"]).read_bytes(), before)
        self.discover.assert_called_once()

    def test_saved_image_is_revalidated_without_regenerating_after_vision_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Vision unavailable"):
            self.execute(validate=mock.Mock(side_effect=RuntimeError("Vision unavailable")))
        pending = studio.load_gallery_items(self.root, self.avatar_id)[0]
        self.assertEqual(pending["status"], "checking")
        self.assertTrue(Path(pending["generated_path"]).is_file())
        self.execute()
        self.assertEqual(sum(call[0] == "generate" for call in self.calls), 3)

    def test_repairs_are_deferred_until_first_attempts_for_every_reference(self):
        validations = iter([{"pass": False, "reasons": ["Wrong pose"]}, {"pass": True}, {"pass": True}, {"pass": True}])
        self.execute(validate=lambda *args: next(validations))
        kinds = [call[0] for call in self.calls]
        self.assertEqual(kinds.count("generate"), 4)
        self.assertEqual(kinds[:kinds.index("repair")].count("generate"), 3)

    def test_repair_503_keeps_all_first_images_and_resumes_without_revalidating_passes(self):
        validations = iter([{"pass": False, "reasons": ["Wrong pose"]}, {"pass": True}, {"pass": True}])
        with self.assertRaises(gallery.GalleryAPIError):
            self.execute(validate=lambda *args: next(validations),
                         repair=mock.Mock(side_effect=gallery.GalleryAPIError("Gemini", 503, "Busy", True)))
        items = studio.load_gallery_items(self.root, self.avatar_id)
        self.assertTrue(all(Path(item["generated_path"]).is_file() for item in items))
        self.assertEqual(items[0]["status"], "repair_pending")
        self.execute()
        self.assertEqual(sum(call[0] == "generate" for call in self.calls), 4)
        self.assertTrue(Path(items[0]["generated_path"]).is_file())
        self.discover.assert_called_once()

    def test_stop_saves_queue_and_resume_does_not_search_again(self):
        def stop_after_image(*args):
            result = self.generate(*args)
            jobs.stop_batch(self.root, self.avatar_id)
            return result

        events = self.execute(generate=stop_after_image)
        self.assertEqual(events[-1]["status"], "stopped")
        self.assertEqual(studio.load_gallery_items(self.root, self.avatar_id)[0]["status"], "checking")
        self.execute()
        self.assertEqual(sum(call[0] == "generate" for call in self.calls), 3)
        self.discover.assert_called_once()

    def test_old_reports_and_unindexed_photos_are_imported_without_deletion(self):
        folder = studio.avatar_path(self.root, self.avatar_id) / "gallery" / "references"
        first = folder / "first.jpg"
        second = folder / "second.jpg"
        image_fixture().save(first)
        image_fixture(1).save(second)
        (folder / "search_legacy.json").write_text(json.dumps({"references": [
            {"image_url": "https://example.test/old.jpg", "path": str(first), "validation": {"accept": True}}
        ]}), encoding="utf-8")
        jobs.import_legacy_references(self.root, self.avatar_id)
        jobs.import_legacy_references(self.root, self.avatar_id)
        rows = jobs.load_catalog(self.root, self.avatar_id)["references"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(first.is_file() and second.is_file())
        self.assertTrue(all(row["state"] == "downloaded" for row in rows))

    def test_existing_approved_unused_references_are_consumed_before_search(self):
        references, _ = self.references(self.avatar_id, 3, "editorial")
        jobs.add_references(self.root, self.avatar_id, references, "editorial")
        for row in jobs.load_catalog(self.root, self.avatar_id)["references"]:
            jobs.update_reference(self.root, self.avatar_id, row["id"], state="approved", validation=approval())
        with mock.patch.object(gallery, "_search_query") as search:
            self.execute(mock_discover=None)
        search.assert_not_called()
        self.discover.assert_not_called()

    def test_snapshot_shows_found_approved_rejected_and_generated_counts(self):
        self.execute()
        rows = jobs.load_catalog(self.root, self.avatar_id)["references"]
        jobs.update_reference(self.root, self.avatar_id, rows[0]["id"], state="rejected", reason="Too blurry")
        view = jobs.snapshot(self.root, self.avatar_id, "Rejected")
        self.assertEqual(view["counts"]["found"], 3)
        self.assertEqual(view["counts"]["generated"], 3)
        self.assertEqual(len(view["rows"]), 1)
        self.assertIn("Too blurry", view["rows"][0][1])

    def test_identity_change_does_not_silently_resume_an_old_batch(self):
        with self.assertRaises(RuntimeError):
            self.execute(make_prompt=mock.Mock(side_effect=RuntimeError("Unavailable")))
        studio.lock_reference(self.root, self.avatar_id, "body", image_fixture(2))
        with self.assertRaisesRegex(ValueError, "different confirmed identity"):
            self.execute()

    def test_corrupt_batch_file_is_not_silently_replaced(self):
        self.execute()
        job = jobs.load_batch(self.root, self.avatar_id)
        jobs._batch_path(self.root, self.avatar_id, job["id"]).write_text("broken", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "cannot be read"):
            jobs.load_batch(self.root, self.avatar_id)
        view = jobs.safe_snapshot(self.root, self.avatar_id)
        self.assertIn("cannot be read", view["error"])

    def test_new_search_cannot_discard_a_committed_unfinished_queue(self):
        with self.assertRaises(RuntimeError):
            self.execute(make_prompt=mock.Mock(side_effect=RuntimeError("Busy")))
        with self.assertRaisesRegex(ValueError, "Resume the committed batch"):
            jobs.new_search(self.root, self.avatar_id)
        self.assertEqual(len(jobs.load_batch(self.root, self.avatar_id)["item_ids"]), 3)

    def real_discovery(self, candidates, validations, **options):
        def downloaded(row, folder):
            index = int(row["image_url"].rsplit("/", 1)[-1].split(".")[0])
            image = Image.new("RGB", (360, 480), "white") if index == 3 else image_fixture(index)
            path = folder / f"reference_{index}.jpg"
            image.save(path)
            return {**row, "path": str(path)}

        with mock.patch.object(gallery, "_search_query", return_value="query") as search, \
                mock.patch.object(gallery, "_tavily_search", return_value={"images": candidates}), \
                mock.patch.object(gallery, "_download_candidate", side_effect=downloaded), \
                mock.patch.object(gallery, "gemini_json", side_effect=validations):
            result = self.execute(mock_discover=None, **options)
        return result, search

    def test_real_discovery_keeps_blank_blurry_and_approved_findings(self):
        candidates = [{"url": f"https://example.test/{index}.jpg"} for index in (1, 2, 3)]
        with mock.patch.dict(gallery.os.environ, {"FFS_AVATAR_SEARCH_ROUNDS": "1"}):
            self.real_discovery(candidates, [{"results": [approval(index=1), approval(index=2, accept=False, image_clear=False, reason="Too blurry")]}], quantity=1)
        view = jobs.snapshot(self.root, self.avatar_id)
        self.assertEqual(view["counts"]["found"], 3)
        self.assertEqual(view["counts"]["approved"], 1)
        self.assertEqual(view["counts"]["rejected"], 2)
        reasons = [row[1] for row in view["rows"]]
        self.assertIn("Blank or nearly uniform image", reasons)
        self.assertIn("Too blurry", reasons)

    def test_face_requirement_rejects_otherwise_good_outfit_reference(self):
        candidates = [{"url": f"https://example.test/{index}.jpg"} for index in (1, 2)]
        self.real_discovery(candidates, [{"results": [approval(index=1, face_visible=False), approval(index=2)]}],
                            quantity=1, require_face=True)
        rows = jobs.load_catalog(self.root, self.avatar_id)["references"]
        self.assertEqual(rows[0]["reason"], "Visible face required")
        self.assertEqual(rows[1]["state"], "approved")

    def test_cleaning_503_preserves_downloaded_images_and_does_not_search_on_resume(self):
        candidates = [{"url": f"https://example.test/{index}.jpg"} for index in (1, 2)]
        with self.assertRaises(gallery.GalleryAPIError):
            self.real_discovery(candidates, gallery.GalleryAPIError("Gemini", 503, "Busy", True), quantity=2)
        rows = jobs.load_catalog(self.root, self.avatar_id)["references"]
        self.assertTrue(all(row["state"] == "downloaded" for row in rows))
        self.assertTrue(all(Path(row["path"]).is_file() for row in rows))
        with mock.patch.object(gallery, "_search_query") as search, \
                mock.patch.object(gallery, "_download_candidate") as download, \
                mock.patch.object(gallery, "gemini_json", return_value={"results": [approval(index=1), approval(index=2)]}):
            self.execute(mock_discover=None)
        search.assert_not_called()
        download.assert_not_called()

    def test_download_budget_is_total_and_all_findings_are_kept(self):
        candidates = [{"url": f"https://example.test/{index}.jpg"} for index in range(20)]
        with mock.patch.object(gallery, "_search_query", return_value="query"), \
                mock.patch.object(gallery, "_tavily_search", return_value={"images": candidates}), \
                mock.patch.object(gallery, "_download_candidate", side_effect=ValueError("Bad image")) as download, \
                mock.patch.dict(gallery.os.environ, {"FFS_AVATAR_MAX_CANDIDATE_DOWNLOADS": "10"}):
            with self.assertRaisesRegex(RuntimeError, "No references passed"):
                self.execute(mock_discover=None)
        self.assertEqual(download.call_count, 10)
        self.assertEqual(len(jobs.load_catalog(self.root, self.avatar_id)["references"]), 20)
        jobs.new_search(self.root, self.avatar_id)
        self.assertEqual(len(jobs.load_catalog(self.root, self.avatar_id)["references"]), 20)


if __name__ == "__main__":
    unittest.main()
