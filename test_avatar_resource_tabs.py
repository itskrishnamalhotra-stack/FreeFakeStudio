"""Resource search / generation separation; all images and model calls are fixtures."""
import hashlib
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import avatar_gallery as gallery
import avatar_gallery_jobs as jobs
import avatar_resource_tabs as tabs
import avatar_studio as studio
from test_avatar_gallery_jobs import image_fixture, approval


class Context:
    def __init__(self, identifier):
        self.id = identifier
        self.saved = {}
        self.store = SimpleNamespace(stopping=threading.Event(), get=lambda _: {})
    def checkpoint(self, **changes):
        self.saved.update(changes)
    def check(self):
        pass
    def emit(self, *args, **kwargs):
        pass


class ResourceTabTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.avatar = studio.create_avatar(self.temp, "Resource tabs QA")
        self.aid = self.avatar["id"]
        for part in ("face", "body"):
            studio.lock_reference(self.temp, self.aid, part, image_fixture())
        self.batch = jobs._create_batch(self.temp, self.aid, 6, "864x1152 (3:4)", "Original", False)
        self.parent = self.batch["id"]
        self.generations = []

    def seed(self, tab_id, quantity, offset=0):
        source = jobs.load_batch(self.temp, self.aid, tab_id)
        folder = jobs._root(self.temp, self.aid) / "references" / tab_id
        folder.mkdir(parents=True, exist_ok=True)
        token = jobs._batch_scope.set((str(Path(self.temp).resolve()), self.aid, tab_id))
        try:
            for index in range(offset, offset + quantity):
                path = folder / f"{index}.png"
                image_fixture(index).save(path)
                row = {"path": str(path), "image_url": f"https://example.test/{tab_id}/{index}.png",
                       "pixel_hash": f"pixels-{tab_id}-{index}"}
                jobs.add_references(self.temp, self.aid, [row], source["theme"])
                jobs.update_reference(self.temp, self.aid, jobs._reference_id(row), state="approved",
                    validation=approval(**gallery.reference_policy(source["theme"], source["require_face"], source.get("avatar_gender", "any"))))
        finally:
            jobs._batch_scope.reset(token)
        return tabs.tab_rows(self.temp, self.aid, source)

    def generate(self, avatar, reference, prompt, aspect, item_id, attempt):
        self.generations.append(reference.get("source_reference_id", reference["id"]))
        return studio.save_gallery_image(self.temp, self.aid, item_id, image_fixture(10), attempt), None, "42"

    def run_generation(self, tab_id, count, context=None, validate=None):
        context = context or Context(hashlib.sha256(str(len(self.generations)).encode()).hexdigest())
        result = tabs.run(self.temp, self.aid, tab_id, count, "864x1152 (3:4)", context,
            lambda *args: "Fixture prompt", self.generate, validate or (lambda *args: {"pass": True}), lambda prompt, _: prompt)
        return result, context

    def new_tab(self, description="Cafe", identifier="abcdef123456"):
        return tabs.create_tab(self.temp, self.aid, self.parent, identifier, description)

    def test_multiple_generation_runs_stay_in_one_batch_and_never_reuse_references(self):
        self.seed(self.parent, 12)
        with mock.patch.object(gallery, "_search_query") as search, mock.patch.object(gallery, "gemini_json") as api:
            self.run_generation(self.parent, 6)
            self.run_generation(self.parent, 6)
        search.assert_not_called()
        api.assert_not_called()
        self.assertEqual(len(self.generations), 12)
        self.assertEqual(len(set(self.generations)), 12)
        items = studio.load_gallery_items(self.temp, self.aid)
        self.assertEqual({row["batch_id"] for row in items}, {self.parent})
        self.assertEqual(len({row["generation_run_id"] for row in items}), 2)
        self.assertEqual(len(jobs.list_batches(self.temp, self.aid)), 1)
        saved_tab = tabs.list_tabs(self.temp, self.aid)[0]
        self.assertEqual(saved_tab["counts"]["used"], 12)
        self.assertEqual(saved_tab["counts"]["available"], 0)
        with self.assertRaisesRegex(ValueError, "No unused approved"):
            self.run_generation(self.parent, 1)

    def test_sources_are_independent_and_outputs_remain_under_the_parent_batch(self):
        self.seed(self.parent, 2)
        source = self.new_tab()
        self.seed(source["id"], 3)
        self.run_generation(source["id"], 2)
        states = {row["id"]: row for row in tabs.list_tabs(self.temp, self.aid)}
        self.assertEqual(states[self.parent]["counts"]["available"], 2)
        self.assertEqual(states[source["id"]]["counts"]["available"], 1)
        self.assertEqual(states[source["id"]]["counts"]["used"], 2)
        self.assertTrue(all(row["resource_tab_id"] == source["id"] for row in studio.load_gallery_items(self.temp, self.aid)))
        self.assertEqual(len(tabs.family_references(self.temp, self.aid, self.parent)), 5)

    def test_used_pixels_cannot_reenter_through_another_tab_or_url(self):
        original = self.seed(self.parent, 1)[0]
        self.run_generation(self.parent, 1)
        source = self.new_tab()
        row = self.seed(source["id"], 1)[0]
        jobs.update_reference(self.temp, self.aid, row["id"], pixel_hash=original["pixel_hash"])
        self.assertEqual(next(row for row in tabs.list_tabs(self.temp, self.aid) if row["id"] == source["id"])["counts"]["available"], 0)
        with self.assertRaisesRegex(ValueError, "No unused approved"):
            self.run_generation(source["id"], 1)

    def test_failed_validation_resumes_existing_output_without_another_generation(self):
        self.seed(self.parent, 2)
        context = Context("vision-failure")
        with self.assertRaisesRegex(RuntimeError, "Vision unavailable"):
            self.run_generation(self.parent, 2, context, mock.Mock(side_effect=RuntimeError("Vision unavailable")))
        first = studio.load_gallery_items(self.temp, self.aid)[0]
        before = Path(first["generated_path"]).read_bytes()
        self.run_generation(self.parent, 2, context)
        self.assertEqual(len(self.generations), 2)
        self.assertEqual(Path(first["generated_path"]).read_bytes(), before)

    def test_completed_run_can_finish_drive_sync_recovery_without_repeating_images(self):
        self.seed(self.parent, 1)
        context = Context("completed-recovery")
        self.run_generation(self.parent, 1, context)
        self.run_generation(self.parent, 1, context)
        self.assertEqual(len(self.generations), 1)

    def test_resources_are_reserved_before_queue_items_exist(self):
        rows = self.seed(self.parent, 2)
        run_id = "111111222222"
        tabs._write_run(self.temp, self.aid, self.batch, self.batch, run_id, rows, 2, "864x1152 (3:4)")
        self.assertEqual(studio.load_gallery_items(self.temp, self.aid), [])
        counts = tabs.list_tabs(self.temp, self.aid)[0]["counts"]
        self.assertEqual(counts["reserved"], 2)
        self.assertEqual(counts["available"], 0)
        self.assertEqual(jobs.list_batches(self.temp, self.aid)[0]["counts"]["pending"], 2)
        jobs.manage_batch(self.temp, self.aid, self.parent, "finish")
        self.assertEqual(tabs.list_tabs(self.temp, self.aid)[0]["counts"]["available"], 2)

    def test_unfinished_queue_reserves_resources_until_parent_is_finished(self):
        self.seed(self.parent, 2)
        context = Context("prompt-failure")
        with mock.patch.object(gallery, "_search_query"):
            with self.assertRaisesRegex(RuntimeError, "Prompt failed"):
                tabs.run(self.temp, self.aid, self.parent, 2, "864x1152 (3:4)", context,
                    mock.Mock(side_effect=RuntimeError("Prompt failed")), self.generate, lambda *args: {}, lambda *args: "")
        self.assertEqual(tabs.list_tabs(self.temp, self.aid)[0]["counts"]["reserved"], 2)
        with self.assertRaisesRegex(ValueError, "reserved"):
            tabs.delete(self.temp, self.aid, self.parent)
        jobs.manage_batch(self.temp, self.aid, self.parent, "finish")
        self.assertEqual(tabs.list_tabs(self.temp, self.aid)[0]["counts"]["available"], 2)

    def test_deletion_is_scoped_and_keeps_generated_image_dependencies(self):
        original = self.seed(self.parent, 2)
        source = self.new_tab()
        rows = self.seed(source["id"], 3)
        self.run_generation(source["id"], 1)
        item = studio.load_gallery_items(self.temp, self.aid)[0]
        generated = Path(item["generated_path"])
        preserved = Path(item["reference_path"])
        result = tabs.delete(self.temp, self.aid, source["id"])
        self.assertEqual(result["removed_resources"], 3)
        self.assertEqual(result["deleted_files"], 2)
        self.assertTrue(generated.is_file() and preserved.is_file())
        self.assertTrue(all(Path(row["path"]).is_file() for row in original))
        self.assertEqual([row["id"] for row in tabs.list_tabs(self.temp, self.aid)], [self.parent])
        self.assertEqual(len(tabs.family_references(self.temp, self.aid, self.parent)), 2)
        with self.assertRaisesRegex(ValueError, "not found"):
            tabs.get(self.temp, self.aid, source["id"])

    def test_delete_original_resource_tab_does_not_delete_its_batch_or_outputs(self):
        rows = self.seed(self.parent, 2)
        self.run_generation(self.parent, 1)
        tabs.delete(self.temp, self.aid, self.parent)
        self.assertIsNotNone(jobs.load_batch(self.temp, self.aid, self.parent))
        self.assertEqual(len(jobs.list_batches(self.temp, self.aid)), 1)
        self.assertEqual(len(studio.load_gallery_items(self.temp, self.aid)), 1)
        self.assertEqual(tabs.list_tabs(self.temp, self.aid), [])
        self.assertTrue(Path(rows[0]["path"]).is_file())

    def test_new_search_does_not_mutate_original_description_or_generate_images(self):
        source = self.new_tab("Beach sunsets")
        mock_search = mock.Mock(return_value=([{"image_url": "https://example.test/beach.png", "path": self.avatar["face_image"]}], {}))
        # Use a genuine fixture path; search-only does not invoke the generator.
        anchor = studio.load_avatar(self.temp, self.aid)["face_image"]
        mock_search.return_value[0][0]["path"] = anchor
        result = tabs.run(self.temp, self.aid, source["id"], 3, "864x1152 (3:4)", Context("search-1"),
            lambda *args: "", self.generate, lambda *args: {}, lambda *args: "", search=True, mock_discover=mock_search)
        self.assertTrue(result["search_only"])
        self.assertEqual(self.generations, [])
        self.assertEqual(jobs.load_batch(self.temp, self.aid, self.parent)["theme"], "Original")
        self.assertEqual(jobs.load_batch(self.temp, self.aid, source["id"])["theme"], "Beach sunsets")
        self.assertEqual(len(tabs.tab_rows(self.temp, self.aid, jobs.load_batch(self.temp, self.aid, source["id"]))), 1)

    def test_generate_more_checks_leftover_images_without_starting_another_search(self):
        rows = self.seed(self.parent, 3)
        for row in rows:
            jobs.update_reference(self.temp, self.aid, row["id"], state="downloaded", validation={})
        with mock.patch.object(gallery, "_search_query") as search, \
                mock.patch.object(gallery, "_tavily_search") as tavily, \
                mock.patch.object(gallery, "gemini_json", return_value={"results": [approval(index=n) for n in (1, 2, 3)]}) as vision:
            self.run_generation(self.parent, 2)
        search.assert_not_called()
        tavily.assert_not_called()
        vision.assert_called_once()
        counts = tabs.list_tabs(self.temp, self.aid)[0]["counts"]
        self.assertEqual(counts["used"], 2)
        self.assertEqual(counts["available"], 1)

    def test_search_503_recovery_keeps_the_same_downloads_and_query(self):
        source = self.new_tab()
        context = Context("search-recovery")
        def downloaded(row, folder):
            path = folder / "fixture.png"
            image_fixture().save(path)
            return {**row, "path": str(path)}
        options = (self.temp, self.aid, source["id"], 1, "864x1152 (3:4)", context,
                   lambda *args: "", self.generate, lambda *args: {}, lambda *args: "")
        with mock.patch.object(gallery, "_search_query", return_value="saved query") as query, \
                mock.patch.object(gallery, "_tavily_search", return_value={"images": [{"url": "https://example.test/ref.png"}]}) as search, \
                mock.patch.object(gallery, "_download_candidate", side_effect=downloaded) as download, \
                mock.patch.object(gallery, "gemini_json", side_effect=gallery.GalleryAPIError("Gemini", 503, "Busy", True)):
            with self.assertRaises(gallery.GalleryAPIError):
                tabs.run(*options, search=True)
        query.assert_called_once()
        search.assert_called_once()
        download.assert_called_once()
        saved = jobs.load_batch(self.temp, self.aid, source["id"])
        self.assertEqual(saved["queries"], ["saved query"])
        self.assertEqual(saved["downloads"], 1)
        with mock.patch.object(gallery, "_search_query") as query, \
                mock.patch.object(gallery, "_download_candidate") as download, \
                mock.patch.object(gallery, "gemini_json", return_value={"results": [approval()]}):
            tabs.run(*options, search=True)
        query.assert_not_called()
        download.assert_not_called()
        self.assertEqual(self.generations, [])

    def test_delete_never_unlinks_a_path_outside_the_reference_directory(self):
        rows = self.seed(self.parent, 1)
        external = Path(self.temp) / "keep.png"
        image_fixture().save(external)
        jobs.update_reference(self.temp, self.aid, rows[0]["id"], path=str(external))
        tabs.delete(self.temp, self.aid, self.parent)
        self.assertTrue(external.is_file())

    def test_generate_more_gets_a_fresh_download_budget_for_previously_found_images(self):
        rows = self.seed(self.parent, 1)
        saved_path = Path(rows[0]["path"])
        saved_path.unlink()
        jobs.update_batch(self.temp, self.aid, batch_id=self.parent, downloads=60, round=3)
        def downloaded(row, folder):
            image_fixture().save(saved_path)
            return {**row, "path": str(saved_path)}
        with mock.patch.object(gallery, "_search_query") as search, \
                mock.patch.object(gallery, "_download_candidate", side_effect=downloaded) as download, \
                mock.patch.object(gallery, "gemini_json", return_value={"results": [approval()]}):
            self.run_generation(self.parent, 1)
        search.assert_not_called()
        download.assert_called_once()
        self.assertEqual(len(self.generations), 1)
        self.assertEqual(jobs.load_batch(self.temp, self.aid, self.parent)["resource_download_total"], 60)

    def test_legacy_unnamed_batch_can_generate_from_its_owned_migrated_resources(self):
        rows = self.seed(self.parent, 2)
        saved = jobs.load_batch(self.temp, self.aid, self.parent)
        saved.update(version=1)
        saved.pop("name", None)
        studio._atomic_json(jobs._batch_path(self.temp, self.aid, self.parent), saved)
        catalog = jobs.load_catalog(self.temp, self.aid)
        for row in catalog["references"]:
            row.pop("batch_id", None)
        jobs._save_catalog(self.temp, self.aid, catalog)
        self.assertEqual(tabs.list_tabs(self.temp, self.aid)[0]["batch_name"], "Original")
        self.run_generation(self.parent, 1)
        self.assertEqual(len(self.generations), 1)
        self.assertEqual(len(jobs.list_batches(self.temp, self.aid)), 1)


if __name__ == "__main__":
    unittest.main()
