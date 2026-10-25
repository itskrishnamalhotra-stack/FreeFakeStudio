"""Offline request-specific reference checks, migration and queue recovery."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import avatar_gallery as gallery
import avatar_gallery_jobs as jobs
import avatar_resource_tabs as tabs
import avatar_resources
import avatar_studio as studio
from test_avatar_gallery_jobs import approval, image_fixture
from test_avatar_resource_tabs import Context


THEME = "Tara Sutaria wearing a bikini at the beach"


class RelevanceGateTests(unittest.TestCase):
    def checked(self, **changes):
        return approval(**gallery.reference_policy(THEME), **changes)

    def test_every_mismatch_or_uncertain_requirement_overrides_a_high_score(self):
        self.assertTrue(gallery.reference_passes(self.checked(), theme=THEME))
        for field in ("request_match", "clothing_match", "scene_match", "subject_match", "source_name_match"):
            for verdict in ("mismatch", "unclear", None):
                with self.subTest(field=field, verdict=verdict):
                    self.assertFalse(gallery.reference_passes(self.checked(**{field: verdict}), theme=THEME))
        self.assertFalse(gallery.reference_passes(self.checked(text_overlay="dominant"), theme=THEME))
        self.assertFalse(gallery.reference_passes(self.checked(text_overlay=None), theme=THEME))
        self.assertFalse(gallery.reference_passes(self.checked(score="invalid"), theme=THEME))

    def test_saved_check_is_scoped_to_exact_request_gender_face_rule_and_version(self):
        check = self.checked()
        self.assertFalse(gallery.reference_passes(approval(), theme=THEME))
        self.assertFalse(gallery.reference_passes(check, theme="A woman in a dress"))
        self.assertFalse(gallery.reference_passes(check, require_face=True, theme=THEME))
        self.assertFalse(gallery.reference_passes(check, avatar_gender="female", theme=THEME))
        self.assertFalse(gallery.reference_passes({**check, "check_version": 1}, theme=THEME))
        self.assertTrue(gallery.reference_passes(check, theme="  Tara Sutaria wearing a bikini  at the beach  "))

    def test_validation_receives_exact_request_and_per_image_source_evidence(self):
        candidates = [{"id": "one", "title": "Tara Sutaria beach photographs", "description": "Black bikini",
                       "source_url": "https://example.test/tara-sutaria", "image_url": "https://cdn.test/photo.jpg"},
                      {"id": "two", "title": "Unrelated event", "description": "Man in a shirt"}]
        checks = [approval(index=1, source_name_match="match"),
                  approval(index=2, clothing_match="mismatch", source_name_match="mismatch", request_reason="Shirt, not bikini")]
        events = []
        with mock.patch.object(gallery, "gemini_json", return_value={"results": checks}) as api:
            result = gallery._validate_references(candidates, theme=THEME,
                on_result=lambda row, check, accepted: events.append((row["id"], accepted)))
        self.assertEqual([row["id"] for row in result], ["one"])
        self.assertEqual(events, [("one", True), ("two", False)])
        prompt = api.call_args.args[0]
        self.assertIn(THEME, prompt)
        self.assertIn("Man in a shirt", prompt)
        self.assertIn("https://example.test/tara-sutaria", prompt)
        self.assertIn("Never identify", prompt)
        schema = api.call_args.args[1]["properties"]["results"]["items"]
        self.assertIn("clothing_match", schema["required"])
        self.assertIn("source_name_match", schema["required"])
        self.assertEqual(result[0]["validation"]["request_fingerprint"], gallery.reference_policy(THEME)["request_fingerprint"])

    def test_duplicate_indices_are_not_persisted_as_approvals(self):
        events = []
        responses = [{"results": [approval(index=1), approval(index=1)]},
                     {"results": [approval(accept=False)]}, {"results": [approval()]}]
        with mock.patch.object(gallery, "gemini_json", side_effect=responses):
            gallery._validate_references([{"id": "a"}, {"id": "b"}], theme=THEME,
                on_result=lambda row, check, accepted: events.append((row["id"], accepted)))
        self.assertEqual(events, [("a", False), ("b", True)])

    def test_candidates_keep_and_merge_metadata_without_duplicate_downloads(self):
        payload = {"images": [{"url": "https://cdn.test/a.jpg", "description": "Black bikini"}],
                   "results": [{"url": "https://source.test/tara-sutaria", "title": "Tara Sutaria",
                                "images": ["https://cdn.test/a.jpg"]}]}
        rows = gallery._candidate_images(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["description"], "Black bikini")
        self.assertEqual(rows[0]["title"], "Tara Sutaria")
        self.assertEqual(rows[0]["source_url"], "https://source.test/tara-sutaria")


class RelevanceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = self.enterContext(tempfile.TemporaryDirectory())
        self.avatar = studio.create_avatar(self.root, "Relevance QA")
        self.aid = self.avatar["id"]
        for kind in ("face", "body"):
            studio.lock_reference(self.root, self.aid, kind, image_fixture())
        self.batch = jobs._create_batch(self.root, self.aid, 1, "864x1152 (3:4)", THEME, False)
        self.bid = self.batch["id"]
        self.token = jobs._batch_scope.set((str(Path(self.root).resolve()), self.aid, self.bid))
        self.addCleanup(jobs._batch_scope.reset, self.token)

    def seed(self, count=1, current=False):
        for index in range(count):
            path = jobs._root(self.root, self.aid) / "references" / f"fixture-{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image_fixture(index).save(path)
            source = {"path": str(path), "image_url": f"https://example.test/{index}.png", "title": "Unrelated event"}
            jobs.add_references(self.root, self.aid, [source], THEME)
            check = approval(**gallery.reference_policy(THEME)) if current else approval()
            jobs.update_reference(self.root, self.aid, jobs._reference_id(source), state="approved", validation=check)
        return jobs.batch_references(self.root, self.aid, self.batch)

    def test_old_approvals_are_pending_in_inventory_counts_and_pipeline_without_writing(self):
        rows = self.seed()
        catalog_path = jobs._root(self.root, self.aid) / "references/catalog.json"
        before = catalog_path.read_bytes()
        self.assertEqual(jobs._approved(self.root, self.aid, self.batch), [])
        counts = tabs.list_tabs(self.root, self.aid)[0]["counts"]
        self.assertEqual((counts["available"], counts["pending"]), (0, 1))
        snapshot = jobs.snapshot(self.root, self.aid)
        self.assertEqual(snapshot["counts"]["approved"], 0)
        chats = mock.Mock()
        chats.list.return_value = []
        inventory = avatar_resources.collect(self.root, self.aid, chats, self.bid, self.bid)
        self.assertEqual(inventory[0]["state"], "pending")
        self.assertIn("Recheck required", inventory[0]["reason"])
        self.assertEqual(before, catalog_path.read_bytes())
        self.assertTrue(Path(rows[0]["path"]).is_file())

    def test_review_existing_resources_does_not_search_or_generate_and_rejects_unrelated(self):
        self.seed(2)
        checks = [approval(index=1, clothing_match="mismatch", request_reason="Man in shirt, not requested outfit"),
                  approval(index=2, request_reason="Requested beach outfit visible")]
        generate = mock.Mock(side_effect=AssertionError("Review must not generate"))
        with mock.patch.object(gallery, "gemini_json", return_value={"results": checks}), \
             mock.patch.object(gallery, "_search_query") as search, \
             mock.patch.object(gallery, "_download_candidate") as download:
            result = tabs.run(self.root, self.aid, self.bid, 1, "864x1152 (3:4)", Context("review"),
                mock.Mock(), generate, mock.Mock(), mock.Mock(), review_only=True)
        self.assertTrue(result["review_only"])
        search.assert_not_called()
        download.assert_not_called()
        generate.assert_not_called()
        rows = jobs.batch_references(self.root, self.aid, self.batch)
        self.assertEqual([row["state"] for row in rows], ["rejected", "approved"])
        self.assertIn("Man in shirt", rows[0]["reason"])
        self.assertEqual(len(jobs._approved(self.root, self.aid, self.batch)), 1)

    def test_review_503_resume_reuses_files_and_never_claims_old_approvals_are_current(self):
        rows = self.seed()
        original = Path(rows[0]["path"]).read_bytes()
        context = Context("review-resume")
        args = (self.root, self.aid, self.bid, 1, "864x1152 (3:4)", context,
                mock.Mock(), mock.Mock(side_effect=AssertionError("No generation")), mock.Mock(), mock.Mock())
        with mock.patch.object(gallery, "gemini_json", side_effect=gallery.GalleryAPIError("Gemini", 503, "Busy", True)):
            with self.assertRaises(gallery.GalleryAPIError):
                tabs.run(*args, review_only=True)
        self.assertEqual(tabs.list_tabs(self.root, self.aid)[0]["counts"]["available"], 0)
        with mock.patch.object(gallery, "gemini_json", return_value={"results": [approval()]}), \
             mock.patch.object(gallery, "_download_candidate") as download:
            tabs.run(*args, review_only=True)
        download.assert_not_called()
        self.assertEqual(Path(rows[0]["path"]).read_bytes(), original)
        self.assertEqual(len(jobs._approved(self.root, self.aid, self.batch)), 1)

    def test_old_committed_queue_rechecks_and_skips_bad_reference_before_prompt_or_generation(self):
        self.seed(current=True)
        options = dict(save_dir=self.root, avatar_id=self.aid, quantity=1, aspect="864x1152 (3:4)",
                       theme=THEME, require_face=False, make_prompt=mock.Mock(side_effect=RuntimeError("Pause before prompt")),
                       generate=mock.Mock(), validate=mock.Mock(), repair=mock.Mock(), batch_id=self.bid)
        with self.assertRaisesRegex(RuntimeError, "Pause before prompt"):
            list(jobs.run(**options))
        row = jobs.batch_references(self.root, self.aid, self.batch)[0]
        jobs.update_reference(self.root, self.aid, row["id"], validation=approval())
        options["make_prompt"] = mock.Mock(side_effect=AssertionError("Do not prompt an unrelated reference"))
        with mock.patch.object(gallery, "gemini_json", return_value={"results": [approval(clothing_match="mismatch")]}), \
             mock.patch.object(gallery, "_search_query") as search:
            states = list(jobs.run(**options))
        options["generate"].assert_not_called()
        search.assert_not_called()
        self.assertTrue(all(isinstance(state, dict) for state in states))
        item = studio.load_gallery_items(self.root, self.aid)[0]
        self.assertEqual(item["status"], "reference_rejected")
        self.assertIn("Saved 0 image(s)", states[-1]["message"])

    def test_review_uses_child_tab_description_not_parent_and_does_not_touch_other_tab(self):
        original = self.seed(current=True)[0]
        child = tabs.create_tab(self.root, self.aid, self.bid, "abcd1234abcd", "Woman in a red evening dress", "Evening")
        token = jobs._batch_scope.set((str(Path(self.root).resolve()), self.aid, child["id"]))
        try:
            jobs.add_references(self.root, self.aid, [{"path": original["path"], "image_url": "https://example.test/child.png"}], child["theme"])
        finally:
            jobs._batch_scope.reset(token)
        with mock.patch.object(gallery, "gemini_json", return_value={"results": [approval()]}) as api:
            tabs.run(self.root, self.aid, child["id"], 1, "864x1152 (3:4)", Context("child-review"),
                     mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock(), review_only=True)
        self.assertIn("Woman in a red evening dress", api.call_args.args[0])
        self.assertNotIn(THEME, api.call_args.args[0])
        self.assertEqual(jobs.batch_references(self.root, self.aid, self.batch)[0], original)

    def test_saved_evidence_is_reviewed_before_new_downloads(self):
        self.seed()
        jobs.add_references(self.root, self.aid, [{"image_url": "https://example.test/pending.png"}], THEME)
        with mock.patch.object(gallery, "gemini_json", side_effect=gallery.GalleryAPIError("Gemini", 503, "Busy", True)), \
             mock.patch.object(gallery, "_download_candidate") as download:
            with self.assertRaises(gallery.GalleryAPIError):
                list(jobs._discover(self.root, self.aid, self.batch, allow_search=False))
        download.assert_not_called()

    def test_rejected_old_reference_keeps_saved_image_and_does_not_run_a_repair_generation(self):
        self.seed(current=True)
        output = Path(self.root) / "saved-output.png"
        image_fixture(4).save(output)
        before = output.read_bytes()
        generate = mock.Mock(return_value=(str(output), None, "42"))
        options = dict(save_dir=self.root, avatar_id=self.aid, quantity=1, aspect="864x1152 (3:4)",
                       theme=THEME, require_face=False, make_prompt=lambda *args: "Saved prompt",
                       generate=generate, validate=mock.Mock(side_effect=RuntimeError("Pause with saved image")),
                       repair=mock.Mock(), batch_id=self.bid)
        with self.assertRaisesRegex(RuntimeError, "Pause with saved image"):
            list(jobs.run(**options))
        row = jobs.batch_references(self.root, self.aid, self.batch)[0]
        jobs.update_reference(self.root, self.aid, row["id"], validation=approval())
        options["validate"] = mock.Mock(return_value={"pass": False})
        with mock.patch.object(gallery, "gemini_json", return_value={"results": [approval(clothing_match="mismatch")]}):
            list(jobs.run(**options))
        self.assertEqual(generate.call_count, 1)
        options["repair"].assert_not_called()
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(studio.load_gallery_items(self.root, self.aid)[0]["status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
