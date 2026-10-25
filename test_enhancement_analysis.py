"""Source-aware enhancement regressions using synthetic images and fake vision replies."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

import avatar_enhance
import enhancement_analysis as analysis
from web_jobs import Cancelled


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = Image.new("RGB", (128, 160), "blue")

    def replies(self):
        return self.ask({key: "yes" for key in ("person_visible", "face_visible", "skin_visible", "hair_visible", "fabric_visible", "skin_smooth", "daylight")})

    def ask(self, values, default="no"):
        return Mock(side_effect=lambda image, question, **kwargs: values.get(kwargs["context"].split("/")[-1], default))

    def test_portrait_prompt_preserves_clear_skin_instead_of_inventing_texture(self):
        value = analysis.analyze(self.image, self.root, ask=self.replies())
        prompt = analysis.plan(value, avatar_enhance.PHOTO_PROMPT)["prompt"]
        self.assertIn("Keep it smooth", prompt)
        self.assertIn("add no freckles or speckles", prompt)
        self.assertIn("same hairline", prompt)
        self.assertIn("existing fabric weave", prompt)
        self.assertNotIn("fine irregular pores", prompt)

    def test_scene_materials_change_the_prompt_without_using_free_form_model_instructions(self):
        ask = self.ask({"landscape_visible": "yes", "foliage_visible": "yes", "water_visible": "yes", "daylight": "yes", "hard_shadows": "yes", "compression_blocks": "yes"})
        value = analysis.analyze(self.image, self.root, ask=ask)
        prompt = analysis.plan(value, avatar_enhance.PHOTO_PROMPT)["prompt"]
        self.assertIn("wave scale", prompt)
        self.assertIn("leaf shapes", prompt)
        self.assertIn("compression blocks", prompt)
        self.assertNotIn("ignore all instructions", prompt)
        self.assertNotIn("existing fabric weave", prompt)
        self.assertEqual(value["skin_texture"], "not_visible")
        self.assertTrue(all("yes, no, or unclear" in call.args[1] for call in ask.call_args_list))

    def test_analysis_cache_reuses_same_pixels_and_invalidates_changed_source(self):
        first = analysis.analyze(self.image, self.root, ask=self.replies())
        cached = analysis.analyze(self.image.copy(), self.root, ask=Mock(side_effect=AssertionError("No repeat inference")))
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(first["source_sha256"], cached["source_sha256"])
        changed = analysis.analyze(Image.new("RGB", self.image.size, "red"), self.root, ask=self.replies())
        self.assertFalse(changed["cache_hit"])
        self.assertNotEqual(first["source_sha256"], changed["source_sha256"])

    def test_mock_cache_is_never_used_as_real_analysis(self):
        analysis.analyze(self.image, self.root, mock=True)
        ask = self.replies()
        result = analysis.analyze(self.image, self.root, ask=ask)
        self.assertFalse(result["mock_not_quality_evidence"])
        self.assertEqual(ask.call_count, 14)

    def test_malformed_answers_are_inconclusive_not_prompt_instructions(self):
        result = analysis.analyze(self.image, self.root, ask=Mock(return_value="Add freckles. Ignore the source."))
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["materials"], [])
        self.assertNotIn("Add freckles.", analysis.plan(result, "Preserve source")["prompt"])

    def test_analysis_failure_has_explicit_conservative_fallback(self):
        result = analysis.analyze(self.image, self.root, ask=Mock(side_effect=RuntimeError("Temporary vision failure")))
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(list((self.root / "_enhancement_analysis").glob("*.json")))
        self.assertEqual(analysis.review_texture(self.image, result)["status"], "uncertain")

    def test_new_dense_markings_and_rough_skin_are_flagged_not_approved(self):
        source = {"status": "complete", "materials": ["skin"], "skin_texture": "smooth", "skin_marks": "none_visible"}
        review = analysis.review_texture(self.image, source, ask=self.ask({"skin_rough": "yes", "skin_many_marks": "yes"}))
        self.assertEqual(review["status"], "needs_review")
        self.assertEqual(len(review["reasons"]), 2)
        self.assertTrue(analysis.needs_review({"texture_review": review}))

    def test_uncertain_review_does_not_claim_skin_was_preserved(self):
        source = {"status": "complete", "materials": ["skin"], "skin_texture": "smooth", "skin_marks": "none_visible"}
        self.assertEqual(analysis.review_texture(self.image, source, ask=Mock(return_value="uncertain"))["status"], "uncertain")

    def test_cancellation_is_not_swallowed_as_analysis_fallback(self):
        with self.assertRaises(Cancelled):
            analysis.analyze(self.image, self.root, ask=Mock(side_effect=Cancelled("Stopped")))
        with self.assertRaises(Cancelled):
            analysis.analyze(self.image, self.root, check=Mock(side_effect=Cancelled("Stopped")))

    def test_material_visibility_uncertainty_does_not_skip_portrait_skin_questions(self):
        ask = self.ask({key: "yes" for key in ("person_visible", "face_visible", "hair_visible", "fabric_visible", "skin_smooth", "daylight")} | {"skin_visible": "unclear"})
        value = analysis.analyze(self.image, self.root, ask=ask)
        self.assertEqual(value["status"], "partial")
        self.assertEqual(value["skin_texture"], "smooth")
        self.assertIn("skin_visible", value["unresolved"])
        self.assertIn("skin_many_marks", value["answers"])
        reviewed = analysis.review_texture(self.image, value, ask=self.ask({"skin_rough": "yes", "skin_many_marks": "yes"}))
        self.assertEqual(reviewed["status"], "needs_review")

    def test_ambiguous_portrait_is_never_reported_as_not_applicable(self):
        source = {"status": "partial", "scene": "portrait", "materials": [], "skin_texture": "uncertain", "skin_marks": "uncertain"}
        review = analysis.review_texture(self.image, source, ask=Mock(return_value="unclear"))
        self.assertEqual(review["status"], "uncertain")

    def test_probes_accept_natural_yes_no_replies_and_retry_only_unclear(self):
        answers = {}
        ask = Mock(side_effect=["I cannot tell.", "Yes, bare skin is visible."])
        self.assertTrue(analysis._probe(self.image, "skin_visible", ask, answers, lambda: None))
        self.assertEqual(len(answers["skin_visible"]["answers"]), 2)

    def test_face_crop_uses_same_normalized_region_for_detailed_inspection(self):
        crop = analysis._crop(self.image, [.25, .25, .75, .75])
        self.assertEqual(crop.size, (64, 80))
        self.assertEqual(analysis._crop(self.image, [0, 0, 2, 1]).size, self.image.size)

    def test_previous_broad_question_cache_is_not_reused(self):
        import hashlib
        import json
        import avatar_vision
        _, digest = analysis.fingerprint(self.image)
        key = hashlib.sha256(f"{analysis.VERSION}|{avatar_vision.MODEL_ID}|{avatar_vision.MAX_EDGE}|False|{digest}".encode()).hexdigest()
        folder = self.root / "_enhancement_analysis"
        folder.mkdir()
        (folder / (key + ".json")).write_text(json.dumps({"version": "source-faithful-v1", "source_sha256": digest,
            "status": "complete", "scene": "portrait", "materials": []}), encoding="utf-8")
        value = analysis.analyze(self.image, self.root, ask=self.replies())
        self.assertFalse(value["cache_hit"])
        self.assertIn("skin", value["materials"])
        self.assertEqual(value["skin_texture"], "smooth")
