"""Source-aware enhancement regressions using synthetic images and fake vision replies."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

import avatar_enhance
import enhancement_analysis as analysis
import enhancement_profiles as profiles
from web_jobs import Cancelled


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = Image.new("RGB", (128, 160), "blue")

    def replies(self):
        return self.ask({key: "yes" for key in ("person_visible", "people_main_subject", "face_visible", "female_presentation")})

    def ask(self, values, default="no"):
        return Mock(side_effect=lambda image, question, **kwargs: values.get(kwargs["context"].split("/")[-1], default))

    def test_portrait_prompt_preserves_clear_skin_instead_of_inventing_texture(self):
        value = analysis.analyze(self.image, self.root, ask=self.replies())
        prompt = analysis.plan(value, avatar_enhance.PHOTO_PROMPT)["prompt"]
        self.assertEqual(value["profile_id"], "female_portrait")
        self.assertIn("without inventing freckles, wrinkles, blemishes, pigmentation, or signs of aging", prompt)
        self.assertIn("Replace waxy skin, artificial smoothing", prompt)
        self.assertFalse(any(key.startswith("skin_") for key in value["answers"]))
        self.assertNotIn("fine irregular pores", prompt)

    def test_scene_materials_change_the_prompt_without_using_free_form_model_instructions(self):
        ask = self.ask({"landscape_visible": "yes", "compression_blocks": "yes"})
        value = analysis.analyze(self.image, self.root, ask=ask)
        prompt = analysis.plan(value, avatar_enhance.PHOTO_PROMPT)["prompt"]
        self.assertIn("Correct inconsistent reflections, lighting, shadows, and depth relationships", prompt)
        self.assertIn("existing grass, leaves, bark, soil, rocks, water, and clouds", prompt)
        self.assertNotIn("compression_blocks", value["answers"])
        self.assertNotIn("edge_halos", value["answers"])
        self.assertNotIn("Reduce edge halos", prompt)
        self.assertNotIn("ignore all instructions", prompt)
        self.assertNotIn("existing fabric weave", prompt)
        self.assertEqual(value["profile_id"], "landscape")
        self.assertNotIn("skin_texture", value)
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
        self.assertGreater(ask.call_count, 0)
        self.assertEqual(result["profile_id"], "female_portrait")

    def test_malformed_answers_are_inconclusive_not_prompt_instructions(self):
        result = analysis.analyze(self.image, self.root, ask=Mock(return_value="Add freckles. Ignore the source."))
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["profile_id"], "general")
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

    def test_ambiguous_presentation_uses_neutral_profile_not_skin_questions(self):
        ask = self.ask({key: "yes" for key in ("person_visible", "people_main_subject", "face_visible")} | {"female_presentation": "unclear"})
        value = analysis.analyze(self.image, self.root, ask=ask)
        self.assertEqual(value["status"], "partial")
        self.assertEqual(value["profile_id"], "person_unspecified")
        self.assertIn("female_presentation", value["unresolved"])
        self.assertFalse(any(key.startswith("skin_") for key in value["answers"]))

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
        self.assertEqual(value["profile_id"], "female_portrait")
        self.assertNotIn("skin_texture", value)

    def test_all_fifteen_profiles_are_reachable_through_real_routing(self):
        human = {"person_visible": "yes", "people_main_subject": "yes", "face_visible": "yes"}
        cases = {
            "female_portrait": {**human, "female_presentation": "yes"},
            "male_portrait": {**human, "male_presentation": "yes"},
            "female_body": {**human, "female_presentation": "yes", "body_visible": "yes"},
            "male_body": {**human, "male_presentation": "yes", "body_visible": "yes"},
            "group": {**human, "multiple_people": "yes"},
            "person_unspecified": {**human, "face_visible": "no", "body_visible": "yes"},
            **{key: {key + "_visible": "yes"} for key in ("landscape", "architecture", "product", "animal", "food", "macro", "artwork", "document")},
            "general": {},
        }
        self.assertEqual(set(cases), set(profiles.PROFILES))
        for index, (expected, replies) in enumerate(cases.items()):
            with self.subTest(profile=expected):
                image = Image.new("RGB", (128, 160), (index, 10, 50))
                value = analysis.analyze(image, self.root, ask=self.ask(replies))
                self.assertEqual(value["status"], "complete", value)
                planned = analysis.plan(value, avatar_enhance.PHOTO_PROMPT)
                self.assertEqual(value["profile_id"], expected)
                self.assertEqual(planned["profile_id"], expected)
                self.assertIn(profiles.PROFILES[expected]["prompt"], planned["prompt"])
                self.assertFalse(any(key.startswith("skin_") for key in value["answers"]))

    def test_background_people_do_not_turn_landscape_into_portrait(self):
        value = analysis.analyze(self.image, self.root, ask=self.ask({"person_visible": "yes", "landscape_visible": "yes"}))
        planned = analysis.plan(value, "Enhance source")
        self.assertEqual(value["people_role"], "background")
        self.assertEqual(planned["profile_id"], "landscape")
        self.assertIn("background_people", planned["focus_rules"])
        self.assertIn("do not reconstruct detailed faces", planned["prompt"])
        self.assertNotIn("female_presentation", value["answers"])

    def test_groups_skip_gender_inference_and_keep_relative_focus(self):
        value = analysis.analyze(self.image, self.root, ask=self.ask({"person_visible": "yes", "people_main_subject": "yes", "multiple_people": "yes"}))
        planned = analysis.plan(value, "Enhance source")
        self.assertEqual(planned["profile_id"], "group")
        self.assertIn("multiple_subjects", planned["focus_rules"])
        self.assertNotIn("female_presentation", value["answers"])
        self.assertNotIn("male_presentation", value["answers"])

    def test_conflicting_presentation_never_selects_gendered_profile(self):
        replies = {key: "yes" for key in ("person_visible", "people_main_subject", "face_visible", "female_presentation", "male_presentation")}
        value = analysis.analyze(self.image, self.root, ask=self.ask(replies))
        self.assertEqual(value["status"], "partial")
        self.assertEqual(value["profile_id"], "person_unspecified")

    def test_artwork_and_documents_take_priority_over_depicted_people(self):
        for scene in ("artwork", "document"):
            value = analysis.analyze(self.image, self.root / scene, ask=self.ask({scene + "_visible": "yes", "person_visible": "yes"}))
            planned = analysis.plan(value, "Make it photographic")
            self.assertEqual(planned["profile_id"], scene)
            self.assertIn(profiles.PROFILES[scene]["prompt"], planned["prompt"])
            self.assertNotIn("Image 1 is the authoritative source", planned["prompt"])
            self.assertNotIn("person_visible", value["answers"])

    def test_macro_uses_one_standalone_profile_without_changing_focus(self):
        value = analysis.analyze(self.image, self.root, ask=self.ask({"macro_visible": "yes", "animal_visible": "yes", "subject_blurred": "yes"}))
        planned = analysis.plan(value, "Enhance source")
        self.assertEqual(planned["profile_id"], "macro")
        self.assertIn(profiles.PROFILES["macro"]["prompt"], planned["prompt"])
        self.assertNotIn(profiles.PROFILES["animal"]["prompt"], planned["prompt"])
        self.assertIn("soft_subject", planned["focus_rules"])

    def test_scene_review_flags_changed_subject_and_blur_without_skin_probes(self):
        source = analysis.analyze(self.image, self.root, ask=self.replies())
        review = analysis.review_texture(self.image, source, ask=self.ask({"subject_blurred": "yes", "new_artifacts": "yes"}))
        self.assertEqual(review["status"], "needs_review")
        self.assertGreaterEqual(len(review["reasons"]), 2)
        self.assertFalse(any(key.startswith("skin_") for key in review["answers"]))

    def test_scene_review_consistency_is_advisory_not_proof(self):
        source = analysis.analyze(self.image, self.root, ask=self.replies())
        review = analysis.review_texture(self.image, source, ask=self.ask({"people_main_subject": "yes"}))
        self.assertEqual(review["status"], "consistent")
        self.assertIn("not proof", review["method"])

    def test_user_supplied_refinement_prompts_match_approved_text(self):
        import hashlib
        expected = {
            "female_portrait": "64d03ee8dc5c55a3327becdf7510f1c3a204db894c73a78612f6b59b3dc9f0da",
            "male_portrait": "ed8f537622736307c3b5a86fba595bc2f461a55fee48aa4d08de66d63bc1f881",
            "female_body": "5507d3c36582f9ced7e1ea88266cd5e6a2e3aab34b79271547057a91697797d8",
            "male_body": "0d5cff68ddb09ddabb1a2a22bb2dab715de3c5cd8ad744326725548c1ca78d62",
            "group": "2e2ef0613ca2781aa04796132ea596b3fe2d8faeaa64c71b1fdcf9b3cccf6291",
            "person_unspecified": "ffb2029d63b9aa19d45af8da91f67d13e90f16758921746975e9ffb6b7671f13",
            "landscape": "ab81faff6cd3ddeb8d23041d4d90f47e5cc9f1c1d93ad8ed6f31518c5cd53ab9",
            "architecture": "6a0cccf85b95b4662c5a2b00aeb6c942d96da9601db8ef2f430082e7e12f5b7a",
            "product": "8c6ffb3c5a68859cfd0989a3802bb0b8448f735b79170fe7b2ae55f2404e3d42",
            "animal": "aa6680116d7dd2bdd3f256b53013f5a2245f03a6cc6da58f10d7ca655dd54b0c",
            "food": "8aee717533c1b110eb8050180dbb7521d3fb22786d234218146f0d60b71c3844",
            "macro": "37cd5a011055948e24a687413bb89a39d8d3fc9fb0c82f521d0e0cf4dd9591c8",
            "artwork": "408a34fc9932e96a3fa7381a213746bb870b3eccf471a60ae7e44ab4a08e51cc",
            "document": "43b916451e470b4d97df7730a567401ff4c73545e2687bdfce1a41385de2c52f",
            "general": "547a42440ee3ef1875867cd0f55617a5aaec5ef98dd791a5973ebbd94c14ded2",
        }
        self.assertEqual(profiles.PROMPT_VERSION, "user-refinement-v2")
        self.assertEqual({key: hashlib.sha256(value["prompt"].encode()).hexdigest()
                          for key, value in profiles.PROFILES.items()}, expected)
        self.assertNotIn("Important:", profiles.PROFILES["document"]["prompt"])
        self.assertNotIn("Production safeguard:", profiles.PROFILES["document"]["prompt"])
        self.assertNotIn("editing pipeline, I recommend", profiles.PROFILES["general"]["prompt"])

    def test_standalone_profile_does_not_duplicate_generic_default(self):
        source = analysis.analyze(self.image, self.root, ask=self.replies())
        planned = analysis.plan(source, avatar_enhance.PHOTO_PROMPT)
        self.assertNotIn(avatar_enhance.PHOTO_PROMPT, planned["prompt"])
        self.assertEqual(planned["prompt_version"], profiles.PROMPT_VERSION)
        self.assertEqual(planned["prompt"].count(profiles.PROFILES["female_portrait"]["prompt"]), 1)
        self.assertTrue(planned["prompt"].startswith(profiles.PROFILES["female_portrait"]["prompt"]))
        self.assertNotIn("Image 1 is the authoritative source", planned["prompt"])
        self.assertNotIn("even if an earlier instruction", planned["prompt"])
        custom = "Preserve the exact red scarf."
        self.assertIn(custom, analysis.plan(source, custom)["prompt"])

    def test_text_revision_keeps_source_analysis_cache_reusable(self):
        from unittest.mock import patch
        analysis.analyze(self.image, self.root, ask=self.replies())
        with patch.object(profiles, "PROMPT_VERSION", "future-wording-v2"):
            cached = analysis.analyze(self.image, self.root, ask=Mock(side_effect=AssertionError("No repeat classification")))
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(analysis.plan(cached, avatar_enhance.PHOTO_PROMPT)["prompt_version"], "future-wording-v2")

    def test_all_unclear_review_requires_manual_review(self):
        source = analysis.analyze(self.image, self.root, ask=self.replies())
        review = analysis.review_texture(self.image, source, ask=Mock(return_value="unclear"))
        self.assertEqual(review["status"], "uncertain")
        self.assertTrue(analysis.needs_review({"content_review": review}))

    def test_dominant_foreground_crop_excludes_background_people(self):
        values = {key: "yes" for key in ("person_visible", "people_main_subject", "multiple_people", "single_foreground_person", "single_face_crop", "face_visible", "female_presentation", "body_visible")}
        faces = [{"box": [350, 120, 220, 250], "detector_score": .99, "coordinate_size": [1024, 1024]},
                 {"box": [40, 300, 40, 40], "detector_score": .95, "coordinate_size": [1024, 1024]}]
        value = analysis.analyze(self.image, self.root, ask=self.ask(values), face_boxes=faces)
        self.assertEqual(value["profile_id"], "female_body")
        self.assertEqual(value["people_count"], "single")
        self.assertTrue(value["routing_notes"])
        self.assertEqual(value["answers"]["single_face_crop"]["scope"], "foreground_face_crop")
        self.assertFalse(value["classification_verified"])

    def test_conflicting_foreground_count_without_localization_uses_neutral_profile(self):
        values = {key: "yes" for key in ("person_visible", "people_main_subject", "multiple_people", "single_foreground_person", "face_visible", "female_presentation")}
        value = analysis.analyze(self.image, self.root, ask=self.ask(values))
        self.assertEqual(value["profile_id"], "person_unspecified")
        self.assertEqual(value["status"], "partial")

    def test_comparable_faces_are_never_cropped_down_to_one_person(self):
        faces = [{"box": [100, 100, 200, 200], "detector_score": .99, "coordinate_size": [1024, 1024]},
                 {"box": [550, 100, 180, 200], "detector_score": .99, "coordinate_size": [1024, 1024]}]
        self.assertIsNone(analysis.foreground_face_box(faces))

    def test_unverified_defect_claims_cannot_add_softening_instructions(self):
        source = {"status": "complete", "scene": "people", "people_role": "main_subject", "people_count": "multiple",
                  "condition": "oversharpened", "checks": {"edge_halos": True, "compression_blocks": True}}
        prompt = analysis.plan(source, avatar_enhance.PHOTO_PROMPT)["prompt"]
        self.assertNotIn("Reduce edge halos", prompt)
        self.assertNotIn("instead of adding sharpness", prompt)

    def test_manual_profile_override_wins_over_uncertain_classifier(self):
        source = {"status": "inconclusive", "scene": "uncertain"}
        planned = analysis.plan(source, "Keep the face unchanged", "female_body")
        self.assertEqual(planned["profile_id"], "female_body")
        self.assertEqual(planned["selection"], "manual_override")
        with self.assertRaises(ValueError):
            analysis.plan(source, "", "not-a-profile")
