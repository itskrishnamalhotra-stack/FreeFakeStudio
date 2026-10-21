"""Exercise the real vision adapter with fake tensors, never torch or model weights."""

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import avatar_studio
import avatar_guide
import avatar_vision as vision


class TokenRow:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Tensor:
    def __init__(self, rows, floating=False):
        self.rows = rows
        self.floating = floating
        self.moves = []

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self

    def __getitem__(self, key):
        row, column = key
        if isinstance(row, int):
            return TokenRow(self.rows[row][column])
        return Tensor([values[column] for values in self.rows[row]])


class VisionGenerationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 32))
        self.processor = mock.Mock()
        self.processor.apply_chat_template.return_value = "formatted prompt"
        self.processor.side_effect = lambda **kwargs: {"input_ids": Tensor([[11, 12, 13]]),
                                                       "pixel_values": Tensor([[0.5]], floating=True)}
        self.model = mock.Mock()
        self.model.generate.return_value = Tensor([[11, 12, 13, 77]])
        self.torch = SimpleNamespace(
            is_tensor=lambda value: isinstance(value, Tensor),
            is_floating_point=lambda value: value.floating,
            float16="float16", inference_mode=contextlib.nullcontext,
        )
        self.load = self.enter(mock.patch.object(vision, "_load_model", return_value=(self.processor, self.model, self.torch)))
        self.cleanup = self.enter(mock.patch.object(vision, "cleanup_memory"))
        self.diagnostic = self.enter(mock.patch.object(vision, "_write_vision_diagnostic", return_value="diagnostic.json"))

    def enter(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def test_decode_removes_only_input_tokens_and_preserves_notebook_settings(self):
        def decode(tokens, skip_special_tokens):
            self.assertEqual(tokens.rows, [[77]])
            return ["Brown hair"]

        self.processor.batch_decode.side_effect = decode
        self.assertEqual(vision._ask_question(self.image, "What color is the hair?"), "Brown hair")
        call = self.model.generate.call_args.kwargs
        self.assertEqual(call["min_new_tokens"], 1)
        self.assertEqual(call["max_new_tokens"], 96)
        self.assertFalse(call["do_sample"])
        self.assertEqual(call["num_beams"], 1)
        self.assertTrue(call["use_cache"])
        self.assertEqual(call["pixel_values"].moves[0][1], {"device": "cuda:0", "dtype": "float16"})
        self.assertEqual(call["input_ids"].moves[0][0], ("cuda:0",))
        self.diagnostic.assert_not_called()

    def test_return_dict_sequences_are_supported(self):
        self.model.generate.return_value = SimpleNamespace(sequences=Tensor([[11, 12, 13, 77]]))
        self.processor.batch_decode.return_value = ["Yes"]
        self.assertEqual(vision._ask_question(self.image, "Is a face visible?"), "Yes")

    def test_special_token_only_reply_retries_without_cache_and_saves_evidence(self):
        self.model.generate.side_effect = [Tensor([[11, 12, 13, 2]]), Tensor([[11, 12, 13, 77]])]
        self.processor.batch_decode.side_effect = [[""], ["<eos>"], ["Yes"]]
        self.assertEqual(vision._ask_question(self.image, "Is a face visible?", context="face/visible"), "Yes")
        self.assertEqual(self.model.generate.call_count, 2)
        self.assertFalse(self.model.generate.call_args.kwargs["use_cache"])
        report = self.diagnostic.call_args.args[0]
        self.assertTrue(report["recovered"])
        self.assertEqual(report["attempts"][0]["token_ids"], [2])
        self.assertEqual(report["attempts"][0]["raw_decoded"], "<eos>")

    def test_two_empty_replies_fail_without_fabricating_unclear_or_yes(self):
        self.model.generate.return_value = Tensor([[11, 12, 13, 2]])
        self.processor.batch_decode.side_effect = [[""], ["<eos>"], [""], ["<eos>"]]
        with self.assertRaisesRegex(RuntimeError, "two short-question attempts.*diagnostic.json"):
            vision._ask_question(self.image, "Is a face visible?")
        self.assertEqual(self.model.generate.call_count, 2)
        self.assertFalse(self.diagnostic.call_args.args[0]["recovered"])
        self.assertEqual(self.cleanup.call_count, 2)

    def test_no_new_token_output_is_diagnosed(self):
        self.model.generate.return_value = Tensor([[11, 12, 13]])
        self.processor.batch_decode.side_effect = [[""], [""], [""], [""]]
        with self.assertRaises(RuntimeError):
            vision._ask_question(self.image, "Describe the face")
        self.assertEqual(self.diagnostic.call_args.args[0]["attempts"][0]["new_tokens"], 0)

    def test_diagnostic_retry_records_numerical_failures_without_full_logits(self):
        class Stat:
            def __init__(self, value):
                self.value = value

            def all(self):
                return self

            def any(self):
                return self

            def item(self):
                return self.value

        self.torch.isfinite = lambda scores: Stat(False)
        self.torch.isnan = lambda scores: Stat(True)
        self.torch.isposinf = lambda scores: Stat(False)
        self.model.generate.side_effect = [Tensor([[11, 12, 13, 2]]),
                                          SimpleNamespace(sequences=Tensor([[11, 12, 13, 2]]), scores=[object()])]
        self.processor.batch_decode.side_effect = [[""], ["<eos>"], [""], ["<eos>"]]
        with self.assertRaises(RuntimeError):
            vision._ask_question(self.image, "Describe the image")
        record = self.diagnostic.call_args.args[0]["attempts"][1]
        self.assertTrue(record["first_step_scores_have_nan"])
        self.assertFalse(record["first_step_scores_have_positive_inf"])
        self.assertTrue(self.model.generate.call_args.kwargs["output_scores"])
        self.assertNotIn("scores", record)

    def test_short_invalid_output_does_not_get_decoded_as_a_real_answer(self):
        self.model.generate.return_value = Tensor([[77]])
        with self.assertRaisesRegex(RuntimeError, "Unexpected vision output shape"):
            vision._ask_question(self.image, "Describe the face")
        self.processor.batch_decode.assert_not_called()
        self.cleanup.assert_called_once()

    def test_warmup_checks_decoded_text(self):
        self.processor.batch_decode.return_value = ["ready"]
        report = vision.warmup(128)
        self.assertEqual(report["answer"], "ready")
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 4)
        self.assertEqual(self.processor.call_args.kwargs["images"][0].size, (128, 128))

    def test_warmup_rejects_empty_inference_instead_of_claiming_ready(self):
        self.processor.batch_decode.side_effect = [[""], ["<eos>"], [""], ["<eos>"]]
        with self.assertRaisesRegex(RuntimeError, "two short-question attempts"):
            vision.warmup(128)

    def test_generation_error_cleans_up_and_is_not_masked_as_a_response(self):
        self.model.generate.side_effect = RuntimeError("CUDA kernel failed")
        with self.assertRaisesRegex(RuntimeError, "CUDA kernel failed"):
            vision._ask_question(self.image, "Describe the face")
        self.cleanup.assert_called_once()


class VisionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 48))

    def test_reference_uses_individual_questions_and_assembles_json_in_python(self):
        def answer(image, question, **kwargs):
            self.assertNotIn("JSON", question)
            if kwargs["context"].endswith("/occlusion"):
                return "No"
            return "Yes" if "Answer yes, no, or unclear" in question else "Visible detail"

        with mock.patch.object(vision, "_ask_question", side_effect=answer) as ask, mock.patch.object(vision, "cleanup_memory"):
            specs = vision.analyze_reference_image("face", self.image, avatar_studio.FACE_SPEC_QUESTIONS)
        self.assertTrue(specs["usable_reference"])
        self.assertEqual(len(specs["answers"]), len(avatar_studio.FACE_SPEC_QUESTIONS))
        self.assertEqual(ask.call_count, len(avatar_studio.FACE_SPEC_QUESTIONS))
        self.assertEqual(specs["analysis_method"], "single-question-v1")

    def test_uncertain_gate_cannot_unlock_reference_or_waste_detail_requests(self):
        with mock.patch.object(vision, "_ask_question", return_value="Unclear") as ask, mock.patch.object(vision, "cleanup_memory"):
            specs = vision.analyze_reference_image("face", self.image, avatar_studio.FACE_SPEC_QUESTIONS)
        self.assertFalse(specs["usable_reference"])
        self.assertEqual(ask.call_count, 3)
        self.assertEqual(len(specs["rejection_reasons"]), 3)

    def test_body_occlusion_gate_has_correct_boolean_polarity(self):
        def answer(image, question, **kwargs):
            return "No" if kwargs["context"].endswith("/occlusion") else "Yes"

        with mock.patch.object(vision, "_ask_question", side_effect=answer), mock.patch.object(vision, "cleanup_memory"):
            specs = vision.analyze_reference_image("body", self.image, avatar_studio.BODY_SPEC_QUESTIONS)
        self.assertTrue(specs["usable_reference"])
        consistency = next(row for row in specs["answers"] if row["key"] == "skin_tone_consistency")
        self.assertIn("unclear", consistency["answer"])

    def test_runtime_failure_is_not_replaced_by_mock_specs(self):
        with mock.patch.object(vision, "_ask_question", side_effect=RuntimeError("no readable text")), mock.patch.object(vision, "cleanup_memory"):
            with self.assertRaisesRegex(RuntimeError, "no readable text"):
                vision.analyze_reference_image("face", self.image, avatar_studio.FACE_SPEC_QUESTIONS)

    def test_legacy_nonstrict_flag_cannot_confirm_mock_identity_in_production(self):
        with mock.patch.dict(os.environ, {"FFS_AVATAR_ANALYZER_STRICT": "0"}), \
                mock.patch.object(vision, "_ask_question", side_effect=RuntimeError("empty reply")), \
                mock.patch.object(vision, "cleanup_memory"):
            with self.assertRaisesRegex(RuntimeError, "empty reply"):
                avatar_studio.analyze_reference_image("face", self.image, dev_mode=False)

    def test_guided_flow_uses_real_adapter_without_model_weights_or_json_replies(self):
        def answer(image, question, **kwargs):
            if kwargs["context"].endswith("/occlusion"):
                return "No"
            return "Yes" if "Answer yes, no, or unclear" in question else "Visible detail"

        with tempfile.TemporaryDirectory() as root, mock.patch.object(vision, "_ask_question", side_effect=answer), \
                mock.patch.object(vision, "cleanup_memory"), mock.patch.object(vision, "_load_model") as load:
            avatar = avatar_guide.create(root, "Vision integration")
            generate = lambda *args: self.image.copy()
            validate = lambda image, saved, prompt: avatar_studio.validate_generated_image(image, saved, prompt, dev_mode=False)
            for _ in range(3):
                list(avatar_guide.advance(root, avatar["id"], None, generate, validate, dev_mode=False))
            restored = avatar_studio.load_avatar(root, avatar["id"])
            self.assertEqual(avatar_guide.stage(restored), "complete")
            self.assertTrue(restored["face_locked"] and restored["body_locked"])
            items = avatar_studio.load_gallery_items(root, avatar["id"])
            self.assertEqual(len(items), 2)
            self.assertTrue(all(item["status"] == "passed" for item in items))
            self.assertTrue(all(item["validation"]["analysis_method"] == "single-question-v1" for item in items))
            load.assert_not_called()

    def test_guided_analysis_failure_preserves_draft_and_does_not_lock_face(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(vision, "_ask_question", side_effect=RuntimeError("empty reply")), \
                mock.patch.object(vision, "cleanup_memory"):
            avatar = avatar_guide.create(root, "Interrupted analysis")
            list(avatar_guide.advance(root, avatar["id"], None, lambda *args: self.image.copy(), None, dev_mode=False))
            with self.assertRaisesRegex(RuntimeError, "empty reply"):
                list(avatar_guide.advance(root, avatar["id"], None, None, None, dev_mode=False))
            restored = avatar_studio.load_avatar(root, avatar["id"])
            self.assertFalse(restored["face_locked"])
            self.assertEqual(avatar_guide.stage(restored), "face_review")
            self.assertTrue(Path(avatar_guide.draft(restored, "face")).is_file())

    def quality(self, replies):
        def answer(image, question, **kwargs):
            key = kwargs["context"].split("/")[1]
            return replies.get(key, "Yes")

        specs = {"answers": [{"key": "hair_color", "answer": "brown"}]}
        with mock.patch.object(vision, "_ask_question", side_effect=answer) as ask, mock.patch.object(vision, "cleanup_memory"):
            result = vision.validate_avatar_generation(self.image, specs, specs, "A person in a park")
        return result, ask

    def test_hidden_face_skips_face_check_but_checks_visible_body(self):
        result, ask = self.quality({"face_test_applicable": "No"})
        self.assertTrue(result["pass"])
        self.assertFalse(result["face_test_applicable"])
        self.assertTrue(result["body_identity_match"])
        self.assertNotIn("quality/face_identity_match", [c.kwargs["context"] for c in ask.call_args_list])

    def test_unclear_visibility_does_not_silently_bypass_identity_check(self):
        result, _ = self.quality({"face_test_applicable": "Unclear"})
        self.assertFalse(result["pass"])
        self.assertTrue(result["face_test_applicable"])
        self.assertTrue(any("face_test_applicable" in reason for reason in result["reasons"]))

    def test_anatomy_or_scene_failure_cannot_pass(self):
        for key in ("anatomy_ok", "request_match", "adult_person_visible", "face_identity_match", "body_identity_match"):
            with self.subTest(key=key):
                result, _ = self.quality({key: "No"})
                self.assertFalse(result["pass"])
                self.assertIn(key, result["repair_instruction"])

    def test_missing_specs_cannot_get_an_automatic_pass(self):
        with mock.patch.object(vision, "_ask_question") as ask:
            with self.assertRaisesRegex(ValueError, "specifications are missing"):
                vision.validate_avatar_generation(self.image, None, {}, "Portrait")
        ask.assert_not_called()

    def test_boolean_parsing_requires_an_explicit_noncontradictory_answer(self):
        for answer, expected in (("Yes, visible", True), ("No.", False), ("Answer: yes", True),
                                 ("unclear", None), ("Yes but I cannot tell", None),
                                 ("yes and no", None), ("The person has hair", None), ("", None)):
            with self.subTest(answer=answer):
                self.assertIs(vision._yes_no(answer), expected)

    def test_diagnostics_are_saved_without_image_bytes_or_credentials(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("workspace.get_save_dir", return_value=folder):
            path = vision._write_vision_diagnostic({"context": "test", "attempts": [{"token_ids": [2]}]})
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(payload["attempts"][0]["token_ids"], [2])
            self.assertEqual(payload["model"], vision.MODEL_ID)
            self.assertNotIn("image", payload)


if __name__ == "__main__":
    unittest.main()
