import ctypes
import gc
import json
import os
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from PIL import Image


MODEL_ID = os.environ.get("FFS_AVATAR_VISION_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
MAX_EDGE = int(os.environ.get("FFS_AVATAR_VISION_MAX_EDGE", "768"))
MAX_NEW_TOKENS = int(os.environ.get("FFS_AVATAR_VISION_MAX_TOKENS", "900"))


def cleanup_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def prepare_image(image, max_edge=MAX_EDGE):
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "Avatar image analysis dependencies are missing. Run the Colab launcher again "
            "with REPAIR_INSTALL enabled so transformers, accelerate, bitsandbytes, and "
            "sentencepiece are installed."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Avatar image analysis needs a CUDA GPU. Use the Colab T4 runtime.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager",
    )
    model.eval()
    cleanup_memory()
    return processor, model, torch


def preload():
    """Load the persistent Avatar Studio analyzer without running a request."""
    _load_model()
    return {"model": MODEL_ID, "quantization": "nf4"}


def unload():
    """Release the cached analyzer before loading a memory-heavy image model."""
    _load_model.cache_clear()
    cleanup_memory()


def warmup(size=128):
    """Run a tiny discarded vision request to initialize lazy CUDA paths."""
    size = min(256, max(64, int(size)))
    image = Image.new("RGB", (size, size), (127, 127, 127))
    try:
        answer = _ask_question(image, "Reply with the single word ready.", max_tokens=4, context="warmup")
        return {"model": MODEL_ID, "size": size, "tokens": 4, "answer": answer}
    finally:
        del image
        cleanup_memory()


def _write_vision_diagnostic(report):
    try:
        import workspace

        folder = Path(workspace.get_save_dir()) / "_debug"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"vision_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        versions = {}
        for package in ("transformers", "torch", "bitsandbytes", "accelerate"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "not installed"
        payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL_ID,
                   "versions": versions, **report}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[vision] Diagnostic: {path}", flush=True)
        return str(path)
    except Exception as exc:
        print(f"[vision] Could not save diagnostic: {exc}", flush=True)
        return None


def _ask_question(image, question, max_tokens=96, context="analysis"):
    """Notebook-style visual Q&A, with bounded empty-response recovery."""
    processor, model, torch = _load_model()
    limit = max(1, min(int(max_tokens), max(1, MAX_NEW_TOKENS)))
    attempts = []
    for attempt in range(2):
        instruction = (
            "Carefully inspect the provided image. Answer only from visible evidence. "
            "Do not invent details. If the detail cannot be determined, say unclear. "
            f"Answer directly and briefly.\n\nQuestion: {question}"
            if attempt == 0 else f"Look at this image. {question}\nGive a short answer, or say unclear."
        )
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": instruction},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        output = generated = sequences = scores = None
        record = {"attempt": attempt + 1, "use_cache": attempt == 0, "max_new_tokens": limit}
        try:
            for key, value in list(inputs.items()):
                if not torch.is_tensor(value):
                    continue
                inputs[key] = (value.to(device="cuda:0", dtype=torch.float16)
                               if torch.is_floating_point(value) else value.to("cuda:0"))
            input_length = inputs["input_ids"].shape[1]
            record["input_tokens"] = input_length
            diagnostic_options = {"return_dict_in_generate": True, "output_scores": True} if attempt else {}
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=limit, min_new_tokens=1,
                    do_sample=False, num_beams=1, use_cache=attempt == 0,
                    **diagnostic_options,
                )
            sequences = getattr(output, "sequences", output)
            if len(sequences.shape) != 2 or sequences.shape[1] < input_length:
                raise RuntimeError("Unexpected vision output shape; cannot safely remove the input prompt.")
            generated = sequences[:, input_length:]
            record["output_tokens"] = sequences.shape[1]
            record["new_tokens"] = generated.shape[1]
            scores = getattr(output, "scores", None)
            if scores:
                record["first_step_scores_finite"] = bool(torch.isfinite(scores[0]).all().item())
                # EOS suppression legitimately uses -inf; NaNs/+inf indicate numerical trouble.
                record["first_step_scores_have_nan"] = bool(torch.isnan(scores[0]).any().item())
                record["first_step_scores_have_positive_inf"] = bool(torch.isposinf(scores[0]).any().item())
            answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            if answer:
                if attempts:
                    _write_vision_diagnostic({"context": context, "question": question[:1000],
                                              "image_size": list(image.size), "recovered": True,
                                              "attempts": attempts + [record]})
                return answer
            record["raw_decoded"] = processor.batch_decode(generated, skip_special_tokens=False)[0][:2000]
            record["token_ids"] = generated[0, :64].detach().cpu().tolist()
            print(f"[vision] Empty reply for {context}; attempt {attempt + 1}/2, "
                  f"input={input_length}, new={record['new_tokens']} tokens", flush=True)
            attempts.append(record)
        finally:
            del inputs, output, generated, sequences, scores
            cleanup_memory()
    tokenizer = getattr(processor, "tokenizer", None)
    path = _write_vision_diagnostic({
        "context": context, "question": question[:1000], "image_size": list(image.size),
        "recovered": False, "attempts": attempts,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    })
    raise RuntimeError("Vision analysis produced no readable text after two short-question attempts. "
                       f"No result was accepted from this analysis. Diagnostic: {path or 'could not be saved'}")


def _yes_no(answer):
    text = str(answer or "").strip().lower()
    text = re.sub(r"^(?:answer\s*:\s*)", "", text).strip(' \"\'`*')
    if re.search(r"\b(unclear|unknown|uncertain|cannot|can't|not sure)\b", text):
        return None
    if re.search(r"\byes\b", text) and re.search(r"\bno\b", text):
        return None
    if re.match(r"yes\b", text):
        return True
    if re.match(r"no\b", text):
        return False
    return None


def _normalise_answers(kind, payload, questions, image_size):
    returned = {}
    for item in payload.get("answers", []):
        if isinstance(item, dict) and item.get("key"):
            returned[item["key"]] = item

    answers = []
    for key, question, strictness in questions:
        item = returned.get(key, {})
        answer = str(item.get("answer") or item.get("value") or "unclear").strip()
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        answers.append(
            {
                "key": key,
                "question": question,
                "answer": answer[:280],
                "confidence": max(0.0, min(1.0, confidence)),
                "strictness": strictness,
            }
        )

    return {
        "kind": kind,
        "analyzer": MODEL_ID,
        "usable_reference": payload.get("usable_reference") is True,
        "rejection_reasons": [str(reason)[:280] for reason in payload.get("rejection_reasons", [])]
        if isinstance(payload.get("rejection_reasons"), list) else [],
        "image_size": list(image_size),
        "answers": answers,
        "raw_summary": str(payload.get("summary", ""))[:1200],
    }


def analyze_reference_image(kind, image, questions):
    if kind not in {"face", "body"}:
        raise ValueError("Reference kind must be face or body.")
    inference_image = prepare_image(image.convert("RGB"))
    gates = {
        f"usable_{kind}_reference": (
            "Does this image show exactly one adult person with a clear face?" if kind == "face"
            else "Does this image show exactly one adult person from head to feet?", True),
        "face_visibility" if kind == "face" else "body_visibility": (
            "Is the whole face clearly visible?" if kind == "face"
            else "Is the full body visible, including the head and feet?", True),
        "adult_presenting": ("Does the person appear to be an adult?", True),
    }
    if kind == "body":
        gates["occlusion"] = ("Is the person's body heavily hidden by objects or cropping?", False)
    answers = {}
    reasons = []
    try:
        for key, (question, expected) in gates.items():
            answer = _ask_question(inference_image, question + " Answer yes, no, or unclear.",
                                   max_tokens=32, context=f"{kind}/{key}")
            answers[key] = {"key": key, "answer": answer, "confidence": 0.0 if _yes_no(answer) is None else 0.5}
            if _yes_no(answer) is not expected:
                reasons.append(f"{question} {answer}")
        if not reasons:
            for index, (key, question, _) in enumerate(questions, 1):
                if key in answers:
                    continue
                print(f"[vision] {kind} detail {index}/{len(questions)}: {key}", flush=True)
                if key == "skin_tone_consistency":
                    # Only the body image is supplied; comparison to another photo is not possible.
                    answer = "unclear; the separate face reference is not supplied to this question"
                else:
                    answer = _ask_question(inference_image, question, context=f"{kind}/{key}")
                answers[key] = {"key": key, "answer": answer, "confidence": 0.0 if "unclear" in answer.lower() else 0.5}
        payload = {"usable_reference": not reasons, "rejection_reasons": reasons,
                   "answers": list(answers.values())}
        specs = _normalise_answers(kind, payload, questions, image.size)
        specs["analysis_method"] = "single-question-v1"
        specs["confidence_source"] = "availability heuristic, not calibrated model probability"
        return specs
    finally:
        del inference_image
        cleanup_memory()


def validate_avatar_generation(image, face_specs, body_specs, requested_prompt):
    from avatar_studio import summarize_specs

    if not face_specs or not body_specs:
        raise ValueError("Saved face or body specifications are missing; confirm the identity references again.")
    inference_image = prepare_image(image.convert("RGB"))
    checks = {}
    raw = {}

    def ask(key, question):
        answer = _ask_question(inference_image, question + " Answer yes, no, or unclear.",
                               max_tokens=48, context=f"quality/{key}")
        raw[key] = answer
        checks[key] = _yes_no(answer)
        return checks[key]

    try:
        ask("adult_person_visible", "Does the image show an adult person?")
        face_visible = ask("face_test_applicable", "Is a face clearly visible and large enough to inspect its features?")
        body_visible = ask("body_test_applicable", "Is enough of the body visible to inspect its build and proportions?")
        if face_visible is not False:
            ask("face_identity_match", "Does the visible face match these appearance notes? " + summarize_specs(face_specs))
        if body_visible is not False:
            ask("body_identity_match", "Does the visible body match these build and proportion notes? " + summarize_specs(body_specs))
        scene = str(requested_prompt or "").rsplit("User request:", 1)[-1].strip()[:4000]
        ask("request_match", "Does the image match this requested scene and action? " + scene)
        ask("anatomy_ok", "Does the visible anatomy look natural, without duplicated limbs or severe distortions?")
        required = ["adult_person_visible", "request_match", "anatomy_ok"]
        for part, visible in (("face", face_visible), ("body", body_visible)):
            if visible is not False:
                required.extend([f"{part}_test_applicable", f"{part}_identity_match"])
        reasons = [f"{key}: {raw[key]}" for key in required if checks.get(key) is not True]
        return {
            "pass": not reasons, "score": round(100 * sum(checks.get(key) is True for key in required) / len(required)),
            "score_source": "fraction of required checks passed, not model confidence",
            "adult_person_visible": checks["adult_person_visible"] is True,
            "face_test_applicable": face_visible is not False,
            "face_identity_match": checks.get("face_identity_match") is True,
            "body_test_applicable": body_visible is not False,
            "body_identity_match": checks.get("body_identity_match") is True,
            "request_match": checks["request_match"] is True, "anatomy_ok": checks["anatomy_ok"] is True,
            "reasons": reasons, "repair_instruction": "; ".join(reasons),
            "answers": raw, "validator": MODEL_ID, "analysis_method": "single-question-v1",
        }
    finally:
        del inference_image
        cleanup_memory()
