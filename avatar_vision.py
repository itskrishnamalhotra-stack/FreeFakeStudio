import ctypes
import gc
import json
import os
import re
from functools import lru_cache

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


def _extract_json(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Vision model returned an empty answer.")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


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
        "image_size": list(image_size),
        "answers": answers,
        "raw_summary": str(payload.get("summary", ""))[:1200],
    }


def analyze_reference_image(kind, image, questions):
    processor, model, torch = _load_model()
    inference_image = prepare_image(image.convert("RGB"))

    question_lines = "\n".join(
        f'- key "{key}" ({strictness}): {question}' for key, question, strictness in questions
    )
    instruction = f"""
Carefully inspect this {kind} reference image for an AI avatar workflow.
Answer only from visible evidence. Do not invent details.
If a detail is not visible, answer "unclear".

Return strict JSON with this shape:
{{
  "summary": "one short visual summary",
  "answers": [
    {{"key": "example", "answer": "short answer", "confidence": 0.0}}
  ]
}}

Questions:
{question_lines}
""".strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[inference_image], return_tensors="pt")
    for key, value in list(inputs.items()):
        if not torch.is_tensor(value):
            continue
        if torch.is_floating_point(value):
            inputs[key] = value.to(device="cuda:0", dtype=torch.float16)
        else:
            inputs[key] = value.to("cuda:0")

    input_length = inputs["input_ids"].shape[1]
    output = None
    generated = None
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        generated = output[:, input_length:]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        payload = _extract_json(answer)
        return _normalise_answers(kind, payload, questions, image.size)
    finally:
        del inputs
        if output is not None:
            del output
        if generated is not None:
            del generated
        if inference_image is not image:
            del inference_image
        cleanup_memory()


def validate_avatar_generation(image, face_specs, body_specs, requested_prompt):
    processor, model, torch = _load_model()
    inference_image = prepare_image(image.convert("RGB"))
    face_notes = json.dumps(face_specs or {}, ensure_ascii=False)[:7000]
    body_notes = json.dumps(body_specs or {}, ensure_ascii=False)[:7000]
    instruction = f"""
Quality-check this generated AI avatar image against the locked identity notes.
Judge only visible evidence. If the face is hidden or too small, do not fail face
identity by itself; set face_test_applicable to false. Apply the same rule to the
body when it is not visible. Always reject broken anatomy, duplicated limbs, severe
artifacts, or an image that does not match the requested scene.

Requested image: {requested_prompt}
Locked face notes: {face_notes}
Locked body notes: {body_notes}

Return strict JSON:
{{
  "pass": true,
  "score": 0,
  "adult_person_visible": true,
  "face_test_applicable": true,
  "face_identity_match": true,
  "body_test_applicable": true,
  "body_identity_match": true,
  "request_match": true,
  "anatomy_ok": true,
  "reasons": ["short reason"],
  "repair_instruction": "one concise prompt correction"
}}
""".strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[inference_image], return_tensors="pt")
    for key, value in list(inputs.items()):
        if not torch.is_tensor(value):
            continue
        if torch.is_floating_point(value):
            inputs[key] = value.to(device="cuda:0", dtype=torch.float16)
        else:
            inputs[key] = value.to("cuda:0")

    input_length = inputs["input_ids"].shape[1]
    output = None
    generated = None
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=min(MAX_NEW_TOKENS, 500),
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        generated = output[:, input_length:]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        payload = _extract_json(answer)
        face_ok = not payload.get("face_test_applicable", True) or payload.get("face_identity_match") is True
        body_ok = not payload.get("body_test_applicable", True) or payload.get("body_identity_match") is True
        passed = (
            payload.get("adult_person_visible") is True
            and face_ok
            and body_ok
            and payload.get("request_match") is True
            and payload.get("anatomy_ok") is True
            and int(payload.get("score", 0)) >= 72
        )
        payload["pass"] = bool(payload.get("pass") is True and passed)
        payload["score"] = max(0, min(100, int(payload.get("score", 0))))
        payload["validator"] = MODEL_ID
        return payload
    finally:
        del inputs
        if output is not None:
            del output
        if generated is not None:
            del generated
        if inference_image is not image:
            del inference_image
        cleanup_memory()
