import base64
import io
import json
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path

import requests
from PIL import Image, ImageOps


GEMINI_MODEL_PRIORITY = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)
MAX_DOWNLOAD_BYTES = 18 * 1024 * 1024
MIN_REFERENCE_EDGE = 300


def _env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_bool(name, default=False):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in ("1", "true", "yes", "on")


def _env_csv(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in re.split(r"[,;\s]+", raw) if item.strip()]


def _time_range():
    value = os.environ.get("FFS_AVATAR_REFERENCE_TIME_RANGE", "").strip().lower()
    allowed = {"day", "week", "month", "year", "d", "w", "m", "y"}
    return value if value in allowed else None


def _secret(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata

        return (userdata.get(name) or "").strip()
    except Exception:
        return ""


def configuration_status():
    missing = [name for name in ("GEMINI_API_KEY", "TAVILY_API_KEY") if not _secret(name)]
    return {
        "ready": not missing,
        "missing": missing,
        "message": (
            "Gallery discovery is ready."
            if not missing
            else "Add these Colab secrets before using Auto Gallery: " + ", ".join(missing)
        ),
    }


@lru_cache(maxsize=1)
def _gemini_model():
    configured = os.environ.get("FFS_GEMINI_MODEL", "").strip()
    if configured:
        return configured.removeprefix("models/")

    key = _secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(configuration_status()["message"])
    response = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": key},
        timeout=45,
    )
    response.raise_for_status()
    supported = {
        item.get("name", "").removeprefix("models/")
        for item in response.json().get("models", [])
        if "generateContent" in item.get("supportedGenerationMethods", [])
    }
    for model in GEMINI_MODEL_PRIORITY:
        if model in supported:
            return model
    flash_models = sorted(model for model in supported if "flash" in model and "image" not in model)
    if flash_models:
        return flash_models[-1]
    raise RuntimeError("No Gemini Flash model with generateContent support is available for this API key.")


def _extract_json(data):
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as exc:
        raise RuntimeError("Gemini returned an invalid JSON response.") from exc


def gemini_json(prompt, schema, images=None, max_output_tokens=2400):
    key = _secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(configuration_status()["message"])
    parts = [{"text": prompt}]
    for index, item in enumerate(images or [], 1):
        parts.append(
            {
                "text": (
                    f"\nCANDIDATE IMAGE {index}\n"
                    f"Source page: {item.get('source_url', '')}\n"
                    f"Description: {item.get('description', '')}"
                )
            }
        )
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.b64encode(item["validation_bytes"]).decode("ascii"),
                }
            }
        )
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{_gemini_model()}:generateContent",
        params={"key": key},
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        json=payload,
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:800]}")
    return _extract_json(response.json())


def _search_query(theme):
    schema = {
        "type": "object",
        "properties": {
            "search_query": {"type": "string", "maxLength": 400},
            "search_focus": {"type": "string"},
        },
        "required": ["search_query", "search_focus"],
    }
    prompt = f"""
Create one broad web image search query for finding clearly adult fashion and pose
reference photographs for an AI avatar gallery. The requested gallery theme is:
{theme or 'varied editorial fashion and lifestyle portraits'}

The references should show a useful outfit, pose, or composition. Prefer full or
mostly full body photographs. Exclude minors, age-ambiguous people, product-only
images, illustrations, collages, and images dominated by text or interface overlays.
Keep the query broad enough to return variety. Return JSON only.
""".strip()
    result = gemini_json(prompt, schema, max_output_tokens=500)
    return re.sub(r"\s+", " ", result["search_query"]).strip()[:400]


def _tavily_search(query, max_results=20):
    key = _secret("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(configuration_status()["message"])
    body = {
        "query": query,
        "search_depth": "basic",
        "auto_parameters": False,
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": True,
        "include_image_descriptions": True,
        "include_usage": True,
        "safe_search": _env_bool("FFS_AVATAR_SAFE_SEARCH", False),
        "max_results": max(1, min(20, int(max_results))),
    }
    domains = _env_csv("FFS_AVATAR_REFERENCE_DOMAINS")
    if domains:
        body["include_domains"] = domains
    time_range = _time_range()
    if time_range:
        body["time_range"] = time_range
    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=90,
    )
    if not response.ok:
        raise RuntimeError(f"Tavily API error {response.status_code}: {response.text[:800]}")
    return response.json()


def _candidate_images(payload):
    candidates = []
    seen = set()

    def add(item, source_url="", title="", source_type="global"):
        if isinstance(item, dict):
            url = str(item.get("url") or item.get("image_url") or "").strip()
            description = str(item.get("description") or item.get("alt") or "").strip()
        else:
            url, description = str(item).strip(), ""
        if not url.startswith(("http://", "https://")) or url in seen:
            return
        seen.add(url)
        candidates.append(
            {
                "image_url": url,
                "source_url": source_url,
                "title": title,
                "description": description,
                "source_type": source_type,
            }
        )

    for item in payload.get("images", []):
        add(item, source_type="global")
    for result in payload.get("results", []):
        source_url = str(result.get("url") or "")
        title = str(result.get("title") or "")
        for item in result.get("images", []):
            add(item, source_url, title, "result")
    return candidates


def _download_candidate(candidate, output_dir):
    response = requests.get(
        candidate["image_url"],
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
        timeout=25,
        allow_redirects=True,
        stream=True,
    )
    response.raise_for_status()
    data = bytearray()
    for chunk in response.iter_content(256 * 1024):
        data.extend(chunk)
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ValueError("reference image is too large")
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if min(image.size) < MIN_REFERENCE_EDGE:
        raise ValueError("reference image is too small")
    filename = f"reference_{uuid.uuid4().hex[:12]}.jpg"
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=94)
    validation = image.copy()
    validation.thumbnail((768, 768), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    validation.save(buffer, "JPEG", quality=80)
    return {
        **candidate,
        "path": str(path),
        "width": image.width,
        "height": image.height,
        "validation_bytes": buffer.getvalue(),
    }


def _validate_references(candidates):
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "accept": {"type": "boolean"},
                        "score": {"type": "integer", "minimum": 0, "maximum": 100},
                        "adult_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                        "adult_person_visible": {"type": "boolean"},
                        "outfit_or_pose_visible": {"type": "boolean"},
                        "person_large_enough": {"type": "boolean"},
                        "major_obstruction": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "index", "accept", "score", "adult_confidence",
                        "adult_person_visible", "outfit_or_pose_visible",
                        "person_large_enough", "major_obstruction", "reason",
                    ],
                },
            }
        },
        "required": ["results"],
    }
    prompt = """
Evaluate every numbered candidate independently as a reference for generating a new
AI avatar image. Accept only a clearly adult person with a useful visible outfit,
pose, or composition. Reject minors or age-ambiguous people, product-only images,
illustrations, tiny subjects, and images where text or UI blocks important details.
A face is not mandatory when the body/outfit reference remains useful. Score 0-100.
Return one result for every supplied candidate and JSON only.
    """.strip()
    validations = {}

    def validate_batch(batch, offset):
        try:
            result = gemini_json(
                prompt,
                schema,
                images=batch,
                max_output_tokens=max(1800, len(batch) * 320),
            )
            for item in result.get("results", []):
                try:
                    validations[offset + int(item["index"]) - 1] = item
                except (KeyError, TypeError, ValueError):
                    continue
        except Exception:
            if len(batch) <= 1:
                return
            midpoint = len(batch) // 2
            validate_batch(batch[:midpoint], offset)
            validate_batch(batch[midpoint:], offset + midpoint)

    for start in range(0, len(candidates), 8):
        validate_batch(candidates[start : start + 8], start)
    accepted = []
    for index, candidate in enumerate(candidates):
        check = validations.get(index, {})
        passes = (
            check.get("accept") is True
            and int(check.get("score", 0)) >= 68
            and int(check.get("adult_confidence", 0)) >= 85
            and check.get("adult_person_visible") is True
            and check.get("outfit_or_pose_visible") is True
            and check.get("person_large_enough") is True
            and check.get("major_obstruction") is False
        )
        if passes:
            accepted.append({**candidate, "validation": check})
    accepted.sort(key=lambda item: int(item["validation"].get("score", 0)), reverse=True)
    return accepted


def discover_references(theme, quantity, output_dir):
    status = configuration_status()
    if not status["ready"]:
        raise RuntimeError(status["message"])
    requested = int(quantity)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_urls = set()
    for report_path in output_dir.glob("search_*.json"):
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            for item in previous.get("references", []):
                if item.get("image_url"):
                    seen_urls.add(item["image_url"])
        except Exception:
            continue

    selected = []
    total_downloaded = 0
    total_approved = 0
    queries = []
    usage = []
    rounds = _env_int("FFS_AVATAR_SEARCH_ROUNDS", 3, minimum=1, maximum=5)
    max_candidates = _env_int("FFS_AVATAR_MAX_CANDIDATE_DOWNLOADS", 60, minimum=10, maximum=120)
    for round_index in range(rounds):
        if len(selected) >= requested:
            break
        variation = (
            theme
            if round_index == 0
            else f"{theme or 'varied editorial fashion'}; use a different outfit, pose, and setting variation"
        )
        query = _search_query(variation)
        payload = _tavily_search(query, max_results=20)
        queries.append(query)
        usage.append(payload.get("usage", {}))
        maximum = min(max_candidates, max(20, (requested - len(selected)) * 5))
        downloaded = []
        for candidate in _candidate_images(payload):
            if candidate["image_url"] in seen_urls:
                continue
            seen_urls.add(candidate["image_url"])
            try:
                downloaded.append(_download_candidate(candidate, output_dir))
            except Exception:
                continue
            if len(downloaded) >= maximum:
                break
        approved = _validate_references(downloaded)
        total_downloaded += len(downloaded)
        total_approved += len(approved)
        selected.extend(approved[: requested - len(selected)])

    report = {
        "queries": queries,
        "requested": requested,
        "found": total_downloaded,
        "approved": total_approved,
        "selected": len(selected),
        "usage": usage,
        "settings": {
            "rounds": rounds,
            "max_candidate_downloads": max_candidates,
            "time_range": _time_range(),
            "include_domains": _env_csv("FFS_AVATAR_REFERENCE_DOMAINS"),
            "safe_search": _env_bool("FFS_AVATAR_SAFE_SEARCH", False),
        },
    }
    report_path = Path(output_dir) / f"search_{uuid.uuid4().hex[:10]}.json"
    saved_references = [
        {key: value for key, value in item.items() if key != "validation_bytes"}
        for item in selected
    ]
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({**report, "references": saved_references}, handle, indent=2, ensure_ascii=False, default=str)
    return selected, report


def create_generation_prompt(avatar_name, theme, reference, face_summary, body_summary):
    schema = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    }
    prompt = f"""
Write one precise FLUX.2 Klein image-editing prompt for a saved AI avatar named
{avatar_name}. Preserve identity using the attached face and body references. Use
the third reference only for outfit, pose, composition, and scene inspiration.

Gallery theme: {theme or 'varied editorial lifestyle'}
Reference description: {reference.get('description') or reference.get('title') or 'visual reference'}
Face identity notes: {face_summary}
Body identity notes: {body_summary}

Describe one coherent photograph. Do not mention these instructions or the order of
the reference images. Return JSON only.
""".strip()
    return gemini_json(prompt, schema, max_output_tokens=900)["prompt"].strip()


def repair_generation_prompt(prompt, validation):
    schema = {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    }
    reasons = validation.get("reasons") or [validation.get("reason", "identity or anatomy mismatch")]
    instruction = validation.get("repair_instruction", "")
    request = f"""
Repair this FLUX.2 Klein avatar prompt after a failed visual quality check.

Original prompt: {prompt}
Failure reasons: {'; '.join(str(item) for item in reasons)}
Suggested repair: {instruction}

Keep the scene intent, but strengthen identity preservation, visible anatomy,
composition, and any failed requirement. Return the full replacement prompt as JSON.
""".strip()
    return gemini_json(request, schema, max_output_tokens=900)["prompt"].strip()
