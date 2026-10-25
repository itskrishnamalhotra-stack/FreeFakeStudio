"""Matched-resolution review evidence. These measurements do not prove realism."""
import numpy as np
from PIL import Image, ImageFilter


def detail_evidence(source, result):
    size = source.size
    factor = min(1, 512 / max(size))
    size = tuple(max(1, round(edge * factor)) for edge in size)
    first = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    second = result.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    def energy(photo):
        gray = photo.convert("L")
        high = np.asarray(gray, dtype=np.float32) - np.asarray(gray.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
        return float(np.sqrt(np.mean(high * high)))
    before, after = energy(first), energy(second)
    ratio = after / before if before >= .1 else (1 if after < .1 else after / .1)
    return {"comparison_size": list(size), "source_detail_energy": round(before, 4),
            "result_detail_energy": round(after, 4), "detail_energy_ratio": round(ratio, 4),
            "advisory": "possible_detail_loss" if ratio < .85 else "little_detail_change" if ratio < 1.1 else "detail_changed_review_needed",
            "quality_verified": False, "meaning": "Matched-resolution high-frequency energy; noise and sharpening can also increase this value"}


def regions(size, faces=None):
    rows = [{"label": "Center detail", "box": [.25, .25, .75, .75]}]
    factor = min(1, 1024 / max(size))
    width, height = max(1, round(size[0] * factor)), max(1, round(size[1] * factor))
    for index, face in enumerate(sorted(faces or [], key=lambda row: row["box"][0])[:3]):
        x, y, w, h = face["box"]
        box = [max(0, (x - .25 * w) / width), max(0, (y - .25 * h) / height),
               min(1, (x + 1.25 * w) / width), min(1, (y + 1.25 * h) / height)]
        if box[0] < box[2] and box[1] < box[3]:
            rows.append({"label": f"Face {index + 1}", "box": [round(v, 5) for v in box]})
    return rows


def enhancement_review(source, result, metrics):
    size = tuple(max(1, round(v * min(1, 512 / max(source.size)))) for v in source.size)
    before = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    after = np.asarray(result.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
    low_before = np.asarray(Image.fromarray(before.astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)
    low_after = np.asarray(Image.fromarray(after.astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)
    tone_change = float(np.abs(low_after - low_before).mean())
    detail_change = float(np.abs((after - low_after) - (before - low_before)).mean())
    reasons = []
    if not metrics.get("near_noop"):
        if tone_change > 2 and detail_change < .3 * tone_change:
            reasons.append("Most measured change is lighting/color rather than fine detail")
        if metrics["detail_evidence"]["detail_energy_ratio"] <= 1.05:
            reasons.append("No measured fine-detail gain at matched resolution")
    return {"status": "needs_review" if reasons else "no_change" if metrics.get("near_noop") else "unverified",
            "reasons": reasons, "low_frequency_change": round(tone_change, 4),
            "fine_detail_change": round(detail_change, 4), "quality_verified": False,
            "method": "Pixel-frequency heuristic; additional noise/sharpening is not proof of improvement"}
