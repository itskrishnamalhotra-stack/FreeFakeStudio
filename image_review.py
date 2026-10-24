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
