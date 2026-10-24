"""Native full-photo editing and optional tiled HD reconstruction; no model imports."""

import math

import numpy as np
from PIL import Image, ImageOps
import image_review


REALISM_PROMPT = (
    "Enhance the supplied image with restrained, source-faithful photographic clarity. "
    "Preserve the exact subjects, identity, apparent age, expression, pose, anatomy, clothing, objects, "
    "composition, colors and lighting. Restore only subtle detail supported by visible forms and "
    "materials. Preserve the existing complexion, skin markings and degree of skin smoothness. "
    "Keep naturally soft areas soft and preserve the original depth of field and background blur. "
    "Correct only clear compression or rendering defects whose intended structure is unambiguous. "
    "Return the same photograph with a modest improvement in clarity, not a redesigned or heavily textured version."
)
TILE_PROMPT = (
    "Image 1 is a cropped detail tile and is the exact editing canvas. Image 2 shows the whole "
    "original photograph for context only. Return only the same crop as image 1, with unchanged "
    "geometry. Never draw the whole photograph, a new face, a collage, or a border inside this tile. "
    "Apply detail instructions only to materials actually visible in this crop; preserve intentional blur. "
)
PHOTO_PROMPT = REALISM_PROMPT
METHODS = ("photo", "hd_tiles")
DEFAULT_STRENGTH = 0.65
MAX_EDGE = 2048
TILE_EDGE = 768
OVERLAP = 128


class EnhancementCancelled(RuntimeError):
    pass


def target_size(size, scale=2):
    if scale not in (1, 2):
        raise ValueError("Choose original size or 2x.")
    width, height = size
    if min(width, height) < 1:
        raise ValueError("The source image has invalid dimensions.")
    factor = min(float(scale), MAX_EDGE / max(width, height))
    return max(1, round(width * factor)), max(1, round(height * factor))


def _starts(length):
    if length <= TILE_EDGE:
        return [0]
    count = math.ceil((length - TILE_EDGE) / (TILE_EDGE - OVERLAP)) + 1
    return [round(index * (length - TILE_EDGE) / (count - 1)) for index in range(count)]


def tile_boxes(size):
    width, height = size
    return [(x, y, min(x + TILE_EDGE, width), min(y + TILE_EDGE, height))
            for y in _starts(height) for x in _starts(width)]


def _pad(tile):
    width, height = tile.size
    extra_x, extra_y = (-width) % 64, (-height) % 64
    return Image.fromarray(np.pad(np.asarray(tile), ((0, extra_y), (0, extra_x), (0, 0)), mode="edge"))


def _weights(box, size):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    x = np.ones(width, dtype=np.float32)
    y = np.ones(height, dtype=np.float32)
    for values, before, after in ((x, left > 0, right < size[0]), (y, top > 0, bottom < size[1])):
        count = min(OVERLAP, len(values) // 2)
        ramp = np.linspace(1 / (count + 1), 1, count, dtype=np.float32)
        if before:
            values[:count] *= ramp
        if after:
            values[-count:] *= ramp[::-1]
    return y[:, None] * x[None, :]


def change_metrics(baseline, result):
    first = np.asarray(baseline.convert("RGB"), dtype=np.float32)
    second = np.asarray(result.convert("RGB"), dtype=np.float32)
    delta = np.abs(first - second)
    mean = float(delta.mean())
    changed = float((delta.max(axis=2) >= 4).mean())
    return {"mean_absolute_pixel_change": round(mean, 4), "changed_pixel_fraction": round(changed, 4),
            "detail_evidence": image_review.detail_evidence(baseline, result), "detail_regions": image_review.regions(baseline.size),
            "near_noop": mean < 1.0 and changed < 0.02,
            "meaning": "pixel change only; not a realism, identity or quality score"}


def reconstruct(image, edit_tile, scale=1, strength=DEFAULT_STRENGTH, seed=0, cancelled=None, method="photo", prompt=None):
    prompt = PHOTO_PROMPT if prompt is None else str(prompt).strip()
    if not prompt:
        raise ValueError("Write a reconstruction prompt first")
    """One native photo edit by default; optional full-photo + tiled HD reconstruction."""
    strength = float(strength)
    if not 0.35 <= strength <= 1.0:
        raise ValueError("Reconstruction strength must be between 0.35 and 1.00.")
    if method not in METHODS:
        raise ValueError("Choose photo or hd_tiles reconstruction.")
    source = ImageOps.exif_transpose(image).convert("RGB")
    size = target_size(source.size, int(scale))
    working = source.copy()
    working.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    baseline = source.resize(size, Image.Resampling.LANCZOS)
    boxes = tile_boxes(size) if method == "hd_tiles" else []
    total = 1 + len(boxes)

    def check_stop():
        if cancelled and cancelled():
            raise EnhancementCancelled("Enhancement stopped. The original image is unchanged.")

    def progress(index, stage):
        return {"tile": index, "total": total, "stage": stage, "size": size,
                "working_size": working.size, "output_images": 1, "image": None}

    check_stop()
    yield progress(1, "full_photo")
    padded = _pad(working)
    rebuilt = edit_tile(padded, None, prompt, int(seed) % (2 ** 63), strength)
    if not isinstance(rebuilt, Image.Image) or rebuilt.size != padded.size:
        raise RuntimeError("FLUX returned an invalid full-photo reconstruction.")
    check_stop()
    rebuilt = rebuilt.convert("RGB").crop((0, 0, working.width, working.height))
    base = rebuilt.resize(size, Image.Resampling.LANCZOS)
    if not boxes:
        yield {**progress(1, "complete"), "image": base, "diagnostics": change_metrics(baseline, base)}
        return
    context = source.copy()
    context.thumbnail((768, 768), Image.Resampling.LANCZOS)
    accum = np.zeros((size[1], size[0], 3), dtype=np.float32)
    weights = np.zeros((size[1], size[0]), dtype=np.float32)
    for index, box in enumerate(boxes):
        check_stop()
        yield progress(index + 2, "hd_tile")
        tile = base.crop(box)
        padded = _pad(tile)
        edited = edit_tile(padded, context, TILE_PROMPT + prompt,
                           (int(seed) + index + 1) % (2 ** 63), strength)
        if not isinstance(edited, Image.Image) or edited.size != padded.size:
            raise RuntimeError("FLUX returned an invalid reconstruction tile.")
        check_stop()
        edited = edited.convert("RGB").crop((0, 0, tile.width, tile.height))
        # Retain FLUX's complete correction instead of filtering it back toward the input.
        pixels = np.asarray(edited, dtype=np.float32)
        mask = _weights(box, size)
        left, top, right, bottom = box
        accum[top:bottom, left:right] += pixels * mask[:, :, None]
        weights[top:bottom, left:right] += mask
    result = Image.fromarray(np.rint(accum / weights[:, :, None]).clip(0, 255).astype(np.uint8))
    yield {**progress(total, "complete"), "image": result, "diagnostics": change_metrics(baseline, result)}
