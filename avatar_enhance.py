"""Bounded, sequential FLUX texture reconstruction; no model imports here."""

import math

import numpy as np
from PIL import Image, ImageFilter, ImageOps


REALISM_PROMPT = (
    "Edit the supplied image to achieve the realism and fine surface detail of a high-resolution "
    "professional RAW photograph. Preserve the exact subjects, facial identity, facial structure, "
    "pose, expression, anatomy, clothing, framing, camera angle, background and overall lighting. "
    "Reconstruct natural human skin microtexture: fine irregular pores, tiny vellus hairs, realistic "
    "hair follicles, subtle skin creases, natural variation in skin tone, realistic highlights and "
    "nonuniform skin reflectance. Add believable individual hair strands, eyebrow hairs, fabric "
    "fibres and fine environmental textures where appropriate. The details must be subtle, physically "
    "plausible and consistent with the existing image. Do not beautify, airbrush or smooth the skin. "
    "Do not change anyone's appearance or add or remove objects. Maintain realistic photographic "
    "detail instead of CGI or waxy smoothness."
)
TILE_PROMPT = (
    "Image 1 is a cropped detail tile and is the exact editing canvas. Image 2 shows the whole "
    "original photograph for context only. Return only the same crop as image 1, with unchanged "
    "geometry. Never draw the whole photograph, a new face, a collage, or a border inside this tile. "
)
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


def reconstruct(image, edit_tile, scale=2, strength=0.3, seed=0, cancelled=None):
    """Yield tile progress, then a PIL result. The injected editor performs inference."""
    strength = float(strength)
    if not 0.15 <= strength <= 0.5:
        raise ValueError("Reconstruction strength must be between 0.15 and 0.50.")
    source = ImageOps.exif_transpose(image).convert("RGB")
    size = target_size(source.size, int(scale))
    base = source.resize(size, Image.Resampling.LANCZOS)
    context = source.copy()
    context.thumbnail((768, 768), Image.Resampling.LANCZOS)
    boxes = tile_boxes(size)
    accum = np.zeros((size[1], size[0], 3), dtype=np.float32)
    weights = np.zeros((size[1], size[0]), dtype=np.float32)
    for index, box in enumerate(boxes):
        if cancelled and cancelled():
            raise EnhancementCancelled("Enhancement stopped. The original image is unchanged.")
        yield {"tile": index + 1, "total": len(boxes), "size": size, "image": None}
        tile = base.crop(box)
        padded = _pad(tile)
        edited = edit_tile(padded, context, TILE_PROMPT + REALISM_PROMPT,
                           (int(seed) + index) % (2 ** 63), strength)
        if not isinstance(edited, Image.Image) or edited.size != padded.size:
            raise RuntimeError("FLUX returned an invalid reconstruction tile.")
        if cancelled and cancelled():
            raise EnhancementCancelled("Enhancement stopped. The original image is unchanged.")
        edited = edited.convert("RGB").crop((0, 0, tile.width, tile.height))
        # Keep source geometry/color at low frequencies; blend only reconstructed detail.
        original = np.asarray(tile, dtype=np.float32)
        original_detail = original - np.asarray(tile.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)
        edited_detail = np.asarray(edited, dtype=np.float32) - np.asarray(
            edited.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)
        pixels = np.clip(original + 0.8 * (edited_detail - original_detail), 0, 255)
        mask = _weights(box, size)
        left, top, right, bottom = box
        accum[top:bottom, left:right] += pixels * mask[:, :, None]
        weights[top:bottom, left:right] += mask
    result = Image.fromarray(np.rint(accum / weights[:, :, None]).clip(0, 255).astype(np.uint8))
    yield {"tile": len(boxes), "total": len(boxes), "size": size, "image": result}
