"""Visualisation helpers for label maps."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np
import torch
from PIL import Image

_GOLDEN = 0.61803398875


def palette_color(i: int, s: float = 0.65, v: float = 0.95) -> list[int]:
    """Deterministic, well-spread RGB colour for index ``i`` (golden-ratio hue)."""
    h = (i * _GOLDEN) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(r * 255), int(g * 255), int(b * 255)]


def conquer_color(div_idx: int, j: int) -> list[int]:
    """Colour for part ``j`` of divide mask ``div_idx`` — a hue near the divide's."""
    base = (div_idx * _GOLDEN) % 1.0
    h = (base + 0.09 * (j + 1)) % 1.0
    s = 0.50 + 0.14 * (j % 2)
    v = 0.78 + 0.16 * ((j + 1) % 2)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(r * 255), int(g * 255), int(b * 255)]


def _up(labels: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour upsample an int label grid to ``(h, w)``."""
    arr = labels.detach().cpu().numpy().astype(np.int32)
    return np.asarray(Image.fromarray(arr, mode="I").resize((w, h), Image.NEAREST))


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45,
                 contour_thickness: int = 2) -> np.ndarray:
    """Blend ``mask`` onto ``image_rgb`` with ``color`` and draw its contour."""
    out = image_rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if contour_thickness > 0 and fg.any():
        ctrs, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, list(color), contour_thickness)
    return out


def overlay_hierarchy(image: Image.Image, result: dict, alpha_divide: float = 0.4,
                      alpha_part: float = 0.5, divide_contour: int = 2,
                      part_contour: int = 1) -> Image.Image:
    """Full-image hierarchical overlay: divide masks as base, conquer parts on top.

    ``result`` is the dict returned by a pipeline's ``run``. Divide masks get a
    thick contour in their palette colour; each part gets a thin contour in a hue
    near its parent divide mask, so the object→parts structure is visible at once.
    """
    base = np.asarray(image.convert("RGB")).copy()
    h, w = base.shape[:2]
    div = _up(result["divide"], h, w)
    out = base
    for k in range(1, int(div.max()) + 1):
        m = div == k
        if m.any():
            out = overlay_mask(out, m, palette_color(k - 1), alpha_divide, divide_contour)
    for entry in result["conquer"]:
        div_id = entry["divide_id"]
        parts = _up(entry["parts"], h, w)
        ids = [p for p in np.unique(parts) if p != 0]
        for j, p in enumerate(ids):
            out = overlay_mask(out, parts == p, conquer_color(div_id - 1, j),
                               alpha_part, part_contour)
    return Image.fromarray(out)


def overlay_labels_crop(image: Image.Image, region: torch.Tensor, labels: torch.Tensor,
                        alpha: float = 0.55, contour_thickness: int = 1) -> Image.Image:
    """Crop to ``region``'s bounding box and overlay ``labels`` (region-local ids)."""
    base = np.asarray(image.convert("RGB"))
    h, w = base.shape[:2]
    reg = _up(region.long(), h, w).astype(bool)
    ys, xs = np.where(reg)
    if len(ys) == 0:
        return image
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lab = _up(labels, h, w)[y0:y1, x0:x1]
    out = base[y0:y1, x0:x1].copy()
    ids = [p for p in np.unique(lab) if p != 0]
    for j, p in enumerate(ids):
        out = overlay_mask(out, lab == p, palette_color(j), alpha, contour_thickness)
    return Image.fromarray(out)


def label_triple(image: Image.Image, labels: torch.Tensor,
                 region: torch.Tensor | None = None, seed: int = 0, pad: int = 0,
                 alpha: float = 0.55, contour_thickness: int = 1
                 ) -> tuple[Image.Image, Image.Image, Image.Image] | None:
    """Return ``(masks_only, original, overlay)`` for ``labels``.

    With ``region`` ``None`` the triple spans the whole image and shows every
    non-zero label. With a boolean ``(H, W)`` ``region`` the triple is **cropped to
    that region's bounding box** (grown by ``pad`` image pixels) and shows only the
    labels of ``labels`` that fall inside the region — i.e. one coarse mask's parts,
    zoomed in. Returns ``None`` if the region is empty.

    Colours come from :func:`color_labels` with the given ``seed``: because that
    palette is seeded deterministically per-label, a part keeps the same colour in
    the whole-image view and in its zoomed crop.
    """
    base = np.asarray(image.convert("RGB"))
    h, w = base.shape[:2]
    lab_full = _up(labels, h, w)
    if region is None:
        y0, y1, x0, x1 = 0, h, 0, w
        lab = lab_full
    else:
        reg = _up(region.long(), h, w).astype(bool)
        ys, xs = np.where(reg)
        if len(ys) == 0:
            return None
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + 1 + pad, h)
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + 1 + pad, w)
        lab = np.where(reg, lab_full, 0)[y0:y1, x0:x1]
    img_crop = base[y0:y1, x0:x1]
    masks = color_labels(torch.from_numpy(lab.astype(np.int64)), seed=seed, bg_label=0)
    overlay = img_crop.copy()
    for p in [int(v) for v in np.unique(lab) if v != 0]:
        col = [int(c) for c in masks[lab == p][0]]
        overlay = overlay_mask(overlay, lab == p, col, alpha, contour_thickness)
    return (Image.fromarray(masks), Image.fromarray(img_crop.copy()),
            Image.fromarray(overlay))


def color_labels(labels: torch.Tensor, seed: int = 0, bg_label: int | None = 0
                 ) -> np.ndarray:
    """Map an ``(H, W)`` label map to an ``(H, W, 3)`` uint8 colour image.

    Each label gets a distinct random colour. ``bg_label`` (if not ``None``) is
    rendered black -- pass ``None`` to colour every label.
    """
    lab = labels.cpu().numpy().astype(np.int64)
    n = int(lab.max()) + 1
    rng = np.random.default_rng(seed)
    palette = rng.integers(40, 256, size=(max(n, 1), 3), dtype=np.uint8)
    if bg_label is not None and 0 <= bg_label < len(palette):
        palette[bg_label] = (0, 0, 0)
    return palette[lab]


def overlay_masks(image: Image.Image, labels: torch.Tensor, alpha: float = 0.5,
                  seed: int = 0, bg_label: int | None = 0) -> Image.Image:
    """Blend coloured masks over ``image`` (resized to the image, NEAREST)."""
    h_img, w_img = image.height, image.width
    color = color_labels(labels, seed=seed, bg_label=bg_label)
    color_img = Image.fromarray(color).resize((w_img, h_img), Image.NEAREST)
    return Image.blend(image.convert("RGB"), color_img, alpha)


def hstack(images: list[Image.Image], pad: int = 4, bg=(255, 255, 255)) -> Image.Image:
    """Concatenate images horizontally with padding (heights assumed equal)."""
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.width + pad
    return canvas
