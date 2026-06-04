"""
Visualization helpers for the hierarchical segmentation output.

All functions return (H, W, 3) uint8 RGB arrays suitable for display.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..masks import smallest_square_containing_mask
from .hierarchical import SegMask, masks_at_level, children_of


# ── colour palette ─────────────────────────────────────────────────────────────

PALETTE: list[list[int]] = [
    [220,  20,  60], [ 30, 144, 255], [ 50, 205,  50], [255, 215,   0],
    [255, 140,   0], [138,  43, 226], [  0, 206, 209], [255, 105, 180],
    [127, 255,   0], [255,  69,   0], [  0, 191, 255], [186,  85, 211],
]

CHILD_PALETTE: list[list[int]] = [
    [255, 230,  50], [ 50, 255, 160], [255, 100,  50], [180,  50, 255],
    [ 50, 220, 255], [255,  50, 150], [200, 255,  50], [255, 160,  50],
    [ 50, 100, 255], [255,  50, 200], [100, 255,  50], [ 50, 200, 180],
]


def level_color(idx: int) -> list[int]:
    return PALETTE[idx % len(PALETTE)]


def child_color(parent_idx: int, child_idx: int) -> list[int]:
    return CHILD_PALETTE[(parent_idx * 3 + child_idx) % len(CHILD_PALETTE)]


# ── core overlay ───────────────────────────────────────────────────────────────

def overlay_mask(
    image_rgb: np.ndarray,
    mask:      np.ndarray,
    color:     Sequence[int],
    alpha:     float = 0.45,
    contour:   int   = 2,
) -> np.ndarray:
    """Blend a binary mask onto image_rgb with a solid colour + optional contour."""
    out = image_rgb.copy()
    fg  = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if contour > 0:
        ctrs, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, ctrs, -1, list(color), contour)
    return out


# ── level-aware visualizations ────────────────────────────────────────────────

def visualize_level1(
    image_rgb: np.ndarray,
    seg_masks: "list[SegMask]",
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay all level-1 masks on the full image."""
    out = image_rgb.copy()
    for sm in masks_at_level(seg_masks, 1):
        out = overlay_mask(out, sm.mask.astype(np.uint8),
                           level_color(sm.child_idx), alpha)
    return out


def visualize_level1_selected(
    image_rgb: np.ndarray,
    seg_masks: "list[SegMask]",
    selected:  int,
    alpha:     float = 0.55,
) -> np.ndarray:
    """Highlight one level-1 mask with a white border; dim the rest."""
    out = image_rgb.copy()
    l1 = masks_at_level(seg_masks, 1)
    for sm in l1:
        a = alpha if sm.child_idx == selected else 0.20
        out = overlay_mask(out, sm.mask.astype(np.uint8),
                           level_color(sm.child_idx), a, contour=0)
    if 0 <= selected < len(l1):
        m = l1[selected].mask.astype(np.uint8)
        ctrs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, [255, 255, 255], 3)
    return out


def visualize_children_crop(
    image_rgb: np.ndarray,
    seg_masks: "list[SegMask]",
    parent_idx: int,
    level:      int = 2,
) -> "np.ndarray | None":
    """
    Crop the bounding box of the parent mask and overlay all child masks on it.

    Returns None if the parent doesn't exist or has no children.
    """
    l_prev = masks_at_level(seg_masks, level - 1)
    if parent_idx >= len(l_prev):
        return None

    parent_mask = l_prev[parent_idx].mask
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(parent_mask)
    ymax, xmax = ymax + 1, xmax + 1
    if ymax <= ymin or xmax <= xmin:
        return None

    crop = image_rgb[ymin:ymax, xmin:xmax].copy()
    for sm in children_of(seg_masks, level, parent_idx):
        m = sm.mask[ymin:ymax, xmin:xmax].astype(np.uint8)
        crop = overlay_mask(crop, m, child_color(parent_idx, sm.child_idx), 0.5)
    return crop


def visualize_child_single(
    image_rgb:  np.ndarray,
    seg_masks:  "list[SegMask]",
    parent_idx: int,
    child_idx:  int,
    level:      int = 2,
    alpha:      float = 0.6,
) -> "np.ndarray | None":
    """
    Crop to the parent bounding box and highlight a single child mask.

    Returns None if the mask doesn't exist.
    """
    l_prev = masks_at_level(seg_masks, level - 1)
    if parent_idx >= len(l_prev):
        return None
    parent_mask = l_prev[parent_idx].mask
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(parent_mask)
    ymax, xmax = ymax + 1, xmax + 1
    if ymax <= ymin or xmax <= xmin:
        return None

    crop = image_rgb[ymin:ymax, xmin:xmax].copy()
    children = children_of(seg_masks, level, parent_idx)
    match = [sm for sm in children if sm.child_idx == child_idx]
    if not match:
        return crop
    m = match[0].mask[ymin:ymax, xmin:xmax].astype(np.uint8)
    return overlay_mask(crop, m, child_color(parent_idx, child_idx), alpha)


def visualize_all(
    image_rgb: np.ndarray,
    seg_masks: "list[SegMask]",
    alpha_l1:  float = 0.35,
    alpha_l2:  float = 0.55,
) -> np.ndarray:
    """Overlay all levels on the full image (level-1 base, level-2 on top)."""
    out = image_rgb.copy()
    for sm in masks_at_level(seg_masks, 1):
        out = overlay_mask(out, sm.mask.astype(np.uint8),
                           level_color(sm.child_idx), alpha_l1)
    for sm in masks_at_level(seg_masks, 2):
        pi = sm.parent_idx if sm.parent_idx is not None else 0
        out = overlay_mask(out, sm.mask.astype(np.uint8),
                           child_color(pi, sm.child_idx), alpha_l2, contour=1)
    return out
