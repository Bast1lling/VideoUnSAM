"""Consolidate the divide+conquer proposal pool into a clean whole-image
segmentation: one mask per real object instead of an unranked, overlapping,
~10-40x over-generated pile.

The tracking pipeline (demo.py, eval_davis2016.py) never needed this — it always
has a click telling it which single proposal to keep. This is the missing piece
for automatic, no-click, whole-frame segmentation.
"""
from __future__ import annotations

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def consolidate(masks: list[np.ndarray], scores: list[float],
                iou_thresh: float = 0.5, min_score: float = 0.0) -> tuple[list[np.ndarray], list[float]]:
    """Greedy class-agnostic instance NMS: keep masks in score order, discard
    any later mask that overlaps (IoU > iou_thresh) an already-kept one.

    min_score: drop candidates below this score before NMS (cheap pre-filter;
    conquer sub-mask scores are divide_score * coverage, so this also acts as
    an implicit confidence floor on top of CuVLER's own score_thresh).
    """
    idx = [i for i in range(len(masks)) if scores[i] >= min_score]
    idx.sort(key=lambda i: scores[i], reverse=True)

    kept_idx: list[int] = []
    for i in idx:
        if all(_iou(masks[i], masks[k]) <= iou_thresh for k in kept_idx):
            kept_idx.append(i)

    return [masks[i] for i in kept_idx], [scores[i] for i in kept_idx]
