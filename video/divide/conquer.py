"""Conquer stage: DINOv3 spectral cut within CuVLER bounding boxes.

Extracts sub-masks from each divide proposal using iterative feature-space
merging (from divide_and_conquer/iterative_merging.py). All masks that cover
>= kept_thresh of the parent proposal are returned alongside the original
divide masks.

This avoids importing divide_conquerV3.py directly (which pulls in detectron2).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms

_REPO = Path(__file__).resolve().parents[2]
_DC = _REPO / "divide_and_conquer"

for _p in [str(_REPO), str(_DC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dinov3 import ViTFeatV3           # noqa: E402
from iterative_merging import iterative_merge  # noqa: E402

_FEAT_DIM = 1024
_PATCH_SIZE = 16

_ToTensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

DEFAULT_PARAMS = SimpleNamespace(
    local_size=512,
    feature_dim=_FEAT_DIM,
    patch_size=_PATCH_SIZE,
    thetas=(0.73, 0.62, 0.51, 0.40, 0.30, 0.20),
    kept_thresh=0.8,
    nms_iou=0.8,
    nms_step=5,
)


def load_backbone() -> ViTFeatV3:
    backbone = ViTFeatV3(feat_dim=_FEAT_DIM, vit_feat="k", patch_size=_PATCH_SIZE)
    backbone.eval()
    return backbone


def _generate_feature_matrix(backbone: ViTFeatV3, image: Image.Image,
                              feat_dim: int, feat_num: int) -> torch.Tensor:
    tensor = _ToTensor(image).unsqueeze(0)
    device = next(backbone.parameters()).device
    tensor = tensor.to(device=device, dtype=next(backbone.parameters()).dtype)
    feat = backbone(tensor)[0].cpu()
    return feat.reshape(feat_dim, feat_num, feat_num).permute(1, 2, 0)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(np.logical_and(a, b))
    union = np.count_nonzero(a) + np.count_nonzero(b) - inter
    return inter / union if union else 0.0


def _area(mask: np.ndarray) -> float:
    return np.count_nonzero(mask) / mask.size


def _nms(pool: list[np.ndarray], threshold: float, step: int) -> list[np.ndarray]:
    sorted_masks = sorted(pool, key=_area, reverse=True)
    kept = set(range(len(sorted_masks)))
    for i in range(len(sorted_masks)):
        if i in kept:
            for j in range(i + 1, min(len(sorted_masks), i + step)):
                if _iou(sorted_masks[i], sorted_masks[j]) > threshold:
                    kept.discard(j)
    return [sorted_masks[i] for i in sorted(kept)]


def _coverage(mask: np.ndarray, parent: np.ndarray) -> float:
    if np.count_nonzero(mask) == 0:
        return 0.0
    return np.count_nonzero(np.logical_and(mask, parent)) / np.count_nonzero(mask)


def _resize_mask(mask: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8)).resize(target_wh)
    arr = np.asarray(img).astype(np.uint8)
    thresh = arr.max() / 2.0
    return (arr > thresh).astype(np.uint8)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    if len(row_idx) == 0 or len(col_idx) == 0:
        return 0, 1, 0, 1
    return row_idx[0], row_idx[-1], col_idx[0], col_idx[-1]


def run_conquer(
    backbone: ViTFeatV3,
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    params: SimpleNamespace = DEFAULT_PARAMS,
) -> list[np.ndarray]:
    """Run conquer stage on a list of divide masks.

    For each divide mask, crops its bounding box, runs DINOv3 spectral
    merging, and returns sub-masks that cover >= kept_thresh of the parent.
    Returns the original divide masks + all accepted conquer masks.
    """
    all_masks = list(divide_masks)
    feat_num = params.local_size // params.patch_size

    for divide_mask in divide_masks:
        ymin, ymax, xmin, xmax = _bbox(divide_mask)
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            continue

        crop = image_rgb[ymin:ymax, xmin:xmax]
        pil_crop = Image.fromarray(crop).resize((params.local_size, params.local_size))
        feature_matrix = _generate_feature_matrix(backbone, pil_crop, params.feature_dim, feat_num)

        merging_masks = iterative_merge(feature_matrix, params.thetas)
        conquer_masks = []
        for layer in merging_masks:
            if layer.shape[0] == 0:
                continue
            for i in range(layer.shape[0]):
                m = _resize_mask(layer[i], (xmax - xmin, ymax - ymin))
                if _coverage(m, divide_mask[ymin:ymax, xmin:xmax]) <= params.kept_thresh:
                    continue
                full = np.zeros_like(divide_mask)
                full[ymin:ymax, xmin:xmax] = m
                conquer_masks.append(full)

        conquer_masks = _nms(conquer_masks, params.nms_iou, params.nms_step)
        all_masks.extend(conquer_masks)

    return all_masks


def run_conquer_scored(
    backbone: ViTFeatV3,
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    divide_scores: list[float],
    params: SimpleNamespace = DEFAULT_PARAMS,
) -> tuple[list[np.ndarray], list[float]]:
    """Same as run_conquer, but also returns a confidence score per mask —
    needed to rank the ~30-40x over-generated proposal pool for whole-image
    (no-click) consolidation. Divide masks keep their detector score; conquer
    sub-masks inherit `divide_score * coverage_of_parent` (coverage is already
    computed to decide accept/reject, just also kept here) as a cheap proxy —
    conquer produces no score of its own, so a submask that barely covers its
    parent is trusted less than a near-total re-derivation of it.
    """
    all_masks = list(divide_masks)
    all_scores = list(divide_scores)
    feat_num = params.local_size // params.patch_size

    for divide_mask, divide_score in zip(divide_masks, divide_scores):
        ymin, ymax, xmin, xmax = _bbox(divide_mask)
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            continue

        crop = image_rgb[ymin:ymax, xmin:xmax]
        pil_crop = Image.fromarray(crop).resize((params.local_size, params.local_size))
        feature_matrix = _generate_feature_matrix(backbone, pil_crop, params.feature_dim, feat_num)

        merging_masks = iterative_merge(feature_matrix, params.thetas)
        conquer_masks, conquer_scores = [], []
        for layer in merging_masks:
            if layer.shape[0] == 0:
                continue
            for i in range(layer.shape[0]):
                m = _resize_mask(layer[i], (xmax - xmin, ymax - ymin))
                cov = _coverage(m, divide_mask[ymin:ymax, xmin:xmax])
                if cov <= params.kept_thresh:
                    continue
                full = np.zeros_like(divide_mask)
                full[ymin:ymax, xmin:xmax] = m
                conquer_masks.append(full)
                conquer_scores.append(divide_score * cov)

        conquer_masks, conquer_scores = _nms_scored(conquer_masks, conquer_scores, params.nms_iou, params.nms_step)
        all_masks.extend(conquer_masks)
        all_scores.extend(conquer_scores)

    return all_masks, all_scores


def run_conquer_one_per_object(
    backbone: ViTFeatV3,
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    divide_scores: list[float] | None = None,
    params: SimpleNamespace = DEFAULT_PARAMS,
) -> tuple[list[np.ndarray], list[float]]:
    """One representative mask per divide-object — for whole-image (no-click)
    output, where run_conquer/run_conquer_scored's every-granularity pool (~10-40
    masks per object, mostly non-overlapping PARTS of the same object) can't be
    cleaned up by NMS (parts don't overlap each other, so IoU-based suppression
    can't merge them back together — see eval_whole_image_ar.py findings).

    iterative_merge's clustering is cumulative across its threshold schedule
    (thetas decreasing => more merging at each step), so merging_masks[-1] is
    the coarsest/most-merged layer and merging_masks[0] the finest. We scan
    coarsest-first and take the first layer with a cluster covering >= kept_thresh
    of the parent divide mask — i.e. the most-merged "whole object" reconstruction
    conquer can produce, not every intermediate part. Falls back to the divide
    mask itself if no layer ever reaches kept_thresh coverage.

    Does NOT replace run_conquer/run_conquer_scored — those still feed the click
    pipeline (demo.py, eval_davis2016.py), which benefits from having many
    granularities to search for the best match to a specific click.
    """
    scores_in = divide_scores if divide_scores is not None else [1.0] * len(divide_masks)
    feat_num = params.local_size // params.patch_size
    out_masks, out_scores = [], []

    for divide_mask, divide_score in zip(divide_masks, scores_in):
        ymin, ymax, xmin, xmax = _bbox(divide_mask)
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            out_masks.append(divide_mask)
            out_scores.append(divide_score)
            continue

        crop = image_rgb[ymin:ymax, xmin:xmax]
        pil_crop = Image.fromarray(crop).resize((params.local_size, params.local_size))
        feature_matrix = _generate_feature_matrix(backbone, pil_crop, params.feature_dim, feat_num)
        merging_masks = iterative_merge(feature_matrix, params.thetas)

        # Rank candidates by AREA (among those clearing the coverage floor), not by
        # coverage itself: _coverage(m, parent) = |m ∩ parent| / |m| is a precision-like
        # measure that a tiny fragment sitting entirely inside the object satisfies
        # trivially (coverage=1.0). Ranking by coverage would pick slivers over the
        # whole object; ranking by area picks the largest mostly-correct region.
        best_mask, best_area, best_cov = None, -1, None
        for layer in reversed(merging_masks):  # coarsest first
            if layer.shape[0] == 0:
                continue
            for i in range(layer.shape[0]):
                m = _resize_mask(layer[i], (xmax - xmin, ymax - ymin))
                cov = _coverage(m, divide_mask[ymin:ymax, xmin:xmax])
                if cov <= params.kept_thresh:
                    continue
                area = int(m.sum())
                if area > best_area:
                    best_area, best_cov = area, cov
                    full = np.zeros_like(divide_mask)
                    full[ymin:ymax, xmin:xmax] = m
                    best_mask = full
            if best_mask is not None:
                break  # this (coarsest qualifying) layer wins; stop descending to finer ones

        if best_mask is not None:
            out_masks.append(best_mask)
            out_scores.append(divide_score * best_cov)
        else:
            out_masks.append(divide_mask)
            out_scores.append(divide_score)

    return out_masks, out_scores


def _nms_scored(masks: list[np.ndarray], scores: list[float],
                threshold: float, step: int) -> tuple[list[np.ndarray], list[float]]:
    """Same greedy suppression as _nms, but ordered by score instead of area."""
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    kept = set(range(len(order)))
    for i in range(len(order)):
        if i in kept:
            for j in range(i + 1, min(len(order), i + step)):
                if _iou(masks[order[i]], masks[order[j]]) > threshold:
                    kept.discard(j)
    return [masks[order[i]] for i in sorted(kept)], [scores[order[i]] for i in sorted(kept)]
