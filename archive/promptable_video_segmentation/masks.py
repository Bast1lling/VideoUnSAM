"""
Self-contained mask utilities: iterative spectral merging, NMS, IoU, coverage,
bounding-box extraction, and mask resizing.

No dependency on any other module in this repo.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image


# ── Geometric helpers ─────────────────────────────────────────────────────────

def smallest_square_containing_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (ymin, ymax, xmin, xmax) tight bounding box of a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return 0, 1, 0, 1
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return int(ymin), int(ymax), int(xmin), int(xmax)


def resize_mask(bipartition: np.ndarray, target_wh: list[int]) -> np.ndarray:
    """Bilinearly resize a soft [0,1] mask to (W, H) = target_wh, then threshold at 0.5·max."""
    img = Image.fromarray(np.uint8(bipartition * 255))
    arr = np.asarray(img.resize(target_wh)).astype(np.uint8)
    upper = int(arr.max())
    thresh = upper / 2.0
    arr[arr > thresh] = upper
    arr[arr <= thresh] = 0
    return arr


# ── Mask scoring ──────────────────────────────────────────────────────────────

def area(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = int(np.count_nonzero(np.logical_and(mask1, mask2)))
    union = int(np.count_nonzero(mask1)) + int(np.count_nonzero(mask2)) - inter
    return inter / union if union > 0 else 0.0


def coverage(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Fraction of mask1 that is also covered by mask2."""
    n1 = int(np.count_nonzero(mask1))
    if n1 == 0:
        return 0.0
    return int(np.count_nonzero(np.logical_and(mask1, mask2))) / n1


# ── Non-maximum suppression ───────────────────────────────────────────────────

def NMS(pool: list[np.ndarray], threshold: float, step: int = 5) -> list[np.ndarray]:
    """
    Non-maximum suppression by IoU.

    Masks are sorted largest-first; for each kept mask, any later mask within
    *step* positions whose IoU exceeds *threshold* is discarded.
    """
    sorted_masks = sorted(pool, key=area, reverse=True)
    keep = list(range(len(sorted_masks)))
    for i in range(len(sorted_masks)):
        if i not in keep:
            continue
        for j in range(i + 1, min(len(sorted_masks), i + step)):
            if j in keep and iou(sorted_masks[i], sorted_masks[j]) > threshold:
                keep.remove(j)
    return [sorted_masks[i] for i in keep]


# ── Iterative spectral merging ────────────────────────────────────────────────

def _merge_two(i: int, j: int, clusters: list[dict]) -> dict:
    c1, c2 = clusters[i], clusters[j]
    n_new    = c1["n"] + c2["n"]
    feat_new = c1["feat"] + c2["feat"]
    mean_f   = (feat_new / n_new).float()
    return {
        "feat":      feat_new,
        "mean_feat": mean_f,
        "norm_feat": F.normalize(mean_f, dim=0),
        "mask":      (c1["mask"] + c2["mask"]) > 0,
        "n":         n_new,
        "neighbors": c1["neighbors"].union(c2["neighbors"]).difference({i, j}),
    }


def iterative_merge(
    features: torch.Tensor,
    threshes: list[float],
    min_size: int = 4,
    similarity_mode: str = "cosine",
    rbf_sigma: "float | None" = None,
    adaptive: bool = False,
) -> list[np.ndarray]:
    """
    Hierarchical spectral merging on a (H, W, C) feature grid.

    Adjacent patches whose similarity exceeds *thresh* are merged iteratively.
    The process runs for each threshold in *threshes* in sequence, producing one
    coarser set of cluster masks per level.

    Parameters
    ----------
    features         : (H, W, C) torch.Tensor of patch features
    threshes         : merging thresholds.
                       adaptive=False (default): absolute cutoffs, decreasing,
                         e.g. [0.73, 0.62, 0.51, …].
                       adaptive=True: quantile fractions in (0,1), decreasing,
                         e.g. [0.95, 0.80, 0.65, …].  Each value is mapped to
                         the corresponding percentile of the initial edge
                         similarity distribution, making cutoffs data-driven.
    min_size         : minimum patch count for a cluster to appear in output.
    similarity_mode  : "cosine" (default) — normalised dot product, ignores
                         feature magnitude.
                       "rbf" — Gaussian kernel exp(−‖a−b‖²/2σ²) on raw cluster
                         centroids, preserving magnitude.
    rbf_sigma        : bandwidth for RBF mode.  None = auto (median adjacent-pair
                         L2 distance in this crop).
    adaptive         : if True, map *threshes* through the empirical CDF of the
                         initial edge similarities so cutoffs are scale-invariant.

    Returns
    -------
    List of (K, H, W) uint8 arrays, one per threshold level.
    """
    H, W = features.shape[:2]

    # ── RBF sigma: auto = median adjacent-pair L2 distance ────────────────────
    if similarity_mode == "rbf" and rbf_sigma is None:
        dists: list[float] = []
        f = features.float()
        for y in range(H):
            for x in range(W):
                if x > 0:
                    dists.append(float((f[y, x] - f[y, x - 1]).norm()))
                if y > 0:
                    dists.append(float((f[y, x] - f[y - 1, x]).norm()))
        rbf_sigma = max(float(np.median(dists)) if dists else 1.0, 1e-8)

    # ── similarity function (operates on cluster dicts) ────────────────────────
    if similarity_mode == "rbf":
        _s2 = 2.0 * rbf_sigma * rbf_sigma  # type: ignore[operator]
        def _sim(a: dict, b: dict) -> float:
            d = a["mean_feat"] - b["mean_feat"]
            return float(torch.exp(-d.dot(d) / _s2).item())
    else:
        def _sim(a: dict, b: dict) -> float:
            return float(torch.dot(a["norm_feat"], b["norm_feat"]).item())

    # ── initialise one cluster per patch ──────────────────────────────────────
    clusters: list[dict] = []
    sims: dict[tuple[int, int], float] = {}
    idx = 0

    for y in range(H):
        for x in range(W):
            patch = features[y, x].float()
            mask  = np.zeros((H, W))
            mask[y, x] = 1
            clusters.append({
                "feat":      patch,
                "mean_feat": patch,
                "norm_feat": F.normalize(patch, dim=0),
                "mask":      mask,
                "n":         1,
                "neighbors": set(),
            })
            if idx % W != 0:
                clusters[idx]["neighbors"].add(idx - 1)
                clusters[idx - 1]["neighbors"].add(idx)
                sims[(idx - 1, idx)] = _sim(clusters[idx - 1], clusters[idx])
            if idx - W >= 0:
                clusters[idx]["neighbors"].add(idx - W)
                clusters[idx - W]["neighbors"].add(idx)
                sims[(idx - W, idx)] = _sim(clusters[idx - W], clusters[idx])
            idx += 1

    # ── adaptive: map threshes as quantile fractions of the initial sim CDF ───
    if adaptive and sims:
        sim_arr  = np.array(list(sims.values()))
        threshes = [float(np.quantile(sim_arr, p)) for p in threshes]

    # ── iterative merge per threshold level ───────────────────────────────────
    all_masks: list[np.ndarray] = []

    for thresh in threshes:
        while sims:
            i, j = max(sims, key=sims.get)
            if sims[(i, j)] < thresh:
                break

            merged = _merge_two(i, j, clusters)
            clusters.append(merged)
            del sims[(i, j)]

            for nb in merged["neighbors"]:
                for old in (i, j):
                    if old in clusters[nb]["neighbors"]:
                        key = (nb, old) if nb < old else (old, nb)
                        sims.pop(key, None)
                        clusters[nb]["neighbors"].discard(old)
                sims[(nb, idx)] = _sim(clusters[nb], merged)
                clusters[nb]["neighbors"].add(idx)

            idx += 1

        seen: set[int] = set()
        level: list[np.ndarray] = []
        for (m, n) in sims:
            for k in (m, n):
                if k not in seen:
                    seen.add(k)
                    if clusters[k]["n"] >= min_size:
                        level.append(clusters[k]["mask"])
        if level:
            all_masks.append(np.stack(level))

    return all_masks
