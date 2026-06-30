"""
Iterative spectral merging for the Conquer phase.

Standalone module — only depends on numpy and torch.

Two innovations over the original cosine-only approach:

1. **RBF similarity** (``similarity_mode="rbf"``)
   Uses a Gaussian kernel ``exp(−‖a−b‖²/2σ²)`` on raw cluster centroids instead
   of the normalised dot product.  This preserves feature magnitude, which
   DINOv3 encodes meaningfully (high-magnitude patches ↔ salient regions).
   ``rbf_sigma=None`` (default) auto-sets σ to the median adjacent-patch L2
   distance in the crop, making the kernel scale-invariant across images.

2. **Adaptive thresholds** (``adaptive=True``)
   Interprets ``threshes`` as quantile fractions of the *initial* edge
   similarity CDF rather than absolute cutoffs.  E.g. ``0.95`` means "merge
   only the most-similar 5 % of adjacent pairs first".  This makes the
   hierarchy data-driven and invariant to the absolute similarity range of
   any particular crop.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ── merge helper ──────────────────────────────────────────────────────────────

def _merge_two(i: int, j: int, clusters: list[dict]) -> dict:
    c1, c2   = clusters[i], clusters[j]
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


# ── main entry point ──────────────────────────────────────────────────────────

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
    The process runs for each threshold in *threshes* in sequence (continuing
    from where the previous threshold left off), producing one coarser set of
    cluster masks per level.

    Parameters
    ----------
    features        : (H, W, C) torch.Tensor of patch features
    threshes        : merging thresholds.
                      ``adaptive=False``: absolute cutoffs, decreasing,
                        e.g. ``[0.73, 0.62, 0.51, …]``.
                      ``adaptive=True``: quantile fractions in (0, 1), decreasing,
                        e.g. ``[0.95, 0.80, 0.65, …]`` — each is mapped to the
                        corresponding percentile of the initial edge similarity
                        distribution, making cutoffs data-driven.
    min_size        : minimum patch count for a cluster to appear in output.
    similarity_mode : ``"cosine"`` — normalised dot product, ignores magnitude.
                      ``"rbf"`` — Gaussian kernel on raw cluster centroids,
                        preserves magnitude.
    rbf_sigma       : bandwidth for RBF.  ``None`` = auto (median adjacent L2).
    adaptive        : if ``True``, treat threshes as quantile fractions.

    Returns
    -------
    List of ``(K, H, W)`` uint8 arrays, one per threshold level.
    """
    H, W = features.shape[:2]

    # ── RBF sigma: auto = median adjacent-pair L2 ─────────────────────────────
    if similarity_mode == "rbf" and rbf_sigma is None:
        f = features.float()
        dists: list[float] = []
        for y in range(H):
            for x in range(W):
                if x > 0:
                    dists.append(float((f[y, x] - f[y, x - 1]).norm()))
                if y > 0:
                    dists.append(float((f[y, x] - f[y - 1, x]).norm()))
        rbf_sigma = max(float(np.median(dists)) if dists else 1.0, 1e-8)

    # ── similarity function ───────────────────────────────────────────────────
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

    # ── adaptive: map threshes as quantile fractions of initial sim CDF ───────
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
