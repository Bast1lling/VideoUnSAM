"""Feature-edge detection: a Sobel-like boundary map over DINOv3 patches.

For every patch we measure how *different* it is from its spatial neighbours in
feature space and sum those differences. High values sit on object/part
boundaries (features change quickly), low values inside smooth regions — i.e. a
feature-space edge detector, the dense analogue of an image gradient.

The distance is cosine by default (DINOv3's native metric — see
[../DINOV3_ARCHITECTURE.md](../DINOV3_ARCHITECTURE.md)); euclidean keeps the
patch-norm magnitude that cosine discards.

Note on borders: neighbours are taken with a wrap-around roll (matching the
reference numpy implementation), so the first/last row & column compare against
the opposite edge. With ~32x32 patch grids this is a negligible 1-patch ring.
"""
from __future__ import annotations

import numpy as np
import torch

from mask_extraction.base import UnionFind, relabel

# 4-connectivity (von Neumann) and the 4 diagonals that extend it to 8 (Moore).
_OFFSETS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_OFFSETS_8 = _OFFSETS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
# one representative per undirected edge (right/down [+ diagonals]) for the graph.
_FORWARD_4 = [(0, 1), (1, 0)]
_FORWARD_8 = _FORWARD_4 + [(1, 1), (1, -1)]


def _pair_distance(a: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    """Per-element feature distance between two ``(..., C)`` tensors."""
    if metric == "cosine":
        num = (a * b).sum(dim=-1)
        denom = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1) + 1e-8
        return 1.0 - num / denom
    if metric == "euclidean":
        return torch.linalg.norm(a - b, dim=-1)
    raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric}")


def feature_edge_map(
    feat: torch.Tensor,
    metric: str = "cosine",
    connectivity: int = 4,
    radii=(1,),
) -> torch.Tensor:
    """Per-patch edge strength from neighbour feature distances.

    Parameters
    ----------
    feat:
        ``(H, W, C)`` patch-feature grid (a DINOv3 hook reshaped to spatial).
    metric:
        ``"cosine"`` -> ``1 - cos(a, b)`` (in ``[0, 2]``); ``"euclidean"`` ->
        ``||a - b||`` (magnitude-aware, unbounded).
    connectivity:
        ``4`` (up/down/left/right) or ``8`` (adds diagonals).
    radii:
        Which neighbour *distances* (in patches) to compare against. ``1`` is the
        immediately adjacent patch, ``2`` the patch two steps away, etc. Passing
        several, e.g. ``(1, 2, 3)``, averages the distance over all of them — a
        thicker, multi-scale edge response. Each radius scales the connectivity
        offsets (so a diagonal at radius 2 is ``(2, 2)``).

    Returns
    -------
    ``(H, W)`` float tensor: the mean neighbour distance per patch, averaged over
    all (radius × connectivity) offsets.
    """
    if connectivity == 8:
        base = _OFFSETS_8
    elif connectivity == 4:
        base = _OFFSETS_4
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    radii = [int(r) for r in radii if int(r) > 0]
    if not radii:
        raise ValueError("radii must contain at least one positive integer")
    offsets = [(dy * r, dx * r) for r in radii for dy, dx in base]

    f = feat.float()
    edge = torch.zeros(f.shape[:2], dtype=torch.float32, device=f.device)

    for dy, dx in offsets:
        shifted = torch.roll(f, shifts=(-dy, -dx), dims=(0, 1))
        edge += _pair_distance(f, shifted, metric)

    return edge / len(offsets)


def graph_segment(
    feat: torch.Tensor,
    metric: str = "cosine",
    connectivity: int = 4,
    radii=(1,),
    threshold: float | None = None,
    quantile: float = 0.9,
    min_size: int = 0,
):
    """Segment by cutting high-dissimilarity edges of the patch neighbour-graph.

    Every patch starts connected to its neighbours (same ``connectivity`` /
    ``radii`` offsets as :func:`feature_edge_map`). A connection is **cut** when
    the feature distance across it exceeds the threshold — these are the
    boundaries. The connected components of what survives are the masks.

    Parameters
    ----------
    threshold:
        Absolute distance cut-off (cut when ``dist > threshold``). If ``None``,
        it is taken as the ``quantile`` of all edge distances — metric-agnostic,
        so it ports across cosine/euclidean and layers (cf. greedy_graph).
    quantile:
        Used only when ``threshold is None``. ``0.9`` cuts the strongest 10% of
        edges (the clearest boundaries).
    min_size:
        Final filter: connected components with fewer than this many patches are
        dropped to background (label 0). ``0`` keeps every component. Label 0 is
        always reserved for background, so kept masks are numbered ``1..K`` —
        this is what lets the grow/merge stages treat 0 as leftover space.

    Returns ``(labels, cut_value, n_masks, n_filtered)`` where ``labels`` is an
    ``(H, W)`` long tensor (0 = background), ``cut_value`` is the distance
    threshold actually applied, ``n_masks`` the number of kept components and
    ``n_filtered`` how many small components the ``min_size`` filter removed.
    """
    base = _FORWARD_8 if connectivity == 8 else _FORWARD_4
    radii = [int(r) for r in radii if int(r) > 0] or [1]
    offsets = [(dy * r, dx * r) for r in radii for dy, dx in base]

    f = feat.float()
    h, w, _ = f.shape
    idx = torch.arange(h * w).reshape(h, w)
    src, dst, dists = [], [], []
    for dy, dx in offsets:                       # dy >= 0 by construction
        if dy >= h or abs(dx) >= w:
            continue
        ya0, ya1 = 0, h - dy
        xa0, xa1 = (0, w - dx) if dx >= 0 else (-dx, w)        # no wrap-around
        a = f[ya0:ya1, xa0:xa1]
        b = f[ya0 + dy:ya1 + dy, xa0 + dx:xa1 + dx]
        src.append(idx[ya0:ya1, xa0:xa1].reshape(-1))
        dst.append(idx[ya0 + dy:ya1 + dy, xa0 + dx:xa1 + dx].reshape(-1))
        dists.append(_pair_distance(a, b, metric).reshape(-1))

    src = torch.cat(src)
    dst = torch.cat(dst)
    dists = torch.cat(dists)
    cut = float(threshold) if threshold is not None else float(torch.quantile(dists, quantile))

    keep = dists <= cut                          # surviving connections
    uf = UnionFind(h * w)
    for a_i, b_i in zip(src[keep].tolist(), dst[keep].tolist()):
        uf.union(a_i, b_i)

    # label 0 is always background; min_size>=1 keeps everything (drops nothing),
    # larger values send small components to background as leftover space
    min_size = max(int(min_size), 1)
    sizes: dict[int, int] = {}
    for i in range(h * w):
        r = uf.find(i)
        sizes[r] = sizes.get(r, 0) + 1
    total = len(sizes)
    n_masks = sum(1 for s in sizes.values() if s >= min_size)
    n_filtered = total - n_masks

    labels = relabel(uf, h, w, min_size=min_size)
    return labels, cut, n_masks, n_filtered


_METRIC_TO_SIM = {"cosine": "cosine", "euclidean": "neg_euclidean"}


def _merge_candidate_sims(ext, grid: torch.Tensor, labels: torch.Tensor,
                          merge_mode: str) -> torch.Tensor:
    """The similarities the chosen merger would test — for a quantile threshold.

    Mirrors the pairing each ``_merge_similar`` uses (whole-mask prototypes vs.
    adjacent shared-seam prototypes), reusing the extractor's own ``sim`` /
    ``_dilate_within`` so the values match exactly what the merge step compares.
    """
    k = int(labels.max())
    if k <= 1:
        return torch.empty(0)
    h, w, c = grid.shape
    flat = grid.reshape(h * w, c)

    if merge_mode != "boundary":                      # whole-mask prototype pairs
        from mask_extraction.similarity import pairwise
        protos = torch.stack([grid[labels == lab].mean(0) for lab in range(1, k + 1)])
        aff = pairwise(protos, metric=ext.sim)        # (K, K)
        iu = torch.triu_indices(k, k, offset=1)
        return aff[iu[0], iu[1]]

    from mask_extraction.greedy_multistage import _NEIGHBOURS_4, _NEIGHBOURS_8
    nb = _NEIGHBOURS_8 if ext.connectivity == 8 else _NEIGHBOURS_4
    seam: dict[tuple[int, int], set[int]] = {}
    ys, xs = torch.where(labels > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        a = int(labels[y, x])
        for dy, dx in nb:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                b = int(labels[ny, nx])
                if b != 0 and b != a:
                    seam.setdefault((a, b), set()).add(y * w + x)
    rings = ext.seam_thickness - 1
    sims = []
    for i, j in {(min(a, b), max(a, b)) for (a, b) in seam}:
        si, sj = seam.get((i, j)), seam.get((j, i))
        if not si or not sj:
            continue
        bi = ext._dilate_within(si, i, labels, rings)
        bj = ext._dilate_within(sj, j, labels, rings)
        pi, pj = flat[list(bi)].mean(0), flat[list(bj)].mean(0)
        sims.append(float(ext.sim(pi.view(1, c), pj.view(1, c))))
    return torch.tensor(sims)


def grow_and_merge(
    feat: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "cosine",
    connectivity: int = 4,
    merge_threshold: float = 0.85,
    merge_mode: str = "boundary",
    seam_thickness: int = 1,
    merge_by: str = "absolute",
    merge_quantile: float = 0.8,
):
    """Apply greedy_multistage stages 2 & 3 on top of existing mask cores.

    * **Stage 2 — grow**: hand the leftover (label-0) patches to the cores, one
      best-first patch at a time, by feature similarity to each core prototype.
    * **Stage 3 — merge**: union cores into top-level hierarchies. Two variants:

      - ``"boundary"`` (greedy_multistage2): only merges **spatially adjacent**
        masks, judged by the features in a band along their shared seam
        (``seam_thickness`` patches wide on each side). Local; avoids fusing
        masks that merely have similar averages but never touch.
      - ``"prototype"`` (greedy_multistage): merges any pair whose **whole-mask**
        prototypes are similar beyond ``merge_threshold``, regardless of adjacency.

    The merge cut-off is either ``"absolute"`` (``merge_threshold`` on the
    similarity scale) or ``"quantile"``: the ``merge_quantile`` of the candidate
    merge similarities for *this* image, so it adapts across images and metrics.
    ``merge_quantile=0.8`` keeps the top 20% most-similar pairs; higher = fewer
    merges, matching the absolute knob's direction.

    Both stages are delegated to the real ``mask_extraction`` extractors so the
    behaviour matches the main pipeline exactly. The distance ``metric`` is mapped
    to the matching similarity (cosine -> ``cosine``, euclidean -> ``neg_euclidean``).

    Returns ``(grown, merged, n_grown, n_merged, merge_cut)`` where ``merge_cut``
    is the similarity threshold actually applied.
    """
    from mask_extraction.greedy_multistage import GreedyMultistageExtractor
    from mask_extraction.greedy_multistage2 import GreedyMultistage2Extractor

    sim = _METRIC_TO_SIM.get(metric, "cosine")
    common = dict(similarity=sim, connectivity=int(connectivity),
                  merge_threshold=float(merge_threshold))
    if merge_mode == "boundary":
        ext = GreedyMultistage2Extractor(seam_thickness=int(seam_thickness), **common)
    else:
        ext = GreedyMultistageExtractor(**common)
    f = feat.float()
    grown = ext._distribute(f, labels)            # stage 2 (shared by both)

    if merge_by == "quantile":
        sims = _merge_candidate_sims(ext, f, grown, merge_mode)
        # merge the top (1 - q) most-similar pairs; empty -> nothing to merge
        ext.merge_threshold = (float(torch.quantile(sims, float(merge_quantile)))
                               if sims.numel() else float("inf"))

    merged = ext._merge_similar(f, grown)         # stage 3 (variant differs)
    return grown, merged, int(grown.max()), int(merged.max()), float(ext.merge_threshold)


def to_greyscale(edge: torch.Tensor, q: float = 0.02, gamma: float = 1.0) -> np.ndarray:
    """Stretch an edge map to a ``(H, W)`` uint8 greyscale array.

    ``q`` is a robust percentile clip: the ``[q, 1 - q]`` quantiles map to
    black/white, taming outlier patches. ``q = 0`` falls back to raw min/max.

    ``gamma`` is a contrast/sharpness curve on the normalised map: ``> 1``
    pushes weak edges toward black while keeping strong ones bright (visually
    "sharper", thinner-looking boundaries); ``< 1`` lifts faint edges; ``1`` is
    a no-op. Display-only — it does not affect the graph-cut segmentation.
    """
    e = edge.float().flatten()
    if q > 0:
        lo = torch.quantile(e, q)
        hi = torch.quantile(e, 1.0 - q)
    else:
        lo, hi = e.min(), e.max()
    g = ((edge - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    if gamma != 1.0:
        g = g.pow(float(gamma))
    return (g * 255).to(torch.uint8).cpu().numpy()
