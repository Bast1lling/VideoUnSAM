"""
Two-stage coarse ▸ fine mask pipeline (greedy_multistage2 for both levels).

A *coarse* extractor splits the image into object-level masks, then a *fine*
extractor runs inside each coarse mask (restricted to its patches) to split it
into parts. The part labels of every coarse mask are concatenated into one global
label map; the per-coarse-mask intermediates are kept so the app can show each
object's sub-segmentation zoomed in.

The two passes may run on **different feature grids** — e.g. heavily-reduced,
smoothed features (256 PCs, σ=0.5) for the coarse split and a higher-resolution,
lightly-smoothed grid (1024-d, more patches) for the fine split. The grids may
differ in channel count **and spatial size**: the coarse label map is upsampled
(nearest) to the fine grid before conquering, so a higher-resolution fine pass
yields finer part boundaries. :meth:`HierPipeline.run` takes one grid per stage.
"""
from __future__ import annotations

from typing import Callable

import torch

from .base import UnionFind
from .greedy_multistage import _NEIGHBOURS_4, _NEIGHBOURS_8, GreedyMultistageExtractor


def _resize_labels(labels: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Nearest-neighbour resize an int ``(H, W)`` label map to ``(h, w)``."""
    if tuple(labels.shape) == (h, w):
        return labels
    t = labels.float().view(1, 1, *labels.shape)
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="nearest")
    return t.view(h, w).long()


def _merge_adjacent_similar(grid: torch.Tensor, labels: torch.Tensor, sim: Callable,
                            threshold: float, connectivity: int = 8) -> torch.Tensor:
    """Merge **spatially-adjacent** masks whose mean prototypes are ≥ ``threshold`` similar.

    Like :meth:`GreedyMultistageExtractor._merge_similar` (whole-mask prototype
    comparison) but restricted to masks that actually touch. Run on the *final*
    label map across the whole image, it repairs slivers that one coarse mask stole
    from a neighbour (because of smoothing) and that the fine stage then re-detected
    as their own part: such a sliver is adjacent to — and feature-similar to — the
    mask it truly belongs to, so it rejoins it. Non-touching look-alikes are left
    alone.
    """
    k = int(labels.max())
    if k <= 1:
        return labels
    h, w, c = grid.shape
    protos = torch.stack([grid[labels == lab].mean(0) for lab in range(1, k + 1)])
    nb = _NEIGHBOURS_8 if connectivity == 8 else _NEIGHBOURS_4
    pairs: set[tuple[int, int]] = set()
    ys, xs = torch.where(labels > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        a = int(labels[y, x])
        for dy, dx in nb:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                b = int(labels[ny, nx])
                if b != 0 and b != a:
                    pairs.add((min(a, b), max(a, b)))
    uf = UnionFind(k)
    for a, b in pairs:
        s = float(sim(protos[a - 1].view(1, c), protos[b - 1].view(1, c)))
        if s >= threshold:
            uf.union(a - 1, b - 1)
    remap: dict[int, int] = {}
    out = torch.zeros_like(labels)
    for lab in range(1, k + 1):
        root = uf.find(lab - 1)
        if root not in remap:
            remap[root] = len(remap) + 1
        out[labels == lab] = remap[root]
    return out


def _conquer_all(grid: torch.Tensor, divide_labels: torch.Tensor,
                 parts: GreedyMultistageExtractor) -> tuple[list, torch.Tensor, dict]:
    """Run ``parts`` inside each divide mask; build flat labels + hierarchy.

    Returns ``(conquer, final, hierarchy)`` where ``conquer`` is a per-divide-mask
    list of ``{divide_id, region, stages, parts}``, ``final`` is the global flat
    label map and ``hierarchy`` maps each divide id to its global part ids. ``grid``
    is the **fine** feature grid (it only drives the per-region subdivision).
    """
    final = torch.zeros_like(divide_labels)
    conquer: list[dict] = []
    hierarchy: dict[int, list[int]] = {}
    nxt = 0
    for k in range(1, int(divide_labels.max()) + 1):
        region = divide_labels == k
        if not region.any():
            continue
        stages = parts.extract_stages(grid, region)       # region-restricted sub-stages
        part_labels = stages[-1][1]
        ids = [p for p in part_labels.unique().tolist() if p != 0]
        gids: list[int] = []
        if not ids:                                       # indivisible -> keep whole
            nxt += 1
            final[region] = nxt
            gids = [nxt]
        else:
            for p in ids:
                nxt += 1
                final[part_labels == p] = nxt
                gids.append(nxt)
            leftover = region & (final == 0)              # disconnected island, if any
            if leftover.any():
                nxt += 1
                final[leftover] = nxt
                gids.append(nxt)
        conquer.append({"divide_id": k, "region": region,
                        "stages": stages, "parts": part_labels})
        hierarchy[k] = gids
    return conquer, final, hierarchy


class HierPipeline:
    """Coarse extractor (on the coarse grid) → per-mask fine extractor (on the fine grid)."""

    def __init__(self, divide: GreedyMultistageExtractor, parts: GreedyMultistageExtractor,
                 final_merge_threshold: float | None = None):
        self.divide = divide
        self.parts = parts
        # if set, a whole-image pass that merges adjacent, feature-similar final masks
        self.final_merge_threshold = final_merge_threshold

    @torch.no_grad()
    def run(self, coarse_grid: torch.Tensor, fine_grid: torch.Tensor | None = None) -> dict:
        coarse_grid = coarse_grid.float()
        fine_grid = coarse_grid if fine_grid is None else fine_grid.float()
        divide_stages = self.divide.extract_stages(coarse_grid)
        # upsample the coarse labels onto the (possibly higher-res) fine grid
        fh, fw = fine_grid.shape[:2]
        divide_labels = _resize_labels(divide_stages[-1][1], fh, fw)
        conquer, final, hierarchy = _conquer_all(fine_grid, divide_labels, self.parts)
        n_premerge = int((final.unique() > 0).sum())
        if self.final_merge_threshold is not None:
            # final cleanup: rejoin adjacent parts (across coarse-mask borders) whose
            # features match — repairs slivers stolen between neighbouring coarse masks
            final = _merge_adjacent_similar(
                fine_grid, final, self.parts.sim, self.final_merge_threshold,
                self.parts.connectivity)
        return {
            "divide_stages": divide_stages,
            "divide": divide_labels,
            "conquer": conquer,
            "final": final,
            "hierarchy": hierarchy,
            "n_divide": int(divide_labels.max()),
            "n_parts": int((final.unique() > 0).sum()),
            "n_premerge": n_premerge,
        }

    @torch.no_grad()
    def extract(self, coarse_grid: torch.Tensor, fine_grid: torch.Tensor | None = None
                ) -> torch.Tensor:
        return self.run(coarse_grid, fine_grid)["final"]
