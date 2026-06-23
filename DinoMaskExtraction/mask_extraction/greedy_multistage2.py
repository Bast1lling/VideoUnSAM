"""
greedy_multistage variant with a *boundary-aware* merge stage.

Identical to :class:`GreedyMultistageExtractor` for stage 1 (cores) and stage 2
(distribute leftovers); only stage 3 — the merge — differs.

The base merge compares whole-mask prototypes and fuses any pair similar beyond
``merge_threshold`` regardless of where they sit in the image. This variant is
strictly local:

* it only ever merges **spatially adjacent** masks (two masks that share a
  border), and
* it judges a pair by the features in a **band along their shared seam**, not by
  the bulk of each mask. For an adjacent pair it collects each side's seam band,
  mean-pools each side into a *boundary prototype*, and merges when those two
  boundary prototypes are similar beyond ``merge_threshold``.

The band width is set by ``seam_thickness``: ``1`` keeps only the pixels directly
on the border (one patch thick, the original behaviour), ``2`` also includes the
next ring inward on each side, and so on — each extra unit dilates the seam by one
more ring **within its own mask**. This trades context for locality: too thin and
the decision is noisy (few pixels), too thick and it drifts back toward the
whole-mask average that the plain ``greedy_multistage`` merge already uses.

Rationale: two patches belonging to the same surface tend to look alike *where
they meet* even when their global means have drifted (lighting, sub-parts).
Comparing a seam band rather than the whole mask keeps the decision local and
avoids fusing masks that merely have similar averages but never actually touch.
"""
from __future__ import annotations

import torch

from .base import UnionFind, register_extractor
from .greedy_multistage import (
    _NEIGHBOURS_4,
    _NEIGHBOURS_8,
    GreedyMultistageExtractor,
)


@register_extractor("greedy_multistage2")
class GreedyMultistage2Extractor(GreedyMultistageExtractor):
    """Like :class:`GreedyMultistageExtractor` but with a boundary-seam merge.

    All base parameters are inherited unchanged; ``merge_threshold`` now applies to
    the similarity of the two *boundary* prototypes of an adjacent mask pair rather
    than to their whole-mask prototypes.

    Parameters
    ----------
    seam_thickness:
        Width (in patches) of the band sampled on each side of a shared border.
        ``1`` = only the border-touching pixels (default); larger values dilate the
        band inward by that many rings within each mask, giving the merge test more
        context at the cost of locality.
    """

    def __init__(self, *args, seam_thickness: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.seam_thickness = max(int(seam_thickness), 1)

    def _dilate_within(self, seeds: set[int], label_val: int, labels: torch.Tensor,
                       rings: int) -> set[int]:
        """Grow ``seeds`` by ``rings`` neighbourhood steps, staying inside ``label_val``."""
        if rings <= 0:
            return seeds
        h, w = labels.shape
        nb = _NEIGHBOURS_8 if self.connectivity == 8 else _NEIGHBOURS_4
        band = set(seeds)
        frontier = set(seeds)
        for _ in range(rings):
            nxt: set[int] = set()
            for idx in frontier:
                y, x = divmod(idx, w)
                for dy, dx in nb:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        nidx = ny * w + nx
                        if nidx not in band and int(labels[ny, nx]) == label_val:
                            nxt.add(nidx)
            band |= nxt
            frontier = nxt
            if not frontier:
                break
        return band

    def _merge_similar(self, grid: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Stage 3: merge adjacent masks whose shared seam band looks the same.

        For every ordered adjacent pair ``(a, b)`` we gather the ``a``-side pixels
        within ``seam_thickness`` of the border with ``b``; mean-pooling those gives
        ``a``'s boundary prototype against ``b``. A pair is merged when its two
        boundary prototypes are at least ``merge_threshold`` similar.
        """
        k = int(labels.max())
        if k <= 1:
            return labels.clone()
        h, w, c = grid.shape
        nb = _NEIGHBOURS_8 if self.connectivity == 8 else _NEIGHBOURS_4

        # seam[(a, b)] = flat indices of a-side patch pixels adjacent to label b
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

        rings = self.seam_thickness - 1
        flat = grid.reshape(h * w, c)
        uf = UnionFind(k)
        pairs = {(min(a, b), max(a, b)) for (a, b) in seam}
        for i, j in pairs:
            side_i = seam.get((i, j))
            side_j = seam.get((j, i))
            if not side_i or not side_j:                      # need a seam on both sides
                continue
            band_i = self._dilate_within(side_i, i, labels, rings)
            band_j = self._dilate_within(side_j, j, labels, rings)
            proto_i = flat[list(band_i)].mean(0)
            proto_j = flat[list(band_j)].mean(0)
            sim = float(self.sim(proto_i.view(1, c), proto_j.view(1, c)))
            if sim >= self.merge_threshold:
                uf.union(i - 1, j - 1)

        remap: dict[int, int] = {}
        out = torch.zeros_like(labels)
        for lab in range(1, k + 1):
            root = uf.find(lab - 1)
            if root not in remap:
                remap[root] = len(remap) + 1
            out[labels == lab] = remap[root]
        return out
