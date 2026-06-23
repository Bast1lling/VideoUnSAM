"""
Sequential best-first region-growing mask extractor.

Where ``frontier_grow`` seeds once and grows every seed, this variant works one
mask at a time and re-seeds on what is left:

1. **Patch guesses.** On the *remaining* (unassigned) pixels, connect very
   similar neighbours (high ``seed_quantile``) and take the connected
   components -- the candidate cores.
2. **Pick the most consistent guess.** Score each candidate by its internal
   feature consistency (mean similarity of its patches to their prototype) and
   take the best one -- the most confident mask candidate right now.
3. **Finalise it.** Grow that candidate with the same frontier rule as
   ``frontier_grow`` (absorb a frontier pixel unless similarity-to-prototype
   drops harshly across the step), then **cut the resulting mask out** of the
   image so later iterations cannot enter it.
4. **Repeat** on the shrinking set of remaining pixels until none are left, no
   valid seed remains, or ``max_masks`` is reached.

Output: label 0 = background/unreached, ``1..K`` = masks in the order they were
finalised (so label 1 is the most-consistent region found first).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import torch

from .base import (
    MaskExtractor, UnionFind, filter_small_masks, grow_region, neighbour_edges,
    prototype_similarity_map, register_extractor,
)
from .similarity import get_similarity


@register_extractor("sequential_grow")
class SequentialGrowExtractor(MaskExtractor):
    """Iteratively extract the most-consistent region, then remove it.

    Parameters
    ----------
    similarity:
        Metric name or callable (default ``"cosine"``). Also used to score
        candidate consistency, so a bounded metric like cosine is recommended.
    seed_quantile:
        Quantile of (remaining) neighbour-edge similarities used to form the
        candidate guesses each iteration.
    seed_min_size:
        Ignore candidate *guesses* smaller than this (avoids trivially "perfect"
        tiny seeds winning the consistency ranking).
    min_mask_size:
        Discard *final grown masks* smaller than this many patches (sent to
        background) after all iterations finish.
    prototype:
        Seed summary for both consistency scoring and growth: ``"mean"``/``"max"``.
    boundary_delta:
        Max (normalised) similarity drop per step before a frontier pixel is a
        boundary (passed to the shared region grower).
    normalize_sim:
        Min-max normalise each candidate's prototype-similarity map before
        growth (keeps ``boundary_delta`` interpretable).
    connectivity:
        4 or 8, for seeding and growth.
    max_masks:
        Stop after this many masks (``None`` = until pixels run out).
    max_iter:
        Per-region growth iteration cap (default grid size).
    sim_kwargs:
        Extra kwargs bound to the metric (e.g. ``{"gamma": 0.01}``).
    """

    def __init__(
        self,
        similarity: str | Callable = "cosine",
        seed_quantile: float = 0.8,
        seed_min_size: int = 4,
        min_mask_size: int = 0,
        prototype: str = "mean",
        boundary_delta: float = 0.15,
        normalize_sim: bool = True,
        connectivity: int = 4,
        max_masks: int | None = None,
        max_iter: int | None = None,
        sim_kwargs: dict | None = None,
    ):
        if prototype not in ("mean", "max"):
            raise ValueError("prototype must be 'mean' or 'max'")
        self.similarity_name = similarity if isinstance(similarity, str) else "custom"
        self.sim = get_similarity(similarity, **(sim_kwargs or {}))
        self.min_mask_size = min_mask_size
        self.seed_quantile = seed_quantile
        self.seed_min_size = seed_min_size
        self.prototype = prototype
        self.boundary_delta = boundary_delta
        self.normalize_sim = normalize_sim
        self.connectivity = connectivity
        self.max_masks = max_masks
        self.max_iter = max_iter

    def _prototype(self, feats: torch.Tensor) -> torch.Tensor:
        return feats.max(dim=0).values if self.prototype == "max" else feats.mean(dim=0)

    def _consistency(self, feats: torch.Tensor, proto: torch.Tensor) -> float:
        """Mean similarity of a candidate's patches to its prototype."""
        return float(self.sim(feats, proto.unsqueeze(0)).mean())

    def _candidates(self, edges, sims, available, h, w) -> list[torch.Tensor]:
        """Connected components (>= seed_min_size) among the available pixels."""
        avail = available.reshape(-1)
        keep = avail[edges[:, 0]] & avail[edges[:, 1]]
        e, s = edges[keep], sims[keep]
        if e.numel() == 0:
            return []
        thr = float(torch.quantile(s, self.seed_quantile))
        uf = UnionFind(h * w)
        for a, b in e[s >= thr].tolist():
            uf.union(a, b)
        comps: dict[int, list[int]] = defaultdict(list)
        for i in avail.nonzero().flatten().tolist():
            comps[uf.find(i)].append(i)
        return [torch.tensor(v) for v in comps.values() if len(v) >= self.seed_min_size]

    @torch.no_grad()
    def extract(self, grid: torch.Tensor) -> torch.Tensor:
        h, w, c = grid.shape
        grid = grid.float()
        flat = grid.reshape(h * w, c)
        labels = torch.zeros(h, w, dtype=torch.long)
        assigned = torch.zeros(h, w, dtype=torch.bool)

        # neighbour-edge similarities are fixed (features don't change) -> once
        edges, sims = neighbour_edges(grid, self.sim, self.connectivity)
        cap = self.max_masks if self.max_masks is not None else (h * w)

        li = 0
        while li < cap:
            cands = self._candidates(edges, sims, ~assigned, h, w)
            if not cands:
                break

            # step 2: pick the most internally consistent candidate
            best_pix, best_proto, best_key = None, None, (-float("inf"), -1)
            for pix in cands:
                proto = self._prototype(flat[pix])
                key = (self._consistency(flat[pix], proto), pix.numel())  # size breaks ties
                if key > best_key:
                    best_key, best_pix, best_proto = key, pix, proto

            # step 3: finalise via region growing, then cut it out
            seed_mask = torch.zeros(h * w, dtype=torch.bool)
            seed_mask[best_pix] = True
            seed_mask = seed_mask.reshape(h, w) & ~assigned
            s = prototype_similarity_map(grid, best_proto, self.sim, self.normalize_sim)
            region = grow_region(s, seed_mask, assigned, self.boundary_delta,
                                 self.connectivity, self.max_iter)
            if not region.any():                       # safety: at least claim the seed
                region = seed_mask
                if not region.any():
                    break

            li += 1
            labels[region] = li
            assigned |= region

        return filter_small_masks(labels, self.min_mask_size)
