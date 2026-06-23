"""
Frontier region-growing mask extractor.

Idea
----
1. **Rough guesses (seeds).** Connect neighbouring patches whose feature
   similarity is *very* high (a high quantile, e.g. 0.8) and take the connected
   components -- small, confident cores of coherent regions.
2. **Prototype per guess.** Summarise each seed by one feature vector (mean or
   element-wise max of its patch features).
3. **Grow each guess to its boundary.** Flood out from the seed by BFS. At each
   step the *frontier* is the thin ring of pixels just outside the current
   region. A frontier pixel is absorbed unless stepping out to it causes a
   **harsh drop** in similarity-to-prototype relative to its interior neighbour:

   ::

       drop(f) = sim(interior_neighbour, proto) − sim(f, proto)
       absorb f   iff   drop(f) <= boundary_delta
       else mark f as boundary

   The key signal is the *rate of change* of similarity as you step from inside
   the object to the frontier, not the absolute similarity -- a real object
   boundary shows a steep drop, the interior is flat. Growth repeats until every
   frontier pixel is a boundary pixel (nothing left to absorb).

Seeds are grown largest-first and claim pixels exclusively, so the output is a
partition: label 0 is background (never reached), labels ``1..K`` are the grown
masks ordered by seed size.

The similarity metric is pluggable (``similarity.py``). Because only the
*gradient* of similarity matters, the prototype-similarity map is min-max
normalised per seed by default, which makes ``boundary_delta`` an interpretable
fraction of dynamic range and roughly metric-agnostic.
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


@register_extractor("frontier_grow")
class FrontierGrowExtractor(MaskExtractor):
    """Seed at high similarity, then grow until similarity gradients turn harsh.

    Parameters
    ----------
    similarity:
        Metric name or callable ``f(a, b) -> sim`` (default ``"cosine"``).
    seed_quantile:
        Quantile of neighbour-edge similarities used to form the rough seeds
        (step 1). High (e.g. 0.8) → small, confident cores.
    seed_min_size:
        Discard *seeds* smaller than this many patches (drops singleton noise).
    min_mask_size:
        Discard *final grown masks* smaller than this many patches (sent to
        background). Independent of ``seed_min_size``: a large seed can still grow
        into a small mask and vice versa.
    prototype:
        How to summarise a seed's features (step 2): ``"mean"`` or ``"max"``.
    boundary_delta:
        Max allowed drop in (normalised) similarity-to-prototype when stepping
        from an interior pixel to a frontier pixel; larger drops mark a boundary
        (step 3). With ``normalize_sim`` this is a fraction of dynamic range.
    normalize_sim:
        Min-max normalise each seed's prototype-similarity map to ``[0, 1]``
        (robust 2/98 percentiles) before measuring drops. Makes
        ``boundary_delta`` interpretable and metric-agnostic.
    connectivity:
        4 (von Neumann) or 8 (Moore) for both seeding and growth.
    max_iter:
        Safety cap on growth iterations per seed (default ``H*W``).
    sim_kwargs:
        Extra kwargs bound to the metric (e.g. ``{"gamma": 0.01}`` for ``rbf``).
    """

    def __init__(
        self,
        similarity: str | Callable = "cosine",
        seed_quantile: float = 0.8,
        seed_min_size: int = 2,
        min_mask_size: int = 0,
        prototype: str = "mean",
        boundary_delta: float = 0.15,
        normalize_sim: bool = True,
        connectivity: int = 4,
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
        self.max_iter = max_iter

    # ------------------------------------------------------------ step 1: seeds
    def _seeds(self, grid: torch.Tensor) -> list[torch.Tensor]:
        h, w, _ = grid.shape
        edges, sims = neighbour_edges(grid, self.sim, self.connectivity)
        thr = float(torch.quantile(sims, self.seed_quantile))
        uf = UnionFind(h * w)
        for a, b in edges[sims >= thr].tolist():
            uf.union(a, b)
        comps: dict[int, list[int]] = defaultdict(list)
        for i in range(h * w):
            comps[uf.find(i)].append(i)
        seeds = [torch.tensor(v) for v in comps.values() if len(v) >= self.seed_min_size]
        seeds.sort(key=lambda t: -t.numel())
        return seeds

    # -------------------------------------------------------- step 2: prototype
    def _prototype(self, feats: torch.Tensor) -> torch.Tensor:
        return feats.max(dim=0).values if self.prototype == "max" else feats.mean(dim=0)

    # ----------------------------------------------------------------- assemble
    @torch.no_grad()
    def extract(self, grid: torch.Tensor) -> torch.Tensor:
        h, w, c = grid.shape
        grid = grid.float()
        flat = grid.reshape(h * w, c)
        labels = torch.zeros(h, w, dtype=torch.long)
        assigned = torch.zeros(h, w, dtype=torch.bool)

        for li, pix in enumerate(self._seeds(grid), start=1):
            seed_mask = torch.zeros(h * w, dtype=torch.bool)
            seed_mask[pix] = True
            seed_mask = seed_mask.reshape(h, w) & ~assigned
            if not seed_mask.any():
                continue

            proto = self._prototype(flat[pix])                                   # step 2
            s = prototype_similarity_map(grid, proto, self.sim, self.normalize_sim)
            region = grow_region(s, seed_mask, assigned, self.boundary_delta,    # step 3
                                 self.connectivity, self.max_iter)
            labels[region] = li
            assigned |= region

        return filter_small_masks(labels, self.min_mask_size)
