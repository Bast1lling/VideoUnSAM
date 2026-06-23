"""
Greedy neighbour-graph connected-components mask extractor.

Algorithm
---------
1. View the ``(H, W)`` feature grid as a graph with one node per patch and
   **no edges**.
2. For every pair of neighbouring patches (4- or 8-connectivity), measure the
   similarity of their features.
3. Greedily add an edge whenever that similarity is "very similar", i.e. at or
   above a threshold. Edges are processed in descending similarity order, so the
   strongest links are committed first (this ordering is what makes the rule
   easy to extend with a stopping criterion later -- e.g. stop at K components).
4. The connected components of the resulting graph are the masks.

The similarity metric is pluggable (see ``similarity.py``); the threshold can be
given absolutely (``threshold=``) or, more robustly across metrics, as a
quantile of the observed neighbour-edge similarities (``quantile=``).
"""
from __future__ import annotations

from typing import Callable

import torch

from .base import MaskExtractor, UnionFind, neighbour_edges, register_extractor, relabel
from .similarity import get_similarity


@register_extractor("greedy_graph")
class GreedyGraphExtractor(MaskExtractor):
    """Connect very-similar neighbours, then label connected components.

    Parameters
    ----------
    similarity:
        Metric name (see ``similarity.SIMILARITIES``) or a callable
        ``f(a, b) -> sim``. Default ``"cosine"`` (DINOv3's native metric).
    threshold:
        Absolute similarity cut-off; an edge is added iff ``sim >= threshold``.
        Mutually exclusive with ``quantile``.
    quantile:
        If ``threshold`` is ``None``, derive the cut-off as this quantile of all
        neighbour-edge similarities (e.g. ``0.5`` keeps the most-similar half of
        edges). Metric-agnostic, which is handy since cosine/dot/euclidean live
        on very different scales.
    connectivity:
        4 (right/down) or 8 (adds diagonals).
    min_size:
        Components smaller than this many patches are merged into background
        (label 0). ``0`` keeps everything.
    sim_kwargs:
        Extra kwargs bound to the similarity metric (e.g. ``{"gamma": 0.01}``).
    """

    def __init__(
        self,
        similarity: str | Callable = "cosine",
        threshold: float | None = None,
        quantile: float = 0.5,
        connectivity: int = 4,
        min_size: int = 0,
        sim_kwargs: dict | None = None,
    ):
        self.similarity_name = similarity if isinstance(similarity, str) else "custom"
        self.sim = get_similarity(similarity, **(sim_kwargs or {}))
        self.threshold = threshold
        self.quantile = quantile
        self.connectivity = connectivity
        self.min_size = min_size

    @torch.no_grad()
    def extract(self, grid: torch.Tensor) -> torch.Tensor:
        h, w, _ = grid.shape
        grid = grid.float()
        edges, sims = neighbour_edges(grid, self.sim, self.connectivity)

        if self.threshold is not None:
            thr = float(self.threshold)
        else:
            thr = float(torch.quantile(sims, self.quantile))

        # greedy: commit strongest edges first, stop once below threshold
        order = torch.argsort(sims, descending=True)
        uf = UnionFind(h * w)
        edges_o, sims_o = edges[order], sims[order]
        for k in range(edges_o.shape[0]):
            if sims_o[k] < thr:
                break
            a, b = edges_o[k]
            uf.union(int(a), int(b))

        return relabel(uf, h, w, self.min_size)
