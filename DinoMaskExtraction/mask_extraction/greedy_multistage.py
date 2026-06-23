"""
Three-stage greedy-graph mask extractor.

A richer variant of :mod:`greedy_graph`. The plain greedy cut is good at finding
coherent cores but leaves a lot of patches unassigned (filtered as small/noisy)
and over-segments one object into several patches. This adds two clean-up stages
on top:

Stage 1 — **seed cores.** Run the greedy neighbour-graph cut to get connected
components, then drop the small ones (``min_mask_size``) and, optionally, the
*internally inconsistent* ones (patches whose member features disagree with their
own mean beyond ``consistency_min``). What survives are trustworthy mask cores;
everything else becomes unlabelled (0).

Stage 2 — **distribute leftovers.** The pixels dropped in stage 1 are handed out
to the surviving cores by **one-pixel-at-a-time** region growing: across *all*
cores at once, repeatedly absorb the single frontier pixel whose feature is most
similar to the core it touches, then recompute frontiers. Growing one pixel per
step (globally best-first, not per-core) means assignment is driven by feature
similarity rather than by which core happened to reach a pixel first.

Stage 3 — **merge duplicates.** Cores that describe the same thing (e.g. two
halves of one object split by the greedy cut) are fused: recompute a prototype
per mask and union any pair whose prototypes are similar beyond
``merge_threshold``.

Because the intermediate states are informative, the extractor exposes
:meth:`extract_stages`, which returns the label map after each stage so the app
can show the full progression rather than only the final masks.
"""
from __future__ import annotations

import heapq
from typing import Callable

import torch

from .base import (
    MaskExtractor,
    UnionFind,
    neighbour_edges,
    prototype_similarity_map,
    register_extractor,
    relabel,
)
from .similarity import get_similarity, pairwise

_NEIGHBOURS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBOURS_8 = _NEIGHBOURS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _renumber(labels: torch.Tensor) -> torch.Tensor:
    """Map nonzero labels to a contiguous ``1..K`` range, preserving order (0 = bg)."""
    out = torch.zeros_like(labels)
    new = 0
    for old in labels.unique().tolist():
        if old == 0:
            continue
        new += 1
        out[labels == old] = new
    return out


@register_extractor("greedy_multistage")
class GreedyMultistageExtractor(MaskExtractor):
    """Greedy cut → consistency filter → similarity-driven growth → semantic merge.

    Parameters
    ----------
    similarity:
        Metric name (see ``similarity.SIMILARITIES``) or a callable. Used for the
        greedy cut, the growth step, the consistency test and the merge test, so
        thresholds below live on this metric's scale (defaults assume cosine).
    threshold, quantile:
        Greedy edge cut-off (stage 1), as in :class:`GreedyGraphExtractor`:
        absolute ``threshold`` if given, else the ``quantile`` of neighbour-edge
        similarities.
    connectivity:
        4 or 8, shared by the cut, the growth frontier and component labelling.
    min_mask_size:
        Stage 1: cores with fewer patches than this become background.
    consistency_filter, consistency_min:
        Stage 1 (optional): drop a core if the mean similarity of its patches to
        their own prototype is below ``consistency_min`` (a core whose features
        disagree internally). Off by default.
    merge_threshold:
        Stage 3: union two masks whose prototypes are at least this similar.
    sim_kwargs:
        Extra kwargs bound to the metric (e.g. ``{"gamma": 0.01}``).
    """

    def __init__(
        self,
        similarity: str | Callable = "cosine",
        threshold: float | None = None,
        quantile: float = 0.6,
        connectivity: int = 4,
        min_mask_size: int = 4,
        consistency_filter: bool = False,
        consistency_min: float = 0.5,
        merge_threshold: float = 0.85,
        sim_kwargs: dict | None = None,
    ):
        self.similarity_name = similarity if isinstance(similarity, str) else "custom"
        self.sim = get_similarity(similarity, **(sim_kwargs or {}))
        self.threshold = threshold
        self.quantile = quantile
        self.connectivity = connectivity
        self.min_mask_size = min_mask_size
        self.consistency_filter = consistency_filter
        self.consistency_min = consistency_min
        self.merge_threshold = merge_threshold

    # --------------------------------------------------------------- public API
    @torch.no_grad()
    def extract(self, grid: torch.Tensor, region: torch.Tensor | None = None) -> torch.Tensor:
        return self._run(grid, region)[-1][1]

    @torch.no_grad()
    def extract_stages(self, grid: torch.Tensor, region: torch.Tensor | None = None
                       ) -> list[tuple[str, torch.Tensor]]:
        """Return ``[(title, labels), ...]`` after each stage (last is final).

        If ``region`` (a boolean ``(H, W)`` mask) is given, the algorithm operates
        **only** on the patches inside it: the greedy cut considers only in-region
        neighbour edges (so its quantile cut-off is computed over those edges
        alone), growth never leaves the region, and all out-of-region patches stay
        background (0). This is what lets the pipeline subdivide one object mask.
        """
        return self._run(grid, region)

    # ------------------------------------------------------------------- stages
    def _run(self, grid: torch.Tensor, region: torch.Tensor | None = None
             ) -> list[tuple[str, torch.Tensor]]:
        grid = grid.float()
        cores = self._greedy_cut(grid, region)
        if self.consistency_filter:
            cores = self._filter_inconsistent(grid, cores)
        grown = self._distribute(grid, cores, region)
        merged = self._merge_similar(grid, grown)
        return [
            ("stage 1 — cores (greedy + filter)", cores),
            ("stage 2 — grown (distributed)", grown),
            ("stage 3 — merged (final)", merged),
        ]

    def _greedy_cut(self, grid: torch.Tensor, region: torch.Tensor | None = None
                    ) -> torch.Tensor:
        """Stage 1 cut: connect very-similar neighbours, keep big-enough cores.

        When ``region`` is given, only neighbour edges with **both** endpoints in
        the region are considered, so the threshold/quantile is derived from the
        in-region similarity distribution only.
        """
        h, w, _ = grid.shape
        edges, sims = neighbour_edges(grid, self.sim, self.connectivity)
        if region is not None:
            rflat = region.reshape(-1)
            keep = rflat[edges[:, 0]] & rflat[edges[:, 1]]
            edges, sims = edges[keep], sims[keep]
        if sims.numel() == 0:
            return torch.zeros(h, w, dtype=torch.long)
        thr = (float(self.threshold) if self.threshold is not None
               else float(torch.quantile(sims, self.quantile)))
        order = torch.argsort(sims, descending=True)
        uf = UnionFind(h * w)
        edges_o, sims_o = edges[order], sims[order]
        for k in range(edges_o.shape[0]):
            if sims_o[k] < thr:
                break
            a, b = edges_o[k]
            uf.union(int(a), int(b))
        # min_size>=1 reserves 0 for filtered patches and labels cores 1..K
        labels = relabel(uf, h, w, min_size=max(self.min_mask_size, 1))
        if region is not None:                            # drop out-of-region singletons
            labels = _renumber(torch.where(region, labels, torch.zeros_like(labels)))
        return labels

    def _filter_inconsistent(self, grid: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Stage 1 (optional): drop cores whose member features disagree internally."""
        c = grid.shape[-1]
        out = labels.clone()
        for lab in labels.unique().tolist():
            if lab == 0:
                continue
            members = grid[labels == lab]                     # (n, C)
            proto = members.mean(0)
            consistency = float(self.sim(members, proto.view(1, c)).mean())
            if consistency < self.consistency_min:
                out[labels == lab] = 0
        return _renumber(out)

    def _distribute(self, grid: torch.Tensor, labels: torch.Tensor,
                    region: torch.Tensor | None = None) -> torch.Tensor:
        """Stage 2: hand out unlabelled pixels to cores, one best-first pixel at a time.

        Only pixels that are unlabelled *and* (if ``region`` is given) inside the
        region are eligible, so growth never escapes the object.
        """
        k = int(labels.max())
        avail = labels == 0
        if region is not None:
            avail = avail & region
        if k == 0 or not avail.any():
            return labels.clone()
        h, w, c = grid.shape
        nb = _NEIGHBOURS_8 if self.connectivity == 8 else _NEIGHBOURS_4

        # raw (un-normalised) similarity of every pixel to every core prototype,
        # so similarities are directly comparable across cores
        sims = torch.empty(k, h, w)
        for lab in range(1, k + 1):
            proto = grid[labels == lab].mean(0)
            sims[lab - 1] = prototype_similarity_map(grid, proto, self.sim,
                                                     normalize_sim=False)

        out = labels.clone()
        heap: list[tuple[float, int, int, int]] = []          # (-sim, y, x, core_idx)

        def eligible(y: int, x: int) -> bool:
            return out[y, x] == 0 and (region is None or bool(region[y, x]))

        def offer(y: int, x: int, core: int) -> None:
            heapq.heappush(heap, (-float(sims[core, y, x]), y, x, core))

        ys, xs = torch.where(out > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            core = int(out[y, x]) - 1
            for dy, dx in nb:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and eligible(ny, nx):
                    offer(ny, nx, core)

        while heap:
            _, y, x, core = heapq.heappop(heap)
            if out[y, x] != 0:
                continue                                      # already claimed
            out[y, x] = core + 1
            for dy, dx in nb:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and eligible(ny, nx):
                    offer(ny, nx, core)
        return out

    def _merge_similar(self, grid: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Stage 3: union masks whose prototypes exceed ``merge_threshold``."""
        k = int(labels.max())
        if k <= 1:
            return labels.clone()
        c = grid.shape[-1]
        protos = torch.stack([grid[labels == lab].mean(0) for lab in range(1, k + 1)])
        aff = pairwise(protos, metric=self.sim)               # (K, K)
        uf = UnionFind(k)
        for i in range(k):
            for j in range(i + 1, k):
                if float(aff[i, j]) >= self.merge_threshold:
                    uf.union(i, j)
        remap: dict[int, int] = {}
        out = torch.zeros_like(labels)
        for lab in range(1, k + 1):
            root = uf.find(lab - 1)
            if root not in remap:
                remap[root] = len(remap) + 1
            out[labels == lab] = remap[root]
        return out
