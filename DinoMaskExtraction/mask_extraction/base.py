"""
Base class + registry for feature-grid mask-extraction algorithms.

A mask extractor maps a dense feature grid ``(H, W, C)`` to an integer label map
``(H, W)`` where each value is a mask id (``0`` is reserved for background /
unassigned). Algorithms are decoupled from the backbone: anything that produces
a feature grid (any DINOv3 hook layer, or any other model) can be segmented.

Add a new algorithm by subclassing :class:`MaskExtractor`, implementing
:meth:`extract`, and decorating it with :func:`register_extractor`; it then
becomes available by name via :func:`get_extractor` and to the CLI in
``run.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn.functional as F

EXTRACTORS: dict[str, type["MaskExtractor"]] = {}


def register_extractor(name: str):
    def deco(cls: type["MaskExtractor"]) -> type["MaskExtractor"]:
        EXTRACTORS[name] = cls
        cls.name = name
        return cls
    return deco


def get_extractor(name: str, **kwargs) -> "MaskExtractor":
    if name not in EXTRACTORS:
        raise KeyError(f"unknown extractor '{name}'. Available: {sorted(EXTRACTORS)}")
    return EXTRACTORS[name](**kwargs)


class MaskExtractor(ABC):
    """Abstract callable that turns an ``(H, W, C)`` feature grid into masks."""

    name: str = "base"

    @abstractmethod
    def extract(self, grid: torch.Tensor) -> torch.Tensor:
        """Return an ``(H, W)`` ``long`` label map (mask id per patch)."""
        raise NotImplementedError

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        return self.extract(grid)


# ----------------------------------------------------------- shared utilities #
class UnionFind:
    """Union-find / disjoint-set with path compression and union by size."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def neighbour_edges(grid: torch.Tensor, sim: Callable, connectivity: int = 4
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Similarities of every adjacent-patch pair on the grid.

    Returns ``(edges, sims)`` where ``edges`` is ``(E, 2)`` of flat node indices
    (``y*W + x``) and ``sims`` is ``(E,)``. ``connectivity`` is 4 (right/down) or
    8 (adds the two diagonals).
    """
    h, w, _ = grid.shape
    idx = torch.arange(h * w).reshape(h, w)
    edge_list, sim_list = [], []

    def add(a_nodes, b_nodes, a_feat, b_feat):
        edge_list.append(torch.stack([a_nodes.reshape(-1), b_nodes.reshape(-1)], dim=1))
        sim_list.append(sim(a_feat, b_feat).reshape(-1))

    # right neighbour
    add(idx[:, :-1], idx[:, 1:], grid[:, :-1], grid[:, 1:])
    # down neighbour
    add(idx[:-1, :], idx[1:, :], grid[:-1, :], grid[1:, :])
    if connectivity == 8:
        add(idx[:-1, :-1], idx[1:, 1:], grid[:-1, :-1], grid[1:, 1:])   # down-right
        add(idx[:-1, 1:], idx[1:, :-1], grid[:-1, 1:], grid[1:, :-1])   # down-left

    return torch.cat(edge_list, dim=0), torch.cat(sim_list, dim=0)


def relabel(uf: UnionFind, h: int, w: int, min_size: int = 0) -> torch.Tensor:
    """Turn a populated union-find into a contiguous ``(H, W)`` label map.

    Components with fewer than ``min_size`` patches are assigned to background
    (label 0); kept components get labels ``1..K`` ordered by descending size.
    With ``min_size == 0`` every component is kept and labelled ``0..K-1``.
    """
    roots = [uf.find(i) for i in range(h * w)]
    sizes: dict[int, int] = {}
    for r in roots:
        sizes[r] = sizes.get(r, 0) + 1

    if min_size > 0:
        kept = sorted((r for r, s in sizes.items() if s >= min_size),
                      key=lambda r: -sizes[r])
        root_to_label = {r: i + 1 for i, r in enumerate(kept)}   # 0 = background
        default = 0
    else:
        order = sorted(sizes, key=lambda r: -sizes[r])
        root_to_label = {r: i for i, r in enumerate(order)}
        default = 0

    labels = torch.tensor([root_to_label.get(r, default) for r in roots])
    return labels.reshape(h, w).long()


def filter_small_masks(labels: torch.Tensor, min_size: int) -> torch.Tensor:
    """Drop masks smaller than ``min_size`` patches to background (label 0).

    Background (0) is left untouched; surviving masks are renumbered contiguously
    in their original label order (preserving any ordering the algorithm encoded,
    e.g. best-first). ``min_size <= 1`` is a no-op.
    """
    if min_size <= 1:
        return labels
    out = torch.zeros_like(labels)
    new = 0
    for old in labels.unique().tolist():
        if old == 0:
            continue
        members = labels == old
        if int(members.sum()) >= min_size:
            new += 1
            out[members] = new
    return out


# ------------------------------------------------------- region-growing utils #
def neighbour_max(x: torch.Tensor, connectivity: int = 4) -> torch.Tensor:
    """Element-wise max of each pixel's neighbours (out-of-grid padded -inf)."""
    h, w = x.shape
    xp = F.pad(x[None, None], (1, 1, 1, 1), value=float("-inf"))[0, 0]
    up, down = xp[0:h, 1:w + 1], xp[2:h + 2, 1:w + 1]
    left, right = xp[1:h + 1, 0:w], xp[1:h + 1, 2:w + 2]
    m = torch.maximum(torch.maximum(up, down), torch.maximum(left, right))
    if connectivity == 8:
        ul, ur = xp[0:h, 0:w], xp[0:h, 2:w + 2]
        dl, dr = xp[2:h + 2, 0:w], xp[2:h + 2, 2:w + 2]
        m = torch.maximum(m, torch.maximum(torch.maximum(ul, ur), torch.maximum(dl, dr)))
    return m


def prototype_similarity_map(grid: torch.Tensor, proto: torch.Tensor, sim: Callable,
                             normalize_sim: bool = True,
                             q: tuple[float, float] = (0.02, 0.98)) -> torch.Tensor:
    """``(H, W)`` similarity of every patch to a prototype feature vector.

    When ``normalize_sim`` is set, the map is robustly min-max scaled to
    ``[0, 1]`` so that downstream thresholds on *changes* in similarity are
    interpretable and roughly metric-agnostic.
    """
    h, w, c = grid.shape
    s = sim(grid, proto.view(1, 1, c))
    if normalize_sim:
        lo = torch.quantile(s, q[0])
        hi = torch.quantile(s, q[1])
        s = ((s - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return s


def grow_region(s: torch.Tensor, seed_mask: torch.Tensor, blocked: torch.Tensor,
                boundary_delta: float, connectivity: int = 4,
                max_iter: int | None = None) -> torch.Tensor:
    """Flood-grow ``seed_mask`` over similarity map ``s`` until frontiers turn harsh.

    A frontier pixel ``f`` is absorbed unless the drop in ``s`` from its strongest
    interior neighbour exceeds ``boundary_delta`` (a steep gradient = boundary).
    ``blocked`` pixels can never be entered. Returns the grown boolean mask.
    """
    region = seed_mask & ~blocked
    if not region.any():
        return region
    neg_inf = torch.full_like(s, float("-inf"))
    max_iter = max_iter or s.numel()
    for _ in range(max_iter):
        s_in = neighbour_max(torch.where(region, s, neg_inf), connectivity)
        frontier = (~region) & (~blocked) & torch.isfinite(s_in)
        if not frontier.any():
            break
        absorb = frontier & ((s_in - s) <= boundary_delta)
        if not absorb.any():
            break                                          # all frontier are boundary
        region |= absorb
    return region
