"""
MaskCut mask extractor (iterative Normalized Cut on patch features).

A clean, self-contained reimplementation of MaskCut (Wang et al., CutLER) adapted
to DINOv3 feature grids and this package's API. No scipy / CRF dependencies: the
generalized NCut eigenproblem is solved with ``torch.linalg.eigh`` on the
symmetric normalised Laplacian, and connected components use a small BFS.

Per iteration:

1. Build a patch-affinity graph: pairwise similarity, **binarised** at ``tau``
   (an edge exists where similarity > tau). Already-extracted patches are
   removed from the graph.
2. **Normalized Cut**: take the second-smallest eigenvector (Fiedler vector) of
   the normalised Laplacian and split patches at its mean → a foreground /
   background bipartition.
3. Orient the bipartition so the object (not the background) is foreground,
   using the eigenvector peak (``seed``) and an object-centric corner prior
   (if ≥3 image corners are foreground, flip it).
4. Keep the connected component containing the seed → one object mask.
5. **Cut it out** (mask those patches from the graph) and repeat up to
   ``n_masks`` times. Stop early on tiny components.

Output: label 0 = background, ``1..K`` = objects in extraction order.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import torch

from .base import MaskExtractor, register_extractor
from .similarity import get_similarity, pairwise


def _ncut_fiedler(affinity: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Second-smallest generalized eigenvector of ``(D - A) x = λ D x``.

    Solved as the second eigenvector of the symmetric normalised Laplacian
    ``L = I - D^{-1/2} A D^{-1/2}`` (then mapped back by ``x = D^{-1/2} y``).
    """
    d = affinity.sum(dim=1).clamp_min(eps)
    d_inv_sqrt = d.rsqrt()
    n = affinity.shape[0]
    lap = torch.eye(n, dtype=affinity.dtype) - d_inv_sqrt[:, None] * affinity * d_inv_sqrt[None, :]
    lap = 0.5 * (lap + lap.T)                         # symmetrise against fp drift
    _, evecs = torch.linalg.eigh(lap)                 # ascending eigenvalues
    return d_inv_sqrt * evecs[:, 1]                   # 2nd smallest -> Fiedler


def _num_fg_corners(bipartition_hw: torch.Tensor) -> int:
    b = bipartition_hw
    return int(b[0, 0]) + int(b[0, -1]) + int(b[-1, 0]) + int(b[-1, -1])


def _component_containing(mask_hw: torch.Tensor, seed: int, connectivity: int) -> torch.Tensor:
    """BFS the foreground component of ``mask_hw`` that contains flat index ``seed``."""
    h, w = mask_hw.shape
    out = torch.zeros_like(mask_hw)
    sy, sx = divmod(seed, w)
    if not mask_hw[sy, sx]:
        return out
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out[sy, sx] = True
    dq = deque([(sy, sx)])
    while dq:
        y, x = dq.popleft()
        for dy, dx in nb:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask_hw[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                dq.append((ny, nx))
    return out


@register_extractor("maskcut")
class MaskCutExtractor(MaskExtractor):
    """Iterative Normalized-Cut foreground extraction on a feature grid.

    Parameters
    ----------
    similarity:
        Metric for the affinity graph (default ``"cosine"`` — MaskCut's choice;
        the Gram-anchoring objective optimises exactly this). Magnitude-aware
        metrics also work but ``tau`` then lives on a different scale.
    tau:
        Affinity threshold: an edge is kept where similarity > ``tau``.
    n_masks:
        Maximum number of objects to extract (NCut iterations).
    min_area_ratio:
        Stop once the selected component covers a smaller fraction of patches
        than this (filters specks / degenerate cuts).
    connectivity:
        4 or 8, for the connected-component step.
    sim_kwargs:
        Extra kwargs bound to the metric (e.g. ``{"gamma": 0.01}``).
    """

    def __init__(
        self,
        similarity: str | Callable = "cosine",
        tau: float = 0.15,
        n_masks: int = 3,
        min_area_ratio: float = 0.01,
        connectivity: int = 4,
        sim_kwargs: dict | None = None,
    ):
        self.similarity_name = similarity if isinstance(similarity, str) else "custom"
        self.sim = get_similarity(similarity, **(sim_kwargs or {}))
        self.tau = tau
        self.n_masks = n_masks
        self.min_area_ratio = min_area_ratio
        self.connectivity = connectivity

    @torch.no_grad()
    def extract(self, grid: torch.Tensor) -> torch.Tensor:
        h, w, c = grid.shape
        flat = grid.reshape(h * w, c).float()
        labels = torch.zeros(h, w, dtype=torch.long)
        painted = torch.zeros(h * w, dtype=torch.bool)        # already-extracted patches
        li = 0

        for _ in range(self.n_masks):
            if painted.all():
                break
            # 1) binarised affinity graph on the remaining patches
            sim = pairwise(flat, metric=self.sim)             # (N, N)
            sim = torch.nan_to_num(sim, nan=0.0)
            aff = (sim > self.tau).float()
            aff[painted, :] = 0.0
            aff[:, painted] = 0.0
            aff = aff + 1e-5                                  # keep graph connected / D>0

            # 2) Normalized Cut bipartition
            vec = _ncut_fiedler(aff)                          # (N,)
            bipartition = vec > vec.mean()

            # 3) orient so the object is foreground
            seed = int(vec.abs().argmax())
            if _num_fg_corners(bipartition.reshape(h, w)) >= 3:
                reverse = True
            else:
                reverse = not bool(bipartition[seed])
            if reverse:
                vec = -vec
                bipartition = ~bipartition
            bipartition = bipartition & ~painted              # never re-pick old patches
            if not bipartition.any():
                break
            # seed = strongest foreground patch
            masked_vec = vec.masked_fill(~bipartition, float("-inf"))
            seed = int(masked_vec.argmax())

            # 4) connected component at the seed
            comp = _component_containing(bipartition.reshape(h, w), seed, self.connectivity)
            if comp.float().mean() <= self.min_area_ratio:
                break

            # 5) emit + cut out
            li += 1
            labels[comp] = li
            painted |= comp.reshape(-1)

        return labels
