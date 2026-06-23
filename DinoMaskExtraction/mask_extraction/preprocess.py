"""
Feature-grid preprocessing for mask extraction.

DINOv3 patch features are high-dimensional (1024-d) and carry a lot of
high-frequency, per-channel detail. Similarity computed on the *full* vectors is
sensitive to that detail, which fragments masks. The PCA-to-RGB visualisation
looks like clean segmentation precisely because it keeps only the **top few
principal components** (the dominant low-rank structure) and throws the rest
away. These transforms expose that denoising as a reusable step in front of any
extractor:

* :func:`pca_reduce` -- *spectral* low-pass: keep the top-``k`` principal
  components of the feature grid (what makes PCA-RGB coherent).
* :func:`spatial_smooth` -- *spatial* low-pass: Gaussian-blur the grid so only
  low spatial-frequency structure survives.

Both return a feature grid of the same ``(H, W, ·)`` layout, so the similarity
metrics and algorithms are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pca_reduce(grid: torch.Tensor, k: int, whiten: bool = False,
               center: bool = True) -> torch.Tensor:
    """Project an ``(H, W, C)`` grid onto its top-``k`` principal components.

    ``whiten`` rescales each kept component to unit variance (equal weighting,
    closer to the per-channel normalisation used by the RGB visualiser).
    """
    h, w, c = grid.shape
    k = min(k, c)
    flat = grid.reshape(h * w, c).float()
    if center:
        flat = flat - flat.mean(dim=0, keepdim=True)
    # deterministic full SVD (randomized pca_lowrank would make masks irreproducible)
    _, _, vh = torch.linalg.svd(flat, full_matrices=False)
    proj = flat @ vh[:k].T                                     # (H*W, k)
    if whiten:
        proj = proj / proj.std(dim=0, keepdim=True).clamp_min(1e-8)
    return proj.reshape(h, w, k)


def spatial_smooth(grid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-blur an ``(H, W, C)`` grid over its spatial dims (reflect pad)."""
    if sigma <= 0:
        return grid
    h, w, c = grid.shape
    radius = max(1, int(round(3 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1 = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    kernel = (k1[:, None] * k1[None, :]).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    kernel = kernel.expand(c, 1, -1, -1).to(grid.dtype)
    t = grid.permute(2, 0, 1).unsqueeze(0)                    # (1, C, H, W)
    t = F.pad(t, (radius, radius, radius, radius), mode="reflect")
    out = F.conv2d(t, kernel, groups=c)
    return out.squeeze(0).permute(1, 2, 0).contiguous()


def reduce_features(grid: torch.Tensor, pca_components: int = 0,
                    smooth_sigma: float = 0.0, whiten: bool = False) -> torch.Tensor:
    """Apply optional PCA reduction then spatial smoothing (both off by default)."""
    if pca_components and pca_components > 0:
        grid = pca_reduce(grid, pca_components, whiten=whiten)
    if smooth_sigma and smooth_sigma > 0:
        grid = spatial_smooth(grid, smooth_sigma)
    return grid
