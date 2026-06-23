"""
Edge-aware mask post-processing (dense-CRF-style refinement).

Mask extractors work on the coarse patch grid (e.g. 28x28), so their masks have
blocky, patch-aligned boundaries. This module upsamples them to pixel resolution
and snaps the boundaries to the image's edges -- the same job the MaskCut
reference delegated to a dense CRF / bilateral solver.

The reference used ``pydensecrf`` (permutohedral-lattice dense CRF), which does
not build on modern Python. Instead we run **mean-field CRF inference** with a
Potts compatibility, where the expensive bilateral (appearance) message is
approximated by a **guided filter** (He et al., 2010) -- an O(N) edge-aware
smoother guided by the image. One pass already behaves like joint-bilateral
upsampling; a few iterations give CRF-like boundary snapping. Pure PyTorch, no
extra dependencies.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Box (mean) filter over a ``(2r+1)`` window, reflect-padded."""
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)


def _guided_filter(guide: torch.Tensor, src: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Guided filter of ``src`` (1,C,H,W) with grayscale ``guide`` (1,1,H,W)."""
    mean_g = _box(guide, r)
    mean_s = _box(src, r)
    mean_gs = _box(guide * src, r)
    cov_gs = mean_gs - mean_g * mean_s
    var_g = _box(guide * guide, r) - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return _box(a, r) * guide + _box(b, r)


def _gray_guide(image) -> torch.Tensor:
    """Grayscale guidance image as a ``(1, 1, H, W)`` tensor in ``[0, 1]``."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(-1)
        if arr.max() > 1.0:
            arr = arr / 255.0
    return torch.from_numpy(arr)[None, None]


def refine_crf(labels: torch.Tensor, image, n_iter: int = 5, radius: int = 3,
               eps: float = 1e-4, weight: float = 1.0, unary_conf: float = 0.9
               ) -> torch.Tensor:
    """Refine a grid label map into an edge-snapped pixel-resolution label map.

    Parameters
    ----------
    labels:
        ``(gh, gw)`` integer mask ids from an extractor (0 = background).
    image:
        The input image (``PIL.Image`` or array) providing edge guidance; its
        size defines the output resolution.
    n_iter:
        Mean-field iterations (0 falls back to plain bilinear upsampling).
    radius, eps:
        Guided-filter window radius and regularisation (larger ``eps`` = softer,
        less edge-sensitive).
    weight:
        Strength of the pairwise (smoothness/appearance) term vs. the data term.
    unary_conf:
        Confidence in the original labels (data term); lower lets the image
        evidence override the coarse mask more.

    Returns an ``(H, W)`` ``long`` label map at the image resolution.
    """
    if isinstance(image, Image.Image):
        w_img, h_img = image.size
    else:
        h_img, w_img = np.asarray(image).shape[:2]
    gh, gw = labels.shape
    k = int(labels.max()) + 1

    onehot = F.one_hot(labels.reshape(-1).long(), k).T.float().reshape(1, k, gh, gw)
    prob = F.interpolate(onehot, size=(h_img, w_img), mode="bilinear", align_corners=False)
    prob = prob.clamp_min(0)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_prob = torch.log(prob * unary_conf + (1 - unary_conf) / k)   # (1, K, H, W)

    guide = _gray_guide(image)
    logits = log_prob.clone()
    for _ in range(n_iter):
        q = torch.softmax(logits, dim=1)
        message = _guided_filter(guide, q, radius, eps)             # edge-aware neighbour vote
        logits = log_prob + weight * message
    return logits.argmax(dim=1)[0].long()
