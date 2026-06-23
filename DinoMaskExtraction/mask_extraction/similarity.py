"""
Plug-and-play feature-similarity functions for DINOv3 patch features.

Every metric has the signature ``f(a, b, **kw) -> sim`` where ``a`` and ``b`` are
broadcastable ``(..., C)`` tensors and the result is ``(...)`` with the
convention **higher == more similar** (distances are therefore returned
negated). Because they only reduce over the last dim and broadcast everything
else, the same function serves both:

* **edge / neighbour** similarity -- pass two aligned ``(N, C)`` tensors;
* **full pairwise** similarity -- use :func:`pairwise` (or pass
  ``a[:, None, :]`` and ``b[None, :, :]`` yourself) to get an ``(N, M)`` matrix.

Why several metrics? The DINOv3 paper's Gram-anchoring objective
(``||X_S X_Sᵀ - X_G X_Gᵀ||²`` with **L2-normalised** ``X``) explicitly trains
the model so that pairwise **cosine** similarities are high quality -- cosine is
the native metric for dense tasks. But DINOv3 patch magnitudes vary a lot with
depth and saliency (see DINOV3_ARCHITECTURE.md), so the magnitude-aware metrics
below (``dot``, ``neg_euclidean``, ``rbf``...) can capture structure that
direction-only cosine discards.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import torch

SIMILARITIES: dict[str, Callable] = {}

# metrics that ignore vector magnitude (direction only) vs. those that use it
MAGNITUDE_AWARE = {"dot", "neg_euclidean", "neg_sq_euclidean", "neg_manhattan", "rbf"}
DIRECTION_ONLY = {"cosine", "correlation", "neg_normalized_euclidean"}


def register_similarity(name: str):
    def deco(fn: Callable) -> Callable:
        SIMILARITIES[name] = fn
        fn.sim_name = name
        return fn
    return deco


# --------------------------------------------------------------- direction-only
@register_similarity("cosine")
def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity in ``[-1, 1]``. DINOv3's native (Gram-trained) metric."""
    num = (a * b).sum(-1)
    den = a.norm(dim=-1).clamp_min(eps) * b.norm(dim=-1).clamp_min(eps)
    return num / den


@register_similarity("correlation")
def correlation(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation == cosine of channel-mean-centred vectors."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return cosine(a, b, eps)


@register_similarity("neg_normalized_euclidean")
def neg_normalized_euclidean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative Euclidean distance between L2-normalised vectors.

    Monotonic in cosine (``||â-b̂||² = 2 - 2·cos``) but on a distance scale.
    """
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    return -(a - b).norm(dim=-1)


# -------------------------------------------------------------- magnitude-aware
@register_similarity("dot")
def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Raw inner product -- the unnormalised Gram-matrix entry (magnitude in)."""
    return (a * b).sum(-1)


@register_similarity("neg_euclidean")
def neg_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Negative L2 distance."""
    return -(a - b).norm(dim=-1)


@register_similarity("neg_sq_euclidean")
def neg_sq_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Negative squared L2 distance (cheaper, no sqrt)."""
    return -((a - b) ** 2).sum(-1)


@register_similarity("neg_manhattan")
def neg_manhattan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Negative L1 distance."""
    return -(a - b).abs().sum(-1)


@register_similarity("rbf")
def rbf(a: torch.Tensor, b: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """Gaussian/RBF kernel ``exp(-gamma·||a-b||²)`` in ``(0, 1]``.

    ``gamma`` defaults to ``1/C`` (feature dimension), a scale-robust choice.
    """
    d2 = ((a - b) ** 2).sum(-1)
    if gamma is None:
        gamma = 1.0 / a.shape[-1]
    return torch.exp(-gamma * d2)


# ----------------------------------------------------------------------- helpers
def get_similarity(name: str | Callable, **kwargs) -> Callable:
    """Resolve a metric name (with bound kwargs) or pass through a callable."""
    if callable(name):
        return name
    if name not in SIMILARITIES:
        raise KeyError(f"unknown similarity '{name}'. Available: {sorted(SIMILARITIES)}")
    fn = SIMILARITIES[name]
    return partial(fn, **kwargs) if kwargs else fn


def pairwise(a: torch.Tensor, b: torch.Tensor | None = None,
             metric: str | Callable = "cosine", **kwargs) -> torch.Tensor:
    """Full pairwise similarity matrix ``(N, M)`` between ``a (N,C)`` and ``b (M,C)``.

    If ``b`` is ``None`` it defaults to ``a`` (self-similarity / Gram matrix).
    """
    sim = get_similarity(metric, **kwargs)
    b = a if b is None else b
    return sim(a[:, None, :], b[None, :, :])
