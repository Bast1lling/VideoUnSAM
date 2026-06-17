"""Optimal-transport mask propagation between two DINOv3 feature grids.

Algorithm:
  - Both frames get *uniform* marginals (μ = 1/N_a, ν = 1/N_b).
  - Cost  C_ij = 1 - <fa_i, fb_j>   (features are L2-normalised, so cosine).
  - Log-domain Sinkhorn → soft assignment T ∈ R^{N_A × N_B}.
  - The source mask is a vector m_A ∈ {0,1}^{N_A} of per-A-patch indicators.
    Push it through the plan:  m_B = m_A @ T   →  reshape, upsample.

Why uniform marginals: if μ encoded mask weights, the column sum of T would be
forced to equal ν, washing out the feature structure. Uniform-on-both lets T
itself carry the geometric matching, and we read off only the mass arriving
from masked A patches.

For 480p DAVIS the plan is ~ (30·54)² ≈ 2.6M floats — small, fully in-memory.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _mask_to_patch_indicator(mask_hw: np.ndarray, grid_h: int, grid_w: int) -> torch.Tensor:
    """Downsample binary mask [H,W] → per-patch coverage [grid_h*grid_w] in [0,1]."""
    m = torch.from_numpy(mask_hw.astype(np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(m, (grid_h, grid_w))[0, 0]
    if pooled.sum() <= 0:
        raise ValueError("Empty mask after pooling — object too small for this grid.")
    return pooled.flatten()


def _sinkhorn_log(cost: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor,
                  eps: float, n_iter: int = 200, tol: float = 1e-6) -> torch.Tensor:
    """Log-domain Sinkhorn. Returns transport plan T = exp((f⊕g - C)/ε + log μ⊗ν)."""
    log_mu = torch.log(mu.clamp_min(1e-30))
    log_nu = torch.log(nu.clamp_min(1e-30))
    K = -cost / eps  # [N_a, N_b]
    f = torch.zeros_like(mu)
    g = torch.zeros_like(nu)
    for _ in range(n_iter):
        f_prev = f
        # u-update:  f = log μ - logsumexp_j (K_ij + g_j)
        f = log_mu - torch.logsumexp(K + g[None, :], dim=1)
        # v-update:  g = log ν - logsumexp_i (K_ij + f_i)
        g = log_nu - torch.logsumexp(K + f[:, None], dim=0)
        if (f - f_prev).abs().max() < tol:
            break
    log_T = K + f[:, None] + g[None, :]
    return torch.exp(log_T)


def propagate_patch(
    feats_a: torch.Tensor,   # [gh_a, gw_a, D] L2-normalised
    feats_b: torch.Tensor,   # [gh_b, gw_b, D] L2-normalised
    m_a_patch: torch.Tensor, # [N_a] soft indicator in [0,1] (NOT normalised)
    blur: float = 0.05,
    sinkhorn_iters: int = 200,
    device: str = "cuda",
) -> torch.Tensor:
    """Core OT push at patch level. Returns [N_b] soft mass, peak-normalised to 1."""
    gh_a, gw_a, D = feats_a.shape
    gh_b, gw_b, _ = feats_b.shape
    N_a, N_b = gh_a * gw_a, gh_b * gw_b

    fa = feats_a.reshape(N_a, D).to(device).float()
    fb = feats_b.reshape(N_b, D).to(device).float()
    mu = torch.full((N_a,), 1.0 / N_a, device=device)
    nu = torch.full((N_b,), 1.0 / N_b, device=device)

    cost = 1.0 - fa @ fb.T
    eps = blur ** 2
    T = _sinkhorn_log(cost, mu, nu, eps=eps, n_iter=sinkhorn_iters)
    cond = T / mu[:, None]  # rows sum to 1
    m_b = m_a_patch.to(device) @ cond  # [N_b]
    return m_b / (m_b.max() + 1e-8)


def propagate(
    feats_a: torch.Tensor,
    feats_b: torch.Tensor,
    mask_a: np.ndarray,
    out_size: tuple[int, int],
    blur: float = 0.05,
    threshold: float = 0.5,
    sinkhorn_iters: int = 200,
    device: str = "cuda",
) -> dict:
    """Pixel-level entry point: takes a binary mask, returns soft heatmap + binary mask."""
    gh_a, gw_a, _ = feats_a.shape
    gh_b, gw_b, _ = feats_b.shape
    m_a = _mask_to_patch_indicator(mask_a, gh_a, gw_a).to(device)
    m_b_patch = propagate_patch(feats_a, feats_b, m_a,
                                blur=blur, sinkhorn_iters=sinkhorn_iters, device=device)

    heat = m_b_patch.reshape(gh_b, gw_b).cpu()
    H_b, W_b = out_size
    heat_up = F.interpolate(
        heat[None, None], size=(H_b, W_b), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    binary = (heat_up > threshold * heat_up.max()).astype(np.uint8)
    return {
        "heatmap": heat_up.astype(np.float32),
        "mask": binary,
        "m_b_patch": m_b_patch.cpu(),
    }
