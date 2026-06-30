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


def _spatial_cost(
    gh_a: int, gw_a: int, gh_b: int, gw_b: int, device: str
) -> torch.Tensor:
    """[N_a, N_b] normalized spatial distance between patch grid positions.

    Coordinates are in [0, 1] per axis; distances are divided by sqrt(2) so the
    matrix lives in [0, 1].  Adding this to the feature cost with a small weight
    (e.g. 0.3) prevents transport from jumping to semantically-similar but
    spatially-distant background regions (bmx-trees, kite-surf).
    """
    def _coords(gh: int, gw: int) -> torch.Tensor:
        ys = torch.arange(gh, device=device).float() / max(gh - 1, 1)
        xs = torch.arange(gw, device=device).float() / max(gw - 1, 1)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy.flatten(), gx.flatten()], dim=1)  # [N, 2]

    ca = _coords(gh_a, gw_a)  # [N_a, 2]
    cb = _coords(gh_b, gw_b)  # [N_b, 2]
    diff = ca[:, None, :] - cb[None, :, :]  # [N_a, N_b, 2]
    return diff.norm(dim=-1) / (2 ** 0.5)   # [N_a, N_b] in [0, 1]


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


def compute_cond(
    feats_a: torch.Tensor,    # [gh_a, gw_a, D] L2-normalised
    feats_b: torch.Tensor,    # [gh_b, gw_b, D] L2-normalised
    blur: float = 0.05,
    sinkhorn_iters: int = 200,
    device: str = "cuda",
    cost_addend: torch.Tensor | None = None,  # [N_a, N_b] extra cost (e.g. color)
) -> torch.Tensor:
    """Compute the forward conditional cond = P(b|a) = T / μ once for reuse.

    Returns [N_a, N_b] with rows summing to 1. Push any source-patch indicator
    m_a [N_a] forward via  m_b = m_a @ cond.  For competitive multi-object label
    propagation, stack K label indicators M [K, N_a] and do M @ cond → [K, N_b],
    then argmax over K. Because rows sum to 1, every target patch receives the same
    total mass (N_a/N_b), so the argmax competition between labels is fair.
    """
    gh_a, gw_a, D = feats_a.shape
    gh_b, gw_b, _ = feats_b.shape
    N_a, N_b = gh_a * gw_a, gh_b * gw_b
    fa = feats_a.reshape(N_a, D).to(device).float()
    fb = feats_b.reshape(N_b, D).to(device).float()
    mu = torch.full((N_a,), 1.0 / N_a, device=device)
    nu = torch.full((N_b,), 1.0 / N_b, device=device)
    cost = 1.0 - fa @ fb.T
    if cost_addend is not None:
        cost = cost + cost_addend.to(device).float()
    T = _sinkhorn_log(cost, mu, nu, eps=blur ** 2, n_iter=sinkhorn_iters)
    return T / mu[:, None]  # [N_a, N_b], rows sum to 1


def propagate_patch(
    feats_a: torch.Tensor,    # [gh_a, gw_a, D] L2-normalised
    feats_b: torch.Tensor,    # [gh_b, gw_b, D] L2-normalised
    m_a_patch: torch.Tensor,  # [N_a] soft indicator in [0,1] (NOT normalised)
    blur: float = 0.05,
    sinkhorn_iters: int = 200,
    device: str = "cuda",
    spatial_weight: float = 0.0,
    motion_a: torch.Tensor | None = None,  # [N_a] per-patch frame-diff score in [0,1]
    motion_b: torch.Tensor | None = None,  # [N_b] per-patch frame-diff score in [0,1]
    motion_weight: float = 0.0,
    cost_addend: torch.Tensor | None = None,  # [N_a, N_b] extra cost to add (e.g. color)
    point_a: torch.Tensor | None = None,  # [N_a] one-hot click indicator — propagated free
    cycle_weight: float = 0.0,  # forward-backward consistency reweight strength
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Core OT push at patch level. Returns [N_b] soft mass, peak-normalised to 1.

    spatial_weight > 0 adds a Euclidean patch-position penalty to the cost matrix.
    motion_weight > 0 adds |motion_b[j] - motion_a[i]| to the cost, which penalises
    transport between patches with very different frame-to-frame motion magnitudes.
    cost_addend: arbitrary [N_a, N_b] additive term — caller is responsible for scaling.
    point_a: if given, propagates a click-point indicator through the same transport plan
             at zero extra cost and returns (heat, point_b) instead of just heat.
             point_b argmax gives the tracked click patch index in frame B.
    cycle_weight > 0: forward-backward (cycle) consistency. For each B patch j, the
             backward conditional P(a|j) = T_aj / ν_j says where j's mass came from in A.
             s_j = Σ_a m_a · P(a|j) is the fraction that traces back into the source mask.
             heat is multiplied by s_j ** cycle_weight, suppressing patches whose mass
             does NOT round-trip to the seed (identity switches, adjacent-background leak).
             At weight 0 the multiplier s**0 = 1 → no effect. Reuses T, near-zero cost.
    """
    gh_a, gw_a, D = feats_a.shape
    gh_b, gw_b, _ = feats_b.shape
    N_a, N_b = gh_a * gw_a, gh_b * gw_b

    fa = feats_a.reshape(N_a, D).to(device).float()
    fb = feats_b.reshape(N_b, D).to(device).float()
    mu = torch.full((N_a,), 1.0 / N_a, device=device)
    nu = torch.full((N_b,), 1.0 / N_b, device=device)

    cost = 1.0 - fa @ fb.T
    if spatial_weight > 0.0:
        cost = cost + spatial_weight * _spatial_cost(gh_a, gw_a, gh_b, gw_b, device)
    if motion_weight > 0.0 and motion_a is not None and motion_b is not None:
        ma = motion_a.to(device).float()   # [N_a]
        mb = motion_b.to(device).float()   # [N_b]
        cost = cost + motion_weight * (mb[None, :] - ma[:, None]).abs()  # [N_a, N_b]
    if cost_addend is not None:
        cost = cost + cost_addend.to(device).float()
    eps = blur ** 2
    T = _sinkhorn_log(cost, mu, nu, eps=eps, n_iter=sinkhorn_iters)
    cond = T / mu[:, None]  # rows sum to 1  → P(b|a)
    m_a_dev = m_a_patch.to(device)
    m_b = m_a_dev @ cond  # [N_b]

    if cycle_weight > 0.0:
        # Backward conditional P(a|b) = T_ab / ν_b; columns sum to 1 over a.
        cond_back = T / nu[None, :]
        s = m_a_dev @ cond_back  # [N_b] round-trip mass into the seed
        # Reference level = the object's OWN round-trip score, weighted by forward mass.
        # Normalising by the global max instead would peak-sharpen (both s and m_b peak at
        # the seed centre) and collapse the mask. Gate relative to the object's level so
        # body patches keep full heat (gate→1) and only sub-object leaks are suppressed.
        w = m_b / (m_b.sum() + 1e-8)
        s_ref = (w * s).sum()
        gate = (s / (s_ref + 1e-8)).clamp(max=1.0)
        m_b = m_b * gate.pow(cycle_weight)

    heat = m_b / (m_b.max() + 1e-8)

    if point_a is not None:
        point_b = point_a.to(device).float() @ cond  # [N_b] — same plan, free ride
        return heat, point_b
    return heat


def propagate_multiscale(
    feats_a: torch.Tensor,   # [gh, gw, D] L2-normalised (fine scale, e.g. 64×64)
    feats_b: torch.Tensor,   # [gh, gw, D] L2-normalised
    m_a_patch: torch.Tensor, # [N_a] soft indicator at fine scale
    blur_fine: float = 0.05,
    blur_coarse: float = 0.10,
    coarse_factor: int = 4,  # pool 64→16 patches
    alpha: float = 0.4,      # weight given to coarse heatmap
    sinkhorn_iters: int = 200,
    device: str = "cuda",
    spatial_weight: float = 0.0,
    motion_a: torch.Tensor | None = None,
    motion_b: torch.Tensor | None = None,
    motion_weight: float = 0.0,
) -> torch.Tensor:
    """Multi-scale OT: run at fine (64×64) and coarse (16×16) grids, blend results.

    The coarse pass uses a larger blur and larger effective patch size, so it
    can track objects that move more than ~2 patches between frames (fast motion,
    small objects). The fine pass keeps localisation accuracy.

    Returns [N_b] combined heatmap, peak-normalised to 1.
    """
    gh, gw, D = feats_a.shape

    # Fine-scale OT (existing behaviour)
    heat_fine = propagate_patch(feats_a, feats_b, m_a_patch,
                                blur=blur_fine, sinkhorn_iters=sinkhorn_iters, device=device,
                                spatial_weight=spatial_weight,
                                motion_a=motion_a, motion_b=motion_b, motion_weight=motion_weight)

    # Coarse features: average-pool D-dim features spatially then re-normalise
    gc = gh // coarse_factor
    fa_c = F.avg_pool2d(feats_a.permute(2, 0, 1).unsqueeze(0).float(),
                        coarse_factor)[0].permute(1, 2, 0)  # [gc, gc, D]
    fb_c = F.avg_pool2d(feats_b.permute(2, 0, 1).unsqueeze(0).float(),
                        coarse_factor)[0].permute(1, 2, 0)
    fa_c = F.normalize(fa_c, dim=-1)
    fb_c = F.normalize(fb_c, dim=-1)

    # Coarse mask: pool the fine indicator then re-normalise to [0, 1]
    m_c = F.avg_pool2d(
        m_a_patch.reshape(1, 1, gh, gw).float(), coarse_factor
    )[0, 0].flatten()
    if m_c.max() > 0:
        m_c = m_c / m_c.max()
    else:
        return heat_fine  # nothing to track

    # Pool motion vectors to coarse grid
    ma_c = mb_c = None
    if motion_weight > 0.0 and motion_a is not None and motion_b is not None:
        ma_c = F.avg_pool2d(motion_a.reshape(1, 1, gh, gw).float(),
                            coarse_factor)[0, 0].flatten()
        mb_c = F.avg_pool2d(motion_b.reshape(1, 1, gh, gw).float(),
                            coarse_factor)[0, 0].flatten()

    # Coarse OT
    heat_coarse = propagate_patch(fa_c, fb_c, m_c,
                                  blur=blur_coarse, sinkhorn_iters=sinkhorn_iters, device=device,
                                  spatial_weight=spatial_weight,
                                  motion_a=ma_c, motion_b=mb_c, motion_weight=motion_weight)

    # Upsample coarse heatmap to fine-scale grid
    heat_coarse_up = F.interpolate(
        heat_coarse.reshape(1, 1, gc, gc).float(),
        size=(gh, gw), mode="bilinear", align_corners=False
    )[0, 0].flatten()

    combined = (1.0 - alpha) * heat_fine + alpha * heat_coarse_up
    return combined / (combined.max() + 1e-8)


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
