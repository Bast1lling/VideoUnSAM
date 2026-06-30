"""
Spectral bipartitioning for foreground/background separation.

Directly inspired by MaskCut (CVPR 2023) but extended with:
  - RBF affinity (preserves feature magnitude, unlike cosine)
  - Adaptive tau: treated as quantile fraction of the current pairwise
    similarity distribution, so the threshold is data-driven and invariant
    to the absolute feature scale of any particular crop / residual.

Key function: ``spectral_bipartition_2d``

    feats   : (H, W, C) float32 patch features
    tau     : affinity threshold.
              Non-adaptive: absolute cosine / RBF value.
              Adaptive:     quantile fraction in [0,1]; lower = easier
                            (connects more pairs → coarser segmentation).
    painting: (H, W) bool mask of already-found foreground patches.
              Those patches are zeroed in feature space so the next call
              naturally finds the *next* most salient object.

Returns a (H, W) bool mask for the principal foreground object.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from scipy.linalg import eigh


# ── affinity matrix builder ───────────────────────────────────────────────────

def _affinity(
    flat: np.ndarray,               # (n, C) float32, painting rows already zeroed
    tau: float,
    mode: str,
    sigma: "float | None",
    adaptive: bool,
    paint_flat: "np.ndarray | None",  # (n,) bool — for quantile over active patches
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build affinity matrix A and degree matrix D."""
    n = flat.shape[0]

    if mode == "rbf":
        f = torch.from_numpy(flat).float()
        sq = (f ** 2).sum(1)
        dist2 = (sq[:, None] + sq[None, :] - 2.0 * (f @ f.T)).clamp(min=0).numpy()
        if sigma is None:
            triu_d = np.sqrt(dist2[np.triu_indices(n, k=1)])
            sigma = max(float(np.median(triu_d)) if triu_d.size else 1.0, 1e-8)
        A_full = np.exp(-dist2 / (2.0 * sigma ** 2))
    else:  # cosine
        nrm = np.linalg.norm(flat, axis=1, keepdims=True)
        fn = flat / (nrm + 1e-8)
        A_full = fn @ fn.T

    # Adaptive: map tau fraction to the corresponding quantile, computed only
    # over non-painted pairs so masking doesn't distort the distribution.
    if adaptive:
        if paint_flat is not None and paint_flat.any():
            active = ~paint_flat
            ni = int(active.sum())
            if ni > 1:
                sub = A_full[np.ix_(active, active)]
                triu = sub[np.triu_indices(ni, k=1)]
                if triu.size:
                    tau = float(np.quantile(triu, tau))
        else:
            triu = A_full[np.triu_indices(n, k=1)]
            if triu.size:
                tau = float(np.quantile(triu, tau))

    A = np.where(A_full > tau, 1.0, eps)
    D = np.diag(A.sum(axis=1))
    return A, D


# ── main entry point ──────────────────────────────────────────────────────────

def spectral_bipartition_2d(
    feats: np.ndarray,              # (H, W, C) float32
    tau: float = 0.15,
    mode: str = "rbf",
    sigma: "float | None" = None,
    adaptive: bool = True,
    painting: "np.ndarray | None" = None,   # (H, W) bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    MaskCut-style spectral bipartition on a 2D patch feature grid.

    Parameters
    ----------
    feats    : (H, W, C) DINOv3 patch features.
    tau      : affinity threshold (absolute or quantile fraction if adaptive=True).
    mode     : "rbf" or "cosine".
    sigma    : RBF bandwidth; None = auto (median pairwise L2 among active patches).
    adaptive : if True, tau is a quantile fraction of the non-painting similarity
               distribution, recomputed each call — data-driven and scale-invariant.
    painting : (H, W) bool mask; True = already assigned foreground; those patches
               are zeroed so the bipartition targets remaining objects.

    Returns
    -------
    fg  : (H, W) bool  — principal foreground connected component.
    eig : (n,)  float  — second-smallest eigenvector (useful for diagnostics).
    """
    H, W, C = feats.shape
    n = H * W
    flat = feats.reshape(n, C).astype(np.float32)
    paint_flat = painting.reshape(n) if painting is not None else None

    # Zero out painted features so painted patches don't attract new foreground
    if paint_flat is not None and paint_flat.any():
        flat = flat.copy()
        flat[paint_flat] = 0.0

    try:
        A, D = _affinity(flat, tau, mode, sigma, adaptive, paint_flat)
        _, vecs = eigh(D - A, D, subset_by_index=[1, 2])
    except Exception:
        return np.zeros((H, W), dtype=bool), np.zeros(n)

    eig = vecs[:, 0].copy()
    fg_flat = eig > eig.mean()

    # Corner heuristic: if ≥3 corners belong to the foreground side, flip —
    # borders are overwhelmingly background in natural images.
    bipart = fg_flat.reshape(H, W)
    nc = int(bipart[0, 0]) + int(bipart[0, -1]) + int(bipart[-1, 0]) + int(bipart[-1, -1])
    if nc >= 3:
        eig = -eig
        fg_flat = ~fg_flat
        bipart = fg_flat.reshape(H, W)
    elif not fg_flat[int(np.argmax(np.abs(eig)))]:
        eig = -eig
        fg_flat = ~fg_flat
        bipart = fg_flat.reshape(H, W)

    # Keep only the connected component that contains the most-salient seed patch
    seed = int(np.argmax(eig))
    sy, sx = divmod(seed, W)
    labeled, _ = ndimage.label(bipart.astype(np.uint8))
    cc = labeled[sy, sx]
    fg = (labeled == cc).astype(bool) if cc > 0 else np.zeros((H, W), dtype=bool)

    # Never include already-painted patches
    if painting is not None:
        fg = fg & ~painting

    return fg, eig
