"""Fully-connected Dense CRF boundary refinement — fully unsupervised.

Two modes:
  - RGB bilateral (default): snaps to colour edges, fast.
  - DINOv3-feature bilateral: PCA-reduces patch features to 3D and uses them
    as the bilateral kernel instead of RGB. Semantically aware — separates
    objects from same-colour backgrounds (goat on grass etc.).

No training, no labels — pure inference-time energy minimisation
(Krähenbühl & Koltun, NeurIPS 2011).

Usage:
    from video.refine.dense_crf import crf_refine
    refined = crf_refine(frame_rgb, soft_map)
    refined = crf_refine(frame_rgb, soft_map, dino_feats=feats_hw_d)
"""
from __future__ import annotations

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
import torch


@torch.no_grad()
def _pca3(features_hwd: np.ndarray) -> np.ndarray:
    """PCA-reduce [H, W, D] patch features → [H, W, 3] uint8 for CRF bilateral.

    Uses a GPU randomized/truncated SVD (torch.pca_lowrank, q=3) instead of a
    full CPU np.linalg.svd over the [n_patches, D] matrix — we only ever need
    the top 3 components, and full SVD (measured ~3.5s/frame at 64×64×1024)
    was the dominant cost of CRF refinement, ~10-30x more than the actual
    dense-CRF mean-field inference. This is O(D·q) on GPU instead of O(D·min(n,D)^2)
    on CPU.
    """
    H, W, D = features_hwd.shape
    flat = torch.from_numpy(features_hwd.reshape(-1, D)).cuda().float()
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(centered, q=3, niter=2)
    proj = centered @ V[:, :3]           # [H*W, 3]
    lo = proj.min(dim=0).values
    hi = proj.max(dim=0).values
    proj = (proj - lo) / (hi - lo).clamp_min(1e-6) * 255.0
    return proj.reshape(H, W, 3).clamp(0, 255).byte().cpu().numpy()


def crf_confidence(soft_up: np.ndarray) -> float:
    """Mean of top-10% pixel values in a normalised [0,1] heatmap.

    Robust gate for selective CRF application: a concentrated OT heatmap
    (clear object) scores ~0.85–0.95; a diffuse or drifted one scores ~0.55–0.70.
    More reliable than raw max() — a single outlier pixel can spike max even on
    a bad mask, whereas top-10% mean requires a coherent blob.
    """
    flat = np.asarray(soft_up).ravel()
    k = max(1, int(len(flat) * 0.10))
    return float(np.partition(flat, -k)[-k:].mean())


def crf_refine(
    image_rgb: np.ndarray,
    soft_map: np.ndarray,
    dino_feats: np.ndarray | None = None,
    n_iter: int = 10,
    gaussian_sxy: float = 3.0,
    gaussian_compat: float = 3.0,
    bilateral_sxy: float = 70.0,
    bilateral_srgb: float = 13.0,
    bilateral_compat: float = 10.0,
    fg_confidence: float = 0.90,
) -> np.ndarray:
    """Refine a soft or binary mask with Dense CRF.

    Args:
        image_rgb: [H, W, 3] uint8 RGB frame.
        soft_map: [H, W] float in [0,1] (OT heatmap) or binary uint8 mask.
        dino_feats: optional [h, w, D] DINOv3 patch features (e.g. 64×64×1024).
            If provided, PCA-reduces to 3D and uses as bilateral kernel instead
            of RGB — semantically aware, handles colour-ambiguous backgrounds.
        n_iter: number of mean-field iterations.
        fg_confidence: when soft_map is binary, pixels clamped to this level.

    Returns:
        [H, W] uint8 binary refined mask.
    """
    H, W = image_rgb.shape[:2]

    # Build bilateral feature image: DINOv3-PCA or RGB
    if dino_feats is not None:
        feat_img = _pca3(dino_feats.numpy() if hasattr(dino_feats, 'numpy') else dino_feats)
        feat_img = cv2.resize(feat_img, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        feat_img = np.ascontiguousarray(image_rgb, dtype=np.uint8)

    # Convert soft_map to probability in [eps, 1-eps]
    soft = soft_map.astype(np.float32)
    if soft.max() <= 1.0 and soft.dtype != np.uint8:
        prob_fg = np.clip(soft, 1e-4, 1.0 - 1e-4)
    else:
        prob_fg = np.where(soft > 0, fg_confidence, 1.0 - fg_confidence).astype(np.float32)
    prob_bg = 1.0 - prob_fg

    unary = np.stack([prob_bg, prob_fg], axis=0)
    unary = (-np.log(unary.reshape(2, -1))).astype(np.float32)

    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=gaussian_sxy, compat=gaussian_compat)
    d.addPairwiseBilateral(
        sxy=bilateral_sxy,
        srgb=bilateral_srgb,
        rgbim=np.ascontiguousarray(feat_img),
        compat=bilateral_compat,
    )

    Q = d.inference(n_iter)
    return np.argmax(np.array(Q), axis=0).reshape(H, W).astype(np.uint8)
