"""Boundary-refinement strategies. Pure numpy/OpenCV — no torch, no GPU, no DINOv3.

Every refiner has the signature

    refine(soft_up, frame_rgb, props, **params) -> binary uint8 [H, W]

where `soft_up` is the OT heatmap bilinearly upsampled to full resolution and
peak-normalised to 1, `frame_rgb` is the RGB frame, and `props` is the list of
CuVLER+conquer proposal masks for that frame (possibly empty).

The problem being attacked: the OT heatmap lives on a 64x64 patch grid, so at 480p
one patch is 16 px. Bilinear upsampling + thresholding makes boundaries snap to that
16 px lattice, producing stepped edges. That is why the pipeline's F score (0.571)
trails its J score (0.635), with the gap worst on car-roundabout (0.234),
car-shadow (0.223) and drift-straight (0.203).
"""

from __future__ import annotations

import cv2
import numpy as np


# ── Guided filter (He et al. 2010), reimplemented so opencv-contrib isn't needed ──

def guided_filter(guide: np.ndarray, src: np.ndarray,
                  radius: int = 16, eps: float = 1e-3) -> np.ndarray:
    """Edge-preserving filter of `src` using `guide` as structure reference.

    Both inputs float32 in [0,1]. Equivalent to cv2.ximgproc.guidedFilter, which is
    what eval_davis2016.py's --guided path uses, but with no contrib dependency.
    """
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.blur(guide, k)
    mean_p = cv2.blur(src, k)
    corr_i = cv2.blur(guide * guide, k)
    corr_ip = cv2.blur(guide * src, k)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.blur(a, k) * guide + cv2.blur(b, k)


def _guide_gray(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


# ── Refiners ─────────────────────────────────────────────────────────────────

def baseline(soft_up, frame_rgb, props, thresh: float = 0.5, **_):
    """What the pipeline does today: threshold the bilinear-upsampled heatmap."""
    return (soft_up > thresh).astype(np.uint8)


def guided(soft_up, frame_rgb, props, thresh: float = 0.5,
           radius: int = 16, eps: float = 1e-3, **_):
    """Guided-filter the heatmap against image structure, then threshold.

    This is eval_davis2016.py's --guided path, which exists in the codebase but has
    never been evaluated — no results for it appear anywhere in the README.
    """
    heat = (soft_up / (soft_up.max() + 1e-8)).astype(np.float32)
    filtered = guided_filter(_guide_gray(frame_rgb), heat, radius=radius, eps=eps)
    return (filtered > thresh).astype(np.uint8)


def snap(soft_up, frame_rgb, props, thresh: float = 0.5, region_thresh: float = 0.35,
         agg: str = "frac", min_area: int = 20, fill_uncovered: bool = True, **_):
    """Snap the mask to conquer sub-mask boundaries instead of thresholding pixels.

    Conquer already produces ~30 tight, object-aligned sub-masks per frame via DINOv3
    spectral clustering. Their boundaries follow real object silhouettes at pixel
    resolution, not the 16 px patch lattice. So rather than asking "is this PIXEL hot?"
    we ask "is this REGION hot?" and accept or reject whole regions.

    agg controls how a region's score is computed from the heatmap:
      "mean"   — mean heat over the region. Simple, but dilutes on large regions.
      "median" — robust to a few hot pixels leaking in.
      "frac"   — fraction of the region's pixels above `thresh`. Usually the best
                 behaved: scale-free and directly comparable to the baseline decision.

    Overlapping proposals are unioned. Pixels covered by no proposal fall back to the
    plain threshold when fill_uncovered, so this can only add boundary precision.
    """
    base = (soft_up > thresh)
    if not props:
        return base.astype(np.uint8)

    out = np.zeros_like(base)
    covered = np.zeros_like(base)
    for m in props:
        m = m.astype(bool)
        n = int(m.sum())
        if n < min_area:
            continue
        covered |= m
        vals = soft_up[m]
        if agg == "mean":
            score = float(vals.mean())
        elif agg == "median":
            score = float(np.median(vals))
        else:  # frac
            score = float((vals > thresh).mean())
        if score >= region_thresh:
            out |= m

    if fill_uncovered:
        out |= (base & ~covered)
    if out.sum() == 0:
        return base.astype(np.uint8)
    return out.astype(np.uint8)


def snap_best(soft_up, frame_rgb, props, thresh: float = 0.5, agg: str = "frac", **_):
    """Take only the single highest-scoring proposal — no unioning.

    Closer to what the reseed step already does, and a useful control: it isolates
    whether snap's gains come from boundary quality or from unioning extra regions.
    """
    base = (soft_up > thresh)
    if not props:
        return base.astype(np.uint8)
    scored = []
    for m in props:
        m = m.astype(bool)
        if m.sum() < 20:
            continue
        vals = soft_up[m]
        score = float((vals > thresh).mean()) if agg == "frac" else float(vals.mean())
        scored.append((score, m))
    if not scored:
        return base.astype(np.uint8)
    return max(scored, key=lambda t: t[0])[1].astype(np.uint8)


def guided_snap(soft_up, frame_rgb, props, thresh: float = 0.5, region_thresh: float = 0.35,
                radius: int = 16, eps: float = 1e-3, agg: str = "frac", **_):
    """Guided filter first (cleans the heat), then region snapping (fixes boundaries)."""
    heat = (soft_up / (soft_up.max() + 1e-8)).astype(np.float32)
    filtered = guided_filter(_guide_gray(frame_rgb), heat, radius=radius, eps=eps)
    return snap(filtered, frame_rgb, props, thresh=thresh,
                region_thresh=region_thresh, agg=agg)


def crf(soft_up, frame_rgb, props, pca3=None, thresh: float = 0.5,
        sxy: float = 70.0, srgb: float = 13.0, compat: float = 20.0,
        n_iter: int = 10, **_):
    """Dense CRF — the pipeline's current refiner. Needs pydensecrf installed.

    Included so the harness can A/B new ideas against what is already shipped.
    Uses the precomputed PCA-3 feature image when available (matching
    video/refine/dense_crf.py's DINOv3-bilateral mode), else falls back to RGB.
    """
    try:
        import pydensecrf.densecrf as dcrf
    except ImportError:
        raise RuntimeError(
            "pydensecrf not installed. On macOS:\n"
            "  pip install 'git+https://github.com/lucasb-eyer/pydensecrf.git'\n"
            "Or just skip it: --refiners baseline,guided,snap,guided_snap"
        )
    h, w = frame_rgb.shape[:2]
    if pca3 is not None:
        feat_img = cv2.resize(pca3, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        feat_img = frame_rgb
    prob_fg = np.clip(soft_up.astype(np.float32), 1e-4, 1 - 1e-4)
    unary = np.stack([1 - prob_fg, prob_fg], axis=0)
    unary = (-np.log(unary.reshape(2, -1))).astype(np.float32)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3.0, compat=3.0)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb,
                           rgbim=np.ascontiguousarray(feat_img, dtype=np.uint8),
                           compat=compat)
    q = d.inference(n_iter)
    return np.argmax(np.array(q), axis=0).reshape(h, w).astype(np.uint8)


def crf_snap(soft_up, frame_rgb, props, pca3=None, thresh: float = 0.5,
             region_thresh: float = 0.35, **kw):
    """CRF for pixel-level sharpening, then region snapping for object-level coherence."""
    refined = crf(soft_up, frame_rgb, props, pca3=pca3, thresh=thresh, **kw)
    return snap(refined.astype(np.float32), frame_rgb, props,
                thresh=0.5, region_thresh=region_thresh, agg="frac")


REFINERS = {
    "baseline": baseline,
    "guided": guided,
    "snap": snap,
    "snap_best": snap_best,
    "guided_snap": guided_snap,
    "crf": crf,
    "crf_snap": crf_snap,
}

# Refiners whose result depends on the precomputed PCA-3 feature image.
NEEDS_PCA = {"crf", "crf_snap"}
