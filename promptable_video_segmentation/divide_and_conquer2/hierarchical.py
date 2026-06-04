"""
Hierarchical DINOv3-only segmentation — no CutLER required.

Algorithm
---------
Level 1 (whole image):
  1. Extract DINOv3 features for the whole image at ``fixed_size × fixed_size``.
  2. Run spectral bipartitioning (``spectral_bipartition_2d``) to find the
     principal foreground mask.
  3. Mark found patches as "painted"; recompute adaptive threshold on the
     remaining features; find the next foreground object.
  4. Repeat up to ``max_k`` times.

Level 2+ (crops):
  For every mask found at the previous level, extract the bounding box,
  re-extract DINOv3 features for that crop (or bilinear-interpolate the
  full-image features), and repeat the same procedure with the tau for
  this level — which should be harder (higher quantile / absolute value)
  to capture finer structure within the parent region.

The tau progression is set by ``tau_per_level``:
  - Non-adaptive: absolute cosine / RBF threshold.
    Lower tau → easier (denser graph, coarser segments).
  - Adaptive (recommended):  quantile fraction in [0, 1].
    Lower fraction → easier (connects top-X% most similar pairs).
    E.g. ``(0.10, 0.40)`` → level 1 connects top 90 % pairs,
                             level 2 connects only top 60 %.

Results are returned as a flat list of :class:`SegMask` objects.
Use :func:`masks_at_level` and :func:`children_of` to query the hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as TF
import PIL.Image as Image

from ..backbone import ViTFeatV3, extract_feature_matrix
from ..masks import smallest_square_containing_mask, iou as mask_iou
from .spectral import spectral_bipartition_2d


# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BACKBONE_ID   = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_TAU_PER_LEVEL = (0.20, 0.40, 0.60)   # quantile fractions; adaptive=True
DEFAULT_MAX_K         = 3
DEFAULT_MAX_J         = 3


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class SegMask:
    """A single segmentation mask from the hierarchical pipeline."""
    mask:       np.ndarray      # (H, W) bool in original image pixel space
    level:      int             # 1-indexed hierarchy level
    parent_idx: "int | None"    # global index in masks_at_level(level-1); None for level-1
    child_idx:  int             # index within this parent's children


# ── hierarchy query helpers ───────────────────────────────────────────────────

def masks_at_level(masks: "list[SegMask]", level: int) -> "list[SegMask]":
    """Return all masks at the given hierarchy level (1-indexed)."""
    return [m for m in masks if m.level == level]


def children_of(
    masks: "list[SegMask]", level: int, parent_idx: int
) -> "list[SegMask]":
    """Return all masks at ``level`` whose parent is ``parent_idx`` at level-1."""
    return [m for m in masks if m.level == level and m.parent_idx == parent_idx]


# ── internal helpers ──────────────────────────────────────────────────────────

def _upscale(patch_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour upscale of a patch-space bool mask to pixel space."""
    t = torch.from_numpy(patch_mask.astype(np.float32))[None, None]
    up = TF.interpolate(t, size=(h, w), mode="nearest").squeeze().numpy()
    return up > 0.5


def _run_level(
    feats:     np.ndarray,
    tau:       float,
    max_k:     int,
    mode:      str,
    sigma:     "float | None",
    adaptive:  bool,
    min_frac:  float,
    max_iou:   float,
) -> "list[np.ndarray]":
    """
    Iterative spectral bipartitioning on a (Hf, Wf, C) feature grid.

    Returns list of (Hf, Wf) bool masks in patch space.
    Each call uses an updated painting mask so subsequent calls find new objects.
    With adaptive=True the threshold is recomputed from the remaining (non-painted)
    feature similarities on every iteration, making it truly data-driven.
    """
    Hf, Wf = feats.shape[:2]
    n = Hf * Wf
    found:    list[np.ndarray] = []
    painting: np.ndarray       = np.zeros((Hf, Wf), dtype=bool)

    for _ in range(max_k):
        if int(painting.sum()) >= n:
            break
        fg, _ = spectral_bipartition_2d(feats, tau, mode, sigma, adaptive, painting)
        if int(fg.sum()) < max(1, int(min_frac * n)):
            break
        if any(mask_iou(fg, prev) > max_iou for prev in found):
            break
        found.append(fg.copy())
        painting |= fg

    return found


# ── model ─────────────────────────────────────────────────────────────────────

class HierarchicalModel:
    """
    DINOv3-only hierarchical segmentation model.

    No CutLER needed: foreground/background separation is achieved via the
    spectral bipartitioning algorithm from MaskCut, using DINOv3 key features
    with RBF affinity and adaptive per-call thresholds.

    Parameters
    ----------
    backbone_model_id : HuggingFace model ID for the DINOv3 backbone.
    tau_per_level     : Tau threshold for each hierarchy level.
                        Adaptive (recommended): quantile fraction in [0,1];
                          lower value = easier = more edges connected.
                        Non-adaptive: absolute cosine / RBF similarity cutoff.
                        If fewer values than max_j, last value is repeated.
    max_k             : Maximum foreground objects extracted per image / crop.
    max_j             : Number of hierarchy levels (1 = whole image only).
    mode              : "rbf" (recommended) or "cosine".
    sigma             : RBF bandwidth; None = auto (median pairwise L2).
    adaptive          : If True, tau values are quantile fractions. Strongly
                        recommended — makes segmentation scale-invariant.
    fixed_size        : Resize images/crops to this (square) before DINOv3.
    use_dino_crops    : True  = re-run DINOv3 on each crop (slower, accurate).
                        False = bilinear-interpolate full-image features (fast).
    min_mask_frac     : Minimum fraction of patches a mask must cover.
    max_iou           : Masks above this IoU with a prior mask are skipped.
    """

    _cache: "dict[str, HierarchicalModel]" = {}

    def __init__(
        self,
        backbone_model_id: str            = DEFAULT_BACKBONE_ID,
        tau_per_level:    Sequence[float] = DEFAULT_TAU_PER_LEVEL,
        max_k:            int             = DEFAULT_MAX_K,
        max_j:            int             = DEFAULT_MAX_J,
        mode:             str             = "rbf",
        sigma:            "float | None"  = None,
        adaptive:         bool            = True,
        fixed_size:       int             = 480,
        use_dino_crops:   bool            = True,
        min_mask_frac:    float           = 0.02,
        max_iou:          float           = 0.5,
    ):
        self.backbone = ViTFeatV3(model_id=backbone_model_id).eval()
        self._patch   = self.backbone.patch_size
        self._dim     = self.backbone.feat_dim

        self.tau_per_level  = list(tau_per_level)
        self.max_k          = max_k
        self.max_j          = max_j
        self.mode           = mode
        self.sigma          = sigma
        self.adaptive       = adaptive
        self.fixed_size     = fixed_size
        self.use_dino_crops = use_dino_crops
        self.min_mask_frac  = min_mask_frac
        self.max_iou        = max_iou

    @classmethod
    def get_or_load(
        cls,
        backbone_model_id: str = DEFAULT_BACKBONE_ID,
    ) -> "HierarchicalModel":
        """Return a cached instance, loading the backbone only on first call."""
        if backbone_model_id not in cls._cache:
            cls._cache[backbone_model_id] = cls(backbone_model_id=backbone_model_id)
        return cls._cache[backbone_model_id]

    # ── feature extraction ────────────────────────────────────────────────────

    def _extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract DINOv3 features for the whole image. Returns (Hf, Wf, C)."""
        sz = self.fixed_size
        pil = Image.fromarray(image_rgb).resize((sz, sz))
        fn  = sz // self._patch
        mat = extract_feature_matrix(self.backbone, pil, self._dim, fn)
        return mat.numpy() if hasattr(mat, "numpy") else np.asarray(mat)

    def _extract_crop(
        self,
        image_rgb:  np.ndarray,
        full_feats: np.ndarray,
        ymin: int, ymax: int, xmin: int, xmax: int,
        use_dino: "bool | None" = None,
    ) -> np.ndarray:
        if (use_dino if use_dino is not None else self.use_dino_crops):
            return self._extract(image_rgb[ymin:ymax, xmin:xmax])

        # Bilinear interpolation: map bbox pixel coords → feature coords
        H_img, W_img = image_rgb.shape[:2]
        Hf, Wf = full_feats.shape[:2]
        fn = self.fixed_size // self._patch

        fy0 = max(0,  int(ymin * Hf / H_img))
        fy1 = min(Hf, int(np.ceil(ymax * Hf / H_img)))
        fx0 = max(0,  int(xmin * Wf / W_img))
        fx1 = min(Wf, int(np.ceil(xmax * Wf / W_img)))

        sub = full_feats[fy0:fy1, fx0:fx1].astype(np.float32)   # (h, w, C)
        t   = torch.from_numpy(sub).permute(2, 0, 1)[None]
        r   = TF.interpolate(t, size=(fn, fn), mode="bilinear", align_corners=False)
        return r.squeeze(0).permute(1, 2, 0).numpy()

    # ── internal segmentation for one region ─────────────────────────────────

    def _segment_region(
        self,
        image_rgb:  np.ndarray,
        full_feats: np.ndarray,
        ymin: int, ymax: int, xmin: int, xmax: int,
        tau:     float,
        max_k:   int,
        mode:    str,
        sigma:   "float | None",
        adaptive: bool,
    ) -> "list[np.ndarray]":
        """
        Run iterative bipartitioning on one rectangular region.

        Returns a list of (H_img, W_img) bool masks in full image space.
        """
        H, W = image_rgb.shape[:2]
        h, w = ymax - ymin, xmax - xmin
        if h < self._patch or w < self._patch:
            return []

        is_full = (ymin == 0 and xmin == 0 and ymax == H and xmax == W)
        feats = full_feats if is_full else self._extract_crop(
            image_rgb, full_feats, ymin, ymax, xmin, xmax
        )

        patch_masks = _run_level(
            feats, tau, max_k, mode, sigma, adaptive,
            self.min_mask_frac, self.max_iou,
        )

        out: list[np.ndarray] = []
        for pm in patch_masks:
            crop_m = _upscale(pm, h, w)
            full_m = np.zeros((H, W), dtype=bool)
            full_m[ymin:ymax, xmin:xmax] = crop_m
            out.append(full_m)
        return out

    # ── public API ────────────────────────────────────────────────────────────

    def segment(
        self,
        image_rgb:    np.ndarray,
        tau_per_level: "Sequence[float] | None" = None,
        max_k:         "int | None"             = None,
        max_j:         "int | None"             = None,
        mode:          "str | None"             = None,
        sigma:         "float | None"           = None,
        adaptive:      "bool | None"            = None,
    ) -> "list[SegMask]":
        """
        Run hierarchical segmentation.

        Keyword arguments override instance defaults for this call only.
        Returns a flat list of :class:`SegMask` objects.  Query with
        :func:`masks_at_level` and :func:`children_of`.

        Level-1 masks use ``tau_per_level[0]``; each deeper level uses the
        next entry (last entry repeated if the list is shorter than max_j).
        """
        taus     = list(tau_per_level) if tau_per_level is not None else list(self.tau_per_level)
        max_k    = max_k    if max_k    is not None else self.max_k
        max_j    = max_j    if max_j    is not None else self.max_j
        mode     = mode     if mode     is not None else self.mode
        sigma    = sigma    if sigma    is not None else self.sigma
        adaptive = adaptive if adaptive is not None else self.adaptive

        while len(taus) < max_j:
            taus.append(taus[-1])

        H, W = image_rgb.shape[:2]
        result: list[SegMask] = []

        # ── Level 1: whole image ─────────────────────────────────────────────
        full_feats = self._extract(image_rgb)

        l1_masks = self._segment_region(
            image_rgb, full_feats, 0, H, 0, W,
            taus[0], max_k, mode, sigma, adaptive,
        )
        for ci, m in enumerate(l1_masks):
            result.append(SegMask(mask=m, level=1, parent_idx=None, child_idx=ci))

        # ── Deeper levels ─────────────────────────────────────────────────────
        for j in range(1, max_j):
            prev = masks_at_level(result, j)  # iterate over level-j parents
            for pi, parent_seg in enumerate(prev):
                ymin, ymax, xmin, xmax = smallest_square_containing_mask(parent_seg.mask)
                ymax, xmax = ymax + 1, xmax + 1

                child_masks = self._segment_region(
                    image_rgb, full_feats, ymin, ymax, xmin, xmax,
                    taus[j], max_k, mode, sigma, adaptive,
                )
                for ci, m in enumerate(child_masks):
                    result.append(SegMask(mask=m, level=j + 1, parent_idx=pi, child_idx=ci))

        return result
