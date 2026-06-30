"""
Divide-and-Conquer segmentation: clean functional + OOP API.

All imports are self-contained within the promptable_video_segmentation package.
No dependency on the old divide_and_conquer directory.

Quickstart
----------
    from promptable_video_segmentation.divide_and_conquer import DiCoModel
    from promptable_video_segmentation.divide_and_conquer import (
        visualize_masks, visualize_conquer_crop, visualize_divide_selected,
        visualize_conquer_single, feature_pca_rgb,
    )
    import numpy as np, PIL.Image as Image

    model = DiCoModel()            # loads defaults; cached across calls
    image = np.array(Image.open("photo.jpg").convert("RGB"))

    divide_masks        = model.divide(image)
    conquer_masks       = model.conquer(image, divide_masks, mask_idx=0)
    conquer_per_divide  = model.conquer_all(image, divide_masks)

    vis_all     = visualize_masks(image, divide_masks)
    vis_crop    = visualize_conquer_crop(image, divide_masks, 0, conquer_masks)
    vis_sel     = visualize_divide_selected(image, divide_masks, 2)
    vis_single  = visualize_conquer_single(image, divide_masks, 0, conquer_masks, 3)
    feat_vis    = model.feature_pca(image, divide_masks, 0)
"""
from __future__ import annotations

import tempfile
from typing import Sequence

import cv2
import numpy as np
import torch
import PIL.Image as Image

from ..backbone import ViTFeatV3, extract_feature_matrix
from ..masks import NMS, coverage, resize_mask, smallest_square_containing_mask
from .iterative_merging import iterative_merge
from .predictor import (
    CutlerPredictor,
    build_cutler_config,
    DEFAULT_CUTLER_CONFIG,
    DEFAULT_CUTLER_WEIGHTS,
)


# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BACKBONE_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_THETAS      = (0.73, 0.62, 0.51, 0.4, 0.3, 0.2)


# ── colour palettes ───────────────────────────────────────────────────────────

PALETTE: list[list[int]] = [
    [220,  20,  60], [ 30, 144, 255], [ 50, 205,  50], [255, 215,   0],
    [255, 140,   0], [138,  43, 226], [  0, 206, 209], [255, 105, 180],
    [127, 255,   0], [255,  69,   0], [  0, 191, 255], [186,  85, 211],
]

CONQUER_PALETTE: list[list[int]] = [
    [255, 230,  50], [ 50, 255, 160], [255, 100,  50], [180,  50, 255],
    [ 50, 220, 255], [255,  50, 150], [200, 255,  50], [255, 160,  50],
    [ 50, 100, 255], [255,  50, 200], [100, 255,  50], [ 50, 200, 180],
]


def palette_color(idx: int) -> list[int]:
    return PALETTE[idx % len(PALETTE)]


def conquer_color(divide_idx: int, conquer_idx: int) -> list[int]:
    return CONQUER_PALETTE[(divide_idx * 3 + conquer_idx) % len(CONQUER_PALETTE)]


# ── core visualization ────────────────────────────────────────────────────────

def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int],
    alpha: float = 0.45,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Blend *mask* onto *image_rgb* with *color* at opacity *alpha*."""
    out = image_rgb.copy()
    fg  = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if contour_thickness > 0:
        ctrs, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, ctrs, -1, list(color), contour_thickness)
    return out


def visualize_masks(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    alpha: float = 0.45,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Overlay every mask in *masks* using palette colours."""
    out = image_rgb.copy()
    for i, mask in enumerate(masks):
        out = overlay_mask(out, mask, palette_color(i), alpha, contour_thickness)
    return out


def visualize_divide_selected(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    mask_idx: int,
    alpha: float = 0.55,
) -> np.ndarray:
    """Highlight one divide mask on the full image with a white contour."""
    if not divide_masks:
        return image_rgb.copy()
    mask = divide_masks[int(mask_idx)].astype(np.uint8)
    out  = overlay_mask(image_rgb.copy(), mask,
                        palette_color(int(mask_idx)), alpha, contour_thickness=0)
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, ctrs, -1, [255, 255, 255], 3)
    return out


def visualize_conquer_crop(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    mask_idx: int,
    conquer_masks: list[np.ndarray],
) -> np.ndarray:
    """
    Hierarchical composite: all conquer masks on the divide-mask crop.

    Rendered largest-first so smaller sub-segments sit on top.
    Overlapping pixels are restored to the original before each new mask is
    painted, preventing double-blending.
    """
    divide_mask = divide_masks[int(mask_idx)]
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
    ymax, xmax = ymax + 1, xmax + 1
    crop_rgb = image_rgb[ymin:ymax, xmin:xmax]

    crop_masks = [
        (m[ymin:ymax, xmin:xmax] > 0.5).astype(np.uint8) for m in conquer_masks
    ]
    order = sorted(range(len(crop_masks)), key=lambda i: -int(np.sum(crop_masks[i])))

    orig = np.array(crop_rgb).copy()
    result = orig.copy()
    painted = np.zeros(orig.shape[:2], dtype=np.uint8)
    rng = np.random.default_rng(42)

    for i in order:
        m = crop_masks[i]
        painted += m
        overlap = painted == 2
        if overlap.any():
            result[overlap] = orig[overlap]
            painted[overlap] -= 1
        color = rng.integers(64, 230, 3).tolist()
        fg = m.astype(bool)
        result[fg] = (result[fg] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

    return result


def visualize_conquer_single(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    divide_idx: int,
    conquer_masks: list[np.ndarray],
    conquer_idx: int,
    alpha: float = 0.55,
) -> "np.ndarray | None":
    """Single conquer mask highlighted on the divide-mask crop."""
    if not conquer_masks or not divide_masks:
        return None
    divide_mask = divide_masks[int(divide_idx)]
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
    ymax, xmax = ymax + 1, xmax + 1
    crop = image_rgb[ymin:ymax, xmin:xmax].copy()
    cm   = conquer_masks[int(conquer_idx)][ymin:ymax, xmin:xmax].astype(np.uint8)
    cidx = int(conquer_idx)
    return overlay_mask(crop, cm, palette_color(cidx), alpha)


def visualize_all(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    conquer_per_divide: "dict[int, list[np.ndarray]] | None" = None,
    alpha_divide: float = 0.45,
    alpha_conquer: float = 0.5,
) -> np.ndarray:
    """Full-image overlay: divide masks as base, conquer sub-masks on top."""
    out = image_rgb.copy()
    for i, mask in enumerate(divide_masks):
        out = overlay_mask(out, mask, palette_color(i), alpha_divide, contour_thickness=2)
    if conquer_per_divide:
        for div_idx, cmasks in conquer_per_divide.items():
            for j, mask in enumerate(cmasks):
                out = overlay_mask(
                    out, mask, conquer_color(int(div_idx), j),
                    alpha_conquer, contour_thickness=1,
                )
    return out


# ── feature PCA visualisation ─────────────────────────────────────────────────

def feature_pca_rgb(
    feat_matrix: np.ndarray,
    upsample_to: "tuple[int, int] | None" = None,
) -> np.ndarray:
    """Project (H, W, C) feature map to RGB via PCA."""
    H, W, C = feat_matrix.shape
    flat = torch.from_numpy(feat_matrix.reshape(-1, C).astype(np.float32))
    flat -= flat.mean(0)
    _, _, V = torch.pca_lowrank(flat, q=3, niter=4)
    proj = flat @ V
    proj -= proj.min(0).values
    proj /= proj.max(0).values + 1e-8
    rgb = (proj * 255).byte().numpy().reshape(H, W, 3)
    if upsample_to is not None:
        rgb = cv2.resize(
            rgb, (upsample_to[1], upsample_to[0]), interpolation=cv2.INTER_NEAREST
        )
    return rgb


# ── synthetic video ───────────────────────────────────────────────────────────

def make_synthetic_video(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    conquer_per_divide: "dict[int, list[np.ndarray]] | None" = None,
    num_frames: int = 8,
    crop_min: float = 0.5,
    flip_prob: float = 0.5,
    fps: int = 4,
    output_path: "str | None" = None,
) -> "tuple[list[np.ndarray], str]":
    """
    Generate a synthetic video via independent per-frame augmentation.

    Each frame applies RandomCrop → ResizeShortestEdge → RandomFlip →
    brightness/contrast/saturation jitter to both the image and all masks.
    """
    from detectron2.data import transforms as T

    conquer_per_divide = conquer_per_divide or {}
    n_div = len(divide_masks)
    conquer_groups = [conquer_per_divide.get(i, []) for i in range(n_div)]
    flat_masks = list(divide_masks) + [m for g in conquer_groups for m in g]

    group_slices: list[tuple[int, int]] = []
    offset = n_div
    for g in conquer_groups:
        group_slices.append((offset, offset + len(g)))
        offset += len(g)

    aug_list = T.AugmentationList([
        T.RandomCrop("relative_range", [crop_min, crop_min]),
        T.ResizeShortestEdge([320, 480], max_size=720, sample_style="range"),
        T.RandomFlip(prob=flip_prob, horizontal=True),
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.RandomSaturation(0.9, 1.1),
    ])

    frames: list[np.ndarray] = []
    for _ in range(int(num_frames)):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        aug_input = T.AugInput(bgr)
        tfm = aug_list(aug_input)
        aug_img = cv2.cvtColor(aug_input.image, cv2.COLOR_BGR2RGB)

        aug_divide = [
            tfm.apply_segmentation(m.astype(np.uint8)) for m in divide_masks
        ]
        aug_conquer = [
            [tfm.apply_segmentation(m.astype(np.uint8)) for m in g]
            for g in conquer_groups
        ]

        frame = aug_img.copy()
        for i, mask in enumerate(aug_divide):
            frame = overlay_mask(frame, mask, palette_color(i), alpha=0.45)
        for div_idx, cmasks in enumerate(aug_conquer):
            for j, mask in enumerate(cmasks):
                frame = overlay_mask(
                    frame, mask, conquer_color(div_idx, j),
                    alpha=0.5, contour_thickness=1,
                )
        frames.append(frame)

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")

    H_out, W_out = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W_out, H_out)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return frames, output_path


# ── model ─────────────────────────────────────────────────────────────────────

class DiCoModel:
    """
    Loaded Divide-and-Conquer model.  Instantiate once and reuse.
    Use :meth:`get_or_load` to share a cached instance across calls.

    Parameters
    ----------
    cutler_config   : path to CutLER YAML config
    cutler_weights  : path to CutLER checkpoint (.pth)
    backbone_model_id : HuggingFace model ID for DINOv3 backbone
    confidence      : CutLER detection score threshold
    """

    _cache: "dict[tuple, DiCoModel]" = {}

    def __init__(
        self,
        cutler_config:    str   = DEFAULT_CUTLER_CONFIG,
        cutler_weights:   str   = DEFAULT_CUTLER_WEIGHTS,
        backbone_model_id: str  = DEFAULT_BACKBONE_ID,
        confidence:       float = 0.1,
    ):
        cfg = build_cutler_config(
            cutler_config, cutler_weights, confidence=confidence
        )
        self.predictor = CutlerPredictor(cfg)
        self.backbone  = ViTFeatV3(model_id=backbone_model_id).eval()
        self._patch_size = self.backbone.patch_size
        self._feat_dim   = self.backbone.feat_dim

    @classmethod
    def get_or_load(
        cls,
        cutler_config:    str   = DEFAULT_CUTLER_CONFIG,
        cutler_weights:   str   = DEFAULT_CUTLER_WEIGHTS,
        backbone_model_id: str  = DEFAULT_BACKBONE_ID,
        confidence:       float = 0.1,
    ) -> "DiCoModel":
        """Return a cached instance, loading models only on first call."""
        key = (cutler_config, cutler_weights, backbone_model_id, confidence)
        if key not in cls._cache:
            cls._cache[key] = cls(
                cutler_config, cutler_weights, backbone_model_id, confidence
            )
        return cls._cache[key]

    # ── divide ────────────────────────────────────────────────────────────────

    def divide(self, image_rgb: np.ndarray) -> list[np.ndarray]:
        """
        Run CutLER on *image_rgb* and return divide masks.

        Returns
        -------
        List of (H, W) bool arrays sorted by detection confidence.
        """
        bgr   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        preds = self.predictor(bgr)
        masks_t = preds["instances"].get("pred_masks")
        return [masks_t[i].cpu().numpy() for i in range(masks_t.shape[0])]

    # ── conquer ───────────────────────────────────────────────────────────────

    def conquer(
        self,
        image_rgb:    np.ndarray,
        divide_masks: list[np.ndarray],
        mask_idx:     int,
        local_size:   int   = 512,
        kept_thresh:  float = 0.8,
        nms_iou:      float = 0.8,
        nms_step:     int   = 5,
        thetas:       Sequence[float] = DEFAULT_THETAS,
        similarity_mode: str          = "cosine",
        rbf_sigma:    "float | None"  = None,
        adaptive:     bool            = False,
    ) -> list[np.ndarray]:
        """
        Run the conquer phase for one divide mask.

        Parameters
        ----------
        image_rgb, divide_masks, mask_idx
            Input image and divide mask list from :meth:`divide`.
        local_size      : resize the crop to this before feature extraction.
        kept_thresh     : minimum coverage to keep a conquer candidate.
        nms_iou         : IoU threshold for NMS.
        nms_step        : NMS window size.
        thetas          : merging thresholds (absolute or quantile fractions).
        similarity_mode : ``"cosine"`` or ``"rbf"``.
        rbf_sigma       : RBF bandwidth; ``None`` = auto.
        adaptive        : if True, threshes are quantile fractions.

        Returns
        -------
        List of (H, W) uint8 arrays at full image size, surviving NMS.
        """
        divide_mask = divide_masks[int(mask_idx)]
        ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
        ymax, xmax = ymax + 1, xmax + 1
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            return []

        local_rgb = image_rgb[ymin:ymax, xmin:xmax]
        crop_pil  = Image.fromarray(local_rgb).resize([local_size, local_size])
        feat_num  = local_size // self._patch_size
        feat_mat  = extract_feature_matrix(
            self.backbone, crop_pil, self._feat_dim, feat_num
        )

        conquer_masks: list[np.ndarray] = []
        for layer in iterative_merge(
            feat_mat, list(thetas),
            similarity_mode=similarity_mode,
            rbf_sigma=rbf_sigma,
            adaptive=adaptive,
        ):
            if layer.shape[0] == 0:
                continue
            for i in range(layer.shape[0]):
                mask     = resize_mask(layer[i], [xmax - xmin, ymax - ymin])
                mask_bin = (mask > 0.5 * 255).astype(np.int32)
                if coverage(mask_bin, divide_mask[ymin:ymax, xmin:xmax]) <= kept_thresh:
                    continue
                full = np.zeros(divide_mask.shape, dtype=np.uint8)
                full[ymin:ymax, xmin:xmax] = mask_bin
                conquer_masks.append(full)

        return NMS(conquer_masks, nms_iou, nms_step)

    def conquer_all(
        self,
        image_rgb:    np.ndarray,
        divide_masks: list[np.ndarray],
        local_size:   int   = 512,
        kept_thresh:  float = 0.8,
        nms_iou:      float = 0.8,
        nms_step:     int   = 5,
        thetas:       Sequence[float] = DEFAULT_THETAS,
        similarity_mode: str          = "cosine",
        rbf_sigma:    "float | None"  = None,
        adaptive:     bool            = False,
    ) -> "dict[int, list[np.ndarray]]":
        """Run :meth:`conquer` on every divide mask; returns ``{idx: [mask, ...]}}``."""
        result: dict[int, list[np.ndarray]] = {}
        for i in range(len(divide_masks)):
            masks = self.conquer(
                image_rgb, divide_masks, i,
                local_size=local_size, kept_thresh=kept_thresh,
                nms_iou=nms_iou, nms_step=nms_step, thetas=thetas,
                similarity_mode=similarity_mode, rbf_sigma=rbf_sigma, adaptive=adaptive,
            )
            if masks:
                result[i] = masks
        return result

    # ── feature PCA ───────────────────────────────────────────────────────────

    def feature_pca(
        self,
        image_rgb:    np.ndarray,
        divide_masks: list[np.ndarray],
        mask_idx:     int,
        local_size:   int = 512,
    ) -> np.ndarray:
        """Extract backbone features for the selected crop and return PCA-RGB."""
        divide_mask = divide_masks[int(mask_idx)]
        ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
        ymax, xmax = ymax + 1, xmax + 1
        local_rgb = image_rgb[ymin:ymax, xmin:xmax]
        crop_pil  = Image.fromarray(local_rgb).resize([local_size, local_size])
        feat_num  = local_size // self._patch_size
        feat_mat  = extract_feature_matrix(
            self.backbone, crop_pil, self._feat_dim, feat_num
        )
        feat_np = feat_mat.numpy() if hasattr(feat_mat, "numpy") else feat_mat
        return feature_pca_rgb(feat_np, upsample_to=local_rgb.shape[:2])
