"""
dico.py — Divide-and-Conquer segmentation: functional API

All methods run from the divide_and_conquer/ directory (same as the other scripts).

Quickstart
----------
    from dico import DiCoModel, visualize_masks, visualize_conquer_crop, visualize_all
    from dico import make_synthetic_video
    import numpy as np
    import PIL.Image as Image

    model = DiCoModel("model_zoo/ckpts/cutler_cascade_final.pth")
    image = np.array(Image.open("photo.jpg").convert("RGB"))

    divide_masks = model.divide(image)
    conquer_masks = model.conquer(image, divide_masks, mask_idx=0)
    conquer_per_divide = model.conquer_all(image, divide_masks)

    vis_divide   = visualize_masks(image, divide_masks)
    vis_crop     = visualize_conquer_crop(image, divide_masks, 0, conquer_masks)
    vis_full     = visualize_all(image, divide_masks, conquer_per_divide)
    feat_vis     = model.feature_pca(image, divide_masks, 0)

    frames, path = make_synthetic_video(image, divide_masks, conquer_per_divide,
                                        output_path="out.mp4")
"""

from __future__ import annotations

import tempfile
from typing import Sequence

import cv2
import numpy as np
import torch
import PIL.Image as Image

from demo_dico import (
    NMS,
    coverage,
    generate_feature_matrix,
    get_parser as _get_dico_parser,
    resize_mask,
    smallest_square_containing_mask,
)
from divide_conquer import setup_cfg
from engine.defaults import DefaultPredictor
from iterative_merging import iterative_merge
import dinov3 as _dinov3
import dino as _dino_module
from synthetic_video import _build_aug_list, _aug_frame, _render_frame


# ── Colour palettes ──────────────────────────────────────────────────────────

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
    """Colour for the i-th divide mask."""
    return PALETTE[idx % len(PALETTE)]


def conquer_color(divide_idx: int, conquer_idx: int) -> list[int]:
    """Colour for the j-th conquer mask inside the i-th divide region."""
    return CONQUER_PALETTE[(divide_idx * 3 + conquer_idx) % len(CONQUER_PALETTE)]


# ── Visualization ────────────────────────────────────────────────────────────

def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int],
    alpha: float = 0.45,
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    Blend *mask* onto *image_rgb* with *color* at *alpha*, optionally drawing
    a contour of the given thickness (0 = no contour).

    Parameters
    ----------
    image_rgb         : (H, W, 3) uint8 RGB array
    mask              : (H, W) binary array
    color             : RGB colour, e.g. [255, 0, 0]
    alpha             : blend weight for the mask fill (0 = invisible, 1 = solid)
    contour_thickness : pixel width of the contour drawn around the mask edge

    Returns
    -------
    (H, W, 3) uint8 RGB array
    """
    out = image_rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if contour_thickness > 0:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, list(color), contour_thickness)
    return out


def visualize_masks(
    image_rgb: np.ndarray,
    masks: list[np.ndarray],
    alpha: float = 0.45,
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    Overlay each mask in *masks* on *image_rgb* using a distinct palette colour.

    Useful for any flat list of masks — divide masks, conquer masks, or any
    other collection.

    Returns
    -------
    (H, W, 3) uint8 RGB array
    """
    out = image_rgb.copy()
    for i, mask in enumerate(masks):
        out = overlay_mask(out, mask, palette_color(i), alpha, contour_thickness)
    return out


def visualize_conquer_crop(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    mask_idx: int,
    conquer_masks: list[np.ndarray],
) -> np.ndarray:
    """
    Hierarchical composite view of one divide crop with all its conquer masks.

    Masks are rendered largest-first so smaller sub-segments sit on top.
    Overlapping pixels are restored to the original image before each new mask
    is painted, preventing double-blending.

    Parameters
    ----------
    image_rgb     : full (H, W, 3) uint8 RGB image
    divide_masks  : list of full-image divide masks
    mask_idx      : which divide mask to use for cropping
    conquer_masks : full-image conquer masks for this divide region

    Returns
    -------
    (crop_H, crop_W, 3) uint8 RGB array — cropped to the divide bounding box
    """
    divide_mask = divide_masks[int(mask_idx)]
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
    crop_rgb = image_rgb[ymin:ymax, xmin:xmax]

    crop_masks = [
        (m[ymin:ymax, xmin:xmax] > 0.5).astype(np.uint8) for m in conquer_masks
    ]
    order = sorted(range(len(crop_masks)), key=lambda i: -int(np.sum(crop_masks[i])))

    I_orig = np.array(crop_rgb).copy()
    result = I_orig.copy()
    already_painted = np.zeros(I_orig.shape[:2], dtype=np.uint8)
    rng = np.random.default_rng(42)

    for i in order:
        mask = crop_masks[i]
        already_painted += mask
        overlap = already_painted == 2
        if np.sum(overlap) > 0:
            result[overlap] = I_orig[overlap]
            already_painted[overlap] -= 1
        color = rng.integers(64, 230, 3).tolist()
        fg = mask.astype(bool)
        result[fg] = (result[fg] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

    return result


def visualize_all(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    conquer_per_divide: dict[int, list[np.ndarray]] | None = None,
    alpha_divide: float = 0.45,
    alpha_conquer: float = 0.5,
) -> np.ndarray:
    """
    Full-image overlay: divide masks as a base layer, conquer sub-masks on top.

    Conquer masks use a visually distinct secondary palette and are drawn after
    the divide fills so they remain visible (conquer regions are always subsets
    of their parent divide region).

    Parameters
    ----------
    image_rgb          : (H, W, 3) uint8 RGB image
    divide_masks       : list of (H, W) binary arrays
    conquer_per_divide : {divide_idx: [conquer_mask, ...]}. Pass None or {}
                         for a divide-only visualisation.
    alpha_divide       : blend strength for divide mask fills
    alpha_conquer      : blend strength for conquer mask fills

    Returns
    -------
    (H, W, 3) uint8 RGB array
    """
    out = image_rgb.copy()

    for i, mask in enumerate(divide_masks):
        out = overlay_mask(out, mask, palette_color(i), alpha_divide, contour_thickness=2)

    if conquer_per_divide:
        for div_idx, c_masks in conquer_per_divide.items():
            for j, mask in enumerate(c_masks):
                out = overlay_mask(
                    out, mask, conquer_color(int(div_idx), j),
                    alpha_conquer, contour_thickness=1,
                )

    return out


def feature_pca_rgb(
    feat_matrix: np.ndarray,
    upsample_to: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Project a (H, W, C) feature map to a 3-channel RGB image via PCA.

    Parameters
    ----------
    feat_matrix : (H, W, C) float array of backbone features
    upsample_to : optional (H_out, W_out) to resize the result

    Returns
    -------
    (H_out, W_out, 3) uint8 RGB array
    """
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


# ── Synthetic video ──────────────────────────────────────────────────────────

def make_synthetic_video(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    conquer_per_divide: dict[int, list[np.ndarray]] | None = None,
    num_frames: int = 8,
    crop_min: float = 0.5,
    flip_prob: float = 0.5,
    fps: int = 4,
    output_path: str | None = None,
) -> tuple[list[np.ndarray], str]:
    """
    Generate a synthetic video via the VideoCutLER augmentation pipeline.

    Each frame independently applies:
      RandomCrop (relative_range, min fraction = crop_min)
      → ResizeShortestEdge [320, 480] (max 720)
      → RandomFlip (horizontal, prob = flip_prob)
      → brightness / contrast / saturation jitter [0.9, 1.1]

    All masks receive the same geometric transform per frame.  Conquer masks
    are drawn on top of divide masks with a distinct secondary palette.

    Parameters
    ----------
    image_rgb          : (H, W, 3) uint8 RGB image
    divide_masks       : list of (H, W) binary arrays from divide()
    conquer_per_divide : {divide_idx: [mask, ...]} from conquer_all().
                         Pass None or {} for divide-only video.
    num_frames         : number of frames to generate
    crop_min           : minimum crop fraction per axis (0 < crop_min <= 1)
    flip_prob          : probability of horizontal flip per frame
    fps                : output video frame rate
    output_path        : where to write the .mp4. Defaults to a temp file.

    Returns
    -------
    (frames, output_path) where frames is a list of (H, W, 3) uint8 RGB arrays.
    """
    conquer_per_divide = conquer_per_divide or {}
    H, W = image_rgb.shape[:2]
    n_divide = len(divide_masks)

    conquer_groups = [conquer_per_divide.get(i, []) for i in range(n_divide)]

    flat_masks = list(divide_masks)
    group_slices: list[tuple[int, int]] = []
    offset = n_divide
    for group in conquer_groups:
        group_slices.append((offset, offset + len(group)))
        flat_masks.extend(group)
        offset += len(group)

    aug_list = _build_aug_list(crop_min, flip_prob)
    frames = []
    for _ in range(int(num_frames)):
        aug_img, aug_flat = _aug_frame(image_rgb, flat_masks, aug_list, (H, W))
        aug_divide = aug_flat[:n_divide]
        aug_conquer_groups = [aug_flat[s:e] for s, e in group_slices]
        frames.append(_render_frame(aug_img, aug_divide, aug_conquer_groups))

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    return frames, output_path


# ── Model ────────────────────────────────────────────────────────────────────

class DiCoModel:
    """
    Loaded Divide-and-Conquer model. Instantiate once and reuse.

    Parameters
    ----------
    weights_path  : path to cutler_cascade_final.pth
    backbone_type : "dinov3" (default) or "dino"
    dino_url      : only used when backbone_type="dino"; URL or local path to
                    DINO-B/8 weights
    """

    def __init__(
        self,
        weights_path: str,
        backbone_type: str = "dinov3",
        dino_url: str = (
            "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/"
            "dino_vitbase8_pretrain.pth"
        ),
    ):
        args = _get_dico_parser(
            yaml_path="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"
        ).parse_args(["--opts", "MODEL.WEIGHTS", weights_path, "MODEL.DEVICE", "cuda"])

        cfg = setup_cfg(args)
        self.predictor = DefaultPredictor(cfg)
        self._args = args

        if backbone_type == "dinov3":
            self.backbone = _dinov3.ViTFeatV3(
                args.backbone_url, args.feature_dim, args.backbone_size, "k", args.patch_size,
            )
            self.backbone.eval().cuda()
        else:
            self.backbone = _dino_module.ViTFeat(dino_url, 768, "base", "k", 8)
            self.backbone.eval().cuda().half()
            self._args.feature_dim = 768
            self._args.patch_size = 8

    # ── Divide ───────────────────────────────────────────────────────────────

    def divide(self, image_rgb: np.ndarray) -> list[np.ndarray]:
        """
        Run CutLER on *image_rgb* and return divide masks.

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 RGB array

        Returns
        -------
        list of (H, W) bool arrays, one per detected instance, sorted by
        detection confidence (highest first, matching CutLER output order).
        """
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        preds = self.predictor(image_bgr)
        masks_t = preds["instances"].get("pred_masks")
        return [masks_t[i].cpu().numpy() for i in range(masks_t.shape[0])]

    # ── Conquer ──────────────────────────────────────────────────────────────

    def conquer(
        self,
        image_rgb: np.ndarray,
        divide_masks: list[np.ndarray],
        mask_idx: int,
        local_size: int = 512,
        kept_thresh: float = 0.8,
        nms_iou: float = 0.8,
        thetas: Sequence[float] = (0.73, 0.62, 0.51, 0.4, 0.3, 0.2),
    ) -> list[np.ndarray]:
        """
        Run the conquer phase for a single divide mask.

        Crops the image to the divide mask's bounding box, extracts backbone
        features, runs iterative spectral merging at each threshold in *thetas*,
        filters by coverage, and applies NMS.

        Parameters
        ----------
        image_rgb    : (H, W, 3) uint8 RGB array
        divide_masks : list of full-image divide masks (from divide())
        mask_idx     : index into divide_masks to process
        local_size   : resize the crop to this before feature extraction (px)
        kept_thresh  : minimum coverage fraction to keep a conquer candidate
        nms_iou      : IoU threshold for NMS
        thetas       : merging similarity thresholds, applied high → low

        Returns
        -------
        list of (H, W) int arrays at full image size, one per conquer region
        surviving NMS. Returns [] if the mask has zero area or all candidates
        are filtered out.
        """
        divide_mask = divide_masks[int(mask_idx)]
        ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            return []

        local_rgb = image_rgb[ymin:ymax, xmin:xmax]
        resized = Image.fromarray(local_rgb).resize([local_size, local_size])

        feat_num = local_size // self._args.patch_size
        feat_matrix = generate_feature_matrix(
            self.backbone, resized, self._args.feature_dim, feat_num
        )

        conquer_masks: list[np.ndarray] = []
        for layer in iterative_merge(feat_matrix, list(thetas)):
            if layer.shape[0] == 0:
                continue
            for i in range(layer.shape[0]):
                mask = resize_mask(layer[i], [xmax - xmin, ymax - ymin])
                mask_bin = (mask > 0.5 * 255).astype(int)
                if coverage(mask_bin, divide_mask[ymin:ymax, xmin:xmax]) <= kept_thresh:
                    continue
                full = np.zeros_like(divide_mask)
                full[ymin:ymax, xmin:xmax] = mask_bin
                conquer_masks.append(full)

        return NMS(conquer_masks, nms_iou, self._args.NMS_step)

    def conquer_all(
        self,
        image_rgb: np.ndarray,
        divide_masks: list[np.ndarray],
        local_size: int = 512,
        kept_thresh: float = 0.8,
        nms_iou: float = 0.8,
        thetas: Sequence[float] = (0.73, 0.62, 0.51, 0.4, 0.3, 0.2),
    ) -> dict[int, list[np.ndarray]]:
        """
        Run conquer on every divide mask and return results grouped by divide index.

        Parameters
        ----------
        Same as conquer(), minus mask_idx.

        Returns
        -------
        dict {divide_idx: [conquer_mask, ...]}. Only divide masks that produced
        at least one surviving conquer mask appear as keys.
        """
        result: dict[int, list[np.ndarray]] = {}
        for i in range(len(divide_masks)):
            masks = self.conquer(
                image_rgb, divide_masks, i,
                local_size=local_size,
                kept_thresh=kept_thresh,
                nms_iou=nms_iou,
                thetas=thetas,
            )
            if masks:
                result[i] = masks
        return result

    # ── Feature visualisation ────────────────────────────────────────────────

    def feature_pca(
        self,
        image_rgb: np.ndarray,
        divide_masks: list[np.ndarray],
        mask_idx: int,
        local_size: int = 512,
    ) -> np.ndarray:
        """
        Extract backbone features for the selected divide crop and return a
        PCA-coloured RGB image at the crop's native resolution.

        Parameters
        ----------
        image_rgb    : (H, W, 3) uint8 RGB array
        divide_masks : list of full-image divide masks (from divide())
        mask_idx     : which divide mask to inspect
        local_size   : internal resize for feature extraction (px)

        Returns
        -------
        (crop_H, crop_W, 3) uint8 RGB array
        """
        divide_mask = divide_masks[int(mask_idx)]
        ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
        local_rgb = image_rgb[ymin:ymax, xmin:xmax]
        resized = Image.fromarray(local_rgb).resize([local_size, local_size])
        feat_num = local_size // self._args.patch_size
        feat_matrix = generate_feature_matrix(
            self.backbone, resized, self._args.feature_dim, feat_num
        )
        feat_np = feat_matrix.numpy() if hasattr(feat_matrix, "numpy") else feat_matrix
        return feature_pca_rgb(feat_np, upsample_to=local_rgb.shape[:2])


if __name__ == "__main__":
    import os

    # ── configure here ────────────────────────────────────────────────────────
    IMAGE_PATH   = "../docs/demos/sa_234337.jpg"
    WEIGHTS      = "model_zoo/ckpts/cutler_cascade_final.pth"
    BACKBONE     = "dinov3"          # "dinov3" or "dino"
    OUT_DIR      = "output"

    LOCAL_SIZE   = 512
    KEPT_THRESH  = 0.8
    NMS_IOU      = 0.8
    THETAS       = [0.73, 0.62, 0.51, 0.4, 0.3, 0.2]

    NUM_FRAMES   = 8
    CROP_MIN     = 0.5
    FLIP_PROB    = 0.5
    FPS          = 4
    # ─────────────────────────────────────────────────────────────────────────

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image: {IMAGE_PATH}  ({image_rgb.shape[1]}×{image_rgb.shape[0]})")

    print("Loading models…")
    model = DiCoModel(WEIGHTS, BACKBONE)

    print("Running divide phase…")
    divide_masks = model.divide(image_rgb)
    print(f"  {len(divide_masks)} divide masks found")

    vis_divide = visualize_masks(image_rgb, divide_masks)
    divide_path = os.path.join(OUT_DIR, f"{stem}_divide.jpg")
    cv2.imwrite(divide_path, cv2.cvtColor(vis_divide, cv2.COLOR_RGB2BGR))
    print(f"  Saved → {divide_path}")

    print("Running conquer phase on all divide masks…")
    conquer_per_divide = model.conquer_all(
        image_rgb, divide_masks,
        local_size=LOCAL_SIZE,
        kept_thresh=KEPT_THRESH,
        nms_iou=NMS_IOU,
        thetas=THETAS,
    )
    n_conquer = sum(len(v) for v in conquer_per_divide.values())
    print(
        f"  {n_conquer} conquer masks across "
        f"{len(conquer_per_divide)}/{len(divide_masks)} divide regions"
    )

    vis_full = visualize_all(image_rgb, divide_masks, conquer_per_divide)
    full_path = os.path.join(OUT_DIR, f"{stem}_all_masks.jpg")
    cv2.imwrite(full_path, cv2.cvtColor(vis_full, cv2.COLOR_RGB2BGR))
    print(f"  Saved → {full_path}")

    print(f"Generating synthetic video ({NUM_FRAMES} frames @ {FPS} fps)…")
    video_path = os.path.join(OUT_DIR, f"{stem}_video.mp4")
    make_synthetic_video(
        image_rgb, divide_masks, conquer_per_divide,
        num_frames=NUM_FRAMES,
        crop_min=CROP_MIN,
        flip_prob=FLIP_PROB,
        fps=FPS,
        output_path=video_path,
    )
    print(f"  Saved → {video_path}")
