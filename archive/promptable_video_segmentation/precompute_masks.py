"""
Precompute Divide-and-Conquer masks for a directory of images.

For each image, the divide phase (CutLER) and conquer phase (DINOv3 +
iterative spectral merging) are run, and all resulting masks are saved as a
compressed .npz file alongside metadata needed to reconstruct the hierarchy.

NPZ schema (one file per image):
    image_path  : bytes  — absolute path to the source image
    masks       : uint8  (N, H, W) — all masks in order: divide, then conquer
    mask_type   : int8   (N,)      — 0 = divide mask, 1 = conquer mask
    divide_idx  : int16  (N,)      — for divide masks: -1
                                     for conquer masks: index of parent divide mask

Edit the configuration block below and run:
    python precompute_masks.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import PIL.Image as Image

# ── resolve paths relative to this file ──────────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, _REPO)

from promptable_video_segmentation.masks import (
    NMS,
    coverage,
    iterative_merge,
    resize_mask,
    smallest_square_containing_mask,
)
from promptable_video_segmentation.backbone import ViTFeatV3, extract_feature_matrix


# ── configure here ────────────────────────────────────────────────────────────
IMAGE_DIR        = "/path/to/images"
OUTPUT_DIR       = "/path/to/output_masks"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

CUTLER_CONFIG  = os.path.join(
    _REPO, "divide_and_conquer",
    "model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
)
CUTLER_WEIGHTS = os.path.join(
    _REPO, "divide_and_conquer",
    "model_zoo/ckpts/cutler_cascade_final.pth",
)

BACKBONE_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEVICE            = "cuda"
CONFIDENCE_THRESH = 0.1

LOCAL_SIZE   = 512    # backbone input resolution for conquer crops
KEPT_THRESH  = 0.8    # min coverage fraction to keep a conquer candidate
NMS_IOU      = 0.8    # IoU threshold for conquer NMS
NMS_STEP     = 5      # NMS window width
THETAS       = [0.73, 0.62, 0.51, 0.4, 0.3, 0.2]  # merging thresholds (high→low)

SKIP_EXISTING = True  # skip images whose .npz already exists in OUTPUT_DIR
# ─────────────────────────────────────────────────────────────────────────────


# ── CutLER predictor setup ────────────────────────────────────────────────────

def _make_cutler_predictor(config_file: str, weights: str, device: str, conf: float):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()

    # Register CutLER-specific fields before merging the YAML.
    cfg.DATALOADER.COPY_PASTE           = False
    cfg.DATALOADER.COPY_PASTE_RATE      = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM   = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE    = False
    cfg.MODEL.ROI_HEADS.USE_DROPLOSS       = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS    = False
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE  = False
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
    cfg.SOLVER.BASE_LR_MULTIPLIER       = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES = []
    cfg.TEST.NO_SEGM                    = False

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights, "MODEL.DEVICE", device])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf
    cfg.freeze()
    return DefaultPredictor(cfg)


# ── Per-image inference ───────────────────────────────────────────────────────

def _run_divide(predictor, image_bgr: np.ndarray) -> list[np.ndarray]:
    preds   = predictor(image_bgr)
    masks_t = preds["instances"].get("pred_masks")
    return [masks_t[i].cpu().numpy() for i in range(masks_t.shape[0])]


def _run_conquer(
    backbone: ViTFeatV3,
    image_rgb: np.ndarray,
    divide_mask: np.ndarray,
) -> list[np.ndarray]:
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
    if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
        return []

    local_rgb = image_rgb[ymin:ymax, xmin:xmax]
    resized   = Image.fromarray(local_rgb).resize([LOCAL_SIZE, LOCAL_SIZE])
    feat_num  = LOCAL_SIZE // backbone.patch_size
    feat      = extract_feature_matrix(backbone, resized, backbone.feat_dim, feat_num)

    candidates: list[np.ndarray] = []
    for layer in iterative_merge(feat, THETAS):
        if layer.shape[0] == 0:
            continue
        for i in range(layer.shape[0]):
            mask     = resize_mask(layer[i], [xmax - xmin, ymax - ymin])
            mask_bin = (mask > 0.5 * 255).astype(int)
            if coverage(mask_bin, divide_mask[ymin:ymax, xmin:xmax]) <= KEPT_THRESH:
                continue
            full = np.zeros_like(divide_mask)
            full[ymin:ymax, xmin:xmax] = mask_bin
            candidates.append(full)

    return NMS(candidates, NMS_IOU, NMS_STEP)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])
    print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

    print("Loading CutLER…")
    predictor = _make_cutler_predictor(CUTLER_CONFIG, CUTLER_WEIGHTS, DEVICE, CONFIDENCE_THRESH)

    print("Loading DINOv3 backbone…")
    backbone = ViTFeatV3(model_id=BACKBONE_MODEL_ID).eval()

    n_done = n_skipped = n_failed = 0

    for img_idx, img_path in enumerate(image_paths):
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.npz")
        prefix   = f"[{img_idx + 1}/{len(image_paths)}]"

        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"{prefix} skip (exists): {stem}")
            n_skipped += 1
            continue

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"{prefix} WARNING: cannot read {img_path}")
            n_failed += 1
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        try:
            divide_masks = _run_divide(predictor, image_bgr)
        except Exception as exc:
            print(f"{prefix} ERROR in divide: {exc}")
            n_failed += 1
            continue

        if not divide_masks:
            print(f"{prefix} no divide masks, skipping: {stem}")
            n_skipped += 1
            continue

        print(f"{prefix} {stem}  —  {len(divide_masks)} divide masks")

        all_masks:      list[np.ndarray] = []
        mask_type_list: list[int]        = []
        divide_idx_list: list[int]       = []

        for d_idx, dmask in enumerate(divide_masks):
            all_masks.append(dmask.astype(np.uint8))
            mask_type_list.append(0)
            divide_idx_list.append(-1)

            try:
                c_masks = _run_conquer(backbone, image_rgb, dmask)
            except Exception as exc:
                print(f"  divide[{d_idx}] conquer ERROR: {exc}")
                c_masks = []

            print(f"  divide[{d_idx}]: {len(c_masks)} conquer masks")
            for cmask in c_masks:
                all_masks.append(cmask.astype(np.uint8))
                mask_type_list.append(1)
                divide_idx_list.append(d_idx)

        np.savez_compressed(
            out_path,
            image_path  = np.bytes_(img_path),
            masks       = np.stack(all_masks, axis=0),                      # (N, H, W) uint8
            mask_type   = np.array(mask_type_list, dtype=np.int8),          # (N,)
            divide_idx  = np.array(divide_idx_list, dtype=np.int16),        # (N,)
        )
        print(f"  saved {len(all_masks)} masks → {out_path}")
        n_done += 1

    print(f"\nDone.  processed={n_done}  skipped={n_skipped}  failed={n_failed}")
