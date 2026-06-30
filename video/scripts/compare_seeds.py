"""Compare frame-0 seed quality: CuVLER+conquer vs Basti's DinoMaskExtraction.

Runs both pipelines on every DAVIS 2016 val clip (or a subset) at frame 0,
picks the mask containing the GT click point, reports IoU vs GT.

Usage:
    python -m video.scripts.compare_seeds
    python -m video.scripts.compare_seeds --clips breakdance,dog,blackswan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))
sys.path.insert(0, str(_REPO / "DinoMaskExtraction"))

from video.loaders import davis
from video.decoder.train_sam_decoder import sample_clicks
from video.divide.cuvler_divide import CuVLERDivider
from video.divide.conquer import load_backbone, run_conquer

from backbone import ViTFeatV3
from mask_extraction import get_extractor
from mask_extraction.preprocess import pca_reduce, spatial_smooth
from mask_extraction.postprocess import refine_crf

DAVIS_VAL_20 = [
    "blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout",
    "car-shadow", "cows", "dance-twirl", "dog", "drift-chicane",
    "drift-straight", "goat", "horsejump-high", "kite-surf", "libby",
    "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox",
]

_BASTI_RES = 512  # Basti's proven resolution

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 0.0


def pick_mask(masks: list[np.ndarray], gt: np.ndarray,
              click_xy: tuple[float, float]) -> np.ndarray | None:
    cx, cy = int(click_xy[0]), int(click_xy[1])
    containing = [m for m in masks if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1]
                  and m[cy, cx] > 0]
    if containing:
        return max(containing, key=lambda m: iou(m, gt))
    return max(masks, key=lambda m: iou(m, gt)) if masks else None


def labels_to_masks(labels: torch.Tensor, H: int, W: int) -> list[np.ndarray]:
    """Convert label map [h, w] → list of full-res binary masks [H, W]."""
    masks = []
    lbl_np = labels.cpu().numpy()
    for lid in np.unique(lbl_np):
        if lid == 0:
            continue
        m_small = (lbl_np == lid).astype(np.uint8)
        m_full = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(m_full)
    return masks


def basti_seeds(backbone: ViTFeatV3, extractor, frame_rgb: np.ndarray) -> list[np.ndarray]:
    H, W = frame_rgb.shape[:2]
    img = cv2.resize(frame_rgb, (_BASTI_RES, _BASTI_RES))
    t = _to_tensor(img).unsqueeze(0).cuda()
    with torch.no_grad():
        feats = backbone(t)  # [1, 1024, h*w]
    h = w = _BASTI_RES // 16  # 32x32 for 512px
    grid = feats[0].T.reshape(h, w, -1).cpu()  # [32, 32, 1024]
    # PCA-256 + spatial smoothing (Basti's proven params)
    grid_r = pca_reduce(grid, k=256)
    grid_s = spatial_smooth(grid_r, sigma=0.5)
    labels = extractor(grid_s)  # [32, 32]
    # optional CRF refinement
    img_gray = cv2.resize(frame_rgb, (_BASTI_RES, _BASTI_RES))
    labels = refine_crf(labels, img_gray, n_iter=5, radius=4, weight=1.5)
    return labels_to_masks(labels, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="",
                    help="Comma-separated clip names (default: all 20 val clips)")
    args = ap.parse_args()
    clips = [c.strip() for c in args.clips.split(",")] if args.clips else DAVIS_VAL_20

    print("Loading CuVLER+conquer…")
    divider = CuVLERDivider(score_thresh=0.35)
    conquer_bb = load_backbone()

    print("Loading Basti backbone…")
    backbone = ViTFeatV3()
    extractor = get_extractor(
        "greedy_multistage2",
        similarity="cosine",
        connectivity=8,
        quantile=0.65,
        min_mask_size=10,
        consistency_filter=True,
        consistency_min=0.5,
        merge_threshold=0.7,
        seam_thickness=3,
    )

    print(f"\n{'Clip':<22} {'CuVLER+conquer':>16} {'Basti gm2':>12} {'Winner':>8}")
    print("-" * 62)

    cuvler_scores, basti_scores = [], []

    for clip in clips:
        frame0 = davis.load_frame(clip, 0)
        H, W = frame0.shape[:2]
        inst_ids = davis.instance_ids(clip, 0)
        if not inst_ids:
            continue
        inst = inst_ids[0]
        gt0 = davis.load_mask(clip, 0, instance_id=inst)

        # click point: distance-transform peak of GT (same as eval script)
        gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        pts = sample_clicks(gt256, 1)
        cx256, cy256 = pts[0]
        click_xy = (cx256 * (W / 256.0), cy256 * (H / 256.0))

        # CuVLER + conquer
        props = divider.predict(frame0)
        props = run_conquer(conquer_bb, frame0, props)
        seed_c = pick_mask(props, gt0, click_xy)
        iou_c = iou(seed_c, gt0) if seed_c is not None else 0.0

        # Basti greedy_multistage2
        try:
            masks_b = basti_seeds(backbone, extractor, frame0)
            seed_b = pick_mask(masks_b, gt0, click_xy)
            iou_b = iou(seed_b, gt0) if seed_b is not None else 0.0
        except Exception as e:
            iou_b = 0.0
            print(f"  [{clip}] Basti error: {e}")

        winner = "BASTI" if iou_b > iou_c + 0.01 else ("CUVLER" if iou_c > iou_b + 0.01 else "tie")
        print(f"  {clip:<20} {iou_c:>16.3f} {iou_b:>12.3f} {winner:>8}")
        cuvler_scores.append(iou_c)
        basti_scores.append(iou_b)

    print("-" * 62)
    print(f"  {'MEAN':<20} {np.mean(cuvler_scores):>16.3f} {np.mean(basti_scores):>12.3f}")


if __name__ == "__main__":
    main()
