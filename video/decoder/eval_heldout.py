"""Held-out box->mask segmentation eval on DAVIS clips with ZERO frames in davis_train.

Clean promptable-segmentation protocol: for each GT instance, feed its GT box, predict a
mask, score IoU vs that GT. Aggregates over many clips/frames so the number is trustworthy
(unlike the 8-instance spot-check). Clip split comes from davis_split.json (content-matched,
leak-free). Box-only by default — the pure segmenter metric for the v1..v3 lineage.

    python -m video.decoder.eval_heldout --ckpt checkpoints/unsup_decoder_v3.pth --frames 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis
from video.scripts.visualize_divide import gt_masks
from video.decoder.train_sam_decoder import UnsupervisedDecoder, _IMG
from video.decoder.infer_decoder import box_from_mask, iou

_SPLIT = Path(__file__).resolve().parent / "davis_split.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unsup_decoder_v3.pth")
    ap.add_argument("--frames", type=int, default=3, help="frames sampled evenly per clip")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--limit-clips", type=int, default=0, help="0 = all clean clips")
    ap.add_argument("--box-jitter", type=float, default=0.0,
                    help="perturb each GT box corner by this fraction of box size (tests "
                         "robustness to imperfect prompts; 0 = exact GT box).")
    args = ap.parse_args()
    np.random.seed(0)                          # deterministic jitter across runs

    clean = json.load(open(_SPLIT))["clean"]
    if args.limit_clips:
        clean = clean[:args.limit_clips]

    dino = DenseDINOv3()
    model = UnsupervisedDecoder().cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])

    all_ious, per_clip = [], {}
    for clip in clean:
        n = davis.num_frames(clip)
        idxs = sorted(set(np.linspace(0, n - 1, args.frames).astype(int).tolist()))
        clip_ious = []
        for idx in idxs:
            rgb = davis.load_frame(clip, idx)
            H, W = rgb.shape[:2]
            gts = gt_masks(clip, idx)
            boxes = [b for b in (box_from_mask(g, W, H) for g in gts) if b is not None]
            if not boxes:
                continue
            if args.box_jitter > 0:
                j = []
                for x0, y0, x1, y1 in boxes:
                    jx, jy = (x1 - x0) * args.box_jitter, (y1 - y0) * args.box_jitter
                    j.append([x0 + np.random.uniform(-jx, jx), y0 + np.random.uniform(-jy, jy),
                              x1 + np.random.uniform(-jx, jx), y1 + np.random.uniform(-jy, jy)])
                boxes = j
            img1024 = cv2.resize(rgb, (_IMG, _IMG))
            with torch.no_grad():
                feats = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()
                low = model(feats, torch.tensor(boxes, dtype=torch.float, device="cuda"))
                up = F.interpolate(low, (H, W), mode="bilinear", align_corners=False).sigmoid()[:, 0].cpu().numpy()
            preds = [(p > args.thresh).astype(np.uint8) for p in up]
            clip_ious += [iou(g, p) for g, p in zip(gts, preds)]
        if clip_ious:
            per_clip[clip] = float(np.mean(clip_ious))
            all_ious += clip_ious

    print(f"[ckpt] {args.ckpt}")
    print(f"[heldout] {len(per_clip)} clips, {len(all_ious)} instances, "
          f"frames/clip={args.frames}")
    print(f"[mIoU] {np.mean(all_ious):.4f}  (median {np.median(all_ious):.4f})")
    worst = sorted(per_clip.items(), key=lambda kv: kv[1])[:5]
    best = sorted(per_clip.items(), key=lambda kv: kv[1])[-5:]
    print("  worst clips:", [f"{c}={v:.2f}" for c, v in worst])
    print("  best  clips:", [f"{c}={v:.2f}" for c, v in best])


if __name__ == "__main__":
    main()
