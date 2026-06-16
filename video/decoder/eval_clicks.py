"""Held-out click->mask eval on DAVIS clips with ZERO frames in davis_train.

Promptable-clicks protocol: for each GT instance, sample K positive clicks from the GT
mask (click 0 = distance-transform peak, the rest uniform interior), feed them as point
prompts, predict a mask, score IoU vs that GT. Aggregates over the leak-free 64-clip split
(davis_split.json) so the number is trustworthy. Reports 1-click and 3-click mIoU, directly
comparable to eval_heldout.py's box number (v3 box-only = 0.538).

    python -m video.decoder.eval_clicks --ckpt checkpoints/unsup_decoder_points_v1.pth --frames 3
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
from video.decoder.train_sam_decoder import UnsupervisedDecoder, sample_clicks, next_click, _IMG
from video.decoder.infer_decoder import iou

_SPLIT = Path(__file__).resolve().parent / "davis_split.json"


def predict(model, feats, gts, k, W, H, thresh):
    """Iterative k-click prediction (SAM protocol). Click 0 = mask centre; clicks 1..k-1 are
    deterministic correction clicks (positive on misses, NEGATIVE on leaks) placed in the
    current error region, with the previous mask fed back as the dense prompt. Returns binary
    masks at original HxW. This mirrors how a user actually refines, and matches training."""
    keep = [g for g in gts if g.sum() > 0]
    if not keep:
        return [], []
    gt256 = [cv2.resize(g, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8) for g in keep]
    N = len(keep)
    coords, labels = [], []
    for g in gt256:
        pts = sample_clicks(g, 1)
        x, y = pts[0] if pts else (128.0, 128.0)
        coords.append([[x * 4.0, y * 4.0]]); labels.append([1.0])

    prev = None
    for it in range(k):
        c = torch.tensor(coords, dtype=torch.float, device="cuda")
        l = torch.tensor(labels, dtype=torch.float, device="cuda")
        with torch.no_grad():
            prev = model(feats, points=(c, l), mask_prompts=prev)
        if it < k - 1:
            pred = (prev[:, 0].sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
            for i in range(N):
                nc = next_click(gt256[i], pred[i])
                if nc is None:
                    coords[i].append(coords[i][-1]); labels[i].append(labels[i][-1])
                else:
                    x, y, lab = nc
                    coords[i].append([x, y]); labels[i].append(lab)
    up = F.interpolate(prev, (H, W), mode="bilinear", align_corners=False).sigmoid()[:, 0].cpu().numpy()
    return [(p > thresh).astype(np.uint8) for p in up], keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unsup_decoder_points_v1.pth")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--clicks", default="1,3", help="comma list of click counts to report")
    ap.add_argument("--limit-clips", type=int, default=0)
    args = ap.parse_args()
    ks = [int(x) for x in args.clicks.split(",")]

    clean = json.load(open(_SPLIT))["clean"]
    if args.limit_clips:
        clean = clean[:args.limit_clips]

    dino = DenseDINOv3()
    model = UnsupervisedDecoder().cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])

    print(f"[ckpt] {args.ckpt}")
    for k in ks:
        np.random.seed(0)                          # deterministic click sampling per k
        all_ious, per_clip = [], {}
        for clip in clean:
            n = davis.num_frames(clip)
            idxs = sorted(set(np.linspace(0, n - 1, args.frames).astype(int).tolist()))
            clip_ious = []
            for idx in idxs:
                rgb = davis.load_frame(clip, idx)
                H, W = rgb.shape[:2]
                gts = gt_masks(clip, idx)
                if not gts:
                    continue
                img1024 = cv2.resize(rgb, (_IMG, _IMG))
                with torch.no_grad():
                    feats = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()
                preds, kept = predict(model, feats, gts, k, W, H, args.thresh)
                clip_ious += [iou(g, p) for g, p in zip(kept, preds)]
            if clip_ious:
                per_clip[clip] = float(np.mean(clip_ious))
                all_ious += clip_ious
        worst = sorted(per_clip.items(), key=lambda kv: kv[1])[:3]
        print(f"[{k}-click] {len(per_clip)} clips, {len(all_ious)} instances | "
              f"mIoU {np.mean(all_ious):.4f}  (median {np.median(all_ious):.4f})  "
              f"worst {[f'{c}={v:.2f}' for c, v in worst]}")


if __name__ == "__main__":
    main()
