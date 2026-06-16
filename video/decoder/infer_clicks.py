"""Visualize the points-prompted unsupervised decoder.

Iterative k-click prediction (same protocol as eval_clicks.py): click 0 = mask
centre (distance-transform peak); clicks 1..k-1 are SAM-style corrections
(positive on misses, negative on leaks), with the previous mask fed back as the
dense prompt. Draws the click points on the input panel alongside the
prediction and GT.

    python -m video.decoder.infer_clicks --ckpt checkpoints/unsup_decoder_points_v2.pth \
        --clips blackswan,bmx-trees,dogs-jump --clicks 3 --out decoder_points_v2.png
"""
from __future__ import annotations

import argparse
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
from video.scripts.visualize_divide import paint, label, gt_masks
from video.decoder.train_sam_decoder import UnsupervisedDecoder, sample_clicks, next_click, _IMG
from video.decoder.infer_decoder import iou


def predict_with_clicks(model, feats, gts, k):
    """Iterative k-click prediction. Returns (logits256, coords_1024, labels, kept_gts)."""
    keep = [g for g in gts if g.sum() > 0]
    if not keep:
        return None, [], [], []
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
    return prev, coords, labels, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unsup_decoder_points_v2.pth")
    ap.add_argument("--clips", default="blackswan,bmx-trees,dogs-jump")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--clicks", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="decoder_points_try.png")
    args = ap.parse_args()

    dino = DenseDINOv3()
    model = UnsupervisedDecoder().cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])
    print(f"[load] {args.ckpt}")

    rows = []
    for clip in [c.strip() for c in args.clips.split(",")]:
        rgb = davis.load_frame(clip, args.frame)
        H, W = rgb.shape[:2]
        gts = gt_masks(clip, args.frame)
        img1024 = cv2.resize(rgb, (_IMG, _IMG))
        with torch.no_grad():
            feats = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()
        logits, coords, labels, kept = predict_with_clicks(model, feats, gts, args.clicks)
        if logits is None:
            continue
        up = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False).sigmoid()[:, 0].cpu().numpy()
        preds = [(p > args.thresh).astype(np.uint8) for p in up]
        ious = [iou(g, p) for g, p in zip(kept, preds)]

        # draw the final click set on the input panel
        clicked = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        sx, sy = W / _IMG, H / _IMG
        for inst_coords, inst_labels in zip(coords, labels):
            for (x, y), lab in zip(inst_coords, inst_labels):
                colour = (0, 255, 0) if lab > 0.5 else (0, 0, 255)
                cv2.circle(clicked, (int(x * sx), int(y * sy)), 6, colour, -1)
                cv2.circle(clicked, (int(x * sx), int(y * sy)), 6, (255, 255, 255), 1)

        rows.append(np.concatenate([
            label(clicked, f"{clip}:{args.frame} {args.clicks}-click prompts"),
            label(paint(rgb, preds), f"decoder pred  mIoU={np.mean(ious):.2f}"),
            label(paint(rgb, kept), "GT"),
        ], axis=1))
        print(f"{clip}: {len(preds)} masks, IoUs={[round(x, 2) for x in ious]}")

    mh = min(r.shape[0] for r in rows)
    rows = [cv2.resize(r, (int(r.shape[1] * mh / r.shape[0]), mh)) for r in rows]
    cv2.imwrite(args.out, np.concatenate(rows, axis=0))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
