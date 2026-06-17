"""Run the trained unsupervised decoder and visualize its masks.

Box-prompted (same as training): give it a box around each object, it predicts a
mask. Prompts come from GT instances (clean eval) or the pseudo-masks.

    python -m video.decoder.infer_decoder --ckpt checkpoints/unsup_decoder.pth \
        --clips blackswan,bmx-trees,dogs-jump --out decoder_try.png
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
from video.decoder.train_sam_decoder import UnsupervisedDecoder, _IMG, degrade_mask


def box_from_mask(m, W, H):
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return None
    sx, sy = _IMG / W, _IMG / H
    return [xs.min() * sx, ys.min() * sy, xs.max() * sx, ys.max() * sy]


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unsup_decoder.pth")
    ap.add_argument("--clips", default="blackswan,bmx-trees,dogs-jump")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--use-mask-prompt", action="store_true",
                    help="feed a degraded copy of each GT mask as a coarse dense prompt "
                         "(matches a box+mask-trained checkpoint; tests refinement).")
    ap.add_argument("--out", default="decoder_try.png")
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
        boxes, mprompts = [], []
        for g in gts:
            b = box_from_mask(g, W, H)
            if b is None:
                continue
            boxes.append(b)
            if args.use_mask_prompt:
                m256 = cv2.resize(g.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                mprompts.append(degrade_mask(m256))
        if not boxes:
            continue
        img1024 = cv2.resize(rgb, (_IMG, _IMG))
        with torch.no_grad():
            feats = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()
            mask_prompts = (torch.tensor(np.stack(mprompts), dtype=torch.float, device="cuda")[:, None]
                            if args.use_mask_prompt else None)
            low = model(feats, torch.tensor(boxes, dtype=torch.float, device="cuda"), mask_prompts)
            up = F.interpolate(low, (H, W), mode="bilinear", align_corners=False).sigmoid()[:, 0].cpu().numpy()
        preds = [(p > args.thresh).astype(np.uint8) for p in up]
        ious = [iou(g, p) for g, p in zip(gts, preds)]

        # draw prompt boxes on the input panel
        boxed = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        for b in boxes:
            x0, y0, x1, y1 = [int(v * (W / _IMG if i % 2 == 0 else H / _IMG)) for i, v in enumerate(b)]
            cv2.rectangle(boxed, (x0, y0), (x1, y1), (0, 255, 255), 2)
        rows.append(np.concatenate([
            label(boxed, f"{clip}:{args.frame} box prompts"),
            label(paint(rgb, preds), f"decoder pred  mIoU={np.mean(ious):.2f}"),
            label(paint(rgb, gts), "GT"),
        ], axis=1))
        print(f"{clip}: {len(preds)} masks, IoUs={[round(x,2) for x in ious]}")

    mh = min(r.shape[0] for r in rows)
    rows = [cv2.resize(r, (int(r.shape[1] * mh / r.shape[0]), mh)) for r in rows]
    cv2.imwrite(args.out, np.concatenate(rows, axis=0))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
