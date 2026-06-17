"""Evaluate the mask-prompt decoder on REAL production pseudo-masks (no degraded GT).

Feeds the actual CuVLER *divide* masks and the full *divide+conquer* masks as the
coarse dense prompt, lets the decoder refine each, and measures IoU-to-GT before vs
after refinement. This is the true-pipeline test: the decoder is meant to clean up
the coarse masks the divide-and-conquer stage actually produces.

Note: the decoder's training target IS the pseudo-mask, so it can only reproduce a
mask of pseudo-mask quality — refinement helps most when the *input* is coarser than
that (the divide masks), and barely when the input already equals the target
(the conquer masks). The before/after numbers make that explicit.

    python -m video.decoder.eval_refine_real --ckpt checkpoints/unsup_decoder_v4.pth \
        --clips horsejump-high,kite-surf,motocross-jump,dance-twirl --out decoder_v4_real.png
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
sys.path.insert(0, str(_REPO / "divide_and_conquer"))

from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis
from video.scripts.visualize_divide import paint, label, gt_masks
from video.scripts.viz_cuvler_hierarchy import run_conquer
from video.divide.cuvler_divide import CuVLERDivider
from video.decoder.train_sam_decoder import UnsupervisedDecoder, _IMG
from video.decoder.infer_decoder import box_from_mask, iou
import dinov3  # noqa: E402


def refine(model, feats, masks, W, H, thresh):
    """Box+mask -> refined mask for each coarse mask. Feeds the REAL mask (no degrade)."""
    boxes, mprompts = [], []
    for m in masks:
        b = box_from_mask(m, W, H)
        if b is None:
            continue
        boxes.append(b)
        mprompts.append(cv2.resize(m.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST))
    if not boxes:
        return []
    with torch.no_grad():
        mp = torch.tensor(np.stack(mprompts), dtype=torch.float, device="cuda")[:, None]
        low = model(feats, torch.tensor(boxes, dtype=torch.float, device="cuda"), mp)
        up = F.interpolate(low, (H, W), mode="bilinear", align_corners=False).sigmoid()[:, 0].cpu().numpy()
    return [(p > thresh).astype(np.uint8) for p in up]


def best_per_gt(gts, masks):
    """For each GT instance, the highest IoU achieved by any mask in the set."""
    return [max((iou(g, m) for m in masks), default=0.0) for g in gts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unsup_decoder_v4.pth")
    ap.add_argument("--clips", default="horsejump-high,kite-surf,motocross-jump,dance-twirl")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.35, help="CuVLER divide score thresh")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="decoder_v4_real.png")
    args = ap.parse_args()

    dense = DenseDINOv3()                                   # decoder's frozen encoder
    model = UnsupervisedDecoder().cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])
    print(f"[load] {args.ckpt}")
    backbone = dinov3.ViTFeatV3(patch_size=16, feat_dim=1024, vit_arch="large", vit_feat="k")
    backbone.eval(); backbone.cuda()                        # conquer backbone
    div = CuVLERDivider(score_thresh=args.score)

    agg = {"divide": ([], []), "conquer": ([], [])}        # (raw_ious, refined_ious)
    rows = []
    for clip in [c.strip() for c in args.clips.split(",")]:
        rgb = davis.load_frame(clip, args.frame)
        H, W = rgb.shape[:2]
        gts = gt_masks(clip, args.frame)
        divide = div.predict(rgb)
        conquer = run_conquer(backbone, rgb, divide)       # divide + finer parts (RGB/fixed)

        img1024 = cv2.resize(rgb, (_IMG, _IMG))
        feats = dense.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()
        ref_div = refine(model, feats, divide, W, H, args.thresh)
        ref_con = refine(model, feats, conquer, W, H, args.thresh)

        for key, raw, ref in [("divide", divide, ref_div), ("conquer", conquer, ref_con)]:
            r, f = best_per_gt(gts, raw), best_per_gt(gts, ref)
            agg[key][0].extend(r); agg[key][1].extend(f)
            print(f"{clip:16s} {key:8s} n={len(raw):3d}  raw mIoU={np.mean(r):.3f}  "
                  f"refined mIoU={np.mean(f):.3f}")

        rows.append(np.concatenate([
            label(paint(rgb, gts), f"{clip}:{args.frame} GT"),
            label(paint(rgb, divide), f"divide raw (n={len(divide)})"),
            label(paint(rgb, ref_div), "divide -> refined"),
            label(paint(rgb, conquer), f"conquer raw (n={len(conquer)})"),
            label(paint(rgb, ref_con), "conquer -> refined"),
        ], axis=1))

    print("\n=== overall best-IoU-per-GT (raw -> refined) ===")
    for key in ("divide", "conquer"):
        raw, ref = agg[key]
        print(f"{key:8s}: {np.mean(raw):.3f} -> {np.mean(ref):.3f}  (n_gt={len(raw)})")

    mh = min(r.shape[0] for r in rows)
    rows = [cv2.resize(r, (int(r.shape[1] * mh / r.shape[0]), mh)) for r in rows]
    cv2.imwrite(args.out, np.concatenate(rows, axis=0))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
