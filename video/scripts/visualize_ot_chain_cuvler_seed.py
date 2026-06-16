"""Same as visualize_ot_chain_video.py, but seeds OT-chain propagation from a
CuVLER+conquer pseudo-mask (Stage-1 pipeline) instead of the small from-scratch
decoder's click prediction. Hypothesis ([[ot-chain-beats-decoder-refine]] follow-up):
propagation barely degrades quality, so a better seed (CuVLER's pretrained-backbone
proposals vs our ~4.6M-param decoder) should translate directly into a better
mean IoU across the whole clip.

The CuVLER proposal is picked by which one contains the SAME click point used for
the decoder-seed run (distance-transform peak of the GT mask) -- still label-free
at the proposal-selection step (a real interactive tool would click, then snap to
whichever CuVLER region contains that point).

Run divide_conquerV3.py --preprocess on frame 0 first (see eval_ot_chain notes):
    cd divide_and_conquer && ../.venv/bin/python divide_conquerV3.py \\
        --input-dir /tmp/dog_seed/in --output-dir /tmp/dog_seed/out \\
        --preprocess True --divide-method cuvler --end-id 2

    python -m video.scripts.visualize_ot_chain_cuvler_seed --clip dog \\
        --cuvler-json /tmp/dog_seed/out/sa_0.json --out video/ot_chain_dog_cuvler.mp4
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
from pycocotools import mask as mask_util

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.loaders import davis  # noqa: E402
from video.decoder.train_sam_decoder import sample_clicks, _IMG  # noqa: E402
from video.propagation.sinkhorn_ot import propagate_patch, _mask_to_patch_indicator  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def overlay_mask(rgb, mask, color=(64, 64, 255), alpha=0.5):
    out = rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def patch_indicator_or_soft(bin_mask, soft_mask, gh=64, gw=64):
    try:
        return _mask_to_patch_indicator(bin_mask, gh, gw)
    except ValueError:
        m = torch.from_numpy(soft_mask.astype(np.float32))[None, None]
        pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
        if pooled.sum() <= 0:
            pooled = pooled + 1.0 / (gh * gw)
        return pooled.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="dog")
    ap.add_argument("--instance-id", type=int, default=1)
    ap.add_argument("--cuvler-json", default="/tmp/dog_seed/out/sa_0.json")
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="video/ot_chain_dog_cuvler.mp4")
    args = ap.parse_args()

    dino = DenseDINOv3()

    n = davis.num_frames(args.clip)
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"[clip] {args.clip}  {n} frames  instance={args.instance_id}")

    frame0 = davis.load_frame(args.clip, 0)
    H, W = frame0.shape[:2]
    gt0 = davis.load_mask(args.clip, 0, instance_id=args.instance_id)

    # Same click point as the decoder-seed run: distance-transform peak of GT.
    gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    pts = sample_clicks(gt256, 1)
    cx256, cy256 = pts[0]
    cx, cy = cx256 * (W / 256.0), cy256 * (H / 256.0)
    print(f"  click point (native px): ({cx:.1f}, {cy:.1f})")

    # Pick the CuVLER proposal containing the click point (largest if multiple).
    d = json.load(open(args.cuvler_json))
    candidates = []
    for a in d["annotations"]:
        m = mask_util.decode(a["segmentation"])
        if m[int(cy), int(cx)] > 0:
            candidates.append(m)
    if not candidates:
        # fall back: nearest mask by centroid distance to the click
        best, best_d = None, 1e18
        for a in d["annotations"]:
            m = mask_util.decode(a["segmentation"])
            ys, xs = np.where(m > 0)
            dist = (xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2
            if dist < best_d:
                best_d, best = dist, m
        candidates = [best]
    seed_mask = max(candidates, key=lambda m: m.sum())
    iou_seed = iou(gt0, seed_mask)
    print(f"  CuVLER seed: {len(d['annotations'])} proposals, picked area={seed_mask.sum()}, "
          f"IoU vs GT = {iou_seed:.3f}")

    img1024 = cv2.resize(frame0, (_IMG, _IMG))
    with torch.no_grad():
        feats_prev = dino.extract(img1024, normalize=False)["feats"].cuda().float()
    feats_prev_norm = F.normalize(feats_prev, dim=-1)

    seed_256 = cv2.resize(seed_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    soft0_256 = seed_256.astype(np.float32)
    bin0_256 = seed_256.astype(np.uint8)
    pred0 = seed_mask.astype(np.uint8)
    patch = patch_indicator_or_soft(bin0_256, soft0_256)

    panel0 = np.concatenate([
        label(overlay_mask(frame0, pred0, (64, 64, 255)), f"frame 0: CuVLER seed  IoU {iou_seed:.2f}"),
        label(overlay_mask(frame0, gt0, (64, 255, 64)), "GT"),
    ], axis=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_h, panel_w = panel0.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (panel_w, panel_h))
    writer.write(cv2.cvtColor(panel0, cv2.COLOR_RGB2BGR))

    ious = [iou_seed]
    for fidx in range(1, n):
        frame = davis.load_frame(args.clip, fidx)
        gt = davis.load_mask(args.clip, fidx, instance_id=args.instance_id)
        img1024 = cv2.resize(frame, (_IMG, _IMG))
        with torch.no_grad():
            feats_b = dino.extract(img1024, normalize=False)["feats"].cuda().float()
        feats_b_norm = F.normalize(feats_b, dim=-1)

        heat = propagate_patch(feats_prev_norm, feats_b_norm, patch, blur=args.blur)
        heat_up = F.interpolate(heat.reshape(1, 1, 64, 64), size=(H, W),
                                 mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        pred = (heat_up > args.thresh * heat_up.max()).astype(np.uint8)
        score = iou(gt, pred) if gt.sum() > 0 else float("nan")
        ious.append(score)
        print(f"  frame {fidx:3d}: IoU {score:.3f}")

        panel = np.concatenate([
            label(overlay_mask(frame, pred, (64, 64, 255)), f"frame {fidx}: OT-chain (CuVLER seed)  IoU {score:.2f}"),
            label(overlay_mask(frame, gt, (64, 255, 64)), "GT"),
        ], axis=1)
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        patch = heat
        feats_prev_norm = feats_b_norm

    writer.release()
    valid = [x for x in ious if not np.isnan(x)]
    print(f"[done] {out_path}  mean IoU {np.mean(valid):.3f}  (n={len(valid)})")


if __name__ == "__main__":
    main()
