"""Render a video of OT-chain mask propagation: click-prompt the v3 decoder on frame 0,
then propagate the resulting mask through every subsequent frame via chained Sinkhorn OT
(no decoder in the loop, per [[ot-chain-beats-decoder-refine]]). Side-by-side panels of
the propagated prediction and GT, with per-frame IoU overlaid.

    python -m video.scripts.visualize_ot_chain_video --clip dog --out video/ot_chain_dog.mp4
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
from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.loaders import davis  # noqa: E402
from video.decoder.train_sam_decoder import UnsupervisedDecoder, sample_clicks, next_click, _IMG  # noqa: E402
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
    ap.add_argument("--ckpt", default="checkpoints/unsup_decoder_points_v3.pth")
    ap.add_argument("--clicks", type=int, default=3)
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="video/ot_chain_demo.mp4")
    args = ap.parse_args()

    dino = DenseDINOv3()
    model = UnsupervisedDecoder().cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])

    n = davis.num_frames(args.clip)
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"[clip] {args.clip}  {n} frames  instance={args.instance_id}")

    frame0 = davis.load_frame(args.clip, 0)
    H, W = frame0.shape[:2]
    gt0 = davis.load_mask(args.clip, 0, instance_id=args.instance_id)
    img1024 = cv2.resize(frame0, (_IMG, _IMG))
    with torch.no_grad():
        feats_prev = dino.extract(img1024, normalize=False)["feats"].cuda().float()
    feats_prev_chw = feats_prev.permute(2, 0, 1)[None]
    feats_prev_norm = F.normalize(feats_prev, dim=-1)

    # K-click iterative prediction on frame 0
    gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    pts = sample_clicks(gt256, 1)
    x, y = pts[0] if pts else (128.0, 128.0)
    coords, labels = [[[x * 4.0, y * 4.0]]], [[1.0]]
    prev, logits0 = None, None
    for it in range(args.clicks):
        c = torch.tensor(coords, dtype=torch.float, device="cuda")
        l = torch.tensor(labels, dtype=torch.float, device="cuda")
        with torch.no_grad():
            logits0 = model(feats_prev_chw, points=(c, l), mask_prompts=prev)
        prev = logits0.detach()
        if it < args.clicks - 1:
            pred = (prev[:, 0].sigmoid() > 0.5).cpu().numpy().astype(np.uint8)[0]
            nc = next_click(gt256, pred)
            if nc is None:
                coords[0].append(coords[0][-1]); labels[0].append(labels[0][-1])
            else:
                px, py, lab = nc
                coords[0].append([px, py]); labels[0].append(lab)

    soft0_256 = logits0.sigmoid().cpu().numpy()[0, 0]
    bin0_256 = (soft0_256 > 0.5).astype(np.uint8)
    pred0 = cv2.resize(bin0_256, (W, H), interpolation=cv2.INTER_NEAREST)
    iou0 = iou(gt0, pred0)
    print(f"  frame   0: IoU {iou0:.3f}  ({args.clicks}-click prediction)")

    patch = patch_indicator_or_soft(bin0_256, soft0_256)

    panel0 = np.concatenate([
        label(overlay_mask(frame0, pred0, (64, 64, 255)), f"frame 0: {args.clicks}-click pred  IoU {iou0:.2f}"),
        label(overlay_mask(frame0, gt0, (64, 255, 64)), "GT"),
    ], axis=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_h, panel_w = panel0.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (panel_w, panel_h))
    writer.write(cv2.cvtColor(panel0, cv2.COLOR_RGB2BGR))

    ious = [iou0]
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
            label(overlay_mask(frame, pred, (64, 64, 255)), f"frame {fidx}: OT-chain pred  IoU {score:.2f}"),
            label(overlay_mask(frame, gt, (64, 255, 64)), "GT"),
        ], axis=1)
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        patch = heat  # chain: propagate from this frame's soft prediction next
        feats_prev_norm = feats_b_norm

    writer.release()
    valid = [x for x in ious if not np.isnan(x)]
    print(f"[done] {out_path}  mean IoU {np.mean(valid):.3f}  (n={len(valid)})")


if __name__ == "__main__":
    main()
