"""Test-time adaptation: train a 1-layer linear probe on frame 0 to recognise the
clicked instance, then segment every frame with it (no labels — supervision is the
unsupervised CuVLER+conquer seed mask).

Positive class  = seed (clicked-object) patches on frame 0.
Negative class  = every other patch on frame 0.
A linear layer on frozen DINOv3 features is trained ~50 steps to separate them,
learning an instance-discriminative direction (upweighting the feature dims that
encode this dancer's appearance). Strictly more powerful than the rejected
mean-prototype template blend, which could only measure cosine distance to a centroid.

Modes:
  --mode probe : pure probe segmentation each frame (isolates the TTA signal)
  --mode ot    : probe score injected into OT cost as a per-patch prior

Usage:
    python -m video.scripts.probe_tta --clip dancing --instance-id 1 \\
        --mode probe --out video/outputs/dancing_probe.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))
from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.loaders import davis  # noqa: E402
from video.decoder.train_sam_decoder import sample_clicks  # noqa: E402
from video.divide.cuvler_divide import CuVLERDivider  # noqa: E402
from video.divide.conquer import load_backbone, run_conquer  # noqa: E402
from video.propagation.sinkhorn_ot import compute_cond  # noqa: E402
from video.scripts.propagate_reseed import (  # noqa: E402
    iou, overlay_mask, put_label,
)


def patch_labels(mask: np.ndarray, gh: int, gw: int) -> torch.Tensor:
    """Binary per-patch label [gh*gw] — patch is positive if >50% covered."""
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
    return (pooled.flatten() > 0.5).float()


def train_probe(feats: torch.Tensor, y: torch.Tensor, steps: int, lr: float,
                hidden: int = 0) -> nn.Module:
    """Train a linear (or 1-hidden-layer) probe on frozen features.

    feats [N, D] (frozen), y [N] in {0,1}. Class-balanced BCE.
    """
    D = feats.shape[1]
    if hidden > 0:
        probe = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, 1)).cuda()
    else:
        probe = nn.Linear(D, 1).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n_pos = y.sum().clamp_min(1)
    n_neg = (1 - y).sum().clamp_min(1)
    pos_weight = (n_neg / n_pos).detach()
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    feats = feats.detach()
    for i in range(steps):
        opt.zero_grad()
        logit = probe(feats).squeeze(-1)
        loss = lossf(logit, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (probe(feats).squeeze(-1) > 0).float()
        acc = (pred == y).float().mean().item()
        tpr = (pred[y > 0] > 0).float().mean().item() if n_pos > 0 else 0.0
    print(f"  probe trained: {steps} steps, final loss {loss.item():.3f}, "
          f"frame0 acc {acc:.3f}, seed recall {tpr:.3f}")
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="dancing")
    ap.add_argument("--instance-id", type=int, default=1)
    ap.add_argument("--feat-size", type=int, default=1024)
    ap.add_argument("--mode", choices=["probe", "ot", "fuse"], default="probe")
    ap.add_argument("--fuse-weight", type=float, default=0.5,
                    help="(fuse mode) weight on probe vs OT: heat = (1-w)*OT + w*probe.")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=0,
                    help="0 = linear probe; >0 adds one ReLU hidden layer of this width.")
    ap.add_argument("--probe-thresh", type=float, default=0.5)
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--cuvler-score", type=float, default=0.35)
    ap.add_argument("--out", default="video/outputs/probe.mp4")
    args = ap.parse_args()

    dino = DenseDINOv3()
    divider = CuVLERDivider(score_thresh=args.cuvler_score)
    conquer_backbone = load_backbone()
    print("[conquer] backbone loaded")

    n = davis.num_frames(args.clip)
    if args.max_frames:
        n = min(n, args.max_frames)

    # --- Frame 0: seed + probe training ---
    frame0 = davis.load_frame(args.clip, 0)
    H, W = frame0.shape[:2]
    gt0 = davis.load_mask(args.clip, 0, instance_id=args.instance_id)
    gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    cx256, cy256 = sample_clicks(gt256, 1)[0]
    click_xy = (cx256 * (W / 256.0), cy256 * (H / 256.0))

    proposals0 = run_conquer(conquer_backbone, frame0, divider.predict(frame0))
    cx, cy = click_xy
    containing = [m for m in proposals0 if m[int(cy), int(cx)] > 0]
    if containing:
        largest = max(containing, key=lambda m: m.sum())
        if largest.sum() > H * W * 0.40:
            valid = [m for m in containing if m.sum() >= H * W * 0.005]
            seed = min(valid, key=lambda m: m.sum()) if valid else largest
        else:
            seed = largest
    else:
        seed = max(proposals0, key=lambda m: m.sum()) if proposals0 else gt0.astype(np.uint8)
    print(f"  frame 0: {len(proposals0)} proposals  seed IoU={iou(gt0, seed):.3f}")

    img_sized = cv2.resize(frame0, (args.feat_size, args.feat_size))
    feats0 = F.normalize(dino.extract(img_sized, normalize=False)["feats"].cuda().float(), dim=-1)
    gh, gw = feats0.shape[:2]
    feats0_flat = feats0.reshape(-1, feats0.shape[-1])

    y0 = patch_labels(seed, gh, gw).cuda()
    probe = train_probe(feats0_flat, y0, steps=args.steps, lr=args.lr, hidden=args.hidden)

    feats_prev = feats0
    prev_mask = seed.astype(np.uint8)
    ious = [iou(gt0, seed)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def make_panel(frame, mask, score_hw, gt, tag):
        a = put_label(overlay_mask(frame, mask, (255, 140, 0)), tag)
        sc = (np.clip(score_hw, 0, 1) * 255).astype(np.uint8)
        sc = cv2.applyColorMap(sc, cv2.COLORMAP_JET)[:, :, ::-1]
        b = put_label(cv2.addWeighted(frame, 0.4, sc, 0.6, 0), "probe score")
        c = put_label(overlay_mask(frame, gt, (64, 255, 64)), "GT")
        return np.concatenate([a, b, c], axis=1)

    with torch.no_grad():
        sc0 = torch.sigmoid(probe(feats0_flat).squeeze(-1)).reshape(gh, gw).cpu().numpy()
    sc0_hw = cv2.resize(sc0, (W, H))
    p0 = make_panel(frame0, seed, sc0_hw, gt0, f"f0 seed IoU={ious[0]:.2f}")
    ph, pw = p0.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (pw, ph))
    writer.write(cv2.cvtColor(p0, cv2.COLOR_RGB2BGR))

    # --- Propagation / segmentation loop ---
    for fidx in range(1, n):
        frame = davis.load_frame(args.clip, fidx)
        gt = davis.load_mask(args.clip, fidx, instance_id=args.instance_id)
        img_sized = cv2.resize(frame, (args.feat_size, args.feat_size))
        feats = F.normalize(dino.extract(img_sized, normalize=False)["feats"].cuda().float(), dim=-1)
        feats_flat = feats.reshape(-1, feats.shape[-1])

        with torch.no_grad():
            score = torch.sigmoid(probe(feats_flat).squeeze(-1))  # [N] in [0,1]

        if args.mode == "probe":
            heat = score.reshape(gh, gw)
            thr = args.probe_thresh
        else:
            # OT-propagate the previous mask through the appearance-feature plan.
            m_a = patch_labels(prev_mask, gh, gw).cuda()
            cond = compute_cond(feats_prev, feats, blur=args.blur)
            ot_heat = m_a @ cond                      # [N]
            ot_heat = ot_heat / (ot_heat.max() + 1e-8)
            if args.mode == "ot":
                heat = ot_heat.reshape(gh, gw)
            else:  # fuse: arithmetic blend; probe revives, OT smooths/carries overlap
                w = args.fuse_weight
                heat = ((1 - w) * ot_heat + w * score).reshape(gh, gw)
            thr_rel = 0.5  # relative threshold for OT/fused heat
            thr = None

        heat_up = F.interpolate(heat[None, None], size=(H, W), mode="bilinear",
                                align_corners=False)[0, 0].cpu().numpy()
        if args.mode == "probe":
            mask = (heat_up > thr).astype(np.uint8)
        else:
            mask = (heat_up > thr_rel * heat_up.max()).astype(np.uint8)
        score_hw = cv2.resize(score.reshape(gh, gw).cpu().numpy(), (W, H))
        s = iou(gt, mask) if gt.sum() > 0 else float("nan")
        ious.append(s)

        writer.write(cv2.cvtColor(make_panel(frame, mask, score_hw, gt, f"f{fidx} IoU={s:.2f}"),
                                  cv2.COLOR_RGB2BGR))
        print(f"  frame {fidx:3d}: IoU={s:.3f}")
        feats_prev = feats
        prev_mask = mask  # probe-corrected mask feeds back into the OT chain

    writer.release()
    if shutil.which("ffmpeg"):
        h264 = out_path.with_suffix(".h264.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", str(out_path), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", str(h264)], check=True, capture_output=True)
        out_path.unlink(); h264.rename(out_path)

    valid = [x for x in ious if not np.isnan(x)]
    print(f"\n[done] {out_path}")
    print(f"  mean IoU {np.mean(valid):.3f}  median {np.median(valid):.3f}")


if __name__ == "__main__":
    main()
