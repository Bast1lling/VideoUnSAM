"""Chained OT mask propagation: hop in small strides, feeding each hop's output
as the next hop's source.

Single-jump propagation degrades on fast motion / multi-instance content
because the cost matrix between distant frames is too noisy. Chained hops
stay in the high-IoU short-stride regime — at the cost of compounding drift
from each hop's segmentation error.

Usage:
    python -m video.scripts.propagate_chain \\
        --clip dogs-jump --frame-a 0 --frame-b 30 \\
        --stride 5 --instance-id 1 --upscale 2.0 \\
        --out chain_dogsjump.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis
from video.propagation.sinkhorn_ot import propagate, propagate_patch, _mask_to_patch_indicator


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha=0.55) -> np.ndarray:
    out = rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def upscale(img, factor):
    if factor == 1.0:
        return img
    H, W = img.shape[:2]
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(img, (int(round(W * factor)), int(round(H * factor))), interpolation=interp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="dogs-jump")
    p.add_argument("--frame-a", type=int, default=0)
    p.add_argument("--frame-b", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--instance-id", type=int, default=1)
    p.add_argument("--blur", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--upscale", type=float, default=2.0)
    p.add_argument("--sharpen", type=float, default=4.0,
                   help="Per-hop sharpening exponent. m <- m^p / max(m^p). "
                        "1.0 disables. 2-6 typical.")
    p.add_argument("--out", default="chain.png")
    args = p.parse_args()

    if args.frame_b <= args.frame_a:
        raise SystemExit("frame-b must be > frame-a")

    hops = list(range(args.frame_a, args.frame_b + 1, args.stride))
    if hops[-1] != args.frame_b:
        hops.append(args.frame_b)
    print(f"[chain] {args.clip}  hops = {hops}  stride={args.stride}  upscale={args.upscale}")

    extractor = DenseDINOv3()
    print("[model] DINOv3 loaded")

    # Cache features per frame.
    feat_cache: dict[int, dict] = {}

    def feats(idx: int):
        if idx not in feat_cache:
            f = davis.load_frame(args.clip, idx)
            feat_cache[idx] = extractor.extract(upscale(f, args.upscale))
        return feat_cache[idx]

    # Init chain in patch space from the GT source mask at frame_a.
    mask_a_native = davis.load_mask(args.clip, hops[0], instance_id=args.instance_id)
    if mask_a_native.sum() == 0:
        raise SystemExit(f"Instance {args.instance_id} absent in frame {hops[0]}")
    mask_a_up = upscale(mask_a_native, args.upscale)
    gh_a = feats(hops[0])["grid_h"]; gw_a = feats(hops[0])["grid_w"]
    m_patch = _mask_to_patch_indicator(mask_a_native, gh_a, gw_a)  # native or upscaled — same grid

    per_hop_iou: list[tuple[int, int, float]] = []
    src_idx = hops[0]
    for dst_idx in hops[1:]:
        m_patch = propagate_patch(
            feats(src_idx)["feats"], feats(dst_idx)["feats"],
            m_patch, blur=args.blur,
        ).cpu()
        if args.sharpen != 1.0:
            m_patch = m_patch.clamp_min(0).pow(args.sharpen)
            m_patch = m_patch / (m_patch.max() + 1e-8)
        # Eval this hop: upsample current soft mass to native and binarise (eval only;
        # the next hop keeps using the soft m_patch).
        gh_b = feats(dst_idx)["grid_h"]; gw_b = feats(dst_idx)["grid_w"]
        H_dst, W_dst = davis.load_frame(args.clip, dst_idx).shape[:2]
        heat = m_patch.reshape(gh_b, gw_b)
        heat_up = F.interpolate(
            heat[None, None], size=(H_dst, W_dst), mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        pred = (heat_up > args.threshold * heat_up.max()).astype(np.uint8)
        gt = davis.load_mask(args.clip, dst_idx, instance_id=args.instance_id)
        score = iou(pred, gt)
        per_hop_iou.append((src_idx, dst_idx, score))
        print(f"  {src_idx:3d} -> {dst_idx:3d}  IoU vs GT = {score:.3f}  "
              f"(mass entropy ~ {float(-(m_patch / m_patch.sum() * (m_patch / m_patch.sum()).clamp_min(1e-12).log()).sum()):.2f})")
        src_idx = dst_idx

    # Final chained prediction in native pixel space (for visualisation).
    chain_pred_native = pred

    # Direct single-jump baseline for comparison
    print("[baseline] single-jump propagation for the same a→b")
    mask_a0 = davis.load_mask(args.clip, args.frame_a, instance_id=args.instance_id)
    mask_a0_up = upscale(mask_a0, args.upscale)
    H_b = davis.load_frame(args.clip, args.frame_b).shape[0]
    W_b = davis.load_frame(args.clip, args.frame_b).shape[1]
    direct = propagate(
        feats(args.frame_a)["feats"], feats(args.frame_b)["feats"],
        mask_a0_up, out_size=(int(round(H_b * args.upscale)), int(round(W_b * args.upscale))),
        blur=args.blur, threshold=args.threshold,
    )
    direct_pred = cv2.resize(direct["mask"], (W_b, H_b), interpolation=cv2.INTER_NEAREST)
    gt_b = davis.load_mask(args.clip, args.frame_b, instance_id=args.instance_id)
    direct_iou_at_b = iou(direct_pred, gt_b)
    chain_iou_at_b = per_hop_iou[-1][2]
    print(f"[result] direct {args.frame_a}->{args.frame_b}: IoU = {direct_iou_at_b:.3f}")
    print(f"[result] chain  {args.frame_a}->{args.frame_b}: IoU = {chain_iou_at_b:.3f}")
    print(f"[delta ] {chain_iou_at_b - direct_iou_at_b:+.3f}")

    # Visualise: source | direct prediction | chained prediction | GT
    frame_a_img = davis.load_frame(args.clip, args.frame_a)
    frame_b_img = davis.load_frame(args.clip, args.frame_b)
    a_vis = overlay_mask(frame_a_img, mask_a0, color=(64, 255, 64))
    b_direct = overlay_mask(frame_b_img, direct_pred, color=(255, 64, 64))
    b_chain = overlay_mask(frame_b_img, chain_pred_native, color=(64, 64, 255))
    b_gt = overlay_mask(frame_b_img, gt_b, color=(64, 255, 64))

    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out

    panel = np.concatenate([
        label(a_vis, f"A:{args.frame_a} src"),
        label(b_direct, f"direct IoU {direct_iou_at_b:.2f}"),
        label(b_chain, f"chain s={args.stride} IoU {chain_iou_at_b:.2f}"),
        label(b_gt, "GT"),
    ], axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
