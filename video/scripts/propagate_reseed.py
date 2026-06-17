"""OT-chain propagation with periodic CuVLER re-seeding.

Every --reseed-interval frames, runs CuVLER on the current frame to get fresh
proposals. Picks the proposal with highest IoU overlap with the current
OT-propagated mask. If the overlap exceeds --reseed-thresh, resets the OT
chain from that proposal; otherwise keeps propagating from the OT mask.

Re-seed events are shown in orange in the output video; normal OT frames in blue.

Usage:
    python -m video.scripts.propagate_reseed --clip dog --instance-id 1 \\
        --reseed-interval 10 --out video/reseed_dog.mp4
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
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))
from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.loaders import davis  # noqa: E402
from video.decoder.train_sam_decoder import sample_clicks, _IMG  # noqa: E402
from video.divide.cuvler_divide import CuVLERDivider  # noqa: E402
from video.divide.conquer import load_backbone, run_conquer  # noqa: E402
from video.propagation.sinkhorn_ot import propagate_patch, _mask_to_patch_indicator  # noqa: E402


def refine_mask(refiner, frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sharpen mask boundaries with CascadePSP around the mask's bounding box."""
    H, W = frame_rgb.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1)          # square crop with 1x margin
    x0c = max(0, int(cx - half))
    y0c = max(0, int(cy - half))
    x1c = min(W, int(cx + half))
    y1c = min(H, int(cy + half))

    img_crop = cv2.cvtColor(frame_rgb[y0c:y1c, x0c:x1c], cv2.COLOR_RGB2BGR)
    msk_crop = (mask[y0c:y1c, x0c:x1c] * 255).astype(np.uint8)
    L = int(np.clip(max(x1c - x0c, y1c - y0c), 100, 900))
    refined_crop = refiner.refine(img_crop, msk_crop, fast=True, L=L)

    out = np.zeros((H, W), dtype=np.uint8)
    out[y0c:y1c, x0c:x1c] = (refined_crop > 128).astype(np.uint8)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def overlay_mask(rgb, mask, color=(64, 64, 255), alpha=0.5):
    out = rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def put_label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def mask_to_patch(mask: np.ndarray, gh: int = 64, gw: int = 64) -> torch.Tensor:
    """Binary mask [H, W] -> soft patch indicator [gh*gw], falls back to uniform if empty."""
    try:
        return _mask_to_patch_indicator(mask, gh, gw)
    except ValueError:
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
        if pooled.sum() <= 0:
            pooled = pooled + 1.0 / (gh * gw)
        return pooled.flatten()


def crop_box(ref, frame_hw: tuple[int, int], context: float) -> tuple[int, int, int, int]:
    """Bounding box for a guided crop.

    ref: binary mask [H,W] or (cx, cy) click point.
    context: expansion factor applied to each half-dimension.
    Returns (x0, y0, x1, y1) clamped to frame bounds.
    """
    H, W = frame_hw
    if isinstance(ref, tuple):
        cx, cy = ref
        base = min(H, W) // 6  # ~80px on 480p
        hx, hy = base, base
        mcx, mcy = cx, cy
    else:
        ys, xs = np.where(ref > 0)
        if len(ys) == 0:
            return 0, 0, W, H
        mcx = (xs.min() + xs.max()) / 2
        mcy = (ys.min() + ys.max()) / 2
        hx = (xs.max() - xs.min()) / 2
        hy = (ys.max() - ys.min()) / 2

    hx = max(hx * context, 32)
    hy = max(hy * context, 32)
    x0 = max(0, int(mcx - hx))
    y0 = max(0, int(mcy - hy))
    x1 = min(W, int(mcx + hx))
    y1 = min(H, int(mcy + hy))
    return x0, y0, x1, y1


def cuvler_on_crop(frame: np.ndarray, divider: CuVLERDivider,
                   box: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Run CuVLER on a subregion and return proposals in full-frame coordinates."""
    x0, y0, x1, y1 = box
    H, W = frame.shape[:2]
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return []
    crop_masks = divider.predict(frame[y0:y1, x0:x1])
    full = []
    for m in crop_masks:
        fm = np.zeros((H, W), dtype=np.uint8)
        fm[y0:y1, x0:x1] = m
        full.append(fm)
    return full


def pick_proposal(masks: list[np.ndarray], ref_mask: np.ndarray,
                  click_xy: tuple[float, float] | None = None) -> np.ndarray | None:
    """Pick the CuVLER proposal with highest IoU vs ref_mask.

    If click_xy is given, restrict to proposals that contain that point first;
    falls back to best-overlap-globally if none contain it.
    """
    if not masks:
        return None
    if click_xy is not None:
        cx, cy = click_xy
        containing = [m for m in masks if m[int(cy), int(cx)] > 0]
        if containing:
            return max(containing, key=lambda m: iou(m, ref_mask))
    return max(masks, key=lambda m: iou(m, ref_mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="dog")
    ap.add_argument("--instance-id", type=int, default=1)
    ap.add_argument("--reseed-interval", type=int, default=10,
                    help="Re-run CuVLER every N frames (0 = never, baseline OT chain)")
    ap.add_argument("--reseed-thresh", type=float, default=0.3,
                    help="Min IoU between CuVLER proposal and current OT mask to accept re-seed")
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="video/reseed_dog.mp4")
    ap.add_argument("--cuvler-score", type=float, default=0.35)
    ap.add_argument("--crop-context", type=float, default=0.0,
                    help="If >0, also run CuVLER on a guided crop at keyframes (around OT "
                         "mask bbox) and merge proposals. Expansion factor on each half-dim.")
    ap.add_argument("--click-crop-radius", type=int, default=0,
                    help="If >0, also run CuVLER on a fixed-pixel crop of this radius "
                         "around the frame-0 click point (e.g. 100 = 200x200 crop).")
    ap.add_argument("--refine", action="store_true",
                    help="Apply CascadePSP boundary refinement to each frame's output mask. "
                         "Adds a third panel (refined) to the output video.")
    ap.add_argument("--conquer", action="store_true",
                    help="Run DINOv3 spectral conquer stage within each CuVLER proposal bbox "
                         "to generate tighter sub-masks before proposal selection.")
    args = ap.parse_args()

    dino = DenseDINOv3()
    divider = CuVLERDivider(score_thresh=args.cuvler_score)
    conquer_backbone = None
    if args.conquer:
        conquer_backbone = load_backbone()
        print("[conquer] DINOv3 conquer backbone loaded")
    refiner = None
    if args.refine:
        import segmentation_refinement as refine_mod
        refiner = refine_mod.Refiner(device="cuda:0")
        print("[refine] CascadePSP loaded")

    n = davis.num_frames(args.clip)
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"[clip] {args.clip}  {n} frames  instance={args.instance_id}  "
          f"reseed_interval={args.reseed_interval}  reseed_thresh={args.reseed_thresh}")

    # --- Frame 0: CuVLER seed ---
    frame0 = davis.load_frame(args.clip, 0)
    H, W = frame0.shape[:2]
    gt0 = davis.load_mask(args.clip, 0, instance_id=args.instance_id)

    # Simulate a user click: distance-transform peak of GT mask
    gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    pts = sample_clicks(gt256, 1)
    cx256, cy256 = pts[0]
    click_xy = (cx256 * (W / 256.0), cy256 * (H / 256.0))
    print(f"  click point (native px): ({click_xy[0]:.1f}, {click_xy[1]:.1f})")

    proposals0 = divider.predict(frame0)
    if args.click_crop_radius > 0:
        cx, cy = click_xy
        r = args.click_crop_radius
        box0 = (max(0, int(cx) - r), max(0, int(cy) - r),
                min(W, int(cx) + r), min(H, int(cy) + r))
        crop_props0 = cuvler_on_crop(frame0, divider, box0)
        print(f"  click crop {box0} ({box0[2]-box0[0]}x{box0[3]-box0[1]}px) "
              f"+{len(crop_props0)} proposals")
        proposals0 = proposals0 + crop_props0
    elif args.crop_context > 0:
        box0 = crop_box(click_xy, (H, W), args.crop_context)
        crop_props0 = cuvler_on_crop(frame0, divider, box0)
        print(f"  guided crop: {box0}  +{len(crop_props0)} proposals")
        proposals0 = proposals0 + crop_props0
    if conquer_backbone is not None:
        n_before = len(proposals0)
        proposals0 = run_conquer(conquer_backbone, frame0, proposals0)
        print(f"  conquer: {n_before} -> {len(proposals0)} proposals")
    seed = pick_proposal(proposals0, gt0, click_xy=click_xy)
    if seed is None:
        print("  WARNING: no CuVLER proposals on frame 0, falling back to GT")
        seed = gt0.astype(np.uint8)
    seed_iou = iou(gt0, seed)
    print(f"  frame 0: {len(proposals0)} proposals -> seed IoU={seed_iou:.3f}")

    img1024 = cv2.resize(frame0, (_IMG, _IMG))
    feats_prev = dino.extract(img1024, normalize=False)["feats"].cuda().float()
    feats_prev_norm = F.normalize(feats_prev, dim=-1)

    cur_mask = seed.astype(np.uint8)
    patch = mask_to_patch(cur_mask)

    # Refine frame-0 seed for display (OT propagation still uses cur_mask/patch)
    display_mask0 = refine_mask(refiner, frame0, cur_mask) if refiner else cur_mask
    refined_seed_iou = iou(gt0, display_mask0)
    if refiner:
        print(f"  frame 0 refined: IoU={refined_seed_iou:.3f}")

    ious_raw = [seed_iou]
    ious_refined = [refined_seed_iou]
    reseed_frames = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels0 = [
        put_label(overlay_mask(frame0, cur_mask, (255, 140, 0)),
                  f"f0 raw IoU={seed_iou:.2f}"),
    ]
    if refiner:
        panels0.append(put_label(overlay_mask(frame0, display_mask0, (200, 80, 200)),
                                 f"f0 refined IoU={refined_seed_iou:.2f}"))
    panels0.append(put_label(overlay_mask(frame0, gt0, (64, 255, 64)), "GT"))
    panel0 = np.concatenate(panels0, axis=1)
    ph, pw = panel0.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (pw, ph))
    writer.write(cv2.cvtColor(panel0, cv2.COLOR_RGB2BGR))

    # --- Propagation loop ---
    for fidx in range(1, n):
        frame = davis.load_frame(args.clip, fidx)
        gt = davis.load_mask(args.clip, fidx, instance_id=args.instance_id)

        img1024 = cv2.resize(frame, (_IMG, _IMG))
        feats_cur = dino.extract(img1024, normalize=False)["feats"].cuda().float()
        feats_cur_norm = F.normalize(feats_cur, dim=-1)

        # OT propagation from previous patch distribution
        heat = propagate_patch(feats_prev_norm, feats_cur_norm, patch, blur=args.blur)
        heat_up = F.interpolate(
            heat.reshape(1, 1, 64, 64), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        ot_mask = (heat_up > args.thresh * heat_up.max()).astype(np.uint8)

        # Periodic re-seed
        did_reseed = False
        if args.reseed_interval > 0 and fidx % args.reseed_interval == 0:
            proposals = divider.predict(frame)
            if args.crop_context > 0:
                box = crop_box(ot_mask, (H, W), args.crop_context)
                proposals = proposals + cuvler_on_crop(frame, divider, box)
            if conquer_backbone is not None:
                proposals = run_conquer(conquer_backbone, frame, proposals)
            candidate = pick_proposal(proposals, ot_mask)
            if candidate is not None:
                overlap = iou(ot_mask, candidate)
                if overlap >= args.reseed_thresh:
                    cur_mask = candidate
                    patch = mask_to_patch(cur_mask)
                    did_reseed = True
                    print(f"  frame {fidx:3d}: RESEED  overlap={overlap:.3f}  "
                          f"({len(proposals)} proposals)")
                else:
                    cur_mask = ot_mask
                    patch = heat
                    print(f"  frame {fidx:3d}: reseed rejected (overlap={overlap:.3f} < "
                          f"{args.reseed_thresh})  keeping OT")
            else:
                cur_mask = ot_mask
                patch = heat
                print(f"  frame {fidx:3d}: no proposals, keeping OT")
        else:
            cur_mask = ot_mask
            patch = heat

        score_raw = iou(gt, cur_mask) if gt.sum() > 0 else float("nan")
        ious_raw.append(score_raw)

        display_mask = refine_mask(refiner, frame, cur_mask) if refiner else cur_mask
        score_ref = iou(gt, display_mask) if gt.sum() > 0 else float("nan")
        ious_refined.append(score_ref)

        if did_reseed:
            reseed_frames.append(fidx)

        color = (255, 140, 0) if did_reseed else (64, 64, 255)
        tag = " [RESEED]" if did_reseed else ""
        score_show = score_ref if refiner else score_raw
        panels = [put_label(overlay_mask(frame, cur_mask, color),
                            f"f{fidx}{tag} raw={score_raw:.2f}")]
        if refiner:
            panels.append(put_label(overlay_mask(frame, display_mask, (200, 80, 200)),
                                    f"f{fidx}{tag} ref={score_ref:.2f}"))
        panels.append(put_label(overlay_mask(frame, gt, (64, 255, 64)), "GT"))
        writer.write(cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))

        feats_prev_norm = feats_cur_norm
        if not did_reseed:
            ref_str = f"  refined={score_ref:.3f}" if refiner else ""
            print(f"  frame {fidx:3d}: IoU={score_raw:.3f}{ref_str}")

    writer.release()

    # Re-encode to H.264 for Mac/browser compatibility (mp4v from OpenCV shows green on QuickTime)
    if shutil.which("ffmpeg"):
        h264_path = out_path.with_suffix(".h264.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_path), "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", str(h264_path)],
            check=True, capture_output=True,
        )
        out_path.unlink()
        h264_path.rename(out_path)
        print(f"  re-encoded to H.264: {out_path}")

    valid_raw = [x for x in ious_raw if not np.isnan(x)]
    valid_ref = [x for x in ious_refined if not np.isnan(x)]
    print(f"\n[done] {out_path}")
    print(f"  raw     mean IoU {np.mean(valid_raw):.3f}  median {np.median(valid_raw):.3f}")
    if refiner:
        print(f"  refined mean IoU {np.mean(valid_ref):.3f}  median {np.median(valid_ref):.3f}")


if __name__ == "__main__":
    main()
