"""Competitive multi-object OT label propagation (whole-image tracking).

Instead of pushing one mask forward, this partitions frame 0 into K labels
(the clicked object + competing scene regions + background) and propagates the
WHOLE label field through the same Sinkhorn plan each frame:

    M [K, N_a]  →  M @ cond  →  heat [K, N_b]  →  argmax over K  →  label_b

Because the labels partition the source (every patch has exactly one label) and
cond's rows sum to 1, every target patch receives equal total mass, so the
argmax is a fair competition. The clicked object is read out as one label.

The point: when a distractor (e.g. a second dancer) approaches the target, the
distractor's OWN label claims the distractor's patches, so the target label's
mass cannot leak onto it — directly attacking the identity-switch failure that
no single-object cost-term tweak could fix.

Usage:
    python -m video.scripts.propagate_multiobject --clip dancing --instance-id 1 \\
        --conquer --color-weight 0.2 --out video/outputs/dancing_multi.mp4
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
from video.decoder.train_sam_decoder import sample_clicks  # noqa: E402
from video.divide.cuvler_divide import CuVLERDivider  # noqa: E402
from video.divide.conquer import load_backbone, run_conquer  # noqa: E402
from video.propagation.sinkhorn_ot import compute_cond  # noqa: E402
from video.scripts.propagate_reseed import (  # noqa: E402
    _color_cost_patches, iou, overlay_mask, put_label, mask_to_patch,
)


# Distinct colors for label visualisation (BGR-ish RGB tuples)
_LABEL_COLORS = np.array([
    [60, 60, 60],     # 0 = background (grey)
    [255, 80, 80], [80, 255, 80], [80, 80, 255], [255, 255, 80],
    [255, 80, 255], [80, 255, 255], [255, 160, 60], [160, 60, 255],
    [60, 255, 160], [200, 200, 200], [255, 120, 180], [120, 200, 80],
], dtype=np.uint8)


def patch_indicator(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Binary per-patch membership [gh*gw] — a patch belongs if >50% covered."""
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
    return (pooled.flatten() > 0.5).numpy().astype(np.int64)


def build_label_map(seed: np.ndarray, proposals: list[np.ndarray],
                    gh: int, gw: int, H: int, W: int,
                    max_labels: int = 10, min_area_frac: float = 0.005,
                    seed_overlap_thresh: float = 0.2) -> tuple[np.ndarray, int]:
    """Partition the source grid into labels: 1 = clicked object, 2..K = competitors,
    0 = background. Returns (label_map [gh*gw] int, n_labels).

    Competitors are large, mutually-distinct proposals that don't overlap the seed.
    The seed is painted LAST so the clicked object stays a single coherent label.
    """
    L = np.zeros(gh * gw, dtype=np.int64)  # 0 = background everywhere

    # Candidate competitors: big, not overlapping the seed
    cands = []
    for m in proposals:
        if m.sum() < H * W * min_area_frac:
            continue
        if iou(m, seed) > seed_overlap_thresh or (m.astype(bool) & seed.astype(bool)).sum() > 0.5 * m.sum():
            continue
        cands.append(m)
    cands.sort(key=lambda m: m.sum(), reverse=True)

    # Greedily keep mutually-low-overlap competitors
    chosen: list[np.ndarray] = []
    for m in cands:
        if all(iou(m, c) < 0.3 for c in chosen):
            chosen.append(m)
        if len(chosen) >= max_labels - 1:
            break

    # Paint competitors (label 2..K), smaller last so tight regions claim their patches
    for k, m in enumerate(sorted(chosen, key=lambda m: m.sum(), reverse=True), start=2):
        L[patch_indicator(m, gh, gw) > 0] = k

    # Paint seed last as label 1 (overwrites any competitor overlap)
    L[patch_indicator(seed, gh, gw) > 0] = 1
    n_labels = 2 + len(chosen)  # background + seed + competitors
    return L, n_labels


def label_map_to_M(L: np.ndarray, n_labels: int, device: str) -> torch.Tensor:
    """[gh*gw] int label map → [n_labels, N] binary one-hot indicator stack."""
    N = L.shape[0]
    M = torch.zeros(n_labels, N, device=device)
    Lt = torch.from_numpy(L).to(device)
    for k in range(n_labels):
        M[k] = (Lt == k).float()
    return M


def colorize(label_hw: np.ndarray) -> np.ndarray:
    """Int label map [H,W] → RGB visualisation."""
    pal = _LABEL_COLORS[np.clip(label_hw, 0, len(_LABEL_COLORS) - 1)]
    return pal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="dancing")
    ap.add_argument("--instance-id", type=int, default=1)
    ap.add_argument("--feat-size", type=int, default=1024)
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--color-weight", type=float, default=0.0,
                    help="LAB color cost weight added to OT (try 0.2).")
    ap.add_argument("--max-labels", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--cuvler-score", type=float, default=0.35)
    ap.add_argument("--out", default="video/outputs/multi.mp4")
    args = ap.parse_args()

    dino = DenseDINOv3()
    divider = CuVLERDivider(score_thresh=args.cuvler_score)
    conquer_backbone = load_backbone()
    print("[conquer] backbone loaded")

    n = davis.num_frames(args.clip)
    if args.max_frames:
        n = min(n, args.max_frames)

    # --- Frame 0 ---
    frame0 = davis.load_frame(args.clip, 0)
    H, W = frame0.shape[:2]
    gt0 = davis.load_mask(args.clip, 0, instance_id=args.instance_id)
    gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    cx256, cy256 = sample_clicks(gt256, 1)[0]
    click_xy = (cx256 * (W / 256.0), cy256 * (H / 256.0))

    proposals0 = run_conquer(conquer_backbone, frame0, divider.predict(frame0))

    # Seed = best proposal containing the click
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
    feats_prev = F.normalize(dino.extract(img_sized, normalize=False)["feats"].cuda().float(), dim=-1)
    gh, gw = feats_prev.shape[:2]

    L, n_labels = build_label_map(seed, proposals0, gh, gw, H, W, max_labels=args.max_labels)
    print(f"  label map: {n_labels} labels "
          f"({int((L == 1).sum())} seed patches, {int((L > 1).sum())} competitor, "
          f"{int((L == 0).sum())} background)")

    ious = [iou(gt0, seed)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def panel(frame, obj_mask, label_hw, gt, tag):
        a = put_label(overlay_mask(frame, obj_mask, (255, 140, 0)), tag)
        b = put_label(cv2.addWeighted(frame, 0.4, colorize(label_hw), 0.6, 0), "all labels")
        c = put_label(overlay_mask(frame, gt, (64, 255, 64)), "GT")
        return np.concatenate([a, b, c], axis=1)

    L_hw0 = cv2.resize(L.reshape(gh, gw).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    p0 = panel(frame0, (L_hw0 == 1).astype(np.uint8), L_hw0, gt0, f"f0 IoU={ious[0]:.2f}")
    ph, pw = p0.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (pw, ph))
    writer.write(cv2.cvtColor(p0, cv2.COLOR_RGB2BGR))

    frame_prev = frame0
    # --- Propagation loop ---
    for fidx in range(1, n):
        frame = davis.load_frame(args.clip, fidx)
        gt = davis.load_mask(args.clip, fidx, instance_id=args.instance_id)
        img_sized = cv2.resize(frame, (args.feat_size, args.feat_size))
        feats_cur = F.normalize(dino.extract(img_sized, normalize=False)["feats"].cuda().float(), dim=-1)

        color_cost = None
        if args.color_weight > 0.0:
            color_cost = _color_cost_patches(frame_prev, frame, gh, gw, gh, gw, weight=args.color_weight)

        cond = compute_cond(feats_prev, feats_cur, blur=args.blur, cost_addend=color_cost)
        M = label_map_to_M(L, n_labels, device=cond.device)  # [K, N_a]
        heat = M @ cond                                        # [K, N_b]
        L_b = heat.argmax(dim=0).cpu().numpy().astype(np.int64)  # [N_b], patch-level (for carry)

        # Pixel-level readout: upsample each label's heat then argmax → sub-patch boundaries
        heat_up = F.interpolate(heat.reshape(n_labels, 1, gh, gw),
                                size=(H, W), mode="bilinear", align_corners=False)[:, 0]  # [K,H,W]
        L_hw = heat_up.argmax(dim=0).cpu().numpy().astype(np.int64)  # [H, W]
        obj_mask = (L_hw == 1).astype(np.uint8)
        score = iou(gt, obj_mask) if gt.sum() > 0 else float("nan")
        ious.append(score)
        writer.write(cv2.cvtColor(
            panel(frame, obj_mask, L_hw, gt, f"f{fidx} IoU={score:.2f}"), cv2.COLOR_RGB2BGR))
        print(f"  frame {fidx:3d}: IoU={score:.3f}")

        L = L_b           # carry the whole label field forward
        feats_prev = feats_cur
        frame_prev = frame

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
