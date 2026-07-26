"""Whole-image segmentation + tracking: seed all objects automatically at frame 0
(divide + conquer + one-per-object, no click), then track each one forward with its
own instance probe + OT propagation — the same mechanism validated for single-object
click tracking, just run N times in parallel instead of once.

Unlike render_whole_image.py (which re-runs divide+conquer from scratch on every
frame, causing the object count to flicker even on easy clips), this only segments
frame 0 automatically; every later frame is tracked, not re-detected.

Usage:
    python -m video.scripts.render_whole_image_tracked --clip dogs-jump --frames 40 \\
        --out video/outputs/whole_image_tracked_dogs_jump.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "divide_and_conquer"))

import demo as D  # noqa: E402 — reuses _dino/_divider/_conquer_bb + probe/OT helpers
from video.divide.conquer import run_conquer_one_per_object  # noqa: E402
from video.propagation.sinkhorn_ot import propagate_patch  # noqa: E402
from video.loaders import davis  # noqa: E402

_PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 215, 0),
    (218, 112, 214), (0, 206, 209), (255, 140, 0), (154, 205, 50),
]


def _overlay_multi(frame: np.ndarray, masks: list[np.ndarray], label: str) -> np.ndarray:
    out = frame.copy()
    for i, m in enumerate(masks):
        color = _PALETTE[i % len(_PALETTE)]
        mb = m.astype(bool)
        for c in range(3):
            out[:, :, c] = np.where(mb, 0.55 * out[:, :, c] + 0.45 * color[c], out[:, :, c])
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--score", type=float, default=0.35, help="CuVLER score_thresh for the frame-0 seed only")
    ap.add_argument("--feat-size", type=int, default=1024)
    ap.add_argument("--compete", action="store_true",
                    help="Per-patch argmax competition across all tracked objects' fused "
                         "scores, so masks are mutually exclusive instead of independently "
                         "thresholded (which lets two tracks claim/blend into the same pixels).")
    ap.add_argument("--spatial-weight", type=float, default=0.0,
                    help="Euclidean patch-position penalty in the OT cost matrix — keeps each "
                         "object's heat anchored near its own previous position instead of "
                         "jumping to a spatially distant but visually-similar twin object "
                         "(try 0.3; only matters when there are look-alike objects).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clip, feat_size = args.clip, args.feat_size
    n = min(args.frames, davis.num_frames(clip))
    frame0 = davis.load_frame(clip, 0)
    H, W = frame0.shape[:2]

    img_sized = cv2.resize(frame0, (feat_size, feat_size))
    with torch.no_grad():
        feats_prev = D._dino.extract(img_sized, normalize=False)["feats"].cuda().float()
    feats_prev_norm = F.normalize(feats_prev, dim=-1)
    gh_feat, gw_feat = feats_prev_norm.shape[:2]

    # Frame-0 whole-image seed: divide + conquer + one-per-object, no click.
    div = D._divider if args.score == 0.35 else __import__(
        "video.divide.cuvler_divide", fromlist=["CuVLERDivider"]
    ).CuVLERDivider(score_thresh=args.score)
    divide_masks, divide_scores = div.predict_scored(frame0)
    obj_masks, obj_scores = run_conquer_one_per_object(D._conquer_bb, frame0, divide_masks, divide_scores)
    obj_masks = [m.astype(np.uint8) for m in obj_masks]
    N = len(obj_masks)
    print(f"[seed] frame 0: {N} objects found")

    # Match each seeded object to its best-overlapping GT instance (frame 0 only,
    # for reporting a real IoU-vs-GT track quality number — not used by the pipeline).
    gt_ids0 = davis.instance_ids(clip, 0)
    gt0 = {i: davis.load_mask(clip, 0, instance_id=i) for i in gt_ids0}
    matched_gt_id = []
    for m in obj_masks:
        best_id = max(gt0, key=lambda i: _iou(m, gt0[i]), default=None) if gt0 else None
        matched_gt_id.append(best_id)

    patches = [D._mask_to_patch(m, gh=gh_feat, gw=gw_feat) for m in obj_masks]
    probes = []
    for m in obj_masks:
        y0 = D._patch_labels(m, gh_feat, gw_feat)
        probes.append(D._train_probe(feats_prev_norm.reshape(-1, feats_prev_norm.shape[-1]), y0))

    frame_prev = frame0
    frames_out = [_overlay_multi(frame0, obj_masks, f"tracked ({N})")]
    ious_per_obj = [[] for _ in range(N)]
    for i, gid in enumerate(matched_gt_id):
        if gid is not None:
            ious_per_obj[i].append(_iou(obj_masks[i], gt0[gid]))

    for fidx in range(1, n):
        frame = davis.load_frame(clip, fidx)
        img_sized = cv2.resize(frame, (feat_size, feat_size))
        with torch.no_grad():
            feats_cur = D._dino.extract(img_sized, normalize=False)["feats"].cuda().float()
        feats_cur_norm = F.normalize(feats_cur, dim=-1)
        gh_cur, gw_cur = feats_cur_norm.shape[:2]
        color_cost = D._color_cost(frame_prev, frame, gh_feat, gw_feat, gh_cur, gw_cur)

        heats = []
        for i in range(N):
            heat = propagate_patch(feats_prev_norm, feats_cur_norm, patches[i], blur=D.OT_BLUR,
                                   cost_addend=color_cost, spatial_weight=args.spatial_weight)
            heat = heat / (heat.max() + 1e-8)
            score = D._probe_score(probes[i], feats_cur_norm)
            heat = (1.0 - D.PROBE_FUSE_WEIGHT) * heat + D.PROBE_FUSE_WEIGHT * score
            heats.append(heat)

        if args.compete:
            # Per-patch argmax: each patch goes to whichever object scores highest
            # there; losers get zeroed at that patch. Masks become mutually
            # exclusive by construction instead of independently thresholded
            # (which let two tracks overlap/blend into the same pixels).
            stacked = torch.stack(heats, dim=0)          # [N, num_patches]
            winner = stacked.argmax(dim=0)                # [num_patches]
            heats = [torch.where(winner == i, stacked[i], torch.zeros_like(stacked[i])) for i in range(N)]

        new_masks = []
        for i in range(N):
            heat_up = F.interpolate(heats[i].reshape(1, 1, gh_cur, gw_cur), size=(H, W),
                                    mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            if args.compete:
                # Do NOT renormalize by this object's own post-competition max — a
                # patch it barely won (e.g. 0.06 vs 0.05 vs 0.04 at some background
                # sliver) would get blown back up to full confidence, defeating the
                # point of competing in the first place. Threshold the raw value.
                soft_up = heat_up
            else:
                soft_up = heat_up / (heat_up.max() + 1e-8)
            mask = (soft_up > D.OT_THRESH).astype(np.uint8)
            new_masks.append(mask)
            patches[i] = D._mask_to_patch(mask, gh=gh_feat, gw=gw_feat)

        frames_out.append(_overlay_multi(frame, new_masks, f"tracked ({N})"))

        gt_ids = davis.instance_ids(clip, fidx)
        gts = {i: davis.load_mask(clip, fidx, instance_id=i) for i in gt_ids}
        for i, gid in enumerate(matched_gt_id):
            if gid is not None and gid in gts:
                ious_per_obj[i].append(_iou(new_masks[i], gts[gid]))

        feats_prev_norm = feats_cur_norm
        frame_prev = frame

    print(f"\n{'object':<8}{'matched GT id':>14}{'mean IoU':>10}{'n frames':>10}")
    for i in range(N):
        vals = ious_per_obj[i]
        gid = matched_gt_id[i]
        print(f"{i:<8}{str(gid):>14}{(np.mean(vals) if vals else float('nan')):>10.3f}{len(vals):>10}")

    tmp = tempfile.NamedTemporaryFile(suffix="_raw.mp4", delete=False)
    tmp.close()
    h, w = frames_out[0].shape[:2]
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 8, (w, h))
    for f in frames_out:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", tmp.name, "-vcodec", "libx264", "-pix_fmt", "yuv420p", args.out],
                  capture_output=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
