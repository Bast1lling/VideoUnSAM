"""DAVIS 2017 val — unsupervised multi-object video segmentation eval.

The whole-image line's missing video-level number: no click, no GT init.
Frame 0 is segmented automatically (divide + conquer + one-per-object
consolidation), then every discovered object is tracked forward with its own
instance probe + Sinkhorn OT, with per-patch argmax competition making the
tracks mutually exclusive. Predicted tracks are matched to GT instance tracks
by Hungarian assignment on per-track mean J (IoU), and the standard J / F /
J&F means are reported over all GT objects (unmatched GT objects score 0) —
the DAVIS 2017 unsupervised protocol, comparable to e.g. VVitCutLER's
24.35% J&F (arXiv 2605.17584, which distills from SAM2; this pipeline uses
no SAM weights anywhere).

Protocol notes: matching is on J only (F computed after matching, for the
matched pairs — computing F for every candidate pair is ~10x slower and
essentially never changes the assignment); frames where a GT object is absent
(fully occluded) are excluded from that object's mean, consistent with
eval_davis2016.py.

Usage:
    python -m video.scripts.eval_davis2017_unsupervised
    python -m video.scripts.eval_davis2017_unsupervised --clips dogs-jump,judo --compete
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))

from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.loaders import davis  # noqa: E402
from video.divide.cuvler_divide import CuVLERDivider  # noqa: E402
from video.divide.conquer import load_backbone, run_conquer_one_per_object  # noqa: E402
from video.propagation.sinkhorn_ot import propagate_patch  # noqa: E402
from video.scripts.eval_davis2016 import (  # noqa: E402
    train_probe, probe_score, _patch_labels, mask_to_patch, j_score, f_score, _write_video,
)

DAVIS2017_VAL = (_REPO / "datasets/davis/DAVIS/ImageSets/2017/val.txt").read_text().split()

_GT_PALETTE = None

# Distinct, stark per-track colors (not the single red-orange used for the
# single-object DAVIS 2016 renderer — here N objects need to stay visually
# separable in one panel).
_TRACK_COLORS = [
    (255, 45, 0), (0, 160, 255), (255, 210, 0), (150, 60, 255),
    (0, 220, 140), (255, 100, 180), (255, 255, 255), (120, 200, 0),
]


def _render_multi_frame(frame: np.ndarray, masks: list[np.ndarray], tag: str) -> np.ndarray:
    """One panel: each mask in `masks` filled with its own color + white outline."""
    panel = frame.copy()
    for idx, m in enumerate(masks):
        color = np.array(_TRACK_COLORS[idx % len(_TRACK_COLORS)])
        mb = m.astype(bool)
        panel[mb] = (0.35 * panel[mb] + 0.65 * color).astype(np.uint8)
    for idx, m in enumerate(masks):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 2)
    w = max(260, 12 * len(tag) + 20)
    cv2.rectangle(panel, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(panel, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _davis_palette() -> list[int]:
    global _GT_PALETTE
    if _GT_PALETTE is None:
        from PIL import Image
        ann = _REPO / "datasets/davis/DAVIS/Annotations/480p/bear/00000.png"
        _GT_PALETTE = Image.open(ann).getpalette()
    return _GT_PALETTE


def _flatten_tracks(masks: list[np.ndarray], scores: list[np.ndarray]) -> np.ndarray:
    """Merge per-track binary masks into one label map (IDs 1..N).

    Overlaps go to the covering track with the highest score at that pixel.
    """
    label = np.zeros(masks[0].shape, dtype=np.uint8)
    best = np.full(masks[0].shape, -np.inf, dtype=np.float32)
    for i, (m, s) in enumerate(zip(masks, scores)):
        sel = (m > 0) & (s > best)
        label[sel] = i + 1
        best[sel] = s[sel]
    return label


def _save_label_png(label: np.ndarray, path: Path) -> None:
    from PIL import Image
    img = Image.fromarray(label, mode="P")
    img.putpalette(_davis_palette())
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def eval_clip(clip: str, dino: DenseDINOv3, divider: CuVLERDivider,
              backbone, args) -> tuple[list[float], list[float], int, int]:
    """Returns (per-GT-object J means, per-GT-object F means, n_tracks, n_gt)."""
    n = davis.num_frames(clip)
    gt_ids = davis.instance_ids(clip, 0)
    if not gt_ids:
        return [], [], 0, 0

    frame0 = davis.load_frame(clip, 0)
    H, W = frame0.shape[:2]

    # Frame-0 automatic whole-image seed — no click anywhere.
    divide_masks, divide_scores = divider.predict_scored(frame0)
    obj_masks, obj_scores = run_conquer_one_per_object(backbone, frame0, divide_masks, divide_scores)
    kept = [(m.astype(np.uint8), s) for m, s in zip(obj_masks, obj_scores) if m.sum() > 0]
    obj_masks = [m for m, _ in kept][:args.max_tracks]
    obj_scores = [s for _, s in kept][:args.max_tracks]
    N = len(obj_masks)
    if N == 0:
        if args.save_dir:  # toolkit requires a PNG for every frame
            sd = Path(args.save_dir)
            for d in (sd, sd.parent / (sd.name + "_prio")):
                for f in range(n):
                    _save_label_png(np.zeros((H, W), dtype=np.uint8),
                                    d / clip / f"{f:05d}.png")
        return [0.0] * len(gt_ids), [0.0] * len(gt_ids), 0, len(gt_ids)

    img_sized = cv2.resize(frame0, (args.feat_size, args.feat_size))
    with torch.no_grad():
        feats_prev = dino.extract(img_sized, normalize=False)["feats"].cuda().float()
    feats_prev_norm = F.normalize(feats_prev, dim=-1)
    gh, gw = feats_prev_norm.shape[:2]
    flat0 = feats_prev_norm.reshape(-1, feats_prev_norm.shape[-1])

    probes = [train_probe(flat0, _patch_labels(m, gh, gw)) for m in obj_masks]
    patches = [mask_to_patch(m, gh=gh, gw=gw) for m in obj_masks]
    pred_tracks: list[list[np.ndarray]] = [[m] for m in obj_masks]

    render_frames = []
    if args.render_dir:
        gt0_masks = [davis.load_mask(clip, 0, instance_id=g) for g in gt_ids]
        left0 = _render_multi_frame(frame0, obj_masks, f"Prediction (N={N}) - f0")
        right0 = _render_multi_frame(frame0, gt0_masks, f"Ground Truth (N={len(gt_ids)})")
        sep0 = np.full((H, 4, 3), 255, dtype=np.uint8)
        render_frames.append(np.concatenate([left0, sep0, right0], axis=1))

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        # Variant A (save_dir): per-pixel max soft score; frame 0 has no soft
        # scores yet, so the smaller (more specific) mask wins via 1/area.
        # Variant B (save_dir + "_prio"): whole-track priority by CuVLER seed
        # score — near-duplicate tracks keep the object intact instead of
        # fragmenting it pixel-by-pixel.
        prio_dir = save_dir.parent / (save_dir.name + "_prio")
        prio = [np.full((H, W), s - 1e-6 * i, dtype=np.float32)
                for i, s in enumerate(obj_scores)]
        areas = [np.full((H, W), 1.0 / max(int(m.sum()), 1), dtype=np.float32)
                 for m in obj_masks]
        _save_label_png(_flatten_tracks(obj_masks, areas),
                        save_dir / clip / "00000.png")
        _save_label_png(_flatten_tracks(obj_masks, prio),
                        prio_dir / clip / "00000.png")

    for fidx in range(1, n):
        frame = davis.load_frame(clip, fidx)
        img_sized = cv2.resize(frame, (args.feat_size, args.feat_size))
        with torch.no_grad():
            feats_cur = dino.extract(img_sized, normalize=False)["feats"].cuda().float()
        feats_cur_norm = F.normalize(feats_cur, dim=-1)

        heats = []
        for i in range(N):
            heat = propagate_patch(feats_prev_norm, feats_cur_norm, patches[i],
                                   blur=args.blur, spatial_weight=args.spatial_weight)
            heat = heat / (heat.max() + 1e-8)
            score = probe_score(probes[i], feats_cur_norm)
            heats.append((1.0 - args.probe_weight) * heat + args.probe_weight * score)

        if args.compete:
            # Mutually exclusive tracks: each patch belongs to its argmax object.
            # Post-competition heats are NOT renormalised per object — blowing a
            # barely-won background sliver back up to full confidence would defeat
            # the competition (see render_whole_image_tracked.py).
            stacked = torch.stack(heats, dim=0)
            winner = stacked.argmax(dim=0)
            heats = [torch.where(winner == i, stacked[i], torch.zeros_like(stacked[i]))
                     for i in range(N)]

        frame_masks, frame_softs = [], []
        for i in range(N):
            heat_up = F.interpolate(heats[i].reshape(1, 1, gh, gw), size=(H, W),
                                    mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            soft = heat_up if args.compete else heat_up / (heat_up.max() + 1e-8)
            mask = (soft > args.thresh).astype(np.uint8)
            pred_tracks[i].append(mask)
            patches[i] = mask_to_patch(mask, gh=gh, gw=gw)
            frame_masks.append(mask)
            frame_softs.append(soft.astype(np.float32))

        if save_dir is not None:
            _save_label_png(_flatten_tracks(frame_masks, frame_softs),
                            save_dir / clip / f"{fidx:05d}.png")
            _save_label_png(_flatten_tracks(frame_masks, prio),
                            prio_dir / clip / f"{fidx:05d}.png")

        if args.render_dir:
            gtf_masks = [davis.load_mask(clip, fidx, instance_id=g) for g in gt_ids]
            leftf = _render_multi_frame(frame, frame_masks, f"Prediction (N={N}) - f{fidx}")
            rightf = _render_multi_frame(frame, gtf_masks, f"Ground Truth (N={len(gt_ids)})")
            sepf = np.full((H, 4, 3), 255, dtype=np.uint8)
            render_frames.append(np.concatenate([leftf, sepf, rightf], axis=1))

        feats_prev_norm = feats_cur_norm

    if args.render_dir and render_frames:
        _write_video(render_frames, Path(args.render_dir) / f"{clip}.mp4")

    # GT tracks (a frame where the object is absent is excluded from its mean).
    gt_tracks = {g: [davis.load_mask(clip, f, instance_id=g) for f in range(n)]
                 for g in gt_ids}

    # Hungarian matching on per-track mean J.
    jmat = np.zeros((N, len(gt_ids)), dtype=np.float64)
    for i in range(N):
        for jg, g in enumerate(gt_ids):
            vals = [j_score(pred_tracks[i][f], gt_tracks[g][f])
                    for f in range(n) if gt_tracks[g][f].sum() > 0]
            jmat[i, jg] = np.mean(vals) if vals else 0.0
    rows, cols = linear_sum_assignment(-jmat)
    assigned = {int(c): int(r) for r, c in zip(rows, cols)}

    js, fs = [], []
    for jg, g in enumerate(gt_ids):
        if jg in assigned:
            i = assigned[jg]
            frames_valid = [f for f in range(n) if gt_tracks[g][f].sum() > 0]
            js.append(float(np.mean([j_score(pred_tracks[i][f], gt_tracks[g][f])
                                     for f in frames_valid])))
            fs.append(float(np.mean([f_score(pred_tracks[i][f], gt_tracks[g][f])
                                     for f in frames_valid])))
        else:  # more GT objects than predicted tracks
            js.append(0.0)
            fs.append(0.0)
    return js, fs, N, len(gt_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="", help="Subset (default: official 30-clip val split)")
    ap.add_argument("--feat-size", type=int, default=1024)
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--probe-weight", type=float, default=0.5)
    ap.add_argument("--spatial-weight", type=float, default=0.0)
    ap.add_argument("--cuvler-score", type=float, default=0.35)
    ap.add_argument("--max-tracks", type=int, default=20,
                    help="Cap on predicted tracks (DAVIS unsupervised protocol allows 20).")
    ap.add_argument("--save-dir", default="",
                    help="Export per-frame indexed-palette PNGs (IDs 1..N) for the "
                         "official davis2017-evaluation toolkit; overlaps between "
                         "independent tracks resolved by max soft score per pixel.")
    ap.add_argument("--compete", action="store_true",
                    help="Per-patch argmax competition across tracks (mutually exclusive "
                         "masks). Off by default: at 30-clip scale it is net negative "
                         "(J&F 0.301 vs 0.317 independent) — junk duplicate tracks from "
                         "frame-0 over-segmentation steal patches from real objects "
                         "(blackswan −0.24, libby −0.20); it only helps where tracks "
                         "genuinely contend (dogs-jump, bike-packing, ~+0.02).")
    ap.add_argument("--render-dir", default="",
                    help="If set, write a per-clip video: prediction panel (each "
                         "track its own color) beside a ground-truth panel.")
    args = ap.parse_args()

    clips = [c.strip() for c in args.clips.split(",")] if args.clips else DAVIS2017_VAL
    print(f"[eval_davis2017_unsupervised] {len(clips)} clips  "
          f"compete={'on' if args.compete else 'off'}  "
          f"spatial_weight={args.spatial_weight}")

    dino = DenseDINOv3()
    divider = CuVLERDivider(score_thresh=args.cuvler_score)
    backbone = load_backbone()
    print("[conquer] backbone loaded")

    all_j, all_f = [], []
    t0 = time.time()
    for clip in clips:
        tc = time.time()
        js, fs, n_pred, n_gt = eval_clip(clip, dino, divider, backbone, args)
        all_j.extend(js)
        all_f.extend(fs)
        jm = np.mean(js) if js else float("nan")
        fm = np.mean(fs) if fs else float("nan")
        print(f"  {clip:<22} n_pred={n_pred:<3} n_gt={n_gt:<2} "
              f"J={jm:.3f}  F={fm:.3f}  J&F={(jm+fm)/2:.3f}  ({time.time()-tc:.0f}s)")

    if all_j:
        mj, mf = float(np.mean(all_j)), float(np.mean(all_f))
        print(f"\n{'─'*60}")
        print(f"  J mean  {mj:.3f}")
        print(f"  F mean  {mf:.3f}")
        print(f"  J&F     {(mj+mf)/2:.3f}   ({len(all_j)} GT objects, "
              f"{(time.time()-t0)/60:.1f} min)")
        print("  reference: VVitCutLER (SAM2-distilled) J&F 0.244 on this protocol")


if __name__ == "__main__":
    main()
