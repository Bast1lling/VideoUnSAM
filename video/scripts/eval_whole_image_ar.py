"""Whole-image (automatic, multi-object) segmentation AR eval — the real test of
"does divide(+conquer) recover every object in the frame", not just the one the
user clicked.

Unlike eval_cuvler.py (3 clips, single-instance-per-clip DAVIS 2016 subset), this
runs on genuinely multi-instance DAVIS 2017 clips (2-8 labeled objects per frame)
across many frames, and reports divide-only AR vs divide+conquer AR so we can see
whether conquer's per-object sub-mask expansion helps or hurts scene-level recall
(conquer is known to over-segment single objects into parts in other contexts —
this checks whether that shows up here as spurious/duplicate proposals dragging
down precision-sensitive metrics, and whether it helps recall at all).

Usage:
    python -m video.scripts.eval_whole_image_ar --n-clips 20 --frames-per-clip 3
    python -m video.scripts.eval_whole_image_ar --clips dancing,lindy-hop --frames 0,20,40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "divide_and_conquer"))

from video.divide.cuvler_divide import CuVLERDivider  # noqa: E402
from video.divide.conquer import load_backbone, run_conquer, run_conquer_scored, run_conquer_one_per_object  # noqa: E402
from video.divide.consolidate import consolidate  # noqa: E402
from video.loaders import davis  # noqa: E402

THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def gt_masks(clip: str, idx: int) -> list[np.ndarray]:
    return [davis.load_mask(clip, idx, instance_id=i) for i in davis.instance_ids(clip, idx)]


def recall_stats(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict | None:
    if not gts:
        return None
    best = np.array([max((iou(g, p) for p in preds), default=0.0) for g in gts])
    return {
        "AR": float(np.mean([(best >= t).mean() for t in THRESHOLDS])),
        "AR50": float((best >= 0.5).mean()),
        "AR75": float((best >= 0.75).mean()),
        "best": float(best.mean()),
        "n_pred": len(preds), "n_gt": len(gts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="", help="Comma-separated clip list (default: sample multi-instance clips)")
    ap.add_argument("--n-clips", type=int, default=20, help="How many multi-instance clips to sample if --clips not given")
    ap.add_argument("--frames-per-clip", type=int, default=3)
    ap.add_argument("--frames", default="", help="Explicit comma-separated frame indices (overrides --frames-per-clip)")
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--consolidate", action="store_true",
                    help="Also run greedy cross-object NMS consolidation and report its AR/count.")
    ap.add_argument("--nms-iou", default="0.5",
                    help="Comma-separated IoU threshold(s) to sweep for --consolidate.")
    ap.add_argument("--one-per-object", action="store_true",
                    help="Also run run_conquer_one_per_object and report its AR/count.")
    ap.add_argument("--divide-nms", type=float, default=0.0,
                    help="If >0, run NMS (this IoU threshold) on the divide-level masks "
                         "before one-per-object — collapses duplicate overlapping boxes "
                         "(useful at a low --score where divide over-detects).")
    args = ap.parse_args()

    all_clips = davis.list_clips()
    if args.clips:
        clips = [c.strip() for c in args.clips.split(",")]
    else:
        multi = [c for c in all_clips if len(davis.instance_ids(c, 0)) > 1]
        # deterministic spread across the sorted list rather than the first N alphabetically
        step = max(1, len(multi) // args.n_clips)
        clips = multi[::step][:args.n_clips]

    print(f"[eval_whole_image_ar] {len(clips)} clips: {clips}")

    div = CuVLERDivider(score_thresh=args.score)
    backbone = load_backbone()
    nms_ious = [float(x) for x in args.nms_iou.split(",")]

    divide_rows, conquer_rows = [], []
    consolidated_rows = {t: [] for t in nms_ious}
    one_per_object_rows = []
    t0 = time.time()
    for clip in clips:
        n = davis.num_frames(clip)
        if args.frames:
            frames = [int(f) for f in args.frames.split(",") if int(f) < n]
        else:
            frames = sorted(set(int(x) for x in np.linspace(0, n - 1, args.frames_per_clip)))
        for f in frames:
            frame = davis.load_frame(clip, f)
            gts = gt_masks(clip, f)
            if not gts:
                continue
            tf = time.time()
            if args.consolidate or args.one_per_object:
                divide_masks, divide_scores = div.predict_scored(frame)
                all_masks, all_scores = run_conquer_scored(backbone, frame, divide_masks, divide_scores)
            else:
                divide_masks = div.predict(frame)
                all_masks = run_conquer(backbone, frame, divide_masks)
            dt = time.time() - tf

            sd = recall_stats(divide_masks, gts)
            sc = recall_stats(all_masks, gts)
            if sd is None:
                continue
            divide_rows.append(sd)
            conquer_rows.append(sc)
            line = (f"  {clip:<18} f{f:>4} ({dt:>5.1f}s)  n_gt={sd['n_gt']:<2} | "
                    f"divide: AR={sd['AR']:.3f} AR50={sd['AR50']:.3f} n_pred={sd['n_pred']:<3} | "
                    f"+conquer: AR={sc['AR']:.3f} AR50={sc['AR50']:.3f} n_pred={sc['n_pred']:<3}")
            if args.consolidate:
                for t in nms_ious:
                    final_masks, _ = consolidate(all_masks, all_scores, iou_thresh=t)
                    sf = recall_stats(final_masks, gts)
                    consolidated_rows[t].append(sf)
            if args.one_per_object:
                dm, ds = divide_masks, divide_scores
                if args.divide_nms > 0:
                    dm, ds = consolidate(dm, ds, iou_thresh=args.divide_nms)
                opo_masks, _ = run_conquer_one_per_object(backbone, frame, dm, ds)
                so = recall_stats(opo_masks, gts)
                one_per_object_rows.append(so)
                tag = "1perobj+nms" if args.divide_nms > 0 else "1perobj"
                line += f" | {tag}: AR={so['AR']:.3f} AR50={so['AR50']:.3f} n_pred={so['n_pred']:<3}"
            print(line)

    def agg(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in ["AR", "AR50", "AR75", "best", "n_pred", "n_gt"]}

    ad, ac = agg(divide_rows), agg(conquer_rows)
    print(f"\n{'='*100}")
    print(f"{len(divide_rows)} frames, {len(clips)} clips, {(time.time()-t0)/60:.1f} min")
    print(f"{'stage':<14}{'AR':>7}{'AR50':>7}{'AR75':>7}{'bestIoU':>9}{'avg#pred':>10}{'avg#gt':>8}")
    print(f"{'divide':<14}{ad['AR']:>7.3f}{ad['AR50']:>7.3f}{ad['AR75']:>7.3f}{ad['best']:>9.3f}{ad['n_pred']:>10.1f}{ad['n_gt']:>8.1f}")
    print(f"{'+conquer':<14}{ac['AR']:>7.3f}{ac['AR50']:>7.3f}{ac['AR75']:>7.3f}{ac['best']:>9.3f}{ac['n_pred']:>10.1f}{ac['n_gt']:>8.1f}")
    if args.consolidate:
        for t in nms_ious:
            af = agg(consolidated_rows[t])
            print(f"{'nms@'+str(t):<14}{af['AR']:>7.3f}{af['AR50']:>7.3f}{af['AR75']:>7.3f}{af['best']:>9.3f}{af['n_pred']:>10.1f}{af['n_gt']:>8.1f}")
    if args.one_per_object:
        ao = agg(one_per_object_rows)
        print(f"{'1perobj':<14}{ao['AR']:>7.3f}{ao['AR50']:>7.3f}{ao['AR75']:>7.3f}{ao['best']:>9.3f}{ao['n_pred']:>10.1f}{ao['n_gt']:>8.1f}")


if __name__ == "__main__":
    main()
