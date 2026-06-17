"""CuVLER divide-mask AR eval on DAVIS (repo-homed; uses the vendored detector).

Mirrors video/scripts/eval_divide_ar.py's metric so numbers are comparable.
Run separately from the CutLER eval — CuVLER's cad.modeling and the local
divide_and_conquer.modeling register the same class names and clash in one process.

    python -m video.scripts.eval_cuvler --clips blackswan,bmx-trees,dogs-jump --frames 0,15,30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.divide.cuvler_divide import CuVLERDivider
from video.loaders import davis

THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def gt_masks(clip, idx):
    return [davis.load_mask(clip, idx, instance_id=i) for i in davis.instance_ids(clip, idx)]


def recall_stats(preds, gts):
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
    ap.add_argument("--clips", default="blackswan,bmx-trees,dogs-jump")
    ap.add_argument("--frames", default="0,15,30")
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    kw = {"score_thresh": args.score}
    if args.weights:
        kw["weights"] = args.weights
    div = CuVLERDivider(**kw)

    clips = [c.strip() for c in args.clips.split(",")]
    frames = [int(f) for f in args.frames.split(",")]
    rows = []
    print(f"\n{'clip':<14}{'frame':>5} | {'AR':>5}{'AR50':>6}{'AR75':>6}{'bestIoU':>8}{'#pred':>6}{'#gt':>4}")
    print("-" * 60)
    for clip in clips:
        for f in frames:
            if f >= davis.num_frames(clip):
                continue
            s = recall_stats(div.predict(davis.load_frame(clip, f)), gt_masks(clip, f))
            if s is None:
                continue
            rows.append(s)
            print(f"{clip:<14}{f:>5} | {s['AR']:>5.3f}{s['AR50']:>6.3f}{s['AR75']:>6.3f}"
                  f"{s['best']:>8.3f}{s['n_pred']:>6}{s['n_gt']:>4}")
    print("-" * 60)
    if rows:
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ["AR", "AR50", "AR75", "best", "n_pred", "n_gt"]}
        print(f"{'MEAN':<14}{'':>5} | {agg['AR']:>5.3f}{agg['AR50']:>6.3f}{agg['AR75']:>6.3f}"
              f"{agg['best']:>8.3f}{agg['n_pred']:>6.1f}{agg['n_gt']:>4.1f}")


if __name__ == "__main__":
    main()
