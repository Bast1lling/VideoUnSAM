"""Aggregate OT mask-propagation eval across DAVIS clips (Stage-3 rigor check).

For each clip, takes the GT instance masks at frame 0 as the source ("keyframe")
mask -- standing in for a Phase-1 CuVLER+conquer pseudo-mask, per propagate_mask.py's
framing -- and OT-propagates directly (single jump, no chaining) to frames at
increasing offsets. Reports mean IoU vs GT at each offset, aggregated over
clips/instances. propagate_mask.py / propagate_chain.py only ever looked at one
clip/instance/offset at a time; this is the first aggregate number.

    python -m video.scripts.eval_propagation --offsets 1,5,10,20,30 --limit-clips 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis
from video.propagation.sinkhorn_ot import propagate

_SPLIT = Path(__file__).resolve().parents[1] / "decoder" / "davis_split.json"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="1,5,10,20,30")
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit-clips", type=int, default=10)
    ap.add_argument("--split", choices=["clean", "all"], default="clean",
                     help="'clean' = the 64-clip leak-free split used by eval_heldout/eval_clicks "
                          "(no training leakage concern for OT itself, but keeps numbers comparable).")
    args = ap.parse_args()
    offsets = [int(x) for x in args.offsets.split(",")]

    clips = json.load(open(_SPLIT))["clean"] if args.split == "clean" else davis.list_clips()
    if args.limit_clips:
        clips = clips[:args.limit_clips]

    extractor = DenseDINOv3()
    print(f"[eval] {len(clips)} clips, offsets={offsets}, blur={args.blur}")

    per_offset: dict[int, list[float]] = {o: [] for o in offsets}

    for clip in clips:
        n = davis.num_frames(clip)
        inst_ids = davis.instance_ids(clip, 0)
        if not inst_ids:
            continue
        frame0 = davis.load_frame(clip, 0)
        feats0 = extractor.extract(frame0)
        clip_scores = []
        for o in offsets:
            if o >= n:
                continue
            frame_b = davis.load_frame(clip, o)
            feats_b = extractor.extract(frame_b)
            for inst in inst_ids:
                mask_a = davis.load_mask(clip, 0, instance_id=inst)
                mask_b_gt = davis.load_mask(clip, o, instance_id=inst)
                if mask_a.sum() == 0 or mask_b_gt.sum() == 0:
                    continue
                result = propagate(
                    feats0["feats"], feats_b["feats"], mask_a,
                    out_size=(frame_b.shape[0], frame_b.shape[1]),
                    blur=args.blur, threshold=args.threshold,
                )
                score = iou(result["mask"], mask_b_gt)
                per_offset[o].append(score)
                clip_scores.append((o, inst, score))
        print(f"  [{clip}] " + "  ".join(f"o={o} i={i}:{s:.2f}" for o, i, s in clip_scores))

    print()
    for o in offsets:
        vals = per_offset[o]
        if vals:
            print(f"offset {o:3d}: mIoU {np.mean(vals):.4f}  median {np.median(vals):.4f}  (n={len(vals)})")
        else:
            print(f"offset {o:3d}: no valid instances")


if __name__ == "__main__":
    main()
