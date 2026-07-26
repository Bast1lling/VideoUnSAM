"""Chained SMURF-flow mask propagation, no re-anchoring -- the actual apples-to-
apples comparison against OT's published decay curve.

video/flow/eval_flow_warp_davis.py measures something different: every frame
re-starts from the REAL ground-truth mask of the previous frame (oracle-
initialised, single hop). That's useful for judging flow quality in isolation,
but it can't show compounding error, because there's nothing to compound --
truth is injected at every step. This script removes that: the mask at frame 0
is seeded from GT once, then propagated frame-by-frame using ONLY the model's
own previous output, all the way to each offset, exactly the condition flow
would run under if it (or something like it) replaced/fed propagation for real.

This is the same protocol video/scripts/eval_propagation_chain.py used to
produce OT's published number: pure chained OT loses ~0.10 IoU over 30 frames,
vs -0.26 to -0.33 for every other trained propagation alternative tried. This
script's job is to put a real, comparable number next to that one for flow.

    python -m video.scripts.eval_flow_propagation_chain \
        --checkpoint ytvis_ft15k.pt --offsets 5,10,20,30 --limit-clips 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.loaders import davis
from video.flow.smurf_infer import load_smurf, compute_backward_flow, warp_mask_backward

_SPLIT = Path(__file__).resolve().parents[1] / "decoder" / "davis_split.json"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--offsets", default="5,10,20,30")
    ap.add_argument("--limit-clips", type=int, default=20)
    ap.add_argument("--split", choices=["clean", "all"], default="clean")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    offsets = sorted(int(x) for x in args.offsets.split(","))
    max_offset = offsets[-1]

    clips = json.load(open(_SPLIT))["clean"] if args.split == "clean" else davis.list_clips()
    if args.limit_clips:
        clips = clips[:args.limit_clips]

    model = load_smurf(args.checkpoint, device=args.device)
    print(f"[eval] {len(clips)} clips, offsets={offsets}, chained frame-by-frame, no re-anchor")

    per_offset_chain: dict[int, list[float]] = {o: [] for o in offsets}
    per_offset_copy: dict[int, list[float]] = {o: [] for o in offsets}

    for clip in clips:
        n = davis.num_frames(clip)
        inst_ids = davis.instance_ids(clip, 0)
        if not inst_ids:
            continue

        frame_cache: dict[int, np.ndarray] = {}

        def frame(idx):
            if idx not in frame_cache:
                frame_cache[idx] = davis.load_frame(clip, idx)
            return frame_cache[idx]

        clip_scores = []
        for inst in inst_ids:
            mask0 = davis.load_mask(clip, 0, instance_id=inst)
            if mask0.sum() == 0:
                continue

            cur_mask = mask0
            last = min(max_offset, n - 1)
            for t in range(1, last + 1):
                flow_bw = compute_backward_flow(model, frame(t), frame(t - 1), device=args.device)
                cur_mask = warp_mask_backward(cur_mask, flow_bw)

                if t in offsets:
                    gt = davis.load_mask(clip, t, instance_id=inst)
                    if gt.sum() == 0:
                        continue
                    chain_score = iou(cur_mask, gt)
                    copy_score = iou(mask0, gt)
                    per_offset_chain[t].append(chain_score)
                    per_offset_copy[t].append(copy_score)
                    clip_scores.append((t, inst, chain_score, copy_score))

        print(f"  [{clip}] " + "  ".join(
            f"o={o} i={i}: flow={c:.2f} copy={d:.2f}" for o, i, c, d in clip_scores))

    print()
    for o in offsets:
        c, d = per_offset_chain[o], per_offset_copy[o]
        if c:
            print(f"offset {o:3d}: flow-chain mIoU {np.mean(c):.4f}  copy mIoU {np.mean(d):.4f}  "
                  f"delta {np.mean(c) - np.mean(d):+.4f}  (n={len(c)})")
        else:
            print(f"offset {o:3d}: no valid instances")

    print("\nCompare against OT's published chain: ~0.10 IoU lost over 30 frames "
          "(propagation-ot-vs-alternatives). This script's frame-0 mIoU is always "
          "1.0 by construction (mask0 == gt at t=0), so read the delta from there.")


if __name__ == "__main__":
    main()
