"""Feasibility test for Stage-5 temporal self-training: is forward-backward OT
cycle-agreement a usable proxy for pseudo-label quality?

Cheap, training-free check before building any self-training loop. For each
clip/instance/offset, propagate the GT frame-0 mask (stand-in for a
CuVLER+conquer pseudo-mask, per eval_propagation.py's framing) forward to
frame t+offset, then propagate that predicted mask BACKWARD to frame 0.
Two numbers per (clip, instance, offset):

  agreement    = IoU(cycle-returned mask, original frame-0 mask)   -- proxy signal
  true_quality = IoU(forward-propagated mask, frame t+offset GT)   -- what we'd want to gate

If agreement correlates with true_quality, cycle-agreement is a usable
training-free filter for selecting which frames' propagated masks are good
enough to add to a self-training pseudo-label set. If not, the signal is
noise and a self-training loop built on top of it is a dead end.

Note: this measures a WHOLE-MASK round-trip IoU as a scalar per-frame gate,
not the previously-rejected per-patch spatial reweight
(propagate_patch's cycle_weight -- see cycle-consistency-erodes-mask memory,
which failed because it modifies the mask itself, patch by patch). This only
asks whether the scalar is informative enough to accept/reject a frame
wholesale, which is a different (coarser, cheaper) use of the same idea.

    python -m video.scripts.eval_cycle_quality_signal --offsets 5,10,20 --limit-clips 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

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
    ap.add_argument("--offsets", default="5,10,20")
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit-clips", type=int, default=20)
    ap.add_argument("--split", choices=["clean", "all"], default="clean")
    args = ap.parse_args()
    offsets = [int(x) for x in args.offsets.split(",")]

    clips = json.load(open(_SPLIT))["clean"] if args.split == "clean" else davis.list_clips()
    if args.limit_clips:
        clips = clips[:args.limit_clips]

    extractor = DenseDINOv3()
    print(f"[eval] {len(clips)} clips, offsets={offsets}, blur={args.blur}")

    agreements: list[float] = []
    qualities: list[float] = []
    rows: list[tuple[str, int, int, float, float]] = []  # clip, inst, offset, agree, quality

    for clip in clips:
        n = davis.num_frames(clip)
        inst_ids = davis.instance_ids(clip, 0)
        if not inst_ids:
            continue
        frame0 = davis.load_frame(clip, 0)
        feats0 = extractor.extract(frame0)
        clip_rows = []
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

                fwd = propagate(
                    feats0["feats"], feats_b["feats"], mask_a,
                    out_size=(frame_b.shape[0], frame_b.shape[1]),
                    blur=args.blur, threshold=args.threshold,
                )
                quality = iou(fwd["mask"], mask_b_gt)
                if fwd["mask"].sum() == 0:
                    continue

                back = propagate(
                    feats_b["feats"], feats0["feats"], fwd["mask"],
                    out_size=(frame0.shape[0], frame0.shape[1]),
                    blur=args.blur, threshold=args.threshold,
                )
                agreement = iou(back["mask"], mask_a)

                agreements.append(agreement)
                qualities.append(quality)
                rows.append((clip, inst, o, agreement, quality))
                clip_rows.append((o, inst, agreement, quality))
        print(f"  [{clip}] " + "  ".join(
            f"o={o} i={i}: agree={a:.2f} qual={q:.2f}" for o, i, a, q in clip_rows))

    agreements_arr = np.array(agreements)
    qualities_arr = np.array(qualities)
    n = len(agreements_arr)
    print(f"\n[n={n}]")
    if n < 3:
        print("Not enough samples for correlation.")
        return

    r, p_r = pearsonr(agreements_arr, qualities_arr)
    rho, p_rho = spearmanr(agreements_arr, qualities_arr)
    print(f"Pearson  r   = {r:.3f}  (p={p_r:.2e})")
    print(f"Spearman rho = {rho:.3f}  (p={p_rho:.2e})")

    order = np.argsort(agreements_arr)
    quartiles = np.array_split(order, 4)
    print("\nQuartile of agreement -> mean true quality (is high agreement -> high quality?):")
    for i, idx in enumerate(quartiles):
        lo, hi = agreements_arr[idx].min(), agreements_arr[idx].max()
        print(f"  Q{i+1} (agreement {lo:.2f}-{hi:.2f}): mean quality={qualities_arr[idx].mean():.3f}"
              f"  (n={len(idx)})")

    baseline = qualities_arr.mean()
    top_half = qualities_arr[order[n // 2:]]
    print(f"\nUnfiltered mean quality:        {baseline:.3f}")
    print(f"Top-50%-agreement mean quality: {top_half.mean():.3f}  "
          f"(delta {top_half.mean() - baseline:+.3f})")


if __name__ == "__main__":
    main()
