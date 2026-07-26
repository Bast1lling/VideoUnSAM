"""Precompute SMURF-flow-warped masks as a refinelab sidecar. Plain PyTorch --
no separate TF venv needed (see video/flow/smurf_infer.py for why: the
ytvis_ft15k.pt checkpoint is loadable directly by torchvision's own RAFT
machinery, via the vendored video/flow/smurf_raft/).

Reads an EXISTING dump written by video/scripts/dump_artifacts.py, and for
every frame t >= 1 warps frame (t-1)'s CONTINUOUS heat (soft_up, the same
bilinear-upsampled-and-peak-normalized field refinelab/bench.py reconstructs
from `heat`) forward into frame t using SMURF backward flow. Writes
dump/flow/<clip>/<fidx>.npz as a float16 field, NOT a binary mask.

Why continuous, not binary: an earlier version of this script warped the
already-thresholded `ot_mask` and stored a bit-packed binary flow_mask. That
breaks flow_snap's blend -- (1-blend)*base + blend*flow_mask, with base ALSO
already binary, collapses to exactly 3 regimes (pure base, intersection, pure
flow) as blend crosses 0.5, not a real interpolation, which only became
obvious after sweeping blend and seeing 5 identical rows on each side of 0.5.
Warping the soft field instead keeps the blend meaningful across its range.

This deliberately warps the CHAIN's own heat (whatever OT actually produced,
right or wrong), not the ground truth -- that's what a real flow_snap refiner
would have available at inference time. It is therefore NOT the same
experiment as eval_flow_warp_davis.py (GT-to-GT, oracle-initialised, single
hop) or eval_flow_propagation_chain.py (GT-seeded, chained, no re-anchor,
measuring drift): this one measures flow quality warping the pipeline's own
possibly-already-wrong previous heat, one hop, display-only.

    # after: python -m video.scripts.dump_artifacts --out-dir dumps/davis2016_default
    python -m video.flow.dump_flow_masks \
        --dump dumps/davis2016_default --checkpoint ytvis_ft15k.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.flow.smurf_infer import load_smurf, compute_backward_flow, warp_soft_backward


def clips_in(dump: str) -> list[str]:
    npz_dir = os.path.join(dump, "npz")
    return sorted(d for d in os.listdir(npz_dir)
                 if os.path.isdir(os.path.join(npz_dir, d)))


def frames_in(dump: str, clip: str) -> list[int]:
    d = os.path.join(dump, "npz", clip)
    return sorted(int(f[:-4]) for f in os.listdir(d) if f.endswith(".npz"))


def load_frame_rgb(dump: str, clip: str, fidx: int) -> np.ndarray:
    bgr = cv2.imread(os.path.join(dump, "frames", clip, f"{fidx:05d}.jpg"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_soft_up(dump: str, clip: str, fidx: int) -> np.ndarray:
    """Reconstructs the exact soft_up field refinelab/bench.py's load_frame
    builds from the raw patch-grid `heat` -- bilinear upsample, peak-normalize."""
    z = np.load(os.path.join(dump, "npz", clip, f"{fidx:05d}.npz"))
    shape = z["shape"]
    h, w = int(shape[0]), int(shape[1])
    gh, gw = (int(x) for x in z["grid"])
    heat = z["heat"].astype(np.float32).reshape(gh, gw)
    heat_up = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    return heat_up / (heat_up.max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_smurf(args.checkpoint, device=args.device)

    clips = clips_in(args.dump)
    for i, clip in enumerate(clips, 1):
        frames = frames_in(args.dump, clip)
        out_dir = os.path.join(args.dump, "flow", clip)
        os.makedirs(out_dir, exist_ok=True)
        n_written = 0
        for t in frames:
            if t == 0 or (t - 1) not in frames:
                continue
            prev_soft = load_soft_up(args.dump, clip, t - 1)
            frame_prev = load_frame_rgb(args.dump, clip, t - 1)
            frame_cur = load_frame_rgb(args.dump, clip, t)

            flow_bw = compute_backward_flow(model, frame_cur, frame_prev, device=args.device)
            flow_heat = warp_soft_backward(prev_soft, flow_bw)

            h, w = flow_heat.shape
            np.savez_compressed(
                os.path.join(out_dir, f"{t:05d}.npz"),
                flow_heat=flow_heat.astype(np.float16),
                shape=np.array([h, w], dtype=np.int32),
            )
            n_written += 1
        print(f"[{i}/{len(clips)}] {clip}: wrote {n_written} flow fields", flush=True)

    print(f"\nDone. Sidecar written under {args.dump}/flow/ -- "
          f"run refinelab/bench.py --refiners flow_snap,flow_guided_snap next.")


if __name__ == "__main__":
    main()
