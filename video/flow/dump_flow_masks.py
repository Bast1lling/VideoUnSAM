"""Precompute SMURF-flow-warped masks as a refinelab sidecar. Plain PyTorch --
no separate TF venv needed (see video/flow/smurf_infer.py for why: the
ytvis_ft15k.pt checkpoint is loadable directly by torchvision's own RAFT
machinery, via the vendored video/flow/smurf_raft/).

Reads an EXISTING dump written by video/scripts/dump_artifacts.py, and for
every frame t >= 1 warps frame (t-1)'s `ot_mask` forward into frame t using
SMURF backward flow. Writes dump/flow/<clip>/<fidx>.npz, bit-packed like the
rest of the dump, so refinelab/bench.py can load it with plain numpy.

This deliberately warps the CHAIN's own ot_mask (whatever OT actually
produced, right or wrong), not the ground truth -- that's what a real
flow_snap refiner would have available at inference time. It is therefore NOT
the same experiment as eval_flow_warp_davis.py (GT-to-GT, oracle-initialised,
single hop) or eval_flow_propagation_chain.py (GT-seeded, chained, no
re-anchor, measuring drift): this one measures flow quality warping the
pipeline's own possibly-already-wrong previous mask, one hop, display-only.

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
from video.flow.smurf_infer import load_smurf, compute_backward_flow, warp_mask_backward


def _unpack(bits: np.ndarray, shape) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    return np.unpackbits(bits)[: h * w].reshape(h, w).astype(np.uint8)


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


def load_ot_mask(dump: str, clip: str, fidx: int) -> np.ndarray:
    z = np.load(os.path.join(dump, "npz", clip, f"{fidx:05d}.npz"))
    return _unpack(z["ot_mask"], z["shape"])


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
            prev_mask = load_ot_mask(args.dump, clip, t - 1)
            frame_prev = load_frame_rgb(args.dump, clip, t - 1)
            frame_cur = load_frame_rgb(args.dump, clip, t)

            flow_bw = compute_backward_flow(model, frame_cur, frame_prev, device=args.device)
            flow_mask = warp_mask_backward(prev_mask, flow_bw)

            h, w = flow_mask.shape
            np.savez_compressed(
                os.path.join(out_dir, f"{t:05d}.npz"),
                flow_mask=np.packbits(flow_mask.astype(bool)),
                shape=np.array([h, w], dtype=np.int32),
            )
            n_written += 1
        print(f"[{i}/{len(clips)}] {clip}: wrote {n_written} flow masks", flush=True)

    print(f"\nDone. Sidecar written under {args.dump}/flow/ -- "
          f"run refinelab/bench.py --refiners flow_snap,flow_guided_snap next.")


if __name__ == "__main__":
    main()
