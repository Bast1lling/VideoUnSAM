"""Render whole-image (automatic, no-click, multi-object) segmentation as a
side-by-side video: predicted instances (left) vs DAVIS ground truth (right).

Uses the default score_thresh=0.35 (the same setting the click pipeline uses),
not the lower experimental threshold from the eval sweep — this shows what the
system does out of the box, not the best-case recall/over-generation tradeoff.

Usage:
    python -m video.scripts.render_whole_image --clip dogs-jump --frames 40 --out video/outputs/whole_image_dogs_jump.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "divide_and_conquer"))

from video.divide.cuvler_divide import CuVLERDivider  # noqa: E402
from video.divide.conquer import load_backbone, run_conquer_one_per_object  # noqa: E402
from video.divide.consolidate import consolidate  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=30, help="Number of frames to render (from frame 0)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--divide-nms", type=float, default=0.0,
                    help="If >0, NMS the divide masks at this IoU threshold before "
                         "one-per-object (use with a low --score to collapse duplicate boxes).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    div = CuVLERDivider(score_thresh=args.score)
    backbone = load_backbone()

    n = min(args.frames, davis.num_frames(args.clip))
    tmp = tempfile.NamedTemporaryFile(suffix="_raw.mp4", delete=False)
    tmp.close()
    writer = None

    for f in range(0, n, args.stride):
        frame = davis.load_frame(args.clip, f)
        H, W = frame.shape[:2]

        divide_masks, divide_scores = div.predict_scored(frame)
        if args.divide_nms > 0:
            divide_masks, divide_scores = consolidate(divide_masks, divide_scores, iou_thresh=args.divide_nms)
        pred_masks, _ = run_conquer_one_per_object(backbone, frame, divide_masks, divide_scores)

        gt_ids = davis.instance_ids(args.clip, f)
        gt_masks = [davis.load_mask(args.clip, f, instance_id=i) for i in gt_ids]

        left = _overlay_multi(frame, pred_masks, f"predicted ({len(pred_masks)})")
        right = _overlay_multi(frame, gt_masks, f"ground truth ({len(gt_masks)})")
        combo = np.concatenate([left, right], axis=1)

        if writer is None:
            writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 8, (combo.shape[1], combo.shape[0]))
        writer.write(cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
        print(f"  f{f:>4}: predicted {len(pred_masks)} masks, GT {len(gt_masks)} objects")

    writer.release()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", tmp.name, "-vcodec", "libx264", "-pix_fmt", "yuv420p", args.out],
                  capture_output=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
