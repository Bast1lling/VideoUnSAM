"""Render CuVLER divide masks on DAVIS clips -> montage PNG (GT | CuVLER per clip).

    python -m video.scripts.viz_cuvler --clips blackswan,bmx-trees,dogs-jump --out cuvler_try.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.divide.cuvler_divide import CuVLERDivider
from video.loaders import davis
from video.scripts.visualize_divide import paint, label, gt_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="blackswan,bmx-trees,dogs-jump")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--out", default="cuvler_try.png")
    args = ap.parse_args()

    div = CuVLERDivider(score_thresh=args.score)
    rows = []
    for clip in [c.strip() for c in args.clips.split(",")]:
        rgb = davis.load_frame(clip, args.frame)
        masks = div.predict(rgb)
        rows.append(np.concatenate([
            label(paint(rgb, gt_masks(clip, args.frame)), f"{clip}:{args.frame} GT"),
            label(paint(rgb, masks), f"CuVLER (n={len(masks)})"),
        ], axis=1))
    mh = min(r.shape[0] for r in rows)
    rows = [cv2.resize(r, (int(r.shape[1] * mh / r.shape[0]), mh)) for r in rows]
    cv2.imwrite(args.out, np.concatenate(rows, axis=0))
    print("saved", args.out)


if __name__ == "__main__":
    main()
