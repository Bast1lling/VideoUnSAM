"""Precompute UNSUPERVISED GrabCut-refined targets for pseudo-masks -> parallel COCO json.

The refiner is OpenCV GrabCut: color-GMM + graph-cut seeded from the pseudo-mask, using
only low-level image cues. No labels, no ImageNet/pretrained weights -> the refinement
signal stays genuinely unsupervised, so a decoder trained on these targets keeps the
label-free guarantee (unlike supervised refiners such as CascadePSP, which are trained on
human masks).

Each output annotation keeps `segmentation` (raw pseudo = decoder INPUT) and adds a
`refined` field (GrabCut-cleaned = decoder TARGET). Identity fallback whenever GrabCut
degenerates, so every input keeps a 1:1 target.

    python -m video.decoder.make_grabcut_targets \
        --coco datasets/davis_train/annotations/train.json \
        --image-root datasets/davis_train/images \
        --out datasets/davis_train/annotations/train_grabcut.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def grabcut_one(image_bgr, m, pad=0.5, iters=5, core_k=5, bg_k=15, min_px=200):
    """GrabCut-refine one binary mask on a context crop; identity fallback on failure.

    Tiny masks (< min_px) are left untouched: GrabCut's color GMMs need enough pixels and
    otherwise collapse the mask to near-empty (a worse target than the raw pseudo-mask)."""
    H, W = m.shape
    ys, xs = np.where(m > 0)
    if len(ys) < min_px:
        return m
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int((x2 - x1) * pad) + 2, int((y2 - y1) * pad) + 2
    rx1, ry1 = max(0, x1 - px), max(0, y1 - py)
    rx2, ry2 = min(W, x2 + px), min(H, y2 + py)
    crop = np.ascontiguousarray(image_bgr[ry1:ry2, rx1:rx2])
    mc = m[ry1:ry2, rx1:rx2]
    if crop.size == 0 or mc.max() == 0:
        return m

    gc = np.full(mc.shape, cv2.GC_PR_BGD, np.uint8)
    gc[mc > 0] = cv2.GC_PR_FGD
    core = cv2.erode(mc, np.ones((core_k, core_k), np.uint8))      # definite-FG core
    gc[core > 0] = cv2.GC_FGD
    ring = cv2.dilate(mc, np.ones((bg_k, bg_k), np.uint8))         # definite-BG outside dilation
    gc[ring == 0] = cv2.GC_BGD
    # GrabCut needs both a FG and a BG seed; bail to identity if either is absent
    if not ((gc == cv2.GC_FGD).any() or (gc == cv2.GC_PR_FGD).any()):
        return m
    if not ((gc == cv2.GC_BGD).any() or (gc == cv2.GC_PR_BGD).any()):
        return m
    try:
        cv2.grabCut(crop, gc, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    rc = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    # reject collapse/explosion: a good refinement stays within 0.5x..2x the raw area
    raw_a, new_a = int(m.sum()), int(rc.sum())
    if new_a < 0.5 * raw_a or new_a > 2.0 * raw_a:
        return m
    out = np.zeros((H, W), np.uint8)
    out[ry1:ry2, rx1:rx2] = rc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="datasets/davis_train/annotations/train.json")
    ap.add_argument("--image-root", default="datasets/davis_train/images")
    ap.add_argument("--out", default="datasets/davis_train/annotations/train_grabcut.json")
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    d = json.load(open(args.coco))
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)

    ious, n = [], 0
    for im in d["images"]:
        bgr = cv2.imread(str(Path(args.image_root) / im["file_name"]))
        for a in by_img[im["id"]]:
            m = mask_util.decode(a["segmentation"])
            r = grabcut_one(bgr, m, pad=args.pad, iters=args.iters)
            enc = mask_util.encode(np.asfortranarray(r))
            enc["counts"] = enc["counts"].decode("ascii")
            a["refined"] = enc
            ious.append(_iou(m, r))
            n += 1
        print(f"[{im['file_name']}] {len(by_img[im['id']])} masks done", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(args.out, "w"))
    print(f"[done] refined {n} masks; mean IoU(raw,refined)={np.mean(ious):.3f}; wrote {args.out}")


if __name__ == "__main__":
    main()
