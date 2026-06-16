"""Show the FULL divide-and-conquer output for CuVLER: GT | divide | divide+conquer.

The 'divide' panel is CuVLER's coarse instance masks (one mask per 'whole thing').
The 'divide+conquer' panel adds the DINOv3 conquer-stage parts — the hierarchy
that splits each coarse mask into finer parts.

Self-contained: imports only the clean iterative_merge (torch/numpy) and the
DINOv3 backbone, reimplementing the small conquer helpers so it avoids the heavy
divide_conquerV3 import chain (segmentation_refinement/coco_annotator -> pandas).

    python -m video.scripts.viz_cuvler_hierarchy --clips blackswan,bmx-trees,dogs-jump --out cuvler_hier.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import PIL.Image as Image
from torchvision import transforms

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))

from video.divide.cuvler_divide import CuVLERDivider
from video.loaders import davis
from video.scripts.visualize_divide import paint, label, gt_masks
import dinov3  # noqa: E402
from iterative_merging import iterative_merge  # noqa: E402 — clean (torch/numpy only)

_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def bbox(mask):
    rows, cols = np.any(mask, 1), np.any(mask, 0)
    if not rows.any() or not cols.any():
        return 0, 1, 0, 1
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return ymin, ymax, xmin, xmax


def coverage(m1, m2):
    n = np.count_nonzero(m1)
    return np.count_nonzero(m1 & m2) / n if n else 0.0


def resize_mask(mask, wh):
    m = np.asarray(Image.fromarray(np.uint8(mask * 255)).resize(wh)).astype(np.uint8)
    hi, lo = m.max(), m.min()
    m[m > hi / 2.0] = hi
    m[m <= hi / 2.0] = lo
    return m


def gen_feat(backbone, pil_img, feat_dim, feat_num):
    t = _TO_TENSOR(pil_img).unsqueeze(0).half().cuda()
    feat = backbone(t)[0].cpu()
    return feat.reshape(feat_dim, feat_num, feat_num).permute(1, 2, 0)


def run_conquer(backbone, image_rgb, divide_masks, local=512, patch=16, fdim=1024,
                thetas=(0.73, 0.62, 0.51, 0.4, 0.3, 0.2), kept=0.8):
    out = list(divide_masks)
    for dm in divide_masks:
        ymin, ymax, xmin, xmax = bbox(dm)
        if ymax - ymin <= 0 or xmax - xmin <= 0:
            continue
        crop = image_rgb[ymin:ymax, xmin:xmax]
        fm = gen_feat(backbone, Image.fromarray(crop).resize([local, local]), fdim, local // patch)
        for layer in iterative_merge(fm, list(thetas)):
            if layer.shape[0] == 0:
                continue
            for i in range(layer.shape[0]):
                m = (resize_mask(layer[i], [xmax - xmin, ymax - ymin]) > 127).astype(np.uint8)
                if coverage(m, dm[ymin:ymax, xmin:xmax].astype(bool)) <= kept:
                    continue
                enl = np.zeros_like(dm, dtype=np.uint8)
                enl[ymin:ymax, xmin:xmax] = m
                out.append(enl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="blackswan,bmx-trees,dogs-jump")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--out", default="cuvler_hier.png")
    args = ap.parse_args()

    backbone = dinov3.ViTFeatV3(patch_size=16, feat_dim=1024, vit_arch="large", vit_feat="k")
    backbone.eval(); backbone.cuda()
    div = CuVLERDivider(score_thresh=args.score)

    rows = []
    for clip in [c.strip() for c in args.clips.split(",")]:
        rgb = davis.load_frame(clip, args.frame)
        divide = div.predict(rgb)
        dc_bgr = run_conquer(backbone, rgb[:, :, ::-1], divide)   # old bug: conquer fed BGR
        dc_rgb = run_conquer(backbone, rgb, divide)               # fixed: conquer fed RGB
        rows.append(np.concatenate([
            label(paint(rgb, gt_masks(clip, args.frame)), f"{clip}:{args.frame} GT"),
            label(paint(rgb, divide), f"divide (n={len(divide)})"),
            label(paint(rgb, dc_bgr), f"conquer BGR/old (n={len(dc_bgr)})"),
            label(paint(rgb, dc_rgb), f"conquer RGB/fixed (n={len(dc_rgb)})"),
        ], axis=1))
    mh = min(r.shape[0] for r in rows)
    rows = [cv2.resize(r, (int(r.shape[1] * mh / r.shape[0]), mh)) for r in rows]
    cv2.imwrite(args.out, np.concatenate(rows, axis=0))
    print("saved", args.out)


if __name__ == "__main__":
    main()
