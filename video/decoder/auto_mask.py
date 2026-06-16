"""Promptless automatic mask generation on ANY image — no GT, no manual box.

Pipeline (fully unsupervised, all pieces you already have):
    CuVLER proposes objects (boxes)  ->  trained decoder refines each into a clean mask.

This is the box-free inference: the prompts come from CuVLER, not a human/GT.

    python -m video.decoder.auto_mask --image path/to/any.jpg --out auto.png
    python -m video.decoder.auto_mask --clip camel --frame 0 --out auto.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from video.features.dinov3_dense import DenseDINOv3
from video.divide.cuvler_divide import CuVLERDivider
from video.scripts.visualize_divide import paint, label
from video.decoder.train_sam_decoder import UnsupervisedDecoder, _IMG


def box_from_mask(m, W, H):
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return None
    sx, sy = _IMG / W, _IMG / H
    return [xs.min() * sx, ys.min() * sy, xs.max() * sx, ys.max() * sy]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="any image path")
    ap.add_argument("--clip", default="camel")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/unsup_decoder_v2.pth")
    ap.add_argument("--cuvler-score", type=float, default=0.35)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="auto.png")
    args = ap.parse_args()

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"Could not read image: {args.image!r} — check the path "
                             f"(it must be a real .jpg/.png file, not the literal 'any.jpg').")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        name = Path(args.image).name
    else:
        from video.loaders import davis
        rgb = davis.load_frame(args.clip, args.frame)
        name = f"{args.clip}:{args.frame}"
    H, W = rgb.shape[:2]

    dino = DenseDINOv3()
    divider = CuVLERDivider(score_thresh=args.cuvler_score)
    model = UnsupervisedDecoder().cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])

    # 1) CuVLER proposes objects (no GT, no human box)
    proposals = divider.predict(rgb)
    boxes = [b for b in (box_from_mask(m, W, H) for m in proposals) if b is not None]
    print(f"{name}: CuVLER proposed {len(boxes)} objects")
    if not boxes:
        print("no proposals"); return

    # 2) decoder refines each proposal into a clean mask
    img1024 = cv2.resize(rgb, (_IMG, _IMG))
    with torch.no_grad():
        feats = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()
        low = model(feats, torch.tensor(boxes, dtype=torch.float, device="cuda"))
        up = F.interpolate(low, (H, W), mode="bilinear", align_corners=False).sigmoid()[:, 0].cpu().numpy()
    refined = [(p > args.thresh).astype(np.uint8) for p in up]

    panel = np.concatenate([
        label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), name),
        label(paint(rgb, proposals), f"CuVLER proposals (n={len(proposals)})"),
        label(paint(rgb, refined), f"decoder refined (n={len(refined)})"),
    ], axis=1)
    cv2.imwrite(args.out, panel)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
