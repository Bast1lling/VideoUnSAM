"""
End-to-end pipeline: Divide → Conquer (DINOv3) → Synthetic video.

Given a single input image:
  1. Divide  — CutLER Cascade Mask R-CNN extracts coarse instance masks.
  2. Conquer — DINOv3 (ViT-L/16, local_size=512) hierarchically refines each
               divide mask using thetas [0.75, 0.65, 0.55, 0.45, 0.35, 0.25].
  3. Video   — All masks (divide + conquer sub-masks) are augmented together
               per frame (RandomCrop → ResizeShortestEdge[320-480] → RandomFlip
               → brightness / contrast / saturation jitter [0.9, 1.1]), producing
               a synthetic multi-frame clip with consistent mask IDs across frames.

Usage:
  python divide_conquer_videoV3.py \\
      --input docs/demos/sa_234337.jpg \\
      --output-video output.mp4 \\
      --output-preview preview.png \\
      --num-frames 8 --fps 4 --crop-min 0.5 --flip-prob 0.5
"""

import argparse
import os
import shutil

import cv2
import numpy as np

from engine.defaults import DefaultPredictor
import dinov3

from divide_conquer import setup_cfg
from divide_conquerV3 import run_conquer
from synthetic_video import run_synth_video

import warnings
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Divide → Conquer (DINOv3) → Synthetic video (single image)"
    )
    # CutLER model
    parser.add_argument(
        "--config-file",
        default="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
        metavar="FILE",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    # DINOv3 backbone
    parser.add_argument("--patch-size", default=16, type=int)
    parser.add_argument("--feature-dim", default=1024, type=int)
    parser.add_argument("--backbone-size", default="large", type=str)
    parser.add_argument("--backbone-url", default="facebook/dinov3-vitl16-pretrain-lvd1689m", type=str)
    # Conquer params
    parser.add_argument("--local-size", default=512, type=int)
    parser.add_argument("--kept-thresh", default=0.9, type=float)
    parser.add_argument("--NMS-iou", default=0.9, type=float)
    parser.add_argument("--NMS-step", default=5, type=int)
    parser.add_argument("--thetas", default=[0.75, 0.65, 0.55, 0.45, 0.35, 0.25], type=list)
    # I/O
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-video", type=str, default="output_video.mp4")
    parser.add_argument("--output-preview", type=str, default=None,
                        help="Path to save a preview PNG showing all masks on frame 0")
    # Video generation params
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Number of synthetic frames to generate")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--crop-min", type=float, default=0.5,
                        help='Min relative crop fraction for T.RandomCrop("relative_range")')
    parser.add_argument("--flip-prob", type=float, default=0.5,
                        help="Horizontal flip probability per frame")
    # detectron2 extras
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser


def main():
    args = get_parser().parse_args()

    # ── 1. Load image ────────────────────────────────────────────────────────
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_bgr.shape[:2]
    print(f"Image: {args.input}  ({W}×{H})")

    # ── 2. Divide — CutLER Cascade Mask R-CNN ───────────────────────────────
    print("Running divide phase...")
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    predictions = predictor(image_bgr)
    divide_masks_tensor = predictions["instances"].get("pred_masks")
    divide_masks = [
        divide_masks_tensor[i].cpu().numpy()
        for i in range(divide_masks_tensor.shape[0])
    ]
    print(f"  → {len(divide_masks)} divide masks")

    # ── 3. Conquer — DINOv3 hierarchical refinement ─────────────────────────
    print("Loading DINOv3 backbone...")
    backbone = dinov3.ViTFeatV3(
        args.backbone_url, args.feature_dim, args.backbone_size, "k", args.patch_size
    )
    backbone.eval()
    if cfg.MODEL.DEVICE == "cpu":
        backbone.to("cpu")
    else:
        backbone.cuda()

    print("Running conquer phase...")
    _, conquer_per_divide = run_conquer(backbone, image_bgr, divide_masks, args)

    n_conquer = sum(len(v) for v in conquer_per_divide.values())
    n_divide_with_conquer = len(conquer_per_divide)
    print(f"  → {n_conquer} conquer masks across {n_divide_with_conquer} divide region(s)")

    # ── 4. Synthetic video generation ────────────────────────────────────────
    print(f"Generating {args.num_frames} synthetic frames...")
    gallery, tmp_path, desc = run_synth_video(
        image_rgb,
        divide_masks,
        conquer_per_divide,
        num_frames=args.num_frames,
        crop_min=args.crop_min,
        flip_prob=args.flip_prob,
        fps=args.fps,
    )
    print(f"  {desc}")

    # Move temp MP4 to requested output path
    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    shutil.move(tmp_path, args.output_video)
    print(f"Video saved → {args.output_video}")

    # ── 5. Optional preview frame ─────────────────────────────────────────────
    if args.output_preview and gallery:
        preview_rgb = gallery[0][0]
        preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_preview, preview_bgr)
        print(f"Preview saved → {args.output_preview}")


if __name__ == "__main__":
    main()
