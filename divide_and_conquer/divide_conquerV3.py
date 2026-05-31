"""
Divide-and-conquer pseudo-mask generation using DINOv3 for the conquer phase.

Key differences from divide_conquer.py:
  - DINOv3 (ViT-L/16) backbone instead of DINO ViT-B/8
  - Local crop size 512 instead of 256
  - Merging thetas: [0.75, 0.65, 0.55, 0.45, 0.35, 0.25]
"""

import argparse
import json
import os

import cv2
import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from tqdm import tqdm

import segmentation_refinement as refine

from coco_annotator import create_image_info, create_annotation_info, output, category_info
from iterative_merging import iterative_merge
from cascadepsp import postprocess
from engine.defaults import DefaultPredictor
import dinov3

from divide_conquer import (
    setup_cfg,
    NMS, coverage, resize_mask, smallest_square_containing_mask,
)

import warnings
warnings.filterwarnings("ignore")


ToTensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def generate_feature_matrix(backbone, image, feat_dim, feat_num):
    if next(backbone.parameters()).device == torch.device('cpu'):
        tensor = ToTensor(image).unsqueeze(0)
        feat = backbone(tensor)[0]
    else:
        tensor = ToTensor(image).unsqueeze(0).to(next(backbone.parameters()).dtype)
        tensor = tensor.cuda()
        feat = backbone(tensor)[0].cpu()
    feat_reshaped = feat.reshape(feat_dim, feat_num, feat_num).permute(1, 2, 0)
    return feat_reshaped


def get_parser():
    parser = argparse.ArgumentParser(description="Divide-and-conquer with DINOv3")
    parser.add_argument(
        "--config-file",
        default="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
        metavar="FILE",
    )
    # DINOv3 backbone
    parser.add_argument("--patch-size", default=16, type=int)
    parser.add_argument("--feature-dim", default=1024, type=int)
    parser.add_argument("--backbone-size", default="large", type=str)
    parser.add_argument("--backbone-url", default="facebook/dinov3-vitl16-pretrain-lvd1689m", type=str)
    # I/O
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="pseudo_masks_output_v3")
    parser.add_argument("--preprocess", default=None, type=bool)
    parser.add_argument("--postprocess", default=None, type=bool)
    # Preprocess / conquer params
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--start-id", default=None, type=int)
    parser.add_argument("--end-id", default=None, type=int)
    parser.add_argument("--local-size", default=512, type=int)
    parser.add_argument("--kept-thresh", default=0.9, type=float)
    parser.add_argument("--NMS-iou", default=0.9, type=float)
    parser.add_argument("--NMS-step", default=5, type=int)
    parser.add_argument("--thetas", default=[0.75, 0.65, 0.55, 0.45, 0.35, 0.25], type=list)
    # Postprocess params
    parser.add_argument("--crop-ratio", default=2.0, type=float)
    parser.add_argument("--refine-scale", default=1, type=int)
    parser.add_argument("--refine-min-L", default=100, type=int)
    parser.add_argument("--refine-max-L", default=900, type=int)
    parser.add_argument("--iou-thresh", default=0.5, type=float)
    parser.add_argument("--min-area-thresh", default=0.0, type=float)
    parser.add_argument("--max-area-thresh", default=0.9, type=float)
    parser.add_argument("--cover-thresh", default=0.9, type=float)
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser


def run_conquer(backbone, image, divide_masks, args):
    """
    Run the conquer phase for all divide masks of one image.

    Returns:
        divide_conquer_masks: flat list — divide masks followed by all conquer masks.
        conquer_per_divide:   dict {divide_idx: [conquer_mask, ...]} for hierarchy tracking.
    """
    divide_conquer_masks = list(divide_masks)
    conquer_per_divide = {}

    for idx, divide_mask in enumerate(divide_masks):
        ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            continue

        local_image = image[ymin:ymax, xmin:xmax]
        resized_local_image = Image.fromarray(local_image).resize(
            [args.local_size, args.local_size]
        )
        feature_matrix = generate_feature_matrix(
            backbone, resized_local_image, args.feature_dim, args.local_size // args.patch_size
        )
        merging_masks = iterative_merge(feature_matrix, args.thetas)

        conquer_masks = []
        for layer in merging_masks:
            if layer.shape[0] == 0:
                continue
            for i in range(layer.shape[0]):
                mask = layer[i]
                mask = resize_mask(mask, [xmax - xmin, ymax - ymin])
                mask = (mask > 0.5 * 255).astype(int)
                if coverage(mask, divide_mask[ymin:ymax, xmin:xmax]) <= args.kept_thresh:
                    continue
                enlarged = np.zeros_like(divide_mask)
                enlarged[ymin:ymax, xmin:xmax] = mask
                conquer_masks.append(enlarged)

        conquer_masks = NMS(conquer_masks, args.NMS_iou, args.NMS_step)
        if conquer_masks:
            conquer_per_divide[idx] = conquer_masks
        divide_conquer_masks.extend(conquer_masks)

    return divide_conquer_masks, conquer_per_divide


def main():
    args = get_parser().parse_args()
    print(args)
    refiner = refine.Refiner(device="cuda:0")

    if args.preprocess:
        if not args.start_id:
            args.start_id = 0
        if not args.end_id:
            args.end_id = len(os.listdir(args.input_dir))
        os.makedirs(args.output_dir, exist_ok=True)

        cfg = setup_cfg(args)
        predictor = DefaultPredictor(cfg)

        backbone = dinov3.ViTFeatV3(
            args.backbone_url, args.feature_dim, args.backbone_size, "k", args.patch_size
        )
        backbone.eval()
        if cfg.MODEL.DEVICE == "cpu":
            backbone.to("cpu")
        else:
            backbone.cuda()

        segmentation_id = 1
        cnt = 0

        for entry in tqdm(os.scandir(args.input_dir)):
            image_name = entry.name
            cnt += 1
            if cnt < args.start_id or cnt >= args.end_id:
                continue

            output["image"], output["annotations"] = {}, []

            image_id = int(image_name.replace(".jpg", "").replace("sa_", ""))
            save_path = os.path.join(args.output_dir, image_name.replace(".jpg", ".json"))
            assert not os.path.exists(save_path), f"annotation already exists: {save_path}"

            image = cv2.imread(os.path.join(args.input_dir, image_name))
            H, W = image.shape[:2]

            predictions = predictor(image)
            divide_masks_tensor = predictions["instances"].get("pred_masks")
            divide_masks = [divide_masks_tensor[i].cpu().numpy() for i in range(divide_masks_tensor.shape[0])]

            divide_conquer_masks, _ = run_conquer(backbone, image, divide_masks, args)

            output["image"] = create_image_info(image_id, image_name, (H, W, 3))
            for m in divide_conquer_masks:
                ann = create_annotation_info(segmentation_id, image_id, category_info, m.astype(np.uint8), None)
                if ann is not None:
                    output["annotations"].append(ann)
                    segmentation_id += 1

            with open(save_path, "w") as f:
                json.dump(output, f, indent=2)

    if args.postprocess:
        if not args.start_id:
            args.start_id = 0
        if not args.end_id:
            args.end_id = len(os.listdir(args.output_dir))
        for ann_name in tqdm(os.listdir(args.output_dir)[args.start_id:args.end_id]):
            annotation_path = os.path.join(args.output_dir, ann_name)
            annotations = json.load(open(annotation_path))
            image_path = os.path.join(args.input_dir, ann_name.replace("json", "jpg"))
            image = cv2.imread(image_path)
            refined = postprocess(args, refiner, annotations, image)
            out_path = os.path.join(args.output_dir, f"p_{ann_name}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(refined, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
