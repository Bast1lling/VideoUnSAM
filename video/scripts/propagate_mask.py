"""Phase-2 visual gate: OT-propagate a mask from frame A to frame B on DAVIS.

We use the DAVIS GT mask of frame A as the *source* mask. This stands in for
the CutLER+conquer mask that would come from Phase 1 in production. The frame
B GT mask is used only for evaluation (IoU), never as input.

Usage:
    python -m video.scripts.propagate_mask \\
        --clip blackswan --frame-a 0 --frame-b 20 \\
        --instance-id 1 --out prop_blackswan_0_20.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis
from video.propagation.sinkhorn_ot import propagate


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha=0.55) -> np.ndarray:
    out = rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def overlay_heat(rgb: np.ndarray, heat: np.ndarray, alpha=0.55) -> np.ndarray:
    h = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    cm = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + cm * alpha).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="blackswan")
    p.add_argument("--frame-a", type=int, default=0)
    p.add_argument("--frame-b", type=int, default=20)
    p.add_argument("--instance-id", type=int, default=1)
    p.add_argument("--blur", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarisation cutoff as fraction of heatmap max")
    p.add_argument("--upscale", type=float, default=1.0,
                   help="Resize frames by this factor before DINOv3 extraction. "
                        "Doubling roughly doubles the feature grid resolution at "
                        "~4x VRAM/compute. Mask + eval still happen at native size.")
    p.add_argument("--sam", action="store_true",
                   help="Diagnostic: also run SAM ViT-H refinement on the OT mask "
                        "and report both IoUs. Supervised prior — not used in the "
                        "pseudo-mask pipeline downstream.")
    p.add_argument("--out", default="propagate.png")
    args = p.parse_args()

    print(f"[load] {args.clip}  A=frame {args.frame_a}  B=frame {args.frame_b}  inst={args.instance_id}")
    frame_a = davis.load_frame(args.clip, args.frame_a)
    frame_b = davis.load_frame(args.clip, args.frame_b)
    mask_a = davis.load_mask(args.clip, args.frame_a, instance_id=args.instance_id)
    mask_b_gt = davis.load_mask(args.clip, args.frame_b, instance_id=args.instance_id)
    if mask_a.sum() == 0:
        raise SystemExit(f"Instance {args.instance_id} absent in frame {args.frame_a}")

    print("[model] DINOv3 ViT-L/16 bf16")
    extractor = DenseDINOv3()

    def _upscale(img):
        if args.upscale == 1.0:
            return img
        H, W = img.shape[:2]
        return cv2.resize(img, (int(round(W * args.upscale)), int(round(H * args.upscale))),
                          interpolation=cv2.INTER_LINEAR)

    frame_a_in = _upscale(frame_a)
    frame_b_in = _upscale(frame_b)
    mask_a_in = (cv2.resize(
        mask_a, (frame_a_in.shape[1], frame_a_in.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) if args.upscale != 1.0 else mask_a)

    out_a = extractor.extract(frame_a_in)
    out_b = extractor.extract(frame_b_in)
    print(f"[feats] A grid {out_a['grid_h']}x{out_a['grid_w']}  "
          f"B grid {out_b['grid_h']}x{out_b['grid_w']}  upscale={args.upscale}")

    print(f"[OT  ] sinkhorn blur={args.blur}")
    result = propagate(
        out_a["feats"], out_b["feats"], mask_a_in,
        out_size=(frame_b.shape[0], frame_b.shape[1]),
        blur=args.blur, threshold=args.threshold,
    )
    pred = result["mask"]
    heat = result["heatmap"]
    score_ot = iou(pred, mask_b_gt)
    print(f"[eval] OT  IoU = {score_ot:.3f}")

    sam_pred = None
    score_sam = None
    if args.sam:
        from video.refine.sam_refiner import SAMRefiner
        print("[sam ] loading SAM ViT-H")
        refiner = SAMRefiner()
        sam_out = refiner.refine(frame_b, pred, heat)
        sam_pred = sam_out["mask"]
        score_sam = iou(sam_pred, mask_b_gt)
        print(f"[eval] SAM IoU = {score_sam:.3f}")

    a_vis = overlay_mask(frame_a, mask_a, color=(64, 255, 64))
    b_heat = overlay_heat(frame_b, heat)
    b_pred = overlay_mask(frame_b, pred, color=(255, 64, 64))
    b_gt = overlay_mask(frame_b, mask_b_gt, color=(64, 255, 64))

    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out

    panels = [
        label(a_vis, f"A:{args.frame_a} src"),
        label(b_heat, f"B:{args.frame_b} OT heat"),
        label(b_pred, f"OT IoU {score_ot:.2f}"),
    ]
    if sam_pred is not None:
        b_sam = overlay_mask(frame_b, sam_pred, color=(64, 200, 255))
        panels.append(label(b_sam, f"SAM IoU {score_sam:.2f}"))
    panels.append(label(b_gt, "GT"))
    row = np.concatenate(panels, axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
