"""Side-by-side visual comparison of divide-stage mask generators.

Renders one row per frame: original | GT instances | maskcut | spectral
(+ CutLER if --cutler given). Each predicted instance gets its own colour.

Usage:
    python -m video.scripts.visualize_divide --clip blackswan --frame 0 --out cmp.png
    python -m video.scripts.visualize_divide --clip dogs-jump --frames 0,30 --out cmp_dogs.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.divide.spectral_cut import SpectralDivide
from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis

# distinct BGR colours for instances
_COLORS = [
    (66, 66, 244), (244, 66, 66), (66, 244, 244), (244, 66, 244),
    (244, 244, 66), (66, 244, 66), (130, 66, 244), (244, 160, 66),
    (66, 130, 130), (200, 120, 200), (120, 200, 120), (60, 180, 250),
]


def paint(rgb: np.ndarray, masks: list[np.ndarray], alpha: float = 0.55) -> np.ndarray:
    """Overlay each mask in a distinct colour on a BGR copy of the frame."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for i, m in enumerate(masks):
        col = np.array(_COLORS[i % len(_COLORS)], dtype=np.float32)
        fg = m.astype(bool)
        out[fg] = (out[fg] * (1 - alpha) + col * alpha).astype(np.uint8)
        # outline for clarity
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, tuple(int(c) for c in col), 2)
    return out


def gt_masks(clip: str, idx: int) -> list[np.ndarray]:
    return [davis.load_mask(clip, idx, instance_id=i) for i in davis.instance_ids(clip, idx)]


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def build_cutler(config: str, weights: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "divide_and_conquer"))
    import modeling  # noqa: F401  — registers CustomCascadeROIHeads in the registry
    from detectron2.config import get_cfg
    from engine.defaults import DefaultPredictor  # CutLER's predictor (matches the repo)
    from divide_conquer import add_cutler_config  # type: ignore

    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35
    predictor = DefaultPredictor(cfg)

    def _predict(rgb: np.ndarray) -> list[np.ndarray]:
        inst = predictor(rgb[:, :, ::-1])["instances"]
        m = inst.get("pred_masks").cpu().numpy()
        return [m[i].astype(np.uint8) for i in range(m.shape[0])]

    return _predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="blackswan")
    p.add_argument("--frame", type=int, default=None, help="single frame (shortcut for --frames)")
    p.add_argument("--frames", default=None, help="comma-separated frame indices")
    p.add_argument("--cutler", action="store_true")
    p.add_argument("--cutler-config", default="")
    p.add_argument("--cutler-weights", default="")
    p.add_argument("--out", default="divide_compare.png")
    args = p.parse_args()

    if args.frames:
        frames = [int(f) for f in args.frames.split(",") if f.strip()]
    elif args.frame is not None:
        frames = [args.frame]
    else:
        frames = [0]

    print("[load] DINOv3")
    ext = DenseDINOv3(with_attention=True)
    methods = [
        ("maskcut", SpectralDivide(ext, mode="maskcut").predict),
        ("spectral", SpectralDivide(ext, mode="spectral").predict),
    ]
    if args.cutler and args.cutler_config and args.cutler_weights:
        try:
            methods.append(("cutler", build_cutler(args.cutler_config, args.cutler_weights)))
            print("[load] CutLER ready")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] CutLER unavailable ({type(e).__name__}: {e})")

    rows = []
    for f in frames:
        rgb = davis.load_frame(args.clip, f)
        panels = [
            label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), f"{args.clip}:{f}"),
            label(paint(rgb, gt_masks(args.clip, f)), "GT"),
        ]
        for name, fn in methods:
            masks = fn(rgb)
            panels.append(label(paint(rgb, masks), f"{name} (n={len(masks)})"))
        rows.append(np.concatenate(panels, axis=1))

    montage = np.concatenate(rows, axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, montage)
    print(f"[save] {args.out}  ({montage.shape[1]}x{montage.shape[0]})")


if __name__ == "__main__":
    main()
