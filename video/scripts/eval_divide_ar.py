"""Average-Recall comparison of divide-stage mask generators on DAVIS.

Compares the training-free DINOv3 SpectralDivide against the frozen ResNet-50
CutLER detector, using DAVIS instance annotations as ground truth. DAVIS only
labels the few salient objects, so AR here measures whether the divide stage
recovers the objects we actually need as propagation seeds — not full SA-1B AR.

Metric (COCO-style mask AR, no score ranking — divide masks are unscored):
  for each GT instance, best IoU over all predicted masks;
  recall@t = fraction of GT with best-IoU >= t;  AR = mean recall over
  t in {0.50, 0.55, ..., 0.95}. We also report AR50, AR75 and mean #preds.

Usage:
    # spectral only (no detectron2 needed)
    python -m video.scripts.eval_divide_ar \\
        --clips blackswan,bmx-trees,dogs-jump --frames 0,15,30 --mode recursive

    # add the frozen CutLER baseline (needs detectron2 + CutLER weights)
    python -m video.scripts.eval_divide_ar --clips blackswan --frames 0 \\
        --cutler --cutler-config divide_and_conquer/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml \\
        --cutler-weights /path/to/cutler_cascade_final.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.divide.spectral_cut import SpectralDivide
from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis

_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def gt_instance_masks(clip: str, idx: int) -> list[np.ndarray]:
    return [davis.load_mask(clip, idx, instance_id=i) for i in davis.instance_ids(clip, idx)]


def recall_stats(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    """Per-frame best-IoU per GT -> recall curve."""
    if not gts:
        return {}
    best = []
    for g in gts:
        best.append(max((_iou(g, p) for p in preds), default=0.0))
    best = np.asarray(best)
    rec = {f"AR{int(t * 100)}": float((best >= t).mean()) for t in (0.5, 0.75)}
    rec["AR"] = float(np.mean([(best >= t).mean() for t in _THRESHOLDS]))
    rec["n_gt"] = len(gts)
    rec["n_pred"] = len(preds)
    rec["best_iou_mean"] = float(best.mean())
    return rec


def build_cutler(config: str, weights: str):
    """Best-effort frozen-CutLER predictor. Returns a callable(image_bgr)->[mask]."""
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

    def _predict(image_bgr: np.ndarray) -> list[np.ndarray]:
        inst = predictor(image_bgr)["instances"]
        m = inst.get("pred_masks").cpu().numpy()
        return [m[i].astype(np.uint8) for i in range(m.shape[0])]

    return _predict


def aggregate(rows: list[dict]) -> dict:
    rows = [r for r in rows if r]
    if not rows:
        return {}
    keys = ["AR", "AR50", "AR75", "best_iou_mean", "n_pred", "n_gt"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", default="blackswan,bmx-trees,dogs-jump")
    p.add_argument("--frames", default="0,15,30")
    p.add_argument("--mode", default="maskcut", choices=["maskcut", "spectral"])
    p.add_argument("--spatial-weight", type=float, default=0.1)
    p.add_argument("--no-attention", action="store_true",
                   help="Ablate the CLS-attention objectness prior (fall back to spectral seed).")
    p.add_argument("--cutler", action="store_true", help="Also evaluate frozen CutLER.")
    p.add_argument("--cutler-config", default="")
    p.add_argument("--cutler-weights", default="")
    args = p.parse_args()

    clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    frames = [int(f) for f in args.frames.split(",") if f.strip()]

    print(f"[load] DINOv3 + SpectralDivide(mode={args.mode})")
    extractor = DenseDINOv3(with_attention=True)
    spectral = SpectralDivide(
        extractor, mode=args.mode,
        spatial_weight=args.spatial_weight, use_attention=not args.no_attention,
    )

    cutler = None
    if args.cutler:
        if not (args.cutler_config and args.cutler_weights):
            print("[warn] --cutler needs --cutler-config and --cutler-weights; skipping CutLER.")
        else:
            try:
                cutler = build_cutler(args.cutler_config, args.cutler_weights)
                print("[load] frozen CutLER predictor ready")
            except Exception as e:  # noqa: BLE001 — environment-dependent; degrade gracefully
                print(f"[warn] could not build CutLER ({type(e).__name__}: {e}); spectral-only.")

    spec_rows: list[dict] = []
    cut_rows: list[dict] = []
    print(f"\n{'clip':<14} {'frame':>5} | {'method':<9} {'AR':>5} {'AR50':>5} {'AR75':>5} "
          f"{'bestIoU':>7} {'#pred':>5} {'#gt':>4}")
    print("-" * 78)
    for clip in clips:
        for f in frames:
            if f >= davis.num_frames(clip):
                continue
            rgb = davis.load_frame(clip, f)
            gts = gt_instance_masks(clip, f)

            s = recall_stats(spectral.predict(rgb), gts)
            spec_rows.append(s)
            print(f"{clip:<14} {f:>5} | {'spectral':<9} {s['AR']:>5.3f} {s['AR50']:>5.3f} "
                  f"{s['AR75']:>5.3f} {s['best_iou_mean']:>7.3f} {s['n_pred']:>5} {s['n_gt']:>4}")

            if cutler is not None:
                c = recall_stats(cutler(rgb[:, :, ::-1]), gts)   # CutLER wants BGR
                cut_rows.append(c)
                print(f"{'':<14} {'':>5} | {'cutler':<9} {c['AR']:>5.3f} {c['AR50']:>5.3f} "
                      f"{c['AR75']:>5.3f} {c['best_iou_mean']:>7.3f} {c['n_pred']:>5} {c['n_gt']:>4}")

    print("-" * 78)
    sa = aggregate(spec_rows)
    if sa:
        print(f"{'MEAN':<14} {'':>5} | {'spectral':<9} {sa['AR']:>5.3f} {sa['AR50']:>5.3f} "
              f"{sa['AR75']:>5.3f} {sa['best_iou_mean']:>7.3f} {sa['n_pred']:>5.1f} {sa['n_gt']:>4.1f}")
    ca = aggregate(cut_rows)
    if ca:
        print(f"{'MEAN':<14} {'':>5} | {'cutler':<9} {ca['AR']:>5.3f} {ca['AR50']:>5.3f} "
              f"{ca['AR75']:>5.3f} {ca['best_iou_mean']:>7.3f} {ca['n_pred']:>5.1f} {ca['n_gt']:>4.1f}")
    if sa and ca:
        print(f"\n[delta] spectral - cutler AR = {sa['AR'] - ca['AR']:+.3f}")


if __name__ == "__main__":
    main()
