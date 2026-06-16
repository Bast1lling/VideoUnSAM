"""CuVLER divide-stage detector — fully-unsupervised drop-in upgrade over frozen CutLER.

CuVLER (Cut-Vote-and-LeaRn, CVPR 2024) is CutLER's unsupervised successor: a
class-agnostic Cascade Mask R-CNN trained only on VoteCut pseudo-masks (NCut +
voting over self-supervised ViTs). No human labels — same SSL status as CutLER.

Measured on DAVIS (blackswan/bmx-trees/dogs-jump, frames 0/15/30), COCO mask AR:
    spectral+attn 0.00 | frozen CutLER 0.37 | CuVLER 0.49 (AR75 0.30 -> 0.56).

Vendored detector code lives in video/cuvler/cad/ (CutLER fork). We import only
cad.modeling (no cad.engine/cad.data) so it does NOT clash with the local
divide_and_conquer.modeling registry — the two can coexist in one process as
long as divide_and_conquer's `modeling` is not also imported.

Output matches the CutLER predictor consumed by run_conquer(): a list of
full-frame binary masks [H, W] uint8.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import PIL.Image as Image

_REPO = Path(__file__).resolve().parents[2]
_CUVLER_SRC = _REPO / "video" / "cuvler"          # contains the `cad` package
_CONFIG = _CUVLER_SRC / "cad" / "model_zoo" / "configs" / "CutVER-ImageNet" / \
    "cascade_mask_rcnn_R_50_FPN_votecut_cad.yaml"
_WEIGHTS = _REPO / "divide_and_conquer" / "model_zoo" / "ckpts" / "cuvler_self_trained.pth"


def _pillow_compat():
    # Pillow>=10 removed these aliases that CuVLER's old detectron2 fork uses.
    for old, new in [("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC")]:
        if not hasattr(Image, old):
            setattr(Image, old, getattr(Image.Resampling, new))


class CuVLERDivider:
    """Build once, call .predict(frame_rgb) per image."""

    def __init__(self, weights: str | Path = _WEIGHTS, config: str | Path = _CONFIG,
                 score_thresh: float = 0.35, device: str = "cuda"):
        _pillow_compat()
        if str(_CUVLER_SRC) not in sys.path:
            sys.path.insert(0, str(_CUVLER_SRC))   # makes `cad` importable

        import torch
        from detectron2.config import get_cfg
        from detectron2.checkpoint import DetectionCheckpointer
        import detectron2.data.transforms as T
        import cad.modeling  # noqa: F401 — registers CustomCascadeROIHeads / mask head
        from cad.modeling.meta_arch.build import build_model

        self._torch = torch

        # add_cuvler_config from file (avoids cad/__init__ import chain)
        spec = importlib.util.spec_from_file_location(
            "cuvler_cfg", str(_CUVLER_SRC / "cad" / "config" / "config.py"))
        cfgmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfgmod)

        cfg = get_cfg()
        cfgmod.add_cuvler_config(cfg)
        cfg.merge_from_file(str(config))
        cfg.MODEL.WEIGHTS = str(weights)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.RESNETS.NORM = "BN"     # single-GPU eval: SyncBN running-stats -> BN
        cfg.MODEL.FPN.NORM = "BN"
        cfg.MODEL.DEVICE = device

        self.model = build_model(cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.rgb_input = cfg.INPUT.FORMAT == "RGB"
        print(f"[CuVLERDivider] loaded {Path(weights).name}  score_thresh={score_thresh}")

    def predict(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """RGB uint8 [H, W, 3] -> list of binary masks [H, W] uint8 (full frame)."""
        torch = self._torch
        with torch.no_grad():
            im = frame_rgb[:, :, ::-1] if self.rgb_input else frame_rgb
            h, w = im.shape[:2]
            t = self.aug.get_transform(im).apply_image(im)
            t = torch.as_tensor(t.astype("float32").transpose(2, 0, 1))
            inst = self.model([{"image": t, "height": h, "width": w}])[0]["instances"]
            m = inst.get("pred_masks").cpu().numpy().astype(np.uint8)
        return [m[i] for i in range(m.shape[0])]


if __name__ == "__main__":
    sys.path.insert(0, str(_REPO))
    from video.loaders import davis
    clip = sys.argv[1] if len(sys.argv) > 1 else "blackswan"
    div = CuVLERDivider()
    masks = div.predict(davis.load_frame(clip, 0))
    print(f"{clip}: {len(masks)} CuVLER divide masks "
          f"(areas: {sorted(round(float(m.mean()), 3) for m in masks)})")
