"""
CutLER predictor: configuration setup and a minimal inference wrapper.

No dependency on the old divide_and_conquer directory.
The custom model heads are registered by importing the local .modeling package.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent   # UnSAM/
_OLD_DAC = _REPO / "divide_and_conquer"

# Pre-register UnSAM/divide_and_conquer/ as the 'divide_and_conquer' package
# before the import below. Without this, Python finds the local
# promptable_video_segmentation/divide_and_conquer/ (which has __init__.py and
# therefore wins over the root directory's namespace package), triggering a
# circular import and a "relative import beyond top-level package" error.
if "divide_and_conquer" not in sys.modules:
    _dac_pkg = types.ModuleType("divide_and_conquer")
    _dac_pkg.__path__ = [str(_OLD_DAC)]
    _dac_pkg.__package__ = "divide_and_conquer"
    sys.modules["divide_and_conquer"] = _dac_pkg

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T

from divide_and_conquer.modeling import build_model  # noqa: F401

DEFAULT_CUTLER_CONFIG  = str(
    _OLD_DAC / "model_zoo" / "configs"
    / "CutLER-ImageNet" / "cascade_mask_rcnn_R_50_FPN.yaml"
)
DEFAULT_CUTLER_WEIGHTS = str(
    _OLD_DAC / "model_zoo" / "ckpts" / "cutler_cascade_final.pth"
)


# ── config helpers ────────────────────────────────────────────────────────────

def _add_cutler_config(cfg) -> None:
    """Register CutLER-specific config keys with their defaults."""
    cfg.DATALOADER.COPY_PASTE               = False
    cfg.DATALOADER.COPY_PASTE_RATE          = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO     = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO     = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM    = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE     = False
    cfg.MODEL.ROI_HEADS.USE_DROPLOSS        = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS     = False
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE   = False
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
    cfg.SOLVER.BASE_LR_MULTIPLIER           = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES     = []
    cfg.TEST.NO_SEGM                        = False


def build_cutler_config(
    config_path:  str   = DEFAULT_CUTLER_CONFIG,
    weights_path: str   = DEFAULT_CUTLER_WEIGHTS,
    confidence:   float = 0.1,
    device:       str   = "cuda",
):
    """Build and freeze a CutLER detectron2 CfgNode."""
    cfg = get_cfg()
    _add_cutler_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights_path, "MODEL.DEVICE", device])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST   = confidence
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST   = confidence
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence
    cfg.freeze()
    return cfg


# ── predictor ─────────────────────────────────────────────────────────────────

class CutlerPredictor:
    """
    Minimal inference wrapper for the CutLER Cascade Mask R-CNN.

    Equivalent to the legacy ``engine.defaults.DefaultPredictor`` from the old
    divide_and_conquer directory, but self-contained.
    """

    def __init__(self, cfg):
        self.cfg   = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST,
        )
        self.input_format = cfg.INPUT.FORMAT

    def __call__(self, bgr_image: np.ndarray) -> dict:
        """Single-image inference; *bgr_image* is a (H, W, 3) uint8 BGR array."""
        with torch.no_grad():
            img = bgr_image[:, :, ::-1] if self.input_format == "RGB" else bgr_image
            img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            image_t = torch.as_tensor(
                self.aug.get_transform(img).apply_image(img)
                .astype("float32").transpose(2, 0, 1)
            )
            return self.model([{"image": image_t, "height": h, "width": w}])[0]
