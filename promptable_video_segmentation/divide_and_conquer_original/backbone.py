"""
DINOv1 ViT key-feature extractor for the *original* Divide-and-Conquer method.

This wraps the original ``dino.ViTFeat`` (ViT-B/8 key features) shipped in the
old ``divide_and_conquer`` directory, so the conquer phase uses exactly the same
backbone and feature extraction as the original ``divide_conquer.py``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from torchvision import transforms

# Register UnSAM/divide_and_conquer/ as the 'divide_and_conquer' package so the
# original top-level ``dino`` module is importable.  Same trick as predictor.py.
_REPO = Path(__file__).resolve().parent.parent.parent   # UnSAM/
_OLD_DAC = _REPO / "divide_and_conquer"
if "divide_and_conquer" not in sys.modules:
    _dac_pkg = types.ModuleType("divide_and_conquer")
    _dac_pkg.__path__ = [str(_OLD_DAC)]
    _dac_pkg.__package__ = "divide_and_conquer"
    sys.modules["divide_and_conquer"] = _dac_pkg

from divide_and_conquer import dino  # noqa: E402


# ── original DINOv1 defaults (from divide_conquer.py get_parser) ───────────────

DEFAULT_BACKBONE_URL  = (
    "https://dl.fbaipublicfiles.com/dino/"
    "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
)
DEFAULT_BACKBONE_ARCH = "base"
DEFAULT_VIT_FEAT      = "k"
DEFAULT_PATCH_SIZE    = 8
DEFAULT_FEAT_DIM      = 768


# Same ImageNet normalisation as the original divide_conquer.py.
_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_dino_v1_backbone(
    url:        str = DEFAULT_BACKBONE_URL,
    feat_dim:   int = DEFAULT_FEAT_DIM,
    arch:       str = DEFAULT_BACKBONE_ARCH,
    vit_feat:   str = DEFAULT_VIT_FEAT,
    patch_size: int = DEFAULT_PATCH_SIZE,
    device:     str = "cuda",
):
    """
    Build the original DINOv1 ViT feature extractor.

    Mirrors divide_conquer.py:
        backbone = dino.ViTFeat(url, feat_dim, arch, 'k', patch_size)
        backbone.eval(); backbone.cuda().to(torch.float16)
    """
    backbone = dino.ViTFeat(url, feat_dim, arch, vit_feat, patch_size)
    backbone.eval()
    if device != "cpu":
        backbone.cuda().to(torch.float16)
    return backbone


def extract_feature_matrix(
    backbone,
    image_pil,
    feat_dim: int,
    feat_num: int,
) -> torch.Tensor:
    """
    Extract a (feat_num, feat_num, feat_dim) feature tensor from a PIL image.

    Identical to the original ``generate_feature_matrix`` in divide_conquer.py:
    the GPU path runs the backbone in half precision; the CPU path in float32.
    """
    if next(backbone.parameters()).device == torch.device("cpu"):
        tensor = _to_tensor(image_pil).unsqueeze(0)
        feat = backbone(tensor)[0]
    else:
        tensor = _to_tensor(image_pil).unsqueeze(0).half().cuda()
        feat = backbone(tensor)[0].cpu()
    return feat.reshape(feat_dim, feat_num, feat_num).permute(1, 2, 0)  # (H, W, C)
