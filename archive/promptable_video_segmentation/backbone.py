"""
Self-contained DINOv3 (ViT-L/16) feature extractor.

No dependency on any other module in this repo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModel


_MODEL_ID    = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_PATCH_SIZE  = 16
_FEAT_DIM    = 1024
_N_PREFIX    = 5   # 1 CLS token + 4 register tokens prepended before spatial patches

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class ViTFeatV3(nn.Module):
    """
    DINOv3 ViT-L/16 spatial feature extractor.

    Accepts ImageNet-normalised NCHW float tensors and returns
    [B, feat_dim, feat_h * feat_w] key features (spatial patches only;
    the CLS and register tokens are discarded).
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        feat_dim: int = _FEAT_DIM,
        patch_size: int = _PATCH_SIZE,
    ):
        super().__init__()
        self.feat_dim   = feat_dim
        self.patch_size = patch_size
        self.model = AutoModel.from_pretrained(
            model_id,
            device_map="cuda",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Loaded DINOv3 ({model_id})")

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img.shape
        feat_h = h // self.patch_size
        feat_w = w // self.patch_size
        out = self.model(pixel_values=img)
        spatial = out.last_hidden_state[:, _N_PREFIX:]           # (B, n_spatial, C)
        return spatial.transpose(1, 2).reshape(b, self.feat_dim, feat_h * feat_w)


def extract_feature_matrix(
    backbone: ViTFeatV3,
    image_pil,
    feat_dim: int,
    feat_num: int,
) -> torch.Tensor:
    """
    Extract a (feat_num, feat_num, feat_dim) feature tensor from a PIL image.

    The image is converted to an ImageNet-normalised tensor, pushed through
    *backbone*, and reshaped into a 2-D spatial grid.
    """
    is_cuda = next(backbone.parameters()).device.type != "cpu"
    dtype   = next(backbone.parameters()).dtype
    tensor  = _to_tensor(image_pil).unsqueeze(0)
    if is_cuda:
        tensor = tensor.to(dtype).cuda()
    feat = backbone(tensor)[0].cpu()                            # (feat_dim, feat_num²)
    return feat.reshape(feat_dim, feat_num, feat_num).permute(1, 2, 0)  # (H, W, C)
