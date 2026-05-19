import torch
import torch.nn as nn
from transformers import AutoModel


_MODEL_ID = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
_PATCH_SIZE = 16
_FEAT_DIM = 1024
# DINOv3 prepends 1 CLS token + 4 register tokens before the spatial patches
_N_PREFIX_TOKENS = 5


class ViTFeatV3(nn.Module):
    """DINOv3 (ViT-L/16) feature extractor.

    Drop-in replacement for ViTFeat from dino.py.  Accepts the same
    constructor signature; the pretrained_pth argument is ignored because
    weights are fetched automatically from HuggingFace.

    Expects ImageNet-normalised float tensors in NCHW format [B, 3, H, W],
    matching the output of torchvision transforms.ToTensor + Normalize —
    the same preprocessing used by generate_feature_matrix in divide_conquer.py.

    Returns:
      feats: [B, feat_dim, feat_h * feat_w]
    """

    def __init__(
        self,
        pretrained_pth: str = _MODEL_ID,
        feat_dim: int = _FEAT_DIM,
        vit_arch: str = 'large',
        vit_feat: str = 'k',               # k / q / v / kqv
        patch_size: int = _PATCH_SIZE,
    ):
        super().__init__()
        self.pretrained_pth = pretrained_pth  # kept for API compat; weights come from HF
        self.vit_arch = vit_arch
        self.feat_dim = feat_dim
        self.vit_feat = vit_feat
        self.patch_size = patch_size

        self.model = AutoModel.from_pretrained(
            _MODEL_ID,
            device_map='cuda',
            attn_implementation='sdpa',
            trust_remote_code=True,
        )
        self.model.eval()
        print(f'Loaded DINOv3 {_MODEL_ID} (feat_dim={feat_dim})')

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img.shape
        feat_h = h // self.patch_size
        feat_w = w // self.patch_size
        n_spatial = feat_h * feat_w

        outputs = self.model(pixel_values=img)
        # last_hidden_state: [B, 1 CLS + 4 registers + n_spatial, feat_dim]
        spatial = outputs.last_hidden_state[:, _N_PREFIX_TOKENS:]  # [B, n_spatial, feat_dim]

        feats = spatial.transpose(1, 2).reshape(b, self.feat_dim, n_spatial)

        if self.vit_feat == 'kqv':
            feats = feats.repeat(1, 3, 1)  # [B, 3*feat_dim, n_spatial]

        return feats


if __name__ == '__main__':
    model = ViTFeatV3(vit_feat='k')
    img = torch.zeros(2, 3, 448, 448, dtype=torch.float32)
    feat = model(img)
    print(feat.shape)  # [2, 1024, 784]  (448/16=28, 28²=784)
