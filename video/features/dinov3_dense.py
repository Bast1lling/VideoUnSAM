"""Full-frame dense DINOv3 features for video.

Differences from the conquer-phase usage in divide_conquerV3.py:
  - Operates on native-resolution frames (e.g. 480p), not 512x512 crops.
  - Pads to a multiple of the patch size; returns the padding so callers can
    map patches back to pixels in the original frame.
  - L2-normalises features per patch — downstream cosine similarity becomes
    a dot product and Sinkhorn cost reduces to (1 - <a,b>).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "divide_and_conquer"))
import dinov3  # noqa: E402


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class DenseDINOv3:
    """Wraps ViTFeatV3 for whole-frame dense feature extraction."""

    def __init__(self, patch_size: int = 16, feat_dim: int = 1024,
                 dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
                 with_attention: bool = False):
        self.patch_size = patch_size
        self.feat_dim = feat_dim
        self.dtype = dtype
        self.device = device
        self.with_attention = with_attention

        self.backbone = dinov3.ViTFeatV3(
            feat_dim=feat_dim, vit_arch="large", vit_feat="k", patch_size=patch_size,
            attn_implementation="eager" if with_attention else "sdpa",
        )
        self.backbone.eval()
        self.backbone.to(device)

    def _prep(self, frame_rgb: np.ndarray) -> tuple[torch.Tensor, int, int, int, int]:
        """HxWx3 uint8 RGB → padded NCHW tensor + (pad_h, pad_w, H, W)."""
        H, W = frame_rgb.shape[:2]
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        t = _to_tensor(frame_rgb).unsqueeze(0).to(self.dtype)
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        return t.to(self.device), ph, pw, H, W

    @torch.no_grad()
    def extract(self, frame_rgb: np.ndarray, normalize: bool = True) -> dict:
        """Returns:
            feats:  [h, w, feat_dim] float32 on CPU (h = (H+ph)/p, w = (W+pw)/p)
            grid_h, grid_w: feature-grid size
            H, W:   original frame size
            ph, pw: bottom/right padding in pixels
        """
        tensor, ph, pw, H, W = self._prep(frame_rgb)
        gh = (H + ph) // self.patch_size
        gw = (W + pw) // self.patch_size

        attn = None
        if self.with_attention:
            feats, cls_attn = self.backbone.forward_with_attention(tensor)
            feats = feats[0]                                  # [feat_dim, n_spatial]
            attn = cls_attn[0].reshape(gh, gw).float().cpu()  # [gh, gw] objectness map
        else:
            feats = self.backbone(tensor)[0]                  # [feat_dim, n_spatial]

        feats = feats.reshape(self.feat_dim, gh, gw).permute(1, 2, 0).float().cpu()
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return {"feats": feats, "grid_h": gh, "grid_w": gw,
                "H": H, "W": W, "ph": ph, "pw": pw, "attn": attn}

    @torch.no_grad()
    def extract_batch(self, frames_rgb: list[np.ndarray], normalize: bool = True) -> list[dict]:
        """Per-frame extraction (frames may differ in size). Loops internally."""
        return [self.extract(f, normalize=normalize) for f in frames_rgb]

    @torch.no_grad()
    def extract_batch_same_size(self, frames_rgb: list[np.ndarray],
                                normalize: bool = True) -> list[dict]:
        """True batched extraction: all frames must already share H, W (e.g. pre-resized
        to the same feat_size). Stacks into one [B, 3, H, W] forward pass instead of B
        separate single-image passes — much better GPU utilization/throughput than
        `extract_batch`, at the cost of B frames' activations resident in VRAM at once."""
        if not frames_rgb:
            return []
        tensors, phs, pws, Hs, Ws = [], [], [], [], []
        for f in frames_rgb:
            t, ph, pw, H, W = self._prep(f)
            tensors.append(t)
            phs.append(ph); pws.append(pw); Hs.append(H); Ws.append(W)
        batch = torch.cat(tensors, dim=0)  # [B, 3, H, W]

        feats = self.backbone(batch)  # [B, feat_dim, n_spatial]
        out = []
        for i in range(len(frames_rgb)):
            gh = (Hs[i] + phs[i]) // self.patch_size
            gw = (Ws[i] + pws[i]) // self.patch_size
            fi = feats[i].reshape(self.feat_dim, gh, gw).permute(1, 2, 0).float().cpu()
            if normalize:
                fi = F.normalize(fi, dim=-1)
            out.append({"feats": fi, "grid_h": gh, "grid_w": gw,
                        "H": Hs[i], "W": Ws[i], "ph": phs[i], "pw": pws[i], "attn": None})
        return out
