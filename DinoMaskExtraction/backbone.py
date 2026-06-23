"""
Self-contained DINOv3 (ViT-L/16) feature extraction.

No dependency on any other module in this repo. Two entry points share the same
model constants and pre-processing:

* :class:`ViTFeatV3` -- the simple extractor. One forward pass returns the
  final-layer spatial patch features ``(B, feat_dim, feat_h*feat_w)``. This is
  the stable API consumed elsewhere (``extract_feature_matrix``).
* :class:`HookedViTFeatV3` -- the richer extractor. Registers forward hooks on
  every transformer block (and the final LayerNorm) so a single pass yields the
  intermediate representation at all 26 depths. Used by the PCA-RGB inspector
  app and for feature-statistics probing.

Token layout per image (for an H x W input, P = patch size = 16)::

    [ CLS | reg_0 reg_1 reg_2 reg_3 | patch_0 ... patch_{N-1} ]
      <-------- N_PREFIX = 5 -------> <------ (H/P)*(W/P) ------>

Only the spatial patch tokens are returned; the CLS and register tokens are
discarded.
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


# --------------------------------------------------------------------------- #
#  Simple final-layer extractor (stable API)                                  #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Hooked multi-location extractor                                            #
# --------------------------------------------------------------------------- #
class HookedViTFeatV3(nn.Module):
    """DINOv3 ViT-L/16 with forward hooks at every block output.

    After :meth:`forward`, :attr:`features` maps a hook name to the raw token
    tensor ``(B, N_tokens, C)`` captured at that location. Use :meth:`extract`
    to get per-location spatial grids with the prefix tokens removed.
    """

    def __init__(self, model_id: str = _MODEL_ID, device: str = "cuda"):
        super().__init__()
        self.patch_size = _PATCH_SIZE
        self.feat_dim   = _FEAT_DIM
        self.device     = device
        self.model = AutoModel.from_pretrained(
            model_id,
            device_map=device,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()

        self.features: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self.hook_names: list[str] = []
        self._register_hooks()
        print(f"Loaded DINOv3 ({model_id}) with {len(self.hook_names)} hooks")

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            # block forward may return a tuple (hidden_states, ...) or a tensor
            t = out[0] if isinstance(out, (tuple, list)) else out
            self.features[name] = t.detach()
        return hook

    def _register_hooks(self):
        # output of the patch-embedding / token-assembly stage
        self._handles.append(
            self.model.embeddings.register_forward_hook(self._make_hook("embeddings"))
        )
        self.hook_names.append("embeddings")
        # output of each transformer block
        for i, layer in enumerate(self.model.model.layer):
            name = f"block_{i:02d}"
            self._handles.append(layer.register_forward_hook(self._make_hook(name)))
            self.hook_names.append(name)
        # output of the final LayerNorm (== model.last_hidden_state)
        self._handles.append(
            self.model.norm.register_forward_hook(self._make_hook("final_norm"))
        )
        self.hook_names.append("final_norm")

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one forward pass and return the captured ``{name: tokens}`` dict."""
        self.features = {}
        self.model(pixel_values=img.to(self.device))
        return self.features

    @torch.no_grad()
    def extract(self, image_pil, apply_norm: bool = False
                ) -> tuple[dict[str, torch.Tensor], int, int]:
        """Extract features from a PIL image.

        Parameters
        ----------
        apply_norm:
            If ``True``, pass each intermediate location's tokens through the
            model's final ``LayerNorm`` (``model.norm``) before returning them.
            This matches the DINOv3 paper's dense-decoder recipe, which applies
            the final layer norm to every intermediate layer it consumes. The
            already-normed ``final_norm`` hook is left untouched to avoid
            double-normalisation.

        Returns ``(grids, feat_h, feat_w)`` where ``grids`` maps each hook name
        to a ``(feat_h, feat_w, C)`` CPU tensor of patch features.
        """
        w, h = image_pil.size
        feat_h, feat_w = h // self.patch_size, w // self.patch_size
        tensor = _to_tensor(image_pil).unsqueeze(0).to(self.device)
        dtype = next(self.model.parameters()).dtype
        self.forward(tensor.to(dtype))
        grids = self._to_grids(feat_h, feat_w, apply_norm)
        return grids, feat_h, feat_w

    @torch.no_grad()
    def extract_tensor(self, batch: torch.Tensor, feat_h: int, feat_w: int,
                       apply_norm: bool = False, names: list[str] | None = None,
                       to_cpu: bool = True) -> dict[str, torch.Tensor]:
        """Run a pre-normalised ``(B, 3, H, W)`` tensor through the backbone.

        Lets the (CPU-heavy) JPEG decode + resize + normalise happen elsewhere —
        e.g. in DataLoader workers — so this method only does the GPU forward.
        Returns ``{name: (B, feat_h, feat_w, C)}`` tensors.

        Parameters
        ----------
        names : restrict the returned grids to these hook names (default: all).
            Materialising only the layer you need avoids reshaping every block.
        to_cpu : copy results to host. Pass ``False`` to keep them on-GPU so a
            caller can do further GPU work (e.g. PCA) before the device→host copy.
        """
        self.forward(batch.to(self.device, next(self.model.parameters()).dtype))
        items = (self.features.items() if names is None
                 else [(n, self.features[n]) for n in names])
        grids = {}
        for name, tok in items:
            if apply_norm and name != "final_norm":
                tok = self.model.norm(tok)                     # final LayerNorm
            patches = tok[:, _N_PREFIX:]                       # (B, N, C)
            b = patches.shape[0]
            g = patches.reshape(b, feat_h, feat_w, -1).float()
            grids[name] = g.cpu() if to_cpu else g
        return grids

    @torch.no_grad()
    def extract_batch(self, image_pils: list, apply_norm: bool = False
                      ) -> tuple[dict[str, torch.Tensor], int, int]:
        """Extract features from a list of **equally-sized** PIL images in one pass.

        All images must share the same ``(w, h)`` (e.g. all centre-cropped to the
        same square ``res``) so their tokens stack. Returns ``(grids, feat_h,
        feat_w)`` where each ``grids[name]`` is a ``(B, feat_h, feat_w, C)`` CPU
        tensor — one batched forward instead of ``B`` single-image passes.
        """
        w, h = image_pils[0].size
        if any(p.size != (w, h) for p in image_pils):
            raise ValueError("extract_batch requires all images to be the same size")
        feat_h, feat_w = h // self.patch_size, w // self.patch_size
        batch = torch.stack([_to_tensor(p) for p in image_pils])
        return self.extract_tensor(batch, feat_h, feat_w, apply_norm), feat_h, feat_w

    def _to_grids(self, feat_h: int, feat_w: int,
                  apply_norm: bool = False) -> dict[str, torch.Tensor]:
        grids = {}
        for name, tok in self.features.items():
            if apply_norm and name != "final_norm":
                tok = self.model.norm(tok)                     # final LayerNorm
            patches = tok[0, _N_PREFIX:]                       # (N, C)
            grids[name] = patches.reshape(feat_h, feat_w, -1).float().cpu()
        return grids


def pca_to_rgb(grid: torch.Tensor, q_lo: float = 0.02, q_hi: float = 0.98) -> torch.Tensor:
    """Project a ``(H, W, C)`` feature grid to a ``(H, W, 3)`` uint8 RGB image.

    The top-3 principal components of the patch features become the R, G, B
    channels, each robustly normalised to ``[0, 1]`` using the given quantiles.
    """
    h, w, c = grid.shape
    ncomp = min(3, c)
    flat = grid.reshape(h * w, c).float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    # deterministic full SVD: leading right singular vectors == top PCs
    _, _, vh = torch.linalg.svd(flat, full_matrices=False)
    proj = flat @ vh[:ncomp].T                                # (H*W, ncomp)
    lo = torch.quantile(proj, q_lo, dim=0)
    hi = torch.quantile(proj, q_hi, dim=0)
    proj = ((proj - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    if ncomp < 3:                                             # pad to RGB
        proj = torch.cat([proj, proj.new_zeros(h * w, 3 - ncomp)], dim=1)
    rgb = (proj.reshape(h, w, 3) * 255).to(torch.uint8)
    return rgb


if __name__ == "__main__":
    # quick smoke test: print captured locations + per-token L2 magnitude
    import torch as _torch
    ext = HookedViTFeatV3(device="cuda" if _torch.cuda.is_available() else "cpu")
    dummy = _torch.randn(1, 3, 224, 224)
    feats = ext.forward(dummy)
    for name in ext.hook_names:
        tok = feats[name][0, _N_PREFIX:]
        print(f"{name:<13} tokens={tuple(tok.shape)}  |x|_mean={tok.norm(dim=1).mean():.2f}")
