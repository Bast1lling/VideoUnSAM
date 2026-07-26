"""Frozen RAFT optical flow -- an independent (non-DINOv3-feature-space) signal
for the Stage-5 pseudo-label quality gate.

Why: cycle-agreement (video/propagation/sinkhorn_ot.py) and CuVLER's detector
score are both computed inside the same DINOv3 feature space the OT
propagation itself uses -- a "wrong track" that's self-consistent in that
space (see cycle-agreement-quality-filter-feasibility memory: drift-chicane,
sheep, surf, snowboard) has no structural reason to also disagree with
itself on either signal. Optical flow is pixel motion, computed from RGB
only -- genuinely orthogonal evidence. RAFT's weights are supervised on
synthetic optical-flow data (FlyingChairs/FlyingThings3D), not on any
segmentation/video-object labels, so using it doesn't reintroduce
segmentation supervision into the label-free pipeline.

Direct single-jump flow (not chained across intermediate frames), matching
this project's established finding that direct single-jump propagation is
as good as chaining for Sinkhorn OT ([[propagation-ot-vs-alternatives]]).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def load_raft(device: str = "cuda"):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    transforms = weights.transforms()
    return model, transforms


def _pad_to_multiple_of_8_np(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


@torch.no_grad()
def compute_flow(model, transforms, frame_from_rgb: np.ndarray, frame_to_rgb: np.ndarray,
                  device: str = "cuda") -> np.ndarray:
    """RGB uint8 [H,W,3] x2 -> flow [H,W,2] (dx, dy) mapping frame_from pixels to
    their location in frame_to. flow[y,x] = (dx,dy) means frame_from[y,x]
    corresponds to frame_to[y+dy, x+dx].

    RAFT's transform preset expects a uint8 tensor (it does the /255 + [-1,1]
    normalisation itself, matching the torchvision.io.read_image convention) --
    passing a pre-converted float tensor skips that rescale and silently feeds
    RAFT [0,255]-range "pixels", which produces garbage flow with no error.
    """
    h, w = frame_from_rgb.shape[:2]
    f1 = _pad_to_multiple_of_8_np(frame_from_rgb)
    f2 = _pad_to_multiple_of_8_np(frame_to_rgb)
    t1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).to(device)  # uint8
    t2 = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).to(device)  # uint8
    t1, t2 = transforms(t1, t2)
    flow = model(t1, t2)[-1]  # [1, 2, Hp, Wp], last refinement iteration
    flow = flow[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()  # crop padding, [H, W, 2]
    return flow


def warp_mask_backward(mask_from: np.ndarray, flow_to_from: np.ndarray) -> np.ndarray:
    """Backward-warp mask_from into the "to" frame using flow_to_from (flow computed
    FROM the "to" frame TO the "from" frame, i.e. compute_flow(frame_to, frame_from)).
    For each pixel in the "to" grid, samples mask_from at pixel + flow -> its
    corresponding source location. Returns binary uint8 mask in the "to" frame."""
    H, W = flow_to_from.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.from_numpy(mask_from.astype(np.float32))[None, None].to(device)  # [1,1,Hf,Wf]
    flow_t = torch.from_numpy(flow_to_from).float().to(device)  # [H, W, 2]

    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                            indexing="ij")
    src_x = xs.float() + flow_t[..., 0]
    src_y = ys.float() + flow_t[..., 1]
    # normalise to [-1, 1] for grid_sample, against mask_from's own resolution
    Hf, Wf = mask_from.shape[:2]
    norm_x = 2.0 * src_x / max(Wf - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(Hf - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)[None]  # [1, H, W, 2]

    warped = F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return (warped[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
