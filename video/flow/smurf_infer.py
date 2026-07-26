"""PyTorch SMURF-YTVIS inference: correct flow direction + channel order.

Wraps video/flow/smurf_raft/ (vendored from ChristophReich1996/SMURF, an
unofficial PyTorch port of SMURF that is literally torchvision's private RAFT
builder with SMURF's hyperparameters — see smurf_raft/smurf.py). That repo
never demonstrates mask warping, so it doesn't surface two conventions that
differ from this project's original TF-SMURF usage
(video/flow/eval_flow_warp_davis.py) and matter for correctness:

1. DIRECTION. model(image1, image2) returns FORWARD flow: for each pixel in
   image1, where it moves to in image2 (standard RAFT/torchvision usage).
   TF-SMURF's eval script instead called smurf.infer(..., infer_bw=True) to
   get BACKWARD flow directly. A gather-based warp (cv2.remap) needs backward
   flow -- for each pixel in the OUTPUT frame, where its content came FROM in
   the input frame -- so here that means calling the model with the frames in
   the opposite order: model(image_dst, image_src).

2. CHANNEL ORDER. TF-SMURF returns (dy, dx) (see the ordering note in
   eval_flow_warp_davis.py:warp_mask). This is plain torchvision RAFT, which
   returns (dx, dy) -- i.e. flow[..., 0] is the x-displacement. Reusing the
   old warp_mask unmodified against this checkpoint would silently transpose
   every warp.

Getting either of these wrong produces a plausible-looking but wrong warp,
which is exactly the failure mode this project already got bitten by once.
"""

from __future__ import annotations

import numpy as np
import cv2
import torch

from video.flow.smurf_raft import raft_smurf


def load_smurf(checkpoint: str, device: str = "cuda") -> torch.nn.Module:
    model = raft_smurf(checkpoint=checkpoint)
    return model.to(device).eval()


def _pad_to_multiple_of_8(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"), pad_h, pad_w


def _to_tensor(img_rgb: np.ndarray, device: str) -> torch.Tensor:
    """uint8 HWC [0,255] -> float32 CHW [-1,1], batched. Matches
    smurf_raft's own perform_inference.py normalization exactly."""
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device)
    t = 2.0 * (t / 255.0) - 1.0
    return t[None]


@torch.no_grad()
def compute_backward_flow(model: torch.nn.Module, frame_dst: np.ndarray,
                          frame_src: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Backward flow for warping frame_src's content into frame_dst's grid.

    Returns flow_bw [H, W, 2] as (dx, dy) float32, at frame_dst's original
    (unpadded) resolution, ready for warp_mask_backward below.
    """
    h, w = frame_dst.shape[:2]
    dst_p, pad_h, pad_w = _pad_to_multiple_of_8(frame_dst)
    src_p, _, _ = _pad_to_multiple_of_8(frame_src)

    # Argument order is swapped relative to naive forward-flow usage -- see
    # module docstring point 1. model(image1, image2): flow FROM image1 TO
    # image2. We want, for each pixel of frame_dst, where it came from in
    # frame_src -- i.e. forward flow dst -> src.
    preds = model(_to_tensor(dst_p, device), _to_tensor(src_p, device))
    flow = preds[-1][0].permute(1, 2, 0).cpu().numpy()  # [H_p, W_p, 2], (dx, dy)

    H_p, W_p = flow.shape[:2]
    return flow[: H_p - pad_h, : W_p - pad_w] if (pad_h or pad_w) else flow


def warp_soft_backward(field_src: np.ndarray, flow_bw: np.ndarray) -> np.ndarray:
    """Gather-warp a CONTINUOUS field (e.g. a soft heatmap) into the grid
    flow_bw is defined on. No thresholding -- see warp_mask_backward below for
    why that matters when the result gets blended with another soft field
    rather than used as a final mask.

    warped(x) = field_src(x + flow_bw(x)), flow_bw channel order (dx, dy) --
    see module docstring point 2.
    """
    h, w = field_src.shape
    dx = flow_bw[..., 0].astype(np.float32)
    dy = flow_bw[..., 1].astype(np.float32)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + dx
    map_y = ys + dy
    return cv2.remap(field_src.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_mask_backward(mask_src: np.ndarray, flow_bw: np.ndarray) -> np.ndarray:
    """Gather-warp a binary mask_src, thresholding the result at 0.5.

    Blending two ALREADY-BINARY masks with a linear weight doesn't behave as
    an interpolation -- e.g. (1-blend)*a + blend*b > 0.5 collapses to exactly
    3 regimes (pure a, intersection, pure b) as blend crosses 0.5, not a
    smooth sweep. Use warp_soft_backward instead for anything that will be
    blended with another continuous field before its own final threshold.
    """
    h, w = mask_src.shape
    dx = flow_bw[..., 0].astype(np.float32)
    dy = flow_bw[..., 1].astype(np.float32)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + dx
    map_y = ys + dy
    warped = cv2.remap(mask_src.astype(np.float32), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (warped > 0.5).astype(np.uint8)
