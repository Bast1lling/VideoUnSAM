"""SAM 1 (ViT-H) boundary refinement on top of OT-propagated masks.

Treats SAM as a *decoder* prompted by what OT discovered. OT picks the right
object, SAM resolves the right boundary at pixel resolution.

Prompt strategy (configurable):
  - bbox: tight box around the propagated mask
  - points: peak-heatmap points sampled inside the mask as positive clicks
  - mask_input: the OT heatmap downsampled to 256x256 as a soft prior

The standard XMem/Cutie+SAM recipe uses all three. Pass `use_mask_input=False`
to A/B against box+points only.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor


_DEFAULT_CKPT = Path("/home/nilsc/VideoUnSAM/checkpoints/sam_vit_h_4b8939.pth")


def _bbox_from_mask(mask: np.ndarray, pad: int = 0) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    H, W = mask.shape
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(H - 1, y1 + pad); x1 = min(W - 1, x1 + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _sample_points_max(score_map: np.ndarray, region: np.ndarray, k: int,
                       min_dist: int = 30) -> np.ndarray:
    """Greedy peak picks of score_map restricted to `region` (bool), spaced apart."""
    work = (score_map * (region > 0)).copy()
    pts = []
    h, w = work.shape
    for _ in range(k):
        if work.max() <= 0:
            break
        y, x = np.unravel_index(np.argmax(work), work.shape)
        pts.append((x, y))
        y0, y1 = max(0, y - min_dist), min(h, y + min_dist + 1)
        x0, x1 = max(0, x - min_dist), min(w, x + min_dist + 1)
        work[y0:y1, x0:x1] = 0
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)


def _negative_region(heatmap: np.ndarray, bbox: np.ndarray, low_quantile: float = 0.3,
                     erode_px: int = 6) -> np.ndarray:
    """Bool region inside the bbox where heatmap is LOW — i.e. background patches
    that the OT plan didn't send mass to. Shrink in a few pixels from the box
    edges so we don't pick obviously-outside points right on the boundary."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    H, W = heatmap.shape
    box = np.zeros_like(heatmap, dtype=bool)
    y0e = min(H - 1, y0 + erode_px); y1e = max(0, y1 - erode_px)
    x0e = min(W - 1, x0 + erode_px); x1e = max(0, x1 - erode_px)
    if y1e <= y0e or x1e <= x0e:
        return box
    box[y0e:y1e, x0e:x1e] = True
    # "Low" relative to the heatmap distribution inside the box
    inside_vals = heatmap[box]
    if inside_vals.size == 0:
        return box
    thresh = np.quantile(inside_vals, low_quantile)
    return box & (heatmap <= thresh)


def _heatmap_to_sam_mask_input(heatmap: np.ndarray) -> np.ndarray:
    """SAM wants a [1, 256, 256] float32 in logit-space (sigmoid pre-image).

    We treat the peak-normalised heatmap as a probability and pass logits."""
    h = torch.from_numpy(heatmap)[None, None].float()
    h = F.interpolate(h, size=(256, 256), mode="bilinear", align_corners=False)[0]
    p = h.clamp(0.02, 0.98)  # avoid inf
    logits = torch.log(p / (1 - p))
    return logits.numpy()  # [1, 256, 256]


class SAMRefiner:
    def __init__(self, ckpt_path: str | Path = _DEFAULT_CKPT,
                 model_type: str = "vit_h", device: str = "cuda"):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {ckpt_path}. "
                "Download with: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )
        sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
        sam.to(device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.device = device

    def refine(
        self,
        frame_rgb: np.ndarray,         # [H, W, 3] uint8
        prop_mask: np.ndarray,         # [H, W] uint8 from OT
        prop_heatmap: np.ndarray,      # [H, W] float32 in [0, 1]
        n_pos: int = 3,
        n_neg: int = 4,
        use_mask_input: bool = True,
        bbox_pad: int = 8,
        neg_quantile: float = 0.25,
        multimask: bool = True,
    ) -> dict:
        if prop_mask.sum() == 0:
            return {"mask": prop_mask, "score": 0.0, "n_candidates": 0}

        self.predictor.set_image(frame_rgb)
        box = _bbox_from_mask(prop_mask, pad=bbox_pad)

        pos = _sample_points_max(prop_heatmap, prop_mask.astype(bool), k=n_pos)
        # Negative anchors: inside the bbox but in low-heat regions (wheel hole etc.)
        neg_region = _negative_region(prop_heatmap, box, low_quantile=neg_quantile)
        # Score negatives by *anti-heat* so we pick the lowest spots.
        anti = (1.0 - prop_heatmap).astype(np.float32)
        neg = _sample_points_max(anti, neg_region, k=n_neg, min_dist=20)

        coords = np.concatenate([pos, neg], axis=0) if len(neg) else pos
        labels = np.concatenate([
            np.ones(len(pos), dtype=np.int32),
            np.zeros(len(neg), dtype=np.int32),
        ]) if len(neg) else np.ones(len(pos), dtype=np.int32)

        kwargs = {"box": box, "multimask_output": multimask}
        if len(coords):
            kwargs["point_coords"] = coords
            kwargs["point_labels"] = labels
        if use_mask_input:
            kwargs["mask_input"] = _heatmap_to_sam_mask_input(prop_heatmap)

        masks, scores, _ = self.predictor.predict(**kwargs)
        best = int(np.argmax(scores))
        return {
            "mask": masks[best].astype(np.uint8),
            "score": float(scores[best]),
            "n_candidates": int(masks.shape[0]),
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "all_masks": masks,
            "all_scores": scores,
        }
