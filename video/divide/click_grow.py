"""Click-seeded mask extraction using DINOv3 feature similarity.

Two modes (selectable via `method`):
  - "threshold": compute cosine-sim map from click patch prototype, threshold
    at `sim_thresh`, keep connected component containing click. Simple and
    consistent across scenes; typically beats CuVLER on animal/object clips.
  - "bfs": BFS region growing (grow_region from Basti's base.py) stopping when
    the similarity gradient turns sharp. Best on high-contrast boundaries
    (e.g. breakdance) but boundary_delta must be tuned per-scene.

Fully unsupervised — no labels, no SAM, no training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "DinoMaskExtraction"))

from mask_extraction.base import grow_region, prototype_similarity_map

PATCH_GRID = 64  # DINOv3 ViT-L/16 at 1024px → 64×64 patch grid


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)


def click_seeded_grow(
    feats: torch.Tensor,
    click_x: float,
    click_y: float,
    image_w: int,
    image_h: int,
    method: str = "threshold",
    sim_thresh: float = 0.50,
    boundary_delta: float = 0.12,
    connectivity: int = 4,
) -> np.ndarray:
    """Grow a mask from a single click using DINOv3 feature similarity.

    Args:
        feats: [H_p, W_p, C] patch feature grid (e.g. 64×64×1024), CPU or GPU.
        click_x, click_y: click in original image pixels.
        image_w, image_h: original image dimensions.
        method: "threshold" (sim > sim_thresh + CC) or "bfs" (grow_region BFS).
        sim_thresh: similarity cutoff for "threshold" mode.
        boundary_delta: max sim drop for BFS stop in "bfs" mode.
        connectivity: 4 or 8 neighbours.

    Returns:
        [image_h, image_w] uint8 binary mask.
    """
    grid = feats.float()
    if grid.device.type != "cpu":
        grid = grid.cpu()

    H_p, W_p, _ = grid.shape
    pi = int(np.clip(click_y / image_h * H_p, 0, H_p - 1))
    pj = int(np.clip(click_x / image_w * W_p, 0, W_p - 1))

    proto = grid[pi, pj]
    sim_map = prototype_similarity_map(grid, proto, _cosine_sim, normalize_sim=True)

    if method == "threshold":
        binary = (sim_map.numpy() > sim_thresh).astype(np.uint8)
        labeled, _ = ndimage.label(binary)
        label_at_click = labeled[pi, pj]
        if label_at_click == 0:
            mask_small = np.zeros((H_p, W_p), dtype=np.uint8)
        else:
            mask_small = (labeled == label_at_click).astype(np.uint8)
    else:  # bfs
        seed = torch.zeros(H_p, W_p, dtype=torch.bool)
        seed[pi, pj] = True
        blocked = torch.zeros(H_p, W_p, dtype=torch.bool)
        region = grow_region(sim_map, seed, blocked, boundary_delta=boundary_delta,
                             connectivity=connectivity)
        mask_small = region.numpy().astype(np.uint8)

    return cv2.resize(mask_small, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
