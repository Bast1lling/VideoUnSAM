"""Phase-2 sanity check: DINOv3 patch-cosine similarity between two video frames.

Picks a query patch in frame A, computes cosine similarity against every patch
in frame B, renders side-by-side: A with a query dot, B with a heatmap overlay.

Usage:
    python -m video.scripts.visualize_frame_similarity \\
        --clip blackswan --frame-a 0 --frame-b 10 \\
        --query-xy 0.5 0.5 --out sim_blackswan_0_10.png

If --query-xy is omitted, defaults to image center.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.features.dinov3_dense import DenseDINOv3


_DAVIS_ROOT = Path("/home/nilsc/VideoUnSAM/datasets/davis/DAVIS/JPEGImages/480p")


def load_frame(clip: str, idx: int) -> np.ndarray:
    path = _DAVIS_ROOT / clip / f"{idx:05d}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def query_patch_feat(feats: torch.Tensor, gh: int, gw: int,
                     qx: float, qy: float, radius: int = 1) -> tuple[torch.Tensor, int, int]:
    """Average features in a (2r+1)x(2r+1) window around (qx, qy) in [0,1] coords."""
    iy = int(round(qy * (gh - 1)))
    ix = int(round(qx * (gw - 1)))
    y0, y1 = max(0, iy - radius), min(gh, iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(gw, ix + radius + 1)
    patch = feats[y0:y1, x0:x1].reshape(-1, feats.shape[-1])
    q = F.normalize(patch.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
    return q, iy, ix


def heatmap_overlay(rgb: np.ndarray, sim: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    H, W = rgb.shape[:2]
    sim_up = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC)
    sim_up = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-8)
    heat = cv2.applyColorMap((sim_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="blackswan")
    p.add_argument("--frame-a", type=int, default=0)
    p.add_argument("--frame-b", type=int, default=10)
    p.add_argument("--query-xy", nargs=2, type=float, default=[0.5, 0.5],
                   help="Query patch location in frame A, normalised [0,1]")
    p.add_argument("--radius", type=int, default=1,
                   help="Average a (2r+1) feature window around the query")
    p.add_argument("--out", default="frame_similarity.png")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--feat-dim", type=int, default=1024)
    args = p.parse_args()

    print(f"[load] {args.clip} frames {args.frame_a} & {args.frame_b}")
    frame_a = load_frame(args.clip, args.frame_a)
    frame_b = load_frame(args.clip, args.frame_b)

    print("[model] DINOv3 ViT-L/16 bf16")
    extractor = DenseDINOv3(patch_size=args.patch_size, feat_dim=args.feat_dim)

    out_a = extractor.extract(frame_a)
    out_b = extractor.extract(frame_b)
    feats_a, feats_b = out_a["feats"], out_b["feats"]
    gh_a, gw_a = out_a["grid_h"], out_a["grid_w"]
    gh_b, gw_b = out_b["grid_h"], out_b["grid_w"]
    print(f"[feats] A={tuple(feats_a.shape)} B={tuple(feats_b.shape)}")

    qx, qy = args.query_xy
    q, iy, ix = query_patch_feat(feats_a, gh_a, gw_a, qx, qy, radius=args.radius)

    sim = (feats_b @ q).numpy()  # [gh_b, gw_b]
    print(f"[sim ] min={sim.min():.3f} max={sim.max():.3f} mean={sim.mean():.3f}")

    # Draw query dot on A in pixel space
    H_a, W_a = frame_a.shape[:2]
    px = int(round(qx * (W_a - 1)))
    py = int(round(qy * (H_a - 1)))
    a_vis = frame_a.copy()
    cv2.circle(a_vis, (px, py), 8, (255, 255, 255), 2)
    cv2.circle(a_vis, (px, py), 4, (255, 0, 0), -1)

    b_vis = heatmap_overlay(frame_b, sim)

    # Side-by-side; pad shorter image if heights differ (shouldn't, same clip)
    H = max(a_vis.shape[0], b_vis.shape[0])
    def pad(img):
        if img.shape[0] == H:
            return img
        out = np.zeros((H, img.shape[1], 3), dtype=np.uint8)
        out[: img.shape[0]] = img
        return out
    panel = np.concatenate([pad(a_vis), pad(b_vis)], axis=1)
    out_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out_bgr)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
