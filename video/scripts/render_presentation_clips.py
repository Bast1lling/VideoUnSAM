"""One-off: render two fresh clips for the 15-min presentation using the
*current* shipped pipeline (demo.py's run_pipeline, hybrid seed-pick,
probe fusion on, CRF on). Click is simulated the same way eval_davis2016.py
does — distance-transform peak of frame-0 GT — so it's not hand-picked.

Usage: python video/scripts/render_presentation_clips.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))

from video.loaders import davis
from video.decoder.train_sam_decoder import sample_clicks
import demo as demo_mod

OUT_DIR = _REPO / "video" / "outputs" / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# clean win: conquer rescues a merged crowd detection
# partial failure: OT drift + reseed lock-in documented in the interim report
CLIPS = {
    "clean_win_breakdance": "breakdance",
    "partial_failure_bmx_trees": "bmx-trees",
}


def simulated_click(clip: str) -> tuple[float, float]:
    """Same click simulation as eval_davis2016.py: distance-transform peak of
    frame-0 GT, returned as (x, y) normalized to [0, 1] for run_pipeline."""
    frame0 = davis.load_frame(clip, 0)
    H, W = frame0.shape[:2]
    gt0 = (davis.load_mask(clip, 0) > 0).astype(np.uint8)
    gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    pts = sample_clicks(gt256, 1)
    cx256, cy256 = pts[0]
    cx, cy = cx256 * (W / 256.0), cy256 * (H / 256.0)
    return cx / W, cy / H


def main():
    for out_name, clip in CLIPS.items():
        cx_norm, cy_norm = simulated_click(clip)
        print(f"[{clip}] simulated click at normalized ({cx_norm:.3f}, {cy_norm:.3f})")
        video_path = demo_mod.run_pipeline(
            clip, cx_norm, cy_norm,
            refine=True, feat_size=1024, use_probe=True,
        )
        dest = OUT_DIR / f"{out_name}.mp4"
        shutil.copy(video_path, dest)
        print(f"[{clip}] -> {dest}")


if __name__ == "__main__":
    main()
