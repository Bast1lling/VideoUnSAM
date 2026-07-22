"""Minimal video frame/mask loader.

Originally written for DAVIS 2017. It now also supports YouTube-VIS-style
layouts, where a clip is a directory of JPEG frames whose filenames are *not*
necessarily contiguous (e.g. ``00000.jpg, 00005.jpg, 00010.jpg`` — every 5th
frame). Frames are addressed by a sequential index ``0 .. num_frames-1`` that is
mapped onto the sorted list of files on disk, so sparse numbering just works.

Dataset root resolution (first match wins):
  1. ``$VIDEO_DATA_ROOT`` if set
  2. the first existing path in ``_CANDIDATE_ROOTS`` below

The chosen root must contain either ``JPEGImages/<clip>/*.jpg`` (YouTube-VIS)
or ``JPEGImages/480p/<clip>/*.jpg`` (DAVIS). Masks are only available for the
DAVIS layout (``Annotations/480p/<clip>/*.png``); YouTube-VIS stores its
annotations as COCO json, so :func:`load_mask` / :func:`instance_ids` raise a
clear error there. The Gradio demo only needs frames, so that is fine.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image


# Searched in order; override with VIDEO_DATA_ROOT to point elsewhere.
_CANDIDATE_ROOTS = [
    "/home/sebastian/data/YouTubeVIS2022/valid",
    "/home/sebastian/data/YouTubeVIS2022/train",
    "/home/nilsc/VideoUnSAM/datasets/davis/DAVIS",
]


def _resolve_root() -> Path:
    candidates = []
    env = os.environ.get("VIDEO_DATA_ROOT")
    if env:
        candidates.append(Path(env))
    candidates += [Path(p) for p in _CANDIDATE_ROOTS]
    for c in candidates:
        if (c / "JPEGImages").is_dir():
            return c
    raise FileNotFoundError(
        "No video dataset root found. Set VIDEO_DATA_ROOT to a directory "
        "containing a 'JPEGImages/' tree. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


_ROOT = _resolve_root()
# DAVIS nests frames under JPEGImages/480p; YouTube-VIS puts them directly under JPEGImages.
_JPEG = _ROOT / "JPEGImages" / "480p"
if not _JPEG.is_dir():
    _JPEG = _ROOT / "JPEGImages"
_ANN = _ROOT / "Annotations" / "480p"


def list_clips() -> list[str]:
    return sorted(p.name for p in _JPEG.iterdir() if p.is_dir())


@lru_cache(maxsize=None)
def _frame_files(clip: str) -> tuple[Path, ...]:
    """Sorted frame paths for a clip. Handles arbitrary / sparse numbering."""
    d = _JPEG / clip
    files = sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No frames found for clip {clip!r} in {d}")
    return tuple(files)


def num_frames(clip: str) -> int:
    return len(_frame_files(clip))


def load_frame(clip: str, idx: int) -> np.ndarray:
    """RGB uint8 [H, W, 3]. ``idx`` is a sequential 0-based frame index."""
    files = _frame_files(clip)
    if not 0 <= idx < len(files):
        raise IndexError(f"frame {idx} out of range for clip {clip!r} ({len(files)} frames)")
    bgr = cv2.imread(str(files[idx]))
    if bgr is None:
        raise FileNotFoundError(files[idx])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(clip: str, idx: int, instance_id: int | None = None) -> np.ndarray:
    """uint8 [H, W]. If instance_id is given, returns a binary mask; otherwise returns
    the raw multi-instance label map (0 = background, 1..N = instances).

    Only available for the DAVIS PNG-annotation layout."""
    if not _ANN.is_dir():
        raise NotImplementedError(
            f"No PNG annotations under {_ANN}. This dataset (e.g. YouTube-VIS) stores "
            "masks as COCO json, which this loader does not read. The demo does not "
            "need masks; only eval against DAVIS ground truth does."
        )
    files = _frame_files(clip)
    stem = files[idx].stem
    path = _ANN / clip / f"{stem}.png"
    label = np.array(Image.open(path))
    if instance_id is None:
        return label
    return (label == instance_id).astype(np.uint8)


def instance_ids(clip: str, idx: int = 0) -> list[int]:
    """Non-background instance IDs present in a given frame."""
    label = load_mask(clip, idx)
    return [int(i) for i in np.unique(label) if i != 0]
