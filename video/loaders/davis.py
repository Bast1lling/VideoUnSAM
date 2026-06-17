"""Minimal DAVIS 2017 frame/mask loader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image


_DAVIS_ROOT = Path("/home/nilsc/VideoUnSAM/datasets/davis/DAVIS")
_JPEG = _DAVIS_ROOT / "JPEGImages" / "480p"
_ANN = _DAVIS_ROOT / "Annotations" / "480p"


def list_clips() -> list[str]:
    return sorted(p.name for p in _JPEG.iterdir() if p.is_dir())


def num_frames(clip: str) -> int:
    return len(list((_JPEG / clip).glob("*.jpg")))


def load_frame(clip: str, idx: int) -> np.ndarray:
    """RGB uint8 [H, W, 3]."""
    bgr = cv2.imread(str(_JPEG / clip / f"{idx:05d}.jpg"))
    if bgr is None:
        raise FileNotFoundError(_JPEG / clip / f"{idx:05d}.jpg")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(clip: str, idx: int, instance_id: int | None = None) -> np.ndarray:
    """uint8 [H, W]. If instance_id is given, returns a binary mask; otherwise returns
    the raw multi-instance label map (0 = background, 1..N = instances)."""
    path = _ANN / clip / f"{idx:05d}.png"
    label = np.array(Image.open(path))
    if instance_id is None:
        return label
    return (label == instance_id).astype(np.uint8)


def instance_ids(clip: str, idx: int = 0) -> list[int]:
    """Non-background instance IDs present in a given frame."""
    label = load_mask(clip, idx)
    return [int(i) for i in np.unique(label) if i != 0]
