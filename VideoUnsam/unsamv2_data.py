"""Loading helpers for the UnSAMv2 granular-mask training data.

The dataset is one COCO-style JSON per image (``sa_<id>.json``) holding every
mask UnSAMv2 produced for that image, annotated with the extra fields the
method cares about: ``granularity``, ``instance_id`` and the
``is_part`` / ``is_divide`` / ``is_conquer`` flags.

Kept separate from the Gradio app so the video experiments can import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

DEFAULT_ANN_DIR = Path("/home/sebastian/Documents/Uni/Practical/UnSAMv2_train_data")

# Places the SA-1B shard may have been extracted to; first hit wins.
IMAGE_DIR_CANDIDATES = [
    Path("/home/sebastian/Documents/Uni/Practical/UnSAMv2_train_images"),
    Path("/home/sebastian/Documents/Uni/Practical/UnSAM/images"),
    Path("/home/sebastian/Documents/Uni/Practical/UnSAM/sa_000023"),
    DEFAULT_ANN_DIR / "images",
]


@dataclass
class Annotation:
    id: int
    area: float
    bbox: tuple[float, float, float, float]  # xywh, full resolution
    granularity: float
    instance_id: int
    is_part: bool
    is_divide: bool
    is_conquer: bool
    rle: dict = field(repr=False)

    @property
    def kind(self) -> str:
        """Short human label for the divide-and-conquer role of this mask."""
        if not self.is_part:
            return "instance"
        return "part+conquer" if self.is_conquer else "part"


@dataclass
class Sample:
    image_id: int
    file_name: str
    width: int
    height: int
    annotations: list[Annotation]
    stats: dict


def list_image_ids(ann_dir: Path | str = DEFAULT_ANN_DIR) -> list[int]:
    """Sorted image ids that have an annotation file."""
    ann_dir = Path(ann_dir)
    ids = []
    for p in ann_dir.glob("sa_*.json"):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


@lru_cache(maxsize=64)
def load_sample(ann_dir: str, image_id: int) -> Sample:
    with open(Path(ann_dir) / f"sa_{image_id}.json") as fh:
        raw = json.load(fh)

    img = raw["image"]
    anns = [
        Annotation(
            id=a["id"],
            area=float(a["area"]),
            bbox=tuple(a["bbox"]),
            granularity=float(a["granularity"]),
            instance_id=int(a["instance_id"]),
            is_part=bool(a["is_part"]),
            is_divide=bool(a["is_divide"]),
            is_conquer=bool(a["is_conquer"]),
            rle=a["segmentation"],
        )
        for a in raw["annotations"]
    ]
    return Sample(
        image_id=img["id"],
        file_name=img["file_name"],
        width=img["width"],
        height=img["height"],
        annotations=anns,
        stats=raw.get("instance_part_stats", {}),
    )


def find_image_dir(explicit: str | None = None) -> Path | None:
    """Resolve where the SA-1B JPEGs live, or None if not downloaded yet."""
    candidates = [Path(explicit)] if explicit else []
    candidates += IMAGE_DIR_CANDIDATES
    for c in candidates:
        if c.is_dir() and any(c.glob("sa_*.jpg")):
            return c
    return None


def decode_mask(ann: Annotation) -> np.ndarray:
    """Decode one RLE to a full-resolution boolean array."""
    rle = dict(ann.rle)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle).astype(bool)


def _shrink(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour downscale of a boolean mask to (w, h)."""
    if (mask.shape[1], mask.shape[0]) == size:
        return mask
    small = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    return np.asarray(small) > 127


def display_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


@lru_cache(maxsize=8)
def decode_all(ann_dir: str, image_id: int, max_side: int) -> list[np.ndarray]:
    """All masks of an image, decoded once and downscaled for display.

    Decoding happens one mask at a time and each is shrunk immediately -- a
    full-resolution stack of ~150 masks at 1500x2258 would be ~500 MB.
    """
    sample = load_sample(ann_dir, image_id)
    size = display_size(sample.width, sample.height, max_side)
    return [_shrink(decode_mask(a), size) for a in sample.annotations]


def load_image(image_dir: Path | None, sample: Sample, max_side: int) -> tuple[np.ndarray, bool]:
    """Return (RGB array at display size, found). Falls back to a grey canvas."""
    w, h = display_size(sample.width, sample.height, max_side)
    if image_dir is not None:
        path = Path(image_dir) / sample.file_name
        if path.is_file():
            img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
            return np.asarray(img), True
    return np.full((h, w, 3), 60, dtype=np.uint8), False
