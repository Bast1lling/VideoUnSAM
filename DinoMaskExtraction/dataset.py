"""
ImageNet-1k dataset for DINOv3 feature extraction.

Directory layout (ImageFolder-style)::

    /home/sebastian/data/imagenet1k/images/
        train/
            0000/  *.JPEG
            0001/  *.JPEG
            ...
            0999/
        validation/
            0000/  *.JPEG
            ...

Each numbered sub-directory is a class; the integer name is used as the label.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

_DEFAULT_ROOT = "/home/sebastian/data/imagenet1k/images"
_IMG_EXTS = (".jpeg", ".jpg", ".png")


class ImageNet1k(Dataset):
    """Iterate over the ImageNet-1k images.

    Parameters
    ----------
    root:
        Path to ``.../imagenet1k/images``.
    split:
        ``"train"`` or ``"validation"``.
    transform:
        Optional callable applied to the loaded ``PIL.Image`` (RGB). When
        ``None`` the raw PIL image is returned, which is what
        ``backbone.extract_feature_matrix`` expects.

    Each item is a ``(image, label)`` tuple where ``label`` is the integer
    class folder name (0-999).
    """

    def __init__(self, root: str = _DEFAULT_ROOT, split: str = "train", transform=None):
        self.root = Path(root) / split
        if not self.root.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.root}")
        self.transform = transform

        self.classes = sorted(
            d.name for d in os.scandir(self.root) if d.is_dir()
        )
        self.class_to_idx = {c: int(c) for c in self.classes}

        self.samples: list[tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            label = self.class_to_idx[cls]
            for entry in os.scandir(cls_dir):
                if entry.is_file() and entry.name.lower().endswith(_IMG_EXTS):
                    self.samples.append((entry.path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":
    ds = ImageNet1k(split="train")
    print(f"train: {len(ds)} images across {len(ds.classes)} classes")
    img, label = ds[0]
    print(f"sample 0: {img.size} px, label={label}")
