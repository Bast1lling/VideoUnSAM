"""
DiCoVideoDataset — synthetic video clips from precomputed DiCo masks.

Each sample is a dict with T independently augmented views of the same image,
where all T frames share the same set of N instances (1-to-1 correspondence by
index, indicated by gt_ids).  By default an instance only needs to be visible in
the chosen *prompt frame*; in the other frames it may be partially or fully
absent (empty mask), so the model learns to emit empty masks where the object
has left the frame.  Pass require_all_frames=True for the legacy "visible in
every frame" behaviour.

Sample format
-------------
    images       : FloatTensor (T, 3, H, W)  ImageNet-normalised, values in ~[-2, 2]
    masks        : BoolTensor  (T, N, H, W)  per-frame masks; may be empty in
                                             non-prompt frames
    gt_ids       : LongTensor  (N,)          instance IDs 0..N-1 (same across frames)
    prompt_frame : int                       frame the instances are guaranteed
                                             visible in (where to anchor the prompt)
    path         : str                       absolute path to the source image

Augmentation pipeline (independent per frame, matching VideoCutLER):
    RandomCrop "relative_range"
    → ResizeShortestEdge [short_min, short_max] (max long-edge = max_size)
    → RandomFlip (horizontal)
    → RandomBrightness / RandomContrast / RandomSaturation

All masks are augmented with the same geometric transform as their frame's
image and then resized back to the original image resolution so that all frames
have identical spatial dimensions — this makes stacking into a (T, N, H, W)
tensor straightforward without padding.
"""
from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from detectron2.data import transforms as T


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DiCoVideoDataset(Dataset):
    """
    Synthetic video dataset built from precomputed Divide-and-Conquer masks.

    Parameters
    ----------
    mask_dir            : directory containing .npz files from precompute_masks.py
    num_frames          : number of independently augmented frames per clip (T)
    use_divide          : include divide-phase masks in the pool
    use_conquer         : include conquer-phase masks in the pool
    crop_min            : minimum crop fraction for RandomCrop (both spatial axes)
    flip_prob           : probability of horizontal flip per frame
    short_edge_lengths  : (min, max) for ResizeShortestEdge sampling
    max_size            : maximum long-edge length after resize
    brightness_range    : (lo, hi) intensity multiplier for RandomBrightness
    contrast_range      : (lo, hi) intensity multiplier for RandomContrast
    saturation_range    : (lo, hi) intensity multiplier for RandomSaturation
    min_mask_area       : minimum pixel count (in the original resolution) for a
                          mask to be considered non-empty after augmentation
    max_retries         : number of fresh augmentation draws to attempt before
                          giving up on finding valid masks for a sample
    image_size          : if set, output every frame + masks at a fixed
                          (image_size, image_size); None keeps each image's own
                          resolution (variable across the dataset)
    """

    def __init__(
        self,
        mask_dir: str | None = None,
        num_frames: int = 2,
        use_divide: bool = True,
        use_conquer: bool = True,
        crop_min: float = 0.5,
        flip_prob: float = 0.5,
        short_edge_lengths: tuple[int, int] = (320, 480),
        max_size: int = 720,
        brightness_range: tuple[float, float] = (0.9, 1.1),
        contrast_range: tuple[float, float] = (0.9, 1.1),
        saturation_range: tuple[float, float] = (0.9, 1.1),
        min_mask_area: int = 64,
        max_retries: int = 5,
        entries: "list[str] | None" = None,
        recursive: bool = False,
        require_all_frames: bool = False,
        image_size: int | None = None,
    ):
        # Either an explicit list of .npz paths (for custom train/val splits) or
        # a directory to glob (optionally recursively, for nested ImageNet dirs).
        if entries is not None:
            self.entries = sorted(entries)
        elif mask_dir is not None:
            pattern = os.path.join(mask_dir, "**", "*.npz") if recursive \
                else os.path.join(mask_dir, "*.npz")
            self.entries = sorted(glob.glob(pattern, recursive=recursive))
        else:
            raise ValueError("Provide either `mask_dir` or `entries`.")
        if not self.entries:
            raise ValueError(
                f"No .npz files found (mask_dir={mask_dir!r}, recursive={recursive}, "
                f"entries={'<list>' if entries is not None else None})."
            )

        self.num_frames         = num_frames
        self.use_divide         = use_divide
        self.use_conquer        = use_conquer
        self.crop_min           = crop_min
        self.flip_prob          = flip_prob
        self.short_edge_lengths = list(short_edge_lengths)
        self.max_size           = max_size
        self.brightness_range   = brightness_range
        self.contrast_range     = contrast_range
        self.saturation_range   = saturation_range
        self.min_mask_area      = min_mask_area
        self.max_retries        = max_retries
        # If set, every frame + its masks are produced at a fixed
        # (image_size, image_size) instead of the source image's resolution.
        # Fixed sizes make every batch uniform → no collate padding waste, stable
        # GPU memory (no allocator fragmentation), and cacheable cuDNN kernels.
        self.image_size         = image_size
        # If True, an instance must be visible in *every* frame (legacy behaviour).
        # If False (default), an instance only needs to be visible in the chosen
        # prompt frame; it may be partially or fully absent (empty mask) in other
        # frames, so the model learns to emit empty masks where the object left.
        self.require_all_frames = require_all_frames

    def __len__(self) -> int:
        return len(self.entries)

    # ── augmentation ──────────────────────────────────────────────────────────

    def _build_aug_list(self) -> T.AugmentationList:
        return T.AugmentationList([
            T.RandomCrop("relative_range", [self.crop_min, self.crop_min]),
            T.ResizeShortestEdge(
                self.short_edge_lengths, max_size=self.max_size, sample_style="range"
            ),
            T.RandomFlip(prob=self.flip_prob, horizontal=True),
            T.RandomBrightness(*self.brightness_range),
            T.RandomContrast(*self.contrast_range),
            T.RandomSaturation(*self.saturation_range),
        ])

    def _augment_frame(
        self,
        image_rgb: np.ndarray,
        masks: list[np.ndarray],
        aug_list: T.AugmentationList,
        out_hw: tuple[int, int],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Apply one independent augmentation draw to the image and all masks,
        then resize everything back to *out_hw* = (H, W).

        detectron2's colour transforms (RandomSaturation) internally use BGR,
        so we convert colour space before and after.
        """
        aug_input = T.AugInput(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        tfms      = aug_list(aug_input)   # fresh random parameters
        aug_img   = cv2.cvtColor(aug_input.image, cv2.COLOR_BGR2RGB)

        aug_masks: list[np.ndarray] = []
        for m in masks:
            aug_m = tfms.apply_segmentation(m.astype(np.uint8))
            if aug_m.shape[:2] != out_hw:
                aug_m = cv2.resize(
                    aug_m.astype(np.float32),
                    (out_hw[1], out_hw[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                aug_m = (aug_m > 0.5).astype(np.uint8)
            aug_masks.append(aug_m)

        if aug_img.shape[:2] != out_hw:
            aug_img = cv2.resize(aug_img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)

        return aug_img, aug_masks

    @staticmethod
    def _to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
        img = image_rgb.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))

    # ── item loading ──────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.entries[idx], allow_pickle=True)

        # Decode image path (stored as numpy bytes scalar).
        raw_path   = data["image_path"].item()
        image_path = raw_path.decode() if isinstance(raw_path, bytes) else str(raw_path)

        masks_all = data["masks"]      # (N, H, W) uint8
        mask_type = data["mask_type"]  # (N,) int8  — 0=divide, 1=conquer

        # Filter mask pool by type.
        keep = np.zeros(len(mask_type), dtype=bool)
        if self.use_divide:
            keep |= (mask_type == 0)
        if self.use_conquer:
            keep |= (mask_type == 1)
        masks_all = masks_all[keep]

        if masks_all.shape[0] == 0:
            raise RuntimeError(f"No masks remain after type filtering for {image_path}")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W      = image_rgb.shape[:2]

        masks_list: list[np.ndarray] = [masks_all[i] for i in range(masks_all.shape[0])]
        N = len(masks_list)

        # Output resolution: fixed square if image_size is set (uniform batches),
        # otherwise the source image's own (H, W).
        out_hw = (self.image_size, self.image_size) if self.image_size else (H, W)

        for _ in range(self.max_retries):
            aug_list = self._build_aug_list()

            # The prompt is anchored to one frame; an instance only needs to be
            # visible *there* to be promptable.  Pick it fresh each retry.
            prompt_frame = random.randrange(self.num_frames)

            frames_img:   list[np.ndarray]       = []
            frames_masks: list[list[np.ndarray]] = []

            for _ in range(self.num_frames):
                f_img, f_masks = self._augment_frame(image_rgb, masks_list, aug_list, out_hw)
                frames_img.append(f_img)
                frames_masks.append(f_masks)

            if self.require_all_frames:
                # Legacy: keep masks above min_mask_area in *every* frame.
                valid = [
                    i for i in range(N)
                    if all(
                        int(frames_masks[t][i].sum()) >= self.min_mask_area
                        for t in range(self.num_frames)
                    )
                ]
            else:
                # Keep masks visible in the prompt frame; they may be partially or
                # fully absent (empty) in the other frames — those become empty
                # targets the model must learn to predict.
                valid = [
                    i for i in range(N)
                    if int(frames_masks[prompt_frame][i].sum()) >= self.min_mask_area
                ]
            if not valid:
                continue   # retry with a different augmentation / prompt frame

            images_t = torch.stack(
                [self._to_tensor(f) for f in frames_img]
            )  # (T, 3, H, W)

            masks_t = torch.from_numpy(
                np.stack(
                    [np.stack([frames_masks[t][i] for i in valid]) for t in range(self.num_frames)]
                )
            ).bool()  # (T, N_valid, H, W) — may be empty in non-prompt frames

            return {
                "images":       images_t,
                "masks":        masks_t,
                "gt_ids":       torch.arange(len(valid), dtype=torch.long),
                "prompt_frame": prompt_frame,
                "path":         image_path,
            }

        # Fallback after all retries: return zero masks at the output resolution.
        # Downstream collate_fn should handle empty-mask samples.
        fb_img = image_rgb
        if fb_img.shape[:2] != out_hw:
            fb_img = cv2.resize(fb_img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
        return {
            "images":       torch.stack([self._to_tensor(fb_img)] * self.num_frames),
            "masks":        torch.zeros(self.num_frames, 0, *out_hw, dtype=torch.bool),
            "gt_ids":       torch.zeros(0, dtype=torch.long),
            "prompt_frame": 0,
            "path":         image_path,
        }


# ── helpers for training ──────────────────────────────────────────────────────

def gather_npz(mask_dir: str, recursive: bool = True) -> "list[str]":
    """Return a sorted list of all .npz files under *mask_dir*."""
    pattern = os.path.join(mask_dir, "**", "*.npz") if recursive \
        else os.path.join(mask_dir, "*.npz")
    return sorted(glob.glob(pattern, recursive=recursive))


def _round_up(x: int, k: int = 32) -> int:
    return ((x + k - 1) // k) * k


def collate_video_clips(batch: "list[dict]", size_divisibility: int = 32) -> dict:
    """
    Collate variable-size video clips into a padded batch.

    Each clip has images (T, 3, H, W) and masks (T, N_b, H, W) where H, W and
    N_b vary across clips.  Images/masks are zero-padded to a common
    (maxH, maxW) rounded up to *size_divisibility* (so the backbone sees sizes
    divisible by 32 and the model's zero spatial-padding-mask path is valid).

    Returns
    -------
    dict with:
      images        : FloatTensor (B, T, 3, Hp, Wp)   — video-first when flattened
      masks         : list[BoolTensor]  length B, each (T, N_b, Hp, Wp)
      gt_ids        : list[LongTensor]  length B
      prompt_frames : list[int]   length B, the prompt frame per clip
      paths         : list[str]
      sizes         : list[(H, W)]      original (pre-pad) clip sizes
    Clips with zero masks are dropped; returns {} if the whole batch is empty.
    """
    batch = [b for b in batch if b["masks"].shape[1] > 0]
    if not batch:
        return {}

    T = batch[0]["images"].shape[0]
    Hp = _round_up(max(b["images"].shape[-2] for b in batch), size_divisibility)
    Wp = _round_up(max(b["images"].shape[-1] for b in batch), size_divisibility)

    images, masks, gt_ids, prompt_frames, paths, sizes = [], [], [], [], [], []
    for b in batch:
        img = b["images"]                       # (T, 3, H, W)
        m   = b["masks"]                         # (T, N, H, W)
        H, W = img.shape[-2:]
        img_p = img.new_zeros((T, 3, Hp, Wp))
        img_p[..., :H, :W] = img
        m_p = m.new_zeros((T, m.shape[1], Hp, Wp))
        m_p[..., :H, :W] = m
        images.append(img_p)
        masks.append(m_p)
        gt_ids.append(b["gt_ids"])
        prompt_frames.append(int(b.get("prompt_frame", 0)))
        paths.append(b["path"])
        sizes.append((H, W))

    return {
        "images": torch.stack(images, 0),   # (B, T, 3, Hp, Wp)
        "masks":  masks,                     # list of (T, N_b, Hp, Wp)
        "gt_ids": gt_ids,
        "prompt_frames": prompt_frames,
        "paths":  paths,
        "sizes":  sizes,
    }
