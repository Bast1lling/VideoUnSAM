"""
Stage-2 ImageCut2Video synthetic video generation, adapted from VideoCutLER.

Augmentation pipeline (each frame uses independent random draws — matches
CocoClipDatasetMapper in data_video/dataset_mapper.py):

  1. RandomCrop         "relative_range" — crop size ∈ [crop_min·H, H] × [crop_min·W, W]
  2. ResizeShortestEdge  shortest edge ∈ [320, 480], max 720  (VideoCutLER training defaults)
  3. RandomFlip          horizontal, p = flip_prob
  4. RandomBrightness    intensity ∈ [0.9, 1.1]
  5. RandomContrast      intensity ∈ [0.9, 1.1]
  6. RandomSaturation    intensity ∈ [0.9, 1.1]

Supports two input modes:
  • divide-only  — pass conquer_per_divide={}
  • hierarchical — pass conquer_per_divide={divide_idx: [mask, ...], ...}
    Conquer masks are augmented together with divide masks (same geometric
    transform per frame) and rendered lighter behind their parent divide mask,
    matching the complete output of divide_conquer.py.

All frames are resized back to the original image resolution for video output.
"""

import tempfile

import cv2
import gradio as gr
import numpy as np

from detectron2.data import transforms as T


# ── colour helpers (mirrors app.py palette) ──────────────────────────────────

_PALETTE = [
    [220,  20,  60], [ 30, 144, 255], [ 50, 205,  50], [255, 215,   0],
    [255, 140,   0], [138,  43, 226], [  0, 206, 209], [255, 105, 180],
    [127, 255,   0], [255,  69,   0], [  0, 191, 255], [186,  85, 211],
]

# Bright secondary palette for conquer sub-masks — visually distinct from divide colours.
_CONQUER_PALETTE = [
    [255, 230,  50], [ 50, 255, 160], [255, 100,  50], [180,  50, 255],
    [ 50, 220, 255], [255,  50, 150], [200, 255,  50], [255, 160,  50],
    [ 50, 100, 255], [255,  50, 200], [100, 255,  50], [ 50, 200, 180],
]


def _color(i: int):
    return _PALETTE[i % len(_PALETTE)]


def _conquer_color(divide_idx: int, conquer_idx: int):
    return _CONQUER_PALETTE[(divide_idx * 3 + conquer_idx) % len(_CONQUER_PALETTE)]


def _overlay(image_rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    """Semi-transparent fill with thick contour — used for divide masks."""
    out = image_rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def _overlay_conquer(image_rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5):
    """Stronger fill with white contour — drawn on top of divide masks so sub-masks are visible."""
    out = image_rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, [255, 255, 255], 1)
    return out


# ── frame rendering ──────────────────────────────────────────────────────────

def _render_frame(aug_img: np.ndarray, aug_divide: list, aug_conquer_groups: list) -> np.ndarray:
    """
    Render one augmented frame.

    aug_conquer_groups: list of lists aligned with aug_divide.
                        aug_conquer_groups[i] contains the conquer masks
                        for divide mask i (may be empty).

    Draw order: divide fills first, then conquer masks on top with distinct colours.
    Conquer masks are always subsets of their parent divide region, so they must be
    drawn after (on top of) the divide fill to remain visible.
    """
    frame = aug_img.copy()

    # divide masks — semi-transparent fill as the base layer
    for i, m in enumerate(aug_divide):
        if m.any():
            frame = _overlay(frame, m.astype(np.uint8), _color(i))

    # conquer masks — drawn on top with distinct colours so they're visible inside
    # the already-coloured divide region
    for i, group in enumerate(aug_conquer_groups):
        for j, m in enumerate(group):
            if m.any():
                frame = _overlay_conquer(frame, m.astype(np.uint8), _conquer_color(i, j))

    return frame


# ── augmentation ─────────────────────────────────────────────────────────────

def _build_aug_list(crop_min: float, flip_prob: float) -> T.AugmentationList:
    """
    Build the per-frame augmentation pipeline, matching VideoCutLER's
    CocoClipDatasetMapper / build_augmentation (data_video/augmentation.py).

    crop_min: minimum relative crop fraction (both H and W).
              T.RandomCrop "relative_range" samples size ∈ [crop_min, 1.0] per axis.
    """
    return T.AugmentationList([
        # equivalent to cfg.INPUT.CROP (type=relative_range, size=[crop_min, crop_min])
        T.RandomCrop("relative_range", [crop_min, crop_min]),
        # equivalent to ResizeShortestEdge with min_size=[320,480], max_size=720
        T.ResizeShortestEdge([320, 480], max_size=720, sample_style="range"),
        # equivalent to RandomFlip (horizontal)
        T.RandomFlip(prob=flip_prob, horizontal=True),
        # equivalent to cfg.INPUT.AUGMENTATIONS brightness/contrast/saturation
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.RandomSaturation(0.9, 1.1),
    ])


def _aug_frame(
    image_rgb: np.ndarray,
    masks: list,
    aug_list: T.AugmentationList,
    out_hw: tuple,
) -> tuple:
    """
    Apply one independent augmentation pass to image + flat mask list, then
    resize everything back to out_hw = (H, W).

    Returns (aug_image_rgb uint8, list[bool mask]).
    """
    # detectron2 colour transforms (RandomSaturation) hardcode cv2.COLOR_BGR2HSV internally
    aug_input = T.AugInput(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    tfms = aug_list(aug_input)          # draws fresh random params each call
    aug_img = cv2.cvtColor(aug_input.image, cv2.COLOR_BGR2RGB)

    aug_masks = []
    for m in masks:
        aug_m = tfms.apply_segmentation(m.astype(np.uint8))   # nearest-neighbour
        if aug_m.shape[:2] != out_hw:
            aug_m = (
                cv2.resize(
                    aug_m.astype(np.float32),
                    (out_hw[1], out_hw[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5
            )
        aug_masks.append(aug_m.astype(bool))

    if aug_img.shape[:2] != out_hw:
        aug_img = cv2.resize(aug_img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)

    return aug_img, aug_masks


# ── Gradio event handler ─────────────────────────────────────────────────────

def run_synth_video(
    image_rgb: np.ndarray,
    divide_masks: list,
    conquer_per_divide: dict,
    num_frames: int,
    crop_min: float,
    flip_prob: float,
    fps: int,
):
    """
    Generate a synthetic video by independently augmenting all masks each frame.

    conquer_per_divide: dict mapping divide-mask index (int) to a list of
                        conquer masks for that divide region.
                        Pass {} to use divide masks only.
                        Sparse keys are fine — only the divide masks that have
                        had conquer run will include sub-masks.
    """
    if image_rgb is None or not divide_masks:
        return [], None, "Upload an image and run the Divide phase first."

    H, W = image_rgb.shape[:2]
    n_divide = len(divide_masks)

    # Build per-divide conquer groups (empty list for indices not yet run).
    conquer_groups = [conquer_per_divide.get(i, []) for i in range(n_divide)]

    # Flatten all masks into one list for a single augmentation call per frame.
    # Layout: [divide_0, ..., divide_{n-1}, conquer_0_0, ..., conquer_0_k, conquer_1_0, ...]
    flat_masks = list(divide_masks)
    group_slices = []           # (start, end) in flat_masks for each divide's conquer group
    offset = n_divide
    for group in conquer_groups:
        group_slices.append((offset, offset + len(group)))
        flat_masks.extend(group)
        offset += len(group)

    aug_list = _build_aug_list(crop_min, flip_prob)

    frames = []
    for _ in range(int(num_frames)):
        aug_img, aug_flat = _aug_frame(image_rgb, flat_masks, aug_list, (H, W))

        aug_divide = aug_flat[:n_divide]
        aug_conquer_groups = [
            aug_flat[s:e] for s, e in group_slices
        ]

        frames.append(_render_frame(aug_img, aug_divide, aug_conquer_groups))

    tmp = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    n_conquer = sum(len(g) for g in conquer_groups)
    desc = (
        f"Generated {num_frames} frames — "
        f"{n_divide} divide masks"
        + (f" + {n_conquer} conquer masks across {sum(1 for g in conquer_groups if g)} region(s)." if n_conquer else ".")
    )
    gallery = [(f, f"Frame {i + 1}") for i, f in enumerate(frames)]
    return gallery, tmp, desc


# ── Gradio UI section ────────────────────────────────────────────────────────

def build_synth_video_section(image_input, divide_masks_state, all_conquer_state, status_box):
    """
    Append the Stage-2 Synthetic Video UI into the current gr.Blocks context.
    Must be called inside a `with gr.Blocks() as demo:` block.

    all_conquer_state: gr.State holding a dict {divide_idx: [mask, ...]} accumulated
                       as the user runs the conquer phase on individual divide masks.
    """
    gr.Markdown(
        "## Stage 2: Synthetic Video (ImageCut2Video)\n"
        "Replicates the VideoCutLER `CocoClipDatasetMapper` pipeline: each frame is "
        "independently augmented with **RandomCrop → ResizeShortestEdge[320-480] → "
        "RandomFlip → brightness / contrast / saturation jitter [0.9, 1.1]**. "
        "All masks (divide + any conquer masks computed so far) are augmented together "
        "with the same geometric transform per frame. "
        "Conquer sub-masks are drawn lighter, behind their parent divide mask."
    )
    with gr.Accordion("Video parameters", open=True):
        with gr.Row():
            num_frames_slider = gr.Slider(2, 12, step=1, value=4, label="Number of frames")
            fps_slider        = gr.Slider(1, 12, step=1, value=4, label="FPS")
        with gr.Row():
            crop_min_slider  = gr.Slider(
                0.3, 1.0, step=0.05, value=0.5,
                label='Min crop fraction — "relative_range" samples size ∈ [this·H, H] × [this·W, W]',
            )
            flip_prob_slider = gr.Slider(0.0, 1.0, step=0.1, value=0.5, label="Flip probability")

    video_btn = gr.Button("Generate Synthetic Video", variant="primary")
    with gr.Row():
        video_gallery = gr.Gallery(
            label="Synthetic frames", columns=6, height=320, type="numpy"
        )
        video_output = gr.Video(label="Video playback", height=320)

    video_btn.click(
        run_synth_video,
        inputs=[
            image_input, divide_masks_state, all_conquer_state,
            num_frames_slider, crop_min_slider, flip_prob_slider, fps_slider,
        ],
        outputs=[video_gallery, video_output, status_box],
    )
