"""
Gradio demo — Divide-and-Conquer → Synthetic Video → Promptable Segmentation.

Tab 1  Generate DiCo Video:
  A) Live:        load random / upload image → run DiCo mask extraction → augmented
                  video with divide+conquer overlays.
  B) Precomputed: load a random precomputed validation .npz → same overlay video.
  Either path stores the raw (no-overlay) augmented frames in shared state for Tab 2.

Tab 2  Segment Video:
  Paint point prompts on the raw frames from Tab 1, then run VideoIMaskDINOHead.
  Displays per-frame mask overlays and a short output video.

Run from the repo root:
    python promptable_video_segmentation/gradio_dico_video.py
    python promptable_video_segmentation/gradio_dico_video.py --port 7863
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image
import torch

warnings.filterwarnings("ignore")

# ── path setup (mirrors gradio_video_seg.py) ──────────────────────────────────
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
# Order matters: promptable_segmentation must be first so bare `from model.body…`
# inside video_mask_dino.py resolves to promptable_segmentation/model, not here.
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "promptable_segmentation"))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))

from detectron2.data import transforms as T  # available in repo env


# ── constants ─────────────────────────────────────────────────────────────────

_DEFAULT_IMAGE_DIR = "/home/sebastian/data/imagenet1k/images"
_IMAGE_EXTS = ("*.JPEG", "*.jpeg", "*.jpg", "*.png", "*.webp")

_METHODS = {
    "original": "promptable_video_segmentation.divide_and_conquer_original",
    "new":      "promptable_video_segmentation.divide_and_conquer",
}
_METHOD_MASKS_DIR = {
    "original": "/home/sebastian/data/imagenet1k/masksV1",
    "new":      "/home/sebastian/data/imagenet1k/masksV3",
}
_PRECOMPUTE_SPLIT = "validation"

# Segmentation model constants
_PIXEL_MEAN   = np.array([123.675, 116.280, 103.530], dtype=np.float32)
_PIXEL_STD    = np.array([58.395,  57.120,  57.375],  dtype=np.float32)
_MODEL_STRIDE = 32
_MIN_EDGE     = 640

# Colour palettes
_DIVIDE_PALETTE: list[list[int]] = [
    [220,  20,  60], [ 30, 144, 255], [ 50, 205,  50], [255, 215,   0],
    [255, 140,   0], [138,  43, 226], [  0, 206, 209], [255, 105, 180],
    [127, 255,   0], [255,  69,   0], [  0, 191, 255], [186,  85, 211],
]
_CONQUER_PALETTE: list[list[int]] = [
    [255, 230,  50], [ 50, 255, 160], [255, 100,  50], [180,  50, 255],
    [ 50, 220, 255], [255,  50, 150], [200, 255,  50], [255, 160,  50],
    [ 50, 100, 255], [255,  50, 200], [100, 255,  50], [ 50, 200, 180],
]
# Segmentation result colours (per prompt) and click-dot colours
_SEG_PALETTE: list[list[int]] = _DIVIDE_PALETTE
_DOT_PALETTE = [
    (255,   0,   0), (  0,   0, 255), (  0, 200,   0), (255, 165,   0),
    (255,   0, 255), (  0, 255, 255), (128,   0, 128), (255, 128,   0),
]


# ── DiCo method module cache ──────────────────────────────────────────────────

_dico_module_cache: dict[str, object] = {}


def _get_module(method: str):
    if method not in _dico_module_cache:
        import importlib
        _dico_module_cache[method] = importlib.import_module(_METHODS[method])
    return _dico_module_cache[method]


# ── segmentation model cache ─────────────────────────────────────────────────

_seg_model_cache: dict = {}


def _to_model_hw(h: int, w: int) -> tuple[int, int]:
    scale = _MIN_EDGE / min(h, w)
    new_h = int(round(h * scale / _MODEL_STRIDE)) * _MODEL_STRIDE
    new_w = int(round(w * scale / _MODEL_STRIDE)) * _MODEL_STRIDE
    return new_h, new_w


def _load_seg_model(config_path: str, weights_path: str, device: str,
                    finetuned_path: str = ""):
    """
    Build + cache VideoIMaskDINOHead.

    Always loads the baseline (backbone + full image-model head).  If
    *finetuned_path* is given, the finetuned checkpoint (temporal layers + any
    unfrozen heads, saved by train_video.py) is overlaid on top with
    strict=False, so picking a checkpoint runs *that* finetuned model.
    """
    key = (config_path, weights_path, device, finetuned_path)
    if _seg_model_cache.get("_key") == key:
        return _seg_model_cache["backbone"], _seg_model_cache["head"]

    from promptable_segmentation.utils.arguments import load_opt_from_config_file
    from promptable_segmentation.model.backbone import build_backbone
    from semantic_sam.language import build_language_encoder
    from promptable_video_segmentation.model.video_mask_dino import (
        VideoIMaskDINOHead, load_image_weights,
    )

    opt = load_opt_from_config_file(config_path)
    # Disable the YAML auto-loader: baseline.pth keys are prefixed
    # ("model.backbone.*") and the swin loader expects bare names, so it
    # silently leaves the backbone random. We load it manually below.
    opt["MODEL"]["BACKBONE"]["LOAD_PRETRAINED"] = False

    backbone = build_backbone(opt)
    backbone.to(device)

    lang_encoder = build_language_encoder(opt)
    if lang_encoder is not None:
        lang_encoder.to(device).eval()

    head = VideoIMaskDINOHead(
        opt, backbone.output_shape(), lang_encoder,
        {"task_switch": {"bbox": True, "mask": True}},
    ).to(device)

    if os.path.exists(weights_path):
        # Backbone: strip "model.backbone." prefix
        ckpt = torch.load(weights_path, map_location=device)
        sd = ckpt.get("model", ckpt)
        prefix = "model.backbone."
        bb_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        backbone.load_state_dict(bb_sd, strict=False)
        # Head: load_image_weights already handles "model.sem_seg_head.*" prefix
        load_image_weights(head, weights_path, map_location=device)
    else:
        print(f"[warn] weights not found: {weights_path!r} — using random init")

    # Overlay the finetuned checkpoint (temporal layers + any unfrozen params).
    if finetuned_path and os.path.exists(finetuned_path):
        ft = torch.load(finetuned_path, map_location=device)
        ft_sd = ft.get("model", ft)
        missing, unexpected = head.load_state_dict(ft_sd, strict=False)
        print(f"[seg] overlaid finetuned ckpt {os.path.basename(finetuned_path)}: "
              f"{len(ft_sd)} tensors (epoch={ft.get('epoch', '?')}, "
              f"{len(unexpected)} unexpected)")
    elif finetuned_path:
        print(f"[warn] finetuned ckpt not found: {finetuned_path!r}")

    backbone.eval()
    head.eval()
    _seg_model_cache.update(_key=key, backbone=backbone, head=head)
    return backbone, head


def _list_checkpoints(ckpt_dir: str) -> "list[str]":
    """Return sorted .pth files under *ckpt_dir* (recursively)."""
    root = Path(ckpt_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*.pth"))


_BASELINE_CHOICE = "— baseline only (no finetuning) —"


# ── general utilities ─────────────────────────────────────────────────────────

def _to_rgb(image) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return np.ascontiguousarray(image[:, :, :3])
    return np.array(Image.fromarray(np.array(image)).convert("RGB"))


def _random_image(image_dir: str) -> np.ndarray | None:
    root = Path(image_dir)
    if not root.exists():
        return None
    imgs: list[Path] = []
    for pat in _IMAGE_EXTS:
        imgs.extend(root.rglob(pat))
    if not imgs:
        return None
    return np.array(Image.open(random.choice(imgs)).convert("RGB"))


def _default_masks_dir(method: str) -> str:
    return _METHOD_MASKS_DIR.get(method, "")


def _load_image_rgb(path: str) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("RGB"))


def _reconstruct_masks(npz_path: Path):
    data       = np.load(npz_path, allow_pickle=False)
    masks      = data["masks"]
    mask_type  = data["mask_type"]
    divide_idx = data["divide_idx"]
    raw = data["image_path"]
    try:
        img_path = raw.item().decode("utf-8")
    except (AttributeError, UnicodeDecodeError):
        img_path = str(raw)
    divide_masks = [masks[i] for i in range(len(mask_type)) if mask_type[i] == 0]
    conquer_per_divide: dict[int, list[np.ndarray]] = {}
    for i in range(len(mask_type)):
        if mask_type[i] == 1:
            conquer_per_divide.setdefault(int(divide_idx[i]), []).append(masks[i])
    return img_path, divide_masks, conquer_per_divide


# ── augmentation + frame rendering ───────────────────────────────────────────

def _make_aug_list(crop_min: float, flip_prob: float) -> T.AugmentationList:
    return T.AugmentationList([
        T.RandomCrop("relative_range", [crop_min, crop_min]),
        T.ResizeShortestEdge([320, 480], max_size=720, sample_style="range"),
        T.RandomFlip(prob=flip_prob, horizontal=True),
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.RandomSaturation(0.9, 1.1),
    ])


def _overlay(img: np.ndarray, mask: np.ndarray,
             color: list, alpha: float, ct: int) -> np.ndarray:
    out = img.copy()
    fg  = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if ct > 0:
        ctrs, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, list(color), ct)
    return out


def _generate_clip(
    image_rgb: np.ndarray,
    divide_masks: list[np.ndarray],
    conquer_per_divide: dict[int, list[np.ndarray]],
    num_frames: int,
    crop_min: float,
    flip_prob: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate *num_frames* independently augmented views of image_rgb + masks.

    Returns
    -------
    raw_frames     : list of (H, W, 3) uint8 RGB — no mask overlays, for model input
    display_frames : list of (H, W, 3) uint8 RGB — with DiCo mask overlays, for Tab 1 gallery
    """
    H, W = image_rgb.shape[:2]
    n_divide = len(divide_masks)
    conquer_groups = [conquer_per_divide.get(i, []) for i in range(n_divide)]

    # Flatten masks so they all get the same geometric transform per frame.
    flat_masks: list[np.ndarray] = list(divide_masks)
    group_slices: list[tuple[int, int]] = []
    offset = n_divide
    for g in conquer_groups:
        group_slices.append((offset, offset + len(g)))
        flat_masks.extend(g)
        offset += len(g)

    aug_list = _make_aug_list(crop_min, flip_prob)
    raw_frames:     list[np.ndarray] = []
    display_frames: list[np.ndarray] = []

    for _ in range(num_frames):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        aug_input = T.AugInput(bgr)
        tfm = aug_list(aug_input)
        raw = cv2.cvtColor(aug_input.image, cv2.COLOR_BGR2RGB)
        if raw.shape[:2] != (H, W):
            raw = cv2.resize(raw, (W, H), cv2.INTER_LINEAR)

        aug_flat: list[np.ndarray] = []
        for m in flat_masks:
            am = tfm.apply_segmentation(m.astype(np.uint8))
            if am.shape[:2] != (H, W):
                am = (cv2.resize(am.astype(np.float32), (W, H),
                                 cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
            aug_flat.append(am)

        aug_divide  = aug_flat[:n_divide]
        aug_conquer = [aug_flat[s:e] for s, e in group_slices]

        display = raw.copy()
        for i, m in enumerate(aug_divide):
            if m.any():
                display = _overlay(display, m,
                                   _DIVIDE_PALETTE[i % len(_DIVIDE_PALETTE)], 0.45, 2)
        for i, cg in enumerate(aug_conquer):
            for j, m in enumerate(cg):
                if m.any():
                    c = _CONQUER_PALETTE[(i * 3 + j) % len(_CONQUER_PALETTE)]
                    display = _overlay(display, m, c, 0.5, 1)

        raw_frames.append(raw)
        display_frames.append(display)

    return raw_frames, display_frames


def _frames_to_video(frames: list[np.ndarray], fps: int) -> str:
    H, W = frames[0].shape[:2]
    tmp = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(
        tmp, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    return tmp


# ── segmentation drawing helpers ─────────────────────────────────────────────

def _draw_clicks(img: np.ndarray, clicks: list[tuple[float, float]]) -> np.ndarray:
    H, W = img.shape[:2]
    out  = img.copy()
    for i, (nx, ny) in enumerate(clicks):
        px, py = int(nx * W), int(ny * H)
        colour = _DOT_PALETTE[i % len(_DOT_PALETTE)]
        cv2.circle(out, (px, py), 10, colour, -1)
        cv2.circle(out, (px, py), 10, (255, 255, 255), 2)
        cv2.putText(out, str(i + 1), (px - 4, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── Tab 1: generation handlers ────────────────────────────────────────────────

def run_extraction_and_video(
    image, method: str,
    num_frames: int, crop_min: float, flip_prob: float, fps: int,
):
    """Divide → conquer_all → generate clip. Returns (frames_state, gallery, video, status)."""
    rgb = _to_rgb(image)
    if rgb is None:
        return None, [], None, "Load or upload an image first."

    try:
        mod   = _get_module(method)
        model = mod.DiCoModel.get_or_load()
    except Exception as exc:
        return None, [], None, f"Failed to load '{method}' models: {exc}"

    divide_masks = model.divide(rgb)
    if not divide_masks:
        return None, [], None, "CutLER found no divide masks — try another image."

    conquer_per_divide = model.conquer_all(rgb, divide_masks)
    n_conquer = sum(len(v) for v in conquer_per_divide.values())

    raw_frames, display_frames = _generate_clip(
        rgb, divide_masks, conquer_per_divide, int(num_frames),
        float(crop_min), float(flip_prob),
    )

    H, W = rgb.shape[:2]
    model_frames = [
        cv2.resize(f, (_to_model_hw(H, W)[1], _to_model_hw(H, W)[0]), cv2.INTER_LINEAR)
        for f in raw_frames
    ]
    frames_state = {"rgb": raw_frames, "model": model_frames}

    gallery    = [(f, f"Frame {i + 1}") for i, f in enumerate(display_frames)]
    video_path = _frames_to_video(display_frames, fps)
    status = (
        f"Method: {method}  |  {len(divide_masks)} divide + {n_conquer} conquer mask(s)  |  "
        f"{len(raw_frames)} frames ready for segmentation."
    )
    return frames_state, gallery, video_path, status


def load_precomputed_and_video(
    masks_dir: str, method: str,
    num_frames: int, crop_min: float, flip_prob: float, fps: int,
):
    """Load precomputed .npz → generate clip. Returns (image, frames_state, gallery, video, status)."""
    root = Path(masks_dir) / _PRECOMPUTE_SPLIT
    if not root.exists():
        return None, None, [], None, f"No precomputed masks dir: {root}"

    npzs = list(root.rglob("*.npz"))
    if not npzs:
        return None, None, [], None, f"No .npz files under {root}"

    npz_path = random.choice(npzs)
    img_path, divide_masks, conquer_per_divide = _reconstruct_masks(npz_path)

    rgb = _load_image_rgb(img_path)
    if rgb is None:
        return None, None, [], None, f"Source image not found: {img_path}"
    if not divide_masks:
        return rgb, None, [], None, f"{npz_path.name}: no divide masks."

    raw_frames, display_frames = _generate_clip(
        rgb, divide_masks, conquer_per_divide, int(num_frames),
        float(crop_min), float(flip_prob),
    )

    H, W = rgb.shape[:2]
    model_frames = [
        cv2.resize(f, (_to_model_hw(H, W)[1], _to_model_hw(H, W)[0]), cv2.INTER_LINEAR)
        for f in raw_frames
    ]
    frames_state = {"rgb": raw_frames, "model": model_frames}

    n_conquer  = sum(len(v) for v in conquer_per_divide.values())
    gallery    = [(f, f"Frame {i + 1}") for i, f in enumerate(display_frames)]
    video_path = _frames_to_video(display_frames, fps)
    status = (
        f"Precomputed ({method}): {npz_path.relative_to(root)}  |  "
        f"{len(divide_masks)} divide + {n_conquer} conquer  |  "
        f"{len(raw_frames)} frames ready for segmentation."
    )
    return rgb, frames_state, gallery, video_path, status


# ── Tab 2: segmentation handlers ──────────────────────────────────────────────

def load_frames_for_seg(frames_state):
    """Refresh the seg-tab gallery and canvas from the current frames_state."""
    if frames_state is None or not frames_state.get("rgb"):
        return [], None, None, [], 0, "Generate a video in Tab 1 first."
    frames = frames_state["rgb"]
    gallery = [(f, f"Frame {i + 1}") for i, f in enumerate(frames)]
    canvas  = frames[0].copy()
    # Reset clicks + prompt frame to 0 whenever frames are (re)loaded.
    return (gallery, canvas, canvas, [], 0,
            f"{len(frames)} frames loaded. Prompt frame: 1  "
            f"(click any frame in the gallery to prompt it instead).")


def select_frame_for_seg(evt: gr.SelectData, frames_state, clicks, prompt_frame):
    """
    Clicking a gallery frame makes it the prompt frame (the temporal model
    anchors the prompt to a single frame).  Selecting a *different* frame resets
    the accumulated prompts, since a prompt is tied to the frame it was painted on.
    """
    if frames_state is None:
        return gr.update(), gr.update(), clicks, prompt_frame, gr.update()
    idx = int(evt.index) % len(frames_state["rgb"])
    frame = frames_state["rgb"][idx]
    if idx == prompt_frame:
        ann = _draw_clicks(frame, clicks)
        return ann, ann, clicks, prompt_frame, f"Prompt frame {idx + 1}: {len(clicks)} prompt(s)."
    # Different frame → start fresh on the new prompt frame.
    return (frame, frame, [], idx,
            f"Prompt frame set to {idx + 1}. Previous prompts cleared — "
            f"paint on this frame and Add Prompt.")


def add_prompt(sketch_input, frames_state, clicks, prompt_frame):
    if sketch_input is None or frames_state is None:
        return gr.update(), gr.update(), clicks, "Generate frames first."

    mask_pil = sketch_input.get("mask")
    if mask_pil is None:
        return gr.update(), gr.update(), clicks, "Paint on the canvas first."

    mask_np = np.asarray(mask_pil)
    paint   = mask_np[:, :, 0] if mask_np.ndim == 3 else mask_np
    yx      = np.argwhere(paint > 10)
    if len(yx) == 0:
        return gr.update(), gr.update(), clicks, "No brush strokes detected — paint then click."

    H_m, W_m = paint.shape[:2]
    ny = float(yx[:, 0].mean()) / H_m
    nx = float(yx[:, 1].mean()) / W_m
    nx, ny = max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    new_clicks = list(clicks) + [(nx, ny)]
    base  = frames_state["rgb"][int(prompt_frame)]
    ann   = _draw_clicks(base, new_clicks)
    status = (
        f"Prompt frame {int(prompt_frame) + 1} — {len(new_clicks)} prompt(s): "
        + ", ".join(f"(x={x:.3f}, y={y:.3f})" for x, y in new_clicks)
    )
    return ann, ann, new_clicks, status


def clear_prompts(frames_state, prompt_frame):
    if frames_state is None or not frames_state.get("rgb"):
        return None, None, [], "No frames loaded."
    base = frames_state["rgb"][int(prompt_frame)].copy()
    return base, base, [], f"All prompts cleared (prompt frame: {int(prompt_frame) + 1})."


def run_segmentation(
    frames_state, clicks, prompt_frame, iou_thresh: float,
    config_path: str, weights_path: str, device_name: str, checkpoint: str,
):
    print(f"[seg] frames={frames_state is not None}, n_clicks={len(clicks) if clicks else 0}, "
          f"prompt_frame={prompt_frame}, ckpt={checkpoint!r}")
    try:
        return _run_segmentation_inner(
            frames_state, clicks, prompt_frame, iou_thresh,
            config_path, weights_path, device_name, checkpoint,
        )
    except Exception:
        msg = traceback.format_exc()
        print(f"[seg] EXCEPTION:\n{msg}")
        return [], None, f"Segmentation error:\n{msg}"


def _run_segmentation_inner(
    frames_state, clicks, prompt_frame, iou_thresh: float,
    config_path: str, weights_path: str, device_name: str, checkpoint: str,
):
    if frames_state is None or not frames_state.get("rgb"):
        return [], None, "Generate frames in Tab 1 first."
    if not clicks:
        return [], None, "Add at least one point prompt."

    # A finetuned checkpoint selection of the baseline sentinel means none.
    finetuned = "" if (not checkpoint or checkpoint == _BASELINE_CHOICE) else checkpoint.strip()
    print(f"[seg] loading model from {weights_path!r} (ckpt={finetuned or 'baseline'}) …")
    try:
        backbone, head = _load_seg_model(
            config_path.strip(), weights_path.strip(), device_name.strip(),
            finetuned_path=finetuned,
        )
    except Exception:
        msg = traceback.format_exc()
        return [], None, f"Model load failed:\n{msg}"

    device  = next(head.parameters()).device
    T_frames = len(frames_state["model"])

    # Build frame tensor (T, 3, mH, mW), normalised
    frames_np = np.stack(frames_state["model"]).astype(np.float32)  # (T, mH, mW, 3)
    frames_t  = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)
    pm = torch.tensor(_PIXEL_MEAN, device=device).view(1, 3, 1, 1)
    ps = torch.tensor(_PIXEL_STD,  device=device).view(1, 3, 1, 1)
    frames_t = (frames_t - pm) / ps

    # Backbone
    print("[seg] backbone forward …")
    try:
        with torch.no_grad():
            features = backbone(frames_t)
    except Exception:
        return [], None, f"Backbone failed:\n{traceback.format_exc()}"

    # Build targets: point prompts, B=1 video
    n_clicks = len(clicks)
    pts = torch.tensor(
        [[nx, ny, 0.005, 0.005] for nx, ny in clicks],
        dtype=torch.float32, device=device,
    )
    pb      = torch.zeros(n_clicks, dtype=torch.long, device=device)
    targets = [{"points": pts, "pb": pb}]

    # Tell the temporal layers which frame the prompt is anchored to, so the
    # other frames know which frame to propagate the prompt's meaning from.
    pf = max(0, min(int(prompt_frame), T_frames - 1))
    prompt_frame_idx = torch.tensor([pf], dtype=torch.long, device=device)

    pred_switch = {"part": False, "whole": False, "seg": True, "det": True}

    # Forward
    print(f"[seg] head forward … (prompt_frame={pf})")
    try:
        with torch.no_grad():
            out, _ = head(
                features,
                targets=targets,
                task="demo",
                extra=pred_switch,
                num_frames=T_frames,
                prompt_frame_idx=prompt_frame_idx,
            )
    except Exception:
        return [], None, f"Model forward failed:\n{traceback.format_exc()}"

    # pred_masks: (T, P*M, h, w)  pred_ious: (T, P, M)
    pred_masks = out["pred_masks"]
    pred_ious  = out["pred_ious"]
    n_tok      = pred_ious.shape[-1]

    # Render per-frame overlays on the raw frames
    result_frames: list[np.ndarray] = []
    for t in range(T_frames):
        frame = frames_state["rgb"][t].copy()
        H_d, W_d = frame.shape[:2]
        for i in range(n_clicks):
            ious     = pred_ious[t, i]
            best_j   = int(ious.argmax())
            best_iou = float(ious[best_j])
            if best_iou < float(iou_thresh):
                continue
            logit = pred_masks[t, i * n_tok + best_j].cpu().float().numpy()
            if logit.shape != (H_d, W_d):
                logit = cv2.resize(logit, (W_d, H_d), cv2.INTER_LINEAR)
            mask  = (logit > 0.0).astype(np.uint8)
            frame = _overlay(frame, mask, _SEG_PALETTE[i % len(_SEG_PALETTE)], 0.45, 2)
        frame = _draw_clicks(frame, clicks)
        result_frames.append(frame)

    video_path = _frames_to_video(result_frames, fps=4) if result_frames else None
    gallery    = [(f, f"Frame {i + 1}") for i, f in enumerate(result_frames)]
    model_name = os.path.basename(finetuned) if finetuned else "baseline"
    status = (
        f"Segmented {T_frames} frame(s)  |  {n_clicks} prompt(s) on frame {pf + 1}  |  "
        f"model: {model_name}  |  IoU threshold: {float(iou_thresh):.2f}"
    )
    print(f"[seg] done — {len(result_frames)} frames rendered")
    return gallery, video_path, status


# ── UI ────────────────────────────────────────────────────────────────────────

def build_demo(
    image_dir:    str,
    config_path:  str,
    weights_path: str,
    device:       str,
    ckpt_dir:     str,
) -> gr.Blocks:
    with gr.Blocks(title="DiCo Video → Segmentation") as demo:
        gr.Markdown("# DiCo Video → Promptable Segmentation")

        # Shared across both tabs: raw frames (no overlay) + model-res frames.
        frames_state = gr.State(None)

        with gr.Tabs():

            # ── Tab 1: Generate DiCo Video ─────────────────────────────────────
            with gr.Tab("Generate DiCo Video"):
                gr.Markdown(
                    "Generate a synthetic video clip with divide-and-conquer mask overlays. "
                    "The raw (no-overlay) frames are stored in shared state and become "
                    "available in the **Segment Video** tab."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        method_radio = gr.Radio(
                            choices=list(_METHODS), value="original",
                            label="Mask extraction method",
                            info="original = DINOv1 ViT-B/8; new = DINOv3 ViT-L/16",
                        )

                        with gr.Accordion("Video parameters", open=False):
                            with gr.Row():
                                num_frames_sl = gr.Slider(2, 16, step=1, value=8, label="Frames (T)")
                                fps_sl        = gr.Slider(1, 12, step=1, value=4,  label="FPS")
                            crop_min_sl  = gr.Slider(0.3, 1.0, step=0.05, value=0.5,
                                                     label="Min crop fraction")
                            flip_prob_sl = gr.Slider(0.0, 1.0, step=0.1,  value=0.5,
                                                     label="Flip probability")

                        gr.Markdown("### A · Live extraction")
                        image_dir_box = gr.Textbox(value=image_dir, label="ImageNet image directory")
                        random_btn    = gr.Button("🎲 Load Random ImageNet Image")
                        image_input   = gr.Image(label="Input image", type="numpy", height=280)
                        run_btn       = gr.Button("Run extraction + generate video", variant="primary")

                        gr.Markdown("### B · Verify precomputed masks")
                        masks_dir_box    = gr.Textbox(
                            value=_default_masks_dir("original"),
                            label=f"Precomputed masks dir  (/{_PRECOMPUTE_SPLIT} is appended)",
                        )
                        precomputed_btn = gr.Button(
                            "🎲 Load Random Precomputed Validation Image", variant="primary",
                        )

                        status_t1 = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=2):
                        frame_gallery = gr.Gallery(
                            label="Frames — divide + conquer masks overlaid",
                            columns=4, height=520, type="numpy",
                        )
                        video_output = gr.Video(label="Video playback", height=280)

                # Wiring
                method_radio.change(_default_masks_dir, [method_radio], [masks_dir_box])
                random_btn.click(_random_image, [image_dir_box], [image_input])

                run_btn.click(
                    run_extraction_and_video,
                    inputs=[image_input, method_radio,
                            num_frames_sl, crop_min_sl, flip_prob_sl, fps_sl],
                    outputs=[frames_state, frame_gallery, video_output, status_t1],
                )
                precomputed_btn.click(
                    load_precomputed_and_video,
                    inputs=[masks_dir_box, method_radio,
                            num_frames_sl, crop_min_sl, flip_prob_sl, fps_sl],
                    outputs=[image_input, frames_state, frame_gallery, video_output, status_t1],
                )

            # ── Tab 2: Segment Video ───────────────────────────────────────────
            with gr.Tab("Segment Video"):
                gr.Markdown(
                    "**Step 1** Click **Load frames** to pull the raw frames from Tab 1.\n\n"
                    "**Step 2** Click a frame in the gallery to choose it as the **prompt frame** "
                    "(loaded into the canvas). The temporal model anchors the prompt to this one "
                    "frame and propagates it to the others — so you can prompt any frame, not just "
                    "the first. Paint the object, then **Add Prompt** (repeat for more objects on "
                    "the *same* frame). Selecting a different frame clears prompts.\n\n"
                    "**Step 3** Click **Segment Video** to run VideoIMaskDINOHead across all frames."
                )
                clicks_state       = gr.State([])
                prompt_frame_state = gr.State(0)   # which frame the prompt is anchored to

                with gr.Row():
                    # ── Left: controls ─────────────────────────────────────────
                    with gr.Column(scale=1, min_width=340):
                        load_frames_btn = gr.Button("↓ Load frames from Tab 1", variant="primary")
                        status_t2       = gr.Textbox(label="Status", interactive=False, lines=2)

                        gr.Markdown("---\n#### Prompts\nPaint → Add Prompt. Click a gallery frame to switch canvas.")
                        canvas = gr.Image(
                            source="upload", tool="sketch", type="pil",
                            label="Paint canvas",
                            brush_color="#FF0000", brush_radius=20,
                        )
                        with gr.Row():
                            add_btn   = gr.Button("Add Prompt")
                            clear_btn = gr.Button("Clear Prompts")
                        annotated_disp = gr.Image(
                            type="numpy", interactive=False,
                            label="Accumulated prompts",
                        )

                        gr.Markdown("---\n#### Segmentation")
                        _ckpts = _list_checkpoints(ckpt_dir)
                        with gr.Row():
                            checkpoint_dd = gr.Dropdown(
                                choices=[_BASELINE_CHOICE] + _ckpts,
                                value=(_ckpts[-1] if _ckpts else _BASELINE_CHOICE),
                                label="Checkpoint to run "
                                      "(baseline + this finetuned overlay)",
                                scale=5,
                            )
                            refresh_ckpts_btn = gr.Button("⟳", scale=1, min_width=40)
                        iou_sl  = gr.Slider(0.0, 1.0, step=0.05, value=0.3,
                                            label="IoU threshold (hides low-confidence masks)")
                        seg_btn = gr.Button("Segment Video", variant="primary")

                        with gr.Accordion("Model settings", open=False):
                            config_box   = gr.Textbox(label="Config path",  value=config_path)
                            weights_box  = gr.Textbox(label="Baseline weights path", value=weights_path)
                            ckpt_dir_box = gr.Textbox(label="Checkpoint directory", value=ckpt_dir)
                            device_box   = gr.Textbox(label="Device",       value=device)

                    # ── Right: galleries ───────────────────────────────────────
                    with gr.Column(scale=2):
                        gr.Markdown("#### Raw frames  *(click any frame to load it into the canvas)*")
                        raw_gallery = gr.Gallery(
                            type="numpy", columns=4, height=280,
                            label="Raw augmented frames",
                        )
                        gr.Markdown("#### Segmentation results")
                        seg_gallery = gr.Gallery(
                            type="numpy", columns=4, height=280,
                            label="Segmented frames",
                        )
                        seg_video = gr.Video(label="Segmented video", height=280)

                # Wiring
                load_frames_btn.click(
                    load_frames_for_seg,
                    inputs=[frames_state],
                    outputs=[raw_gallery, canvas, annotated_disp,
                             clicks_state, prompt_frame_state, status_t2],
                )
                raw_gallery.select(
                    select_frame_for_seg,
                    inputs=[frames_state, clicks_state, prompt_frame_state],
                    outputs=[canvas, annotated_disp,
                             clicks_state, prompt_frame_state, status_t2],
                )
                add_btn.click(
                    add_prompt,
                    inputs=[canvas, frames_state, clicks_state, prompt_frame_state],
                    outputs=[canvas, annotated_disp, clicks_state, status_t2],
                )
                clear_btn.click(
                    clear_prompts,
                    inputs=[frames_state, prompt_frame_state],
                    outputs=[canvas, annotated_disp, clicks_state, status_t2],
                )
                refresh_ckpts_btn.click(
                    lambda d: gr.update(choices=[_BASELINE_CHOICE] + _list_checkpoints(d)),
                    inputs=[ckpt_dir_box], outputs=[checkpoint_dd],
                )
                seg_btn.click(
                    run_segmentation,
                    inputs=[frames_state, clicks_state, prompt_frame_state, iou_sl,
                            config_box, weights_box, device_box, checkpoint_dd],
                    outputs=[seg_gallery, seg_video, status_t2],
                )

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir",  default=_DEFAULT_IMAGE_DIR)
    p.add_argument("--config",     default=str(_THIS / "configs" / "video_sam_swinT.yaml"))
    p.add_argument("--weights",    default=str(_THIS / "ckpts"   / "baseline.pth"))
    p.add_argument("--ckpt-dir",   default=str(_THIS / "output" / "video_ft" / "checkpoints"),
                   help="directory of finetuned checkpoints to choose from")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port",  type=int, default=7863)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    demo = build_demo(args.image_dir, args.config, args.weights, args.device, args.ckpt_dir)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()


"""
python promptable_video_segmentation/gradio_dico_video.py \
  --ckpt-dir promptable_video_segmentation/output/video_ft/checkpoints

"""