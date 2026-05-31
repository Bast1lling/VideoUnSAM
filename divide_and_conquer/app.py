"""
Gradio visualisation app for the Divide-and-Conquer segmentation pipeline.

Run from the divide_and_conquer/ directory:
    python app.py [--weights path/to/cutler_cascade_final.pth]
"""

import argparse

import cv2
import gradio as gr
import numpy as np
import torch
import PIL.Image as Image

from demo_dico import (
    NMS,
    coverage,
    generate_feature_matrix,
    get_parser as get_dico_parser,
    resize_mask,
    smallest_square_containing_mask,
)
from divide_conquer import setup_cfg
from engine.defaults import DefaultPredictor
from iterative_merging import iterative_merge
import dinov3
import dino as dino_module
from synthetic_video import build_synth_video_section


# ── feature PCA visualisation ───────────────────────────────────────────────

def _pca_rgb(feat_matrix: np.ndarray, upsample_to: tuple | None = None) -> np.ndarray:
    H, W, C = feat_matrix.shape
    flat = torch.from_numpy(feat_matrix.reshape(-1, C).astype(np.float32))
    flat -= flat.mean(0)
    _, _, V = torch.pca_lowrank(flat, q=3, niter=4)
    proj = flat @ V
    proj -= proj.min(0).values
    proj /= proj.max(0).values + 1e-8
    rgb = (proj * 255).byte().numpy().reshape(H, W, 3)
    if upsample_to is not None:
        rgb = cv2.resize(rgb, (upsample_to[1], upsample_to[0]), interpolation=cv2.INTER_NEAREST)
    return rgb


# ── colour palette ──────────────────────────────────────────────────────────
_PALETTE = [
    [220,  20,  60], [ 30, 144, 255], [ 50, 205,  50], [255, 215,   0],
    [255, 140,   0], [138,  43, 226], [  0, 206, 209], [255, 105, 180],
    [127, 255,   0], [255,  69,   0], [  0, 191, 255], [186,  85, 211],
]


def _color(i: int):
    return _PALETTE[i % len(_PALETTE)]


def _overlay(image_rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45):
    out = image_rgb.copy()
    fg = mask.astype(bool)
    out[fg] = (out[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


# ── single-image navigation helpers ─────────────────────────────────────────

def _render_divide(idx: int, image_rgb, divide_masks):
    """Returns (display_img, selected_preview, count_label)."""
    if not divide_masks or image_rgb is None:
        return None, None, "–"
    idx = max(0, min(len(divide_masks) - 1, int(idx)))
    mask = divide_masks[idx]
    display = _overlay(image_rgb, mask, _color(idx))
    preview = _overlay(image_rgb, mask, _color(idx), alpha=0.6)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, [255, 255, 255], 3)
    return display, preview, f"Mask {idx + 1} / {len(divide_masks)}"


def _nav_divide(direction: int, idx, image_rgb, divide_masks):
    """Prev/Next for divide: returns (new_idx, display, preview, label)."""
    if not divide_masks:
        return idx, None, None, "–"
    new_idx = max(0, min(len(divide_masks) - 1, int(idx) + direction))
    display, preview, label = _render_divide(new_idx, image_rgb, divide_masks)
    return new_idx, display, preview, label


def _render_conquer(idx: int, divide_mask_idx, image_rgb, divide_masks, conquer_masks):
    """Returns (display_img, count_label)."""
    if not conquer_masks or image_rgb is None or not divide_masks:
        return None, "–"
    idx = max(0, min(len(conquer_masks) - 1, int(idx)))
    divide_mask = divide_masks[int(divide_mask_idx)]
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
    crop_rgb = image_rgb[ymin:ymax, xmin:xmax]
    crop_mask = conquer_masks[idx][ymin:ymax, xmin:xmax]
    return _overlay(crop_rgb, crop_mask, _color(idx)), f"Mask {idx + 1} / {len(conquer_masks)}"


def _nav_conquer(direction: int, idx, divide_mask_idx, image_rgb, divide_masks, conquer_masks):
    """Prev/Next for conquer: returns (new_idx, display, label)."""
    if not conquer_masks:
        return idx, None, "–"
    new_idx = max(0, min(len(conquer_masks) - 1, int(idx) + direction))
    display, label = _render_conquer(new_idx, divide_mask_idx, image_rgb, divide_masks, conquer_masks)
    return new_idx, display, label


def _render_all_conquer(image_rgb, divide_mask_idx, divide_masks, conquer_masks_for_divide):
    """Composite overlay: all conquer masks stacked on the selected divide crop (small on top of large)."""
    if image_rgb is None or not divide_masks or not conquer_masks_for_divide:
        return None
    divide_mask = divide_masks[int(divide_mask_idx)]
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
    crop_rgb = image_rgb[ymin:ymax, xmin:xmax]
    crop_masks = [(m[ymin:ymax, xmin:xmax] > 0.5).astype(np.uint8) for m in conquer_masks_for_divide]
    # largest masks first so smaller masks end up painted on top
    order = sorted(range(len(crop_masks)), key=lambda i: -int(np.sum(crop_masks[i])))
    I_orig = np.array(crop_rgb).copy()
    result = I_orig.copy()
    already_painted = np.zeros(I_orig.shape[:2], dtype=np.uint8)
    rng = np.random.default_rng(42)
    for i in order:
        mask = crop_masks[i]
        already_painted += mask
        overlap = already_painted == 2
        if np.sum(overlap) > 0:
            result[overlap] = I_orig[overlap]
            already_painted[overlap] -= 1
        color = rng.integers(64, 230, 3).tolist()
        fg = mask.astype(bool)
        result[fg] = (result[fg] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    return result


def _refresh_all_conquer_from_state(image_rgb, divide_mask_idx, divide_masks, all_conquer):
    """Pull the stored conquer masks for the current divide index and render the composite view."""
    masks = all_conquer.get(int(divide_mask_idx), [])
    return _render_all_conquer(image_rgb, divide_mask_idx, divide_masks, masks)


# ── lazy model loading ───────────────────────────────────────────────────────
_models: dict = {}


def _load_models(weights_path: str, backbone_type: str = "dinov3", dino_url: str = ""):
    cache_key = (weights_path, backbone_type, dino_url)
    if _models.get("key") == cache_key:
        return _models["predictor"], _models["backbone"], _models["args"]

    dico_args = get_dico_parser(
        yaml_path="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"
    ).parse_args(["--opts", "MODEL.WEIGHTS", weights_path, "MODEL.DEVICE", "cuda"])
    cfg = setup_cfg(dico_args)
    predictor = DefaultPredictor(cfg)

    if backbone_type == "dinov3":
        backbone = dinov3.ViTFeatV3(
            dico_args.backbone_url, dico_args.feature_dim, dico_args.backbone_size, "k", dico_args.patch_size,
        )
        backbone.eval().cuda()
    else:
        backbone = dino_module.ViTFeat(dino_url, 768, "base", "k", 8)
        backbone.eval().cuda().half()
        dico_args.feature_dim = 768
        dico_args.patch_size = 8

    _models["key"] = cache_key
    _models["predictor"] = predictor
    _models["backbone"] = backbone
    _models["args"] = dico_args
    return predictor, backbone, dico_args


# ── Gradio event handlers ────────────────────────────────────────────────────

def run_divide(image_rgb: np.ndarray, weights_path: str, backbone_type: str, dino_url: str):
    if image_rgb is None:
        return [], gr.update(visible=False), None, None, "–", "Upload an image first.", {}, None
    try:
        predictor, backbone, args = _load_models(weights_path, backbone_type, dino_url)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        preds = predictor(image_bgr)
        masks_t = preds["instances"].get("pred_masks")
        divide_masks = [masks_t[i].cpu().numpy() for i in range(masks_t.shape[0])]
        print(f"[divide] {len(divide_masks)} masks found")

        slider_upd = gr.update(
            maximum=max(len(divide_masks) - 1, 0), value=0, visible=len(divide_masks) > 0
        )
        if divide_masks:
            display, preview, count = _render_divide(0, image_rgb, divide_masks)
        else:
            display, preview, count = None, None, "No masks found"

        # reset accumulated conquer masks whenever divide is re-run
        return divide_masks, slider_upd, display, preview, count, f"Divide: {len(divide_masks)} masks.", {}, None

    except Exception:
        import traceback; tb = traceback.format_exc()
        print(tb)
        return [], gr.update(visible=False), None, None, "–", f"ERROR in divide:\n{tb}", {}, None


def run_conquer(
    image_rgb: np.ndarray,
    mask_idx: int,
    divide_masks: list,
    weights_path: str,
    backbone_type: str,
    dino_url: str,
    local_size: int,
    kept_thresh: float,
    nms_iou: float,
    thetas_str: str,
    all_conquer: dict,
):
    if image_rgb is None or not divide_masks:
        return [], gr.update(visible=False), None, "–", None, None, "No image or divide masks.", all_conquer
    try:
        predictor, backbone, args = _load_models(weights_path, backbone_type, dino_url)

        try:
            thetas = [float(t.strip()) for t in thetas_str.split(",") if t.strip()]
        except ValueError:
            return [], gr.update(visible=False), None, "–", None, None, f"Invalid thetas: {thetas_str!r}", all_conquer

        divide_mask = divide_masks[int(mask_idx)]
        ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            return [], gr.update(visible=False), None, "–", None, None, "Selected mask has zero area.", all_conquer

        print(f"[conquer] mask {mask_idx}, local_size={local_size}, kept_thresh={kept_thresh}, nms_iou={nms_iou}")

        local_rgb = image_rgb[ymin:ymax, xmin:xmax]
        resized = Image.fromarray(local_rgb).resize([local_size, local_size])

        feat_num = local_size // args.patch_size
        feat_matrix = generate_feature_matrix(backbone, resized, args.feature_dim, feat_num)
        print(f"[conquer] feature matrix shape: {feat_matrix.shape}")

        feat_np = feat_matrix.numpy() if hasattr(feat_matrix, "numpy") else feat_matrix
        feature_vis = _pca_rgb(feat_np, upsample_to=local_rgb.shape[:2])

        conquer_masks = []
        for layer in iterative_merge(feat_matrix, thetas):
            if layer.shape[0] == 0:
                continue
            for i in range(layer.shape[0]):
                mask = resize_mask(layer[i], [xmax - xmin, ymax - ymin])
                mask_bin = (mask > 0.5 * 255).astype(int)
                if coverage(mask_bin, divide_mask[ymin:ymax, xmin:xmax]) <= kept_thresh:
                    continue
                full = np.zeros_like(divide_mask)
                full[ymin:ymax, xmin:xmax] = mask_bin
                conquer_masks.append(full)

        print(f"[conquer] {len(conquer_masks)} masks before NMS")
        conquer_masks = NMS(conquer_masks, nms_iou, args.NMS_step)
        print(f"[conquer] {len(conquer_masks)} masks after NMS")

        # accumulate into the all_conquer dict so the video section can use all computed masks
        updated_all_conquer = {**all_conquer, int(mask_idx): conquer_masks}

        if conquer_masks:
            display, count = _render_conquer(0, mask_idx, image_rgb, divide_masks, conquer_masks)
            slider_upd = gr.update(maximum=len(conquer_masks) - 1, value=0, visible=True)
        else:
            display, count = None, "No masks"
            slider_upd = gr.update(visible=False)

        all_conquer_view = _render_all_conquer(image_rgb, mask_idx, divide_masks, conquer_masks)

        status = f"Conquer: {len(conquer_masks)} masks for divide mask {mask_idx}."
        if not conquer_masks:
            status += " (all filtered by coverage threshold)"
        return conquer_masks, slider_upd, display, count, feature_vis, all_conquer_view, status, updated_all_conquer

    except Exception:
        import traceback; tb = traceback.format_exc()
        print(tb)
        return [], gr.update(visible=False), None, "–", None, None, f"ERROR in conquer:\n{tb}", all_conquer


# ── UI ───────────────────────────────────────────────────────────────────────

def build_ui(default_weights: str) -> gr.Blocks:
    with gr.Blocks(title="DiCo Visualiser", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Divide-and-Conquer Segmentation Visualiser\n"
            "**Step 1** — upload an image and run the divide phase.  \n"
            "**Step 2** — navigate to a divide mask, then run the conquer phase."
        )

        divide_masks_state  = gr.State([])
        conquer_masks_state = gr.State([])
        all_conquer_state   = gr.State({})   # {divide_idx: [mask, ...]} accumulated across conquer runs

        # ── settings ────────────────────────────────────────────────────────
        with gr.Accordion("Settings", open=False):
            weights_input = gr.Textbox(value=default_weights, label="CutLER weights path")
            backbone_radio = gr.Radio(choices=["dinov3", "dino"], value="dinov3", label="Conquer backbone")
            dino_url_input = gr.Textbox(
                value="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
                label="DINO weights URL (backbone=dino only)",
                visible=False,
            )
            backbone_radio.change(
                lambda b: gr.update(visible=(b == "dino")),
                inputs=[backbone_radio], outputs=[dino_url_input],
            )

        status_box = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=8)

        # ── divide phase ─────────────────────────────────────────────────────
        gr.Markdown("## Divide Phase")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Input image", type="numpy")
                divide_btn  = gr.Button("Run Divide Phase", variant="primary")
            with gr.Column(scale=2):
                divide_count = gr.Markdown("–")
                divide_display = gr.Image(label="Divide mask", type="numpy", interactive=False, height=480)
                with gr.Row():
                    prev_div_btn  = gr.Button("◀ Prev", size="sm")
                    mask_slider   = gr.Slider(minimum=0, maximum=0, step=1, value=0,
                                              label="Mask index", visible=False)
                    next_div_btn  = gr.Button("Next ▶", size="sm")

        # ── conquer phase ────────────────────────────────────────────────────
        gr.Markdown("## Conquer Phase")
        with gr.Accordion("Conquer parameters", open=True):
            with gr.Row():
                local_size_slider  = gr.Slider(128, 1024, step=128, value=512,
                    label="Local crop size (px) — try 512 for DINOv3 to match DINO-B/8 resolution")
                kept_thresh_slider = gr.Slider(0.0, 1.0, step=0.05, value=0.8,
                    label="Coverage threshold (kept_thresh)")
            with gr.Row():
                nms_iou_slider = gr.Slider(0.0, 1.0, step=0.05, value=0.8, label="NMS IoU threshold")
                thetas_input   = gr.Textbox(value="0.73,0.62,0.51,0.4,0.3,0.2",
                    label="Merging thresholds (thetas, comma-separated, high→low)")

        with gr.Row():
            with gr.Column(scale=1):
                selected_preview = gr.Image(label="Selected divide mask", type="numpy", interactive=False)
                feature_map_img  = gr.Image(label="Backbone features (PCA → RGB)", type="numpy", interactive=False)
                conquer_btn = gr.Button("Run Conquer Phase", variant="secondary")
            with gr.Column(scale=2):
                conquer_count   = gr.Markdown("–")
                conquer_display = gr.Image(label="Conquer mask (crop)", type="numpy", interactive=False, height=480)
                with gr.Row():
                    prev_con_btn   = gr.Button("◀ Prev", size="sm")
                    conquer_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0,
                                               label="Mask index", visible=False)
                    next_con_btn   = gr.Button("Next ▶", size="sm")

        # ── hierarchical conquer view ─────────────────────────────────────────
        gr.Markdown("## Hierarchical Conquer View")
        gr.Markdown(
            "All conquer masks for the selected divide crop, stacked with small masks on top of large masks."
        )
        all_conquer_view_img = gr.Image(
            label="All conquer masks (hierarchical overlay)",
            type="numpy", interactive=False, height=480,
        )

        # ── event wiring ─────────────────────────────────────────────────────

        divide_btn.click(
            run_divide,
            inputs=[image_input, weights_input, backbone_radio, dino_url_input],
            outputs=[divide_masks_state, mask_slider, divide_display, selected_preview, divide_count, status_box, all_conquer_state, all_conquer_view_img],
        )

        # divide navigation — also refresh the composite view from cached conquer results
        mask_slider.change(
            _render_divide,
            inputs=[mask_slider, image_input, divide_masks_state],
            outputs=[divide_display, selected_preview, divide_count],
        ).then(
            _refresh_all_conquer_from_state,
            inputs=[image_input, mask_slider, divide_masks_state, all_conquer_state],
            outputs=[all_conquer_view_img],
        )
        prev_div_btn.click(
            lambda idx, img, masks: _nav_divide(-1, idx, img, masks),
            inputs=[mask_slider, image_input, divide_masks_state],
            outputs=[mask_slider, divide_display, selected_preview, divide_count],
        ).then(
            _refresh_all_conquer_from_state,
            inputs=[image_input, mask_slider, divide_masks_state, all_conquer_state],
            outputs=[all_conquer_view_img],
        )
        next_div_btn.click(
            lambda idx, img, masks: _nav_divide(+1, idx, img, masks),
            inputs=[mask_slider, image_input, divide_masks_state],
            outputs=[mask_slider, divide_display, selected_preview, divide_count],
        ).then(
            _refresh_all_conquer_from_state,
            inputs=[image_input, mask_slider, divide_masks_state, all_conquer_state],
            outputs=[all_conquer_view_img],
        )

        conquer_btn.click(
            run_conquer,
            inputs=[
                image_input, mask_slider, divide_masks_state,
                weights_input, backbone_radio, dino_url_input,
                local_size_slider, kept_thresh_slider, nms_iou_slider, thetas_input,
                all_conquer_state,
            ],
            outputs=[conquer_masks_state, conquer_slider, conquer_display, conquer_count, feature_map_img, all_conquer_view_img, status_box, all_conquer_state],
        )

        # conquer navigation
        conquer_slider.change(
            _render_conquer,
            inputs=[conquer_slider, mask_slider, image_input, divide_masks_state, conquer_masks_state],
            outputs=[conquer_display, conquer_count],
        )
        prev_con_btn.click(
            lambda idx, didx, img, dm, cm: _nav_conquer(-1, idx, didx, img, dm, cm),
            inputs=[conquer_slider, mask_slider, image_input, divide_masks_state, conquer_masks_state],
            outputs=[conquer_slider, conquer_display, conquer_count],
        )
        next_con_btn.click(
            lambda idx, didx, img, dm, cm: _nav_conquer(+1, idx, didx, img, dm, cm),
            inputs=[conquer_slider, mask_slider, image_input, divide_masks_state, conquer_masks_state],
            outputs=[conquer_slider, conquer_display, conquer_count],
        )

        # ── stage 2: synthetic video ──────────────────────────────────────────
        build_synth_video_section(image_input, divide_masks_state, all_conquer_state, status_box)

    return demo


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model_zoo/ckpts/cutler_cascade_final.pth")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    cli_args = ap.parse_args()
    build_ui(cli_args.weights).launch(server_port=cli_args.port, share=cli_args.share)
