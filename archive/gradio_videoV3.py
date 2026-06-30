"""
Gradio demo: single-image Divide → Conquer (DINOv3-V3) → Synthetic video.

Drop in one image, click Run — the pipeline extracts coarse instance masks
(CutLER), hierarchically refines each region with DINOv3 (ViT-L/16,
local_size=512, thetas=[0.75…0.25]), then generates a synthetic multi-frame
video by independently augmenting the image and all masks each frame.

Run from divide_and_conquer/:
    python gradio_videoV3.py --weights path/to/cutler_cascade_final.pth
"""

import argparse
import pathlib
import types
import warnings

import cv2
import gradio as gr
import numpy as np

from divide_conquer import setup_cfg
from divide_conquerV3 import run_conquer
from synthetic_video import run_synth_video
from engine.defaults import DefaultPredictor
import dinov3

warnings.filterwarnings("ignore")


_SCRIPT_DIR  = pathlib.Path(__file__).parent
_DEFAULT_CFG = str(_SCRIPT_DIR / "model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml")

_PATCH_SIZE = 16
_FEAT_DIM   = 1024
_BACKBONE_URL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_THETAS_V3  = [0.75, 0.65, 0.55, 0.45, 0.35, 0.25]

# ── lazy model cache ──────────────────────────────────────────────────────────
_cache: dict = {}


def _load_models(weights_path: str, confidence: float = 0.1):
    key = (weights_path, confidence)
    if _cache.get("key") == key:
        return _cache["predictor"], _cache["backbone"]

    opts = ["MODEL.WEIGHTS", weights_path] if weights_path else []
    args = types.SimpleNamespace(
        config_file=_DEFAULT_CFG,
        confidence_threshold=confidence,
        opts=opts,
    )
    cfg = setup_cfg(args)
    _cache["cfg"]       = cfg
    _cache["predictor"] = DefaultPredictor(cfg)

    backbone = dinov3.ViTFeatV3(_BACKBONE_URL, _FEAT_DIM, "large", "k", _PATCH_SIZE)
    backbone.eval()
    if cfg.MODEL.DEVICE == "cpu":
        backbone.to("cpu")
    else:
        backbone.cuda()
    _cache["backbone"] = backbone
    _cache["key"]      = key

    return _cache["predictor"], _cache["backbone"]


# ── pipeline (generator for streaming status updates) ────────────────────────

def run_pipeline(
    image_rgb,
    weights_path,
    confidence,
    local_size,
    kept_thresh,
    nms_iou,
    num_frames,
    fps,
    crop_min,
    flip_prob,
):
    if image_rgb is None:
        return None, [], "Upload an image first."

    # ── load models ──────────────────────────────────────────────────────────
    try:
        predictor, backbone = _load_models(weights_path.strip() if weights_path else "", confidence)
    except Exception:
        import traceback
        return None, [], f"Model load failed:\n{traceback.format_exc()}"

    # ── divide ───────────────────────────────────────────────────────────────
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        preds = predictor(image_bgr)
        tensor = preds["instances"].get("pred_masks")
        divide_masks = [tensor[i].cpu().numpy() for i in range(tensor.shape[0])]
    except Exception:
        import traceback
        return None, [], f"Divide phase failed:\n{traceback.format_exc()}"

    n_divide = len(divide_masks)

    # ── conquer ──────────────────────────────────────────────────────────────
    try:
        conquer_args = types.SimpleNamespace(
            patch_size=_PATCH_SIZE,
            feature_dim=_FEAT_DIM,
            local_size=int(local_size),
            kept_thresh=float(kept_thresh),
            NMS_iou=float(nms_iou),
            NMS_step=5,
            thetas=_THETAS_V3,
        )
        _, conquer_per_divide = run_conquer(backbone, image_bgr, divide_masks, conquer_args)
    except Exception:
        import traceback
        return None, [], f"Conquer phase failed:\n{traceback.format_exc()}"

    n_conquer = sum(len(v) for v in conquer_per_divide.values())

    # ── synthetic video ───────────────────────────────────────────────────────
    try:
        gallery, tmp_path, desc = run_synth_video(
            image_rgb,
            divide_masks,
            conquer_per_divide,
            num_frames=int(num_frames),
            crop_min=float(crop_min),
            flip_prob=float(flip_prob),
            fps=int(fps),
        )
    except Exception:
        import traceback
        return None, [], f"Video generation failed:\n{traceback.format_exc()}"

    return tmp_path, gallery, f"{n_divide} divide + {n_conquer} conquer mask(s) | {desc}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_demo():
    with gr.Blocks(title="Divide → Conquer → Video") as demo:
        gr.Markdown(
            "## Divide → Conquer (DINOv3-V3) → Synthetic Video\n"
            "Upload a single image. The pipeline runs **CutLER** (divide) to extract "
            "coarse instance masks, then **DINOv3 ViT-L/16** (conquer) to hierarchically "
            "refine each region, and finally generates a synthetic multi-frame video by "
            "independently augmenting the image and all masks per frame — replicating "
            "VideoCutLER's *ImageCut2Video* Stage 2."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="numpy", label="Input image")

                with gr.Accordion("Model settings", open=False):
                    weights_input = gr.Textbox(
                        label="CutLER weights path",
                        placeholder="path/to/cutler_cascade_final.pth",
                        value="",
                    )
                    confidence_slider = gr.Slider(
                        0.01, 0.5, step=0.01, value=0.1, label="Confidence threshold"
                    )

                with gr.Accordion("Conquer parameters", open=False):
                    local_size_slider = gr.Slider(
                        128, 512, step=64, value=512,
                        label="Local crop size (px) — fed to DINOv3",
                    )
                    kept_thresh_slider = gr.Slider(
                        0.5, 1.0, step=0.05, value=0.9,
                        label="Coverage threshold — min overlap of conquer mask with divide region",
                    )
                    nms_iou_slider = gr.Slider(
                        0.3, 1.0, step=0.05, value=0.9,
                        label="NMS IoU threshold",
                    )

                with gr.Accordion("Video parameters", open=True):
                    with gr.Row():
                        num_frames_slider = gr.Slider(2, 16, step=1, value=8, label="Frames")
                        fps_slider        = gr.Slider(1, 12, step=1, value=4,  label="FPS")
                    with gr.Row():
                        crop_min_slider  = gr.Slider(
                            0.3, 1.0, step=0.05, value=0.5,
                            label='Min crop fraction (relative_range)',
                        )
                        flip_prob_slider = gr.Slider(
                            0.0, 1.0, step=0.1, value=0.5, label="Flip probability"
                        )

                run_btn    = gr.Button("Run Pipeline", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Column(scale=2):
                video_output  = gr.Video(label="Synthetic video", height=400)
                frame_gallery = gr.Gallery(
                    label="Individual frames",
                    columns=4,
                    height=300,
                    type="numpy",
                )

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                image_input,
                weights_input,
                confidence_slider,
                local_size_slider,
                kept_thresh_slider,
                nms_iou_slider,
                num_frames_slider,
                fps_slider,
                crop_min_slider,
                flip_prob_slider,
            ],
            outputs=[video_output, frame_gallery, status_box],
        )

    return demo


# ── entry point ───────────────────────────────────────────────────────────────

def _get_parser():
    p = argparse.ArgumentParser(description="Gradio demo: Divide → Conquer → Video")
    p.add_argument("--weights", default="", help="Path to CutLER weights (.pth)")
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--port",    type=int, default=7861)
    p.add_argument("--share",   action="store_true")
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    demo = build_demo()

    # Pre-populate weights field if passed via CLI
    if args.weights:
        print(f"Weights: {args.weights}")

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
