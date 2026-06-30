"""
Gradio demo for hierarchical DINOv3 segmentation (divide_and_conquer2).

Each press of "⊕ Next Mask" extracts one mask from the BFS queue.
Level-1 comes first (whole image), then level-2 crops (BFS), etc.

Features:
  - Largest-connected-component post-processing per mask.
  - Dynamic tau sliders: one per level, shown/hidden as max_j changes.
  - Per-mask visibility checkboxes in the overview.

Run:
    python promptable_video_segmentation/divide_and_conquer2/app.py
    python promptable_video_segmentation/divide_and_conquer2/app.py --port 7862
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import PIL.Image as Image
from scipy import ndimage

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from promptable_video_segmentation.divide_and_conquer2 import (
    HierarchicalModel,
    SegMask,
    masks_at_level,
    children_of,
    visualize_level1_selected,
    visualize_children_crop,
    visualize_child_single,
    DEFAULT_BACKBONE_ID,
    DEFAULT_TAU_PER_LEVEL,
    DEFAULT_MAX_K,
    DEFAULT_MAX_J,
)
from promptable_video_segmentation.divide_and_conquer2.hierarchical import _upscale
from promptable_video_segmentation.divide_and_conquer2.spectral import spectral_bipartition_2d
from promptable_video_segmentation.divide_and_conquer2.viz import overlay_mask, level_color, child_color
from promptable_video_segmentation.masks import (
    smallest_square_containing_mask,
    coverage as _coverage,
    iou as _iou,
)

_DEFAULT_IMAGE_DIR = "/home/sebastian/data/imagenet1k/images"

MAX_LEVELS    = 3                         # max hierarchy depth the UI supports
_DEFAULT_TAUS = [0.20, 0.40, 0.60] # default tau per level slot


# ── utilities ──────────────────────────────────────────────────────────────────

def _to_rgb(image) -> "np.ndarray | None":
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image[:, :, :3]
    return np.array(Image.fromarray(np.array(image)).convert("RGB"))


def _random_image(image_dir: str) -> "Image.Image | None":
    root = Path(image_dir)
    if not root.exists():
        return None
    imgs = (list(root.rglob("*.JPEG")) + list(root.rglob("*.jpg"))
            + list(root.rglob("*.png")))
    return Image.open(random.choice(imgs)).convert("RGB") if imgs else None


def _largest_connected(mask: np.ndarray) -> np.ndarray:
    """Return the largest 4-connected component of a binary mask."""
    labeled, n = ndimage.label(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return (labeled == int(np.argmax(sizes)) + 1)


def _mask_label(sm: SegMask) -> str:
    if sm.level == 1:
        return f"L1 #{sm.child_idx}"
    return f"L{sm.level} p{sm.parent_idx} #{sm.child_idx}"


def visualize_filtered(
    image_rgb: np.ndarray,
    found:     "list[SegMask]",
    visible:   "set[int]",
) -> np.ndarray:
    """Overlay only masks whose index is in *visible*, preserving colour assignments."""
    out = image_rgb.copy()
    for i, sm in enumerate(found):
        if i not in visible:
            continue
        if sm.level == 1:
            color, alpha, ct = level_color(sm.child_idx), 0.35, 2
        else:
            color, alpha, ct = child_color(sm.parent_idx or 0, sm.child_idx), 0.55, 1
        out = overlay_mask(out, sm.mask.astype(np.uint8), color, alpha, ct)
    return out


# ── step-state helpers ─────────────────────────────────────────────────────────

def _new_state(image_rgb: np.ndarray, full_feats: np.ndarray, cfg: dict) -> dict:
    return {
        "image_rgb":  image_rgb,
        "full_feats": full_feats,
        "found":      [],
        "queue": [{
            "level":      1,
            "parent_idx": None,
            "feats":      full_feats,
            "painting":   np.zeros(full_feats.shape[:2], dtype=bool),
            "count":      0,
            "bbox":       None,   # None = full image
        }],
        "cfg": cfg,
    }


def _queue_status(state: dict) -> str:
    found = state["found"]
    n1 = len(masks_at_level(found, 1))
    n2 = len(masks_at_level(found, 2))
    q  = state["queue"]
    if not q:
        return f"🏁 Done — {n1} level-1, {n2} level-2 mask(s)"
    e = q[0]
    return (f"Level {e['level']} (parent {e['parent_idx']})  "
            f"[{e['count']}/{state['cfg']['max_k']}]  ·  "
            f"{n1} L1 + {n2} L2  ·  {len(q)} region(s) queued")


def _no_result(msg: str) -> tuple:
    upd = gr.update(maximum=0, value=0)
    return (None, None, None, msg, None, None, None, None, None,
            upd, upd, upd,
            gr.update(choices=[], value=[]))


def _render(
    state:        "dict | None",
    status:       str,
    last:         "SegMask | None" = None,
    prev_visible: "list | None"    = None,   # current mask_select.value; None = select all
) -> tuple:
    """Build all 13 outputs from the current state."""
    found     = state["found"]     if state else []
    image_rgb = state["image_rgb"] if state else None
    upd0      = gr.update(maximum=0, value=0)

    if not found or image_rgb is None:
        return (state, image_rgb, found, status,
                None, None, None, None, None,
                upd0, upd0, upd0,
                gr.update(choices=[], value=[]))

    # ── mask-select: build choices, preserve existing selection, auto-add newest ─
    choices = [(_mask_label(sm), str(i)) for i, sm in enumerate(found)]
    new_idx = len(found) - 1
    if prev_visible is None:
        sel = [str(i) for i in range(len(found))]
    else:
        prev = list(prev_visible)
        sel  = prev if str(new_idx) in prev else prev + [str(new_idx)]
    visible = {int(v) for v in sel}

    # ── highlight indices ─────────────────────────────────────────────────────
    l1   = masks_at_level(found, 1)
    n_l1 = len(l1)

    if last and last.level == 2:
        par_idx = last.parent_idx or 0
        ch_idx  = last.child_idx
    else:
        par_idx = n_l1 - 1
        ch_idx  = 0
    par_idx = min(par_idx, n_l1 - 1) if n_l1 else 0

    # ── images ────────────────────────────────────────────────────────────────
    overview   = visualize_filtered(image_rgb, found, visible)
    l1_img     = visualize_level1_selected(image_rgb, found, n_l1 - 1)
    l2_par_img = visualize_level1_selected(image_rgb, found, par_idx)
    l2_crop    = visualize_children_crop(image_rgb, found, par_idx)
    ch_list    = children_of(found, 2, par_idx)
    ch_idx     = min(ch_idx, len(ch_list) - 1) if ch_list else 0
    l2_single  = visualize_child_single(image_rgb, found, par_idx, ch_idx) if ch_list else None
    n_ch       = len(ch_list)

    return (
        state, image_rgb, found, status,
        overview,
        l1_img,
        l2_par_img, l2_crop, l2_single,
        gr.update(maximum=max(0, n_l1 - 1), value=n_l1 - 1),
        gr.update(maximum=max(0, n_l1 - 1), value=par_idx),
        gr.update(maximum=max(0, n_ch  - 1), value=ch_idx),
        gr.update(choices=choices, value=sel),
    )


# ── main step callback ─────────────────────────────────────────────────────────

def do_next(
    step_state,
    image,
    backbone_id:    str,
    tau1: float, tau2: float, tau3: float,
    max_k:          int,
    max_j:          int,
    mode:           str,
    sigma_str:      str,
    adaptive:       bool,
    use_dino_crops: bool,
    use_largest_cc: bool,
    kept_thresh:    float,
    nms_iou:        float,
    current_visible,   # current mask_select.value (list[str])
):
    """
    Run one bipartition step.  Auto-initialises on the first call or after Reset.
    Hyperparameters are baked into the config at init; use Reset to change them.
    """
    sigma = None
    if sigma_str.strip():
        try:
            sigma = float(sigma_str.strip())
        except ValueError:
            return _no_result(f"Invalid σ: {sigma_str!r}")

    # ── initialise on first call ──────────────────────────────────────────────
    if step_state is None:
        image_rgb = _to_rgb(image)
        if image_rgb is None:
            return _no_result("Upload an image first")
        try:
            model      = HierarchicalModel.get_or_load(backbone_id)
            full_feats = model._extract(image_rgb)
        except Exception as exc:
            return _no_result(f"Feature extraction failed: {exc}")

        all_taus = [tau1, tau2, tau3]
        cfg = {
            "backbone_id":    backbone_id,
            "tau_per_level":  [float(t) for t in all_taus[:int(max_j)]],
            "max_k":          int(max_k),
            "max_j":          int(max_j),
            "mode":           mode,
            "sigma":          sigma,
            "adaptive":       adaptive,
            "use_dino_crops": bool(use_dino_crops),
            "use_largest_cc": bool(use_largest_cc),
            "kept_thresh":    float(kept_thresh),
            "nms_iou":        float(nms_iou),
            "min_mask_frac":  0.02,
        }
        step_state = _new_state(image_rgb, full_feats, cfg)

    # ── done? ──────────────────────────────────────────────────────────────────
    if not step_state["queue"]:
        return _render(step_state, _queue_status(step_state),
                       prev_visible=current_visible)

    # ── one bipartition step ───────────────────────────────────────────────────
    entry = step_state["queue"][0]
    cfg   = step_state["cfg"]
    taus  = cfg["tau_per_level"]
    tau   = taus[entry["level"] - 1] if entry["level"] - 1 < len(taus) else taus[-1]

    feats     = entry["feats"]
    Hf, Wf   = feats.shape[:2]
    n_patches = Hf * Wf

    try:
        fg, _ = spectral_bipartition_2d(
            feats, tau, cfg["mode"], cfg["sigma"], cfg["adaptive"],
            entry["painting"],
        )
    except Exception as exc:
        step_state["queue"].pop(0)
        return _render(step_state, f"Bipartition failed: {exc}, skipping",
                       prev_visible=current_visible)

    if int(fg.sum()) < max(1, int(cfg["min_mask_frac"] * n_patches)):
        step_state["queue"].pop(0)
        return _render(step_state,
                       f"Level {entry['level']}: no foreground left, moving on…",
                       prev_visible=current_visible)

    # ── patch mask → full-image bool mask ─────────────────────────────────────
    H, W = step_state["image_rgb"].shape[:2]
    if entry["bbox"] is None:
        img_m = _upscale(fg, H, W)
    else:
        ymin, ymax, xmin, xmax = entry["bbox"]
        img_m = np.zeros((H, W), dtype=bool)
        img_m[ymin:ymax, xmin:xmax] = _upscale(fg, ymax - ymin, xmax - xmin)

    if cfg["use_largest_cc"]:
        img_m = _largest_connected(img_m)

    # ── coverage filter: child must overlap sufficiently with its parent ───────
    # (Only meaningful for level 2+; level-1 has no parent to compare against.)
    if entry["level"] > 1 and entry["parent_idx"] is not None:
        parent_list = masks_at_level(step_state["found"], entry["level"] - 1)
        if entry["parent_idx"] < len(parent_list):
            cov = _coverage(img_m, parent_list[entry["parent_idx"]].mask)
            if cov < cfg["kept_thresh"]:
                entry["painting"] = entry["painting"] | fg
                entry["count"] += 1
                if entry["count"] >= cfg["max_k"]:
                    step_state["queue"].pop(0)
                return _render(step_state,
                               f"Filtered: coverage {cov:.2f} < kept_thresh {cfg['kept_thresh']:.2f}",
                               prev_visible=current_visible)

    # ── NMS / IoU filter: no two sibling masks should overlap too much ─────────
    siblings = [sm for sm in step_state["found"]
                if sm.level == entry["level"] and sm.parent_idx == entry["parent_idx"]]
    if any(_iou(img_m, sm.mask) > cfg["nms_iou"] for sm in siblings):
        entry["painting"] = entry["painting"] | fg
        entry["count"] += 1
        if entry["count"] >= cfg["max_k"]:
            step_state["queue"].pop(0)
        return _render(step_state,
                       f"Filtered: IoU with sibling > nms_iou {cfg['nms_iou']:.2f}",
                       prev_visible=current_visible)

    new_mask = SegMask(
        mask=img_m, level=entry["level"],
        parent_idx=entry["parent_idx"], child_idx=entry["count"],
    )
    step_state["found"].append(new_mask)
    entry["painting"] = entry["painting"] | fg
    entry["count"]   += 1
    if entry["count"] >= cfg["max_k"]:
        step_state["queue"].pop(0)

    # ── schedule a deeper level (BFS: append to back of queue) ────────────────
    if entry["level"] < cfg["max_j"]:
        parent_global = len(masks_at_level(step_state["found"], entry["level"])) - 1
        y0, y1, x0, x1 = smallest_square_containing_mask(img_m)
        y1, x1 = y1 + 1, x1 + 1
        model = HierarchicalModel.get_or_load(cfg["backbone_id"])
        if (y1 - y0) >= model._patch and (x1 - x0) >= model._patch:
            crop_feats = model._extract_crop(
                step_state["image_rgb"], step_state["full_feats"],
                y0, y1, x0, x1, use_dino=cfg["use_dino_crops"],
            )
            step_state["queue"].append({
                "level":      entry["level"] + 1,
                "parent_idx": parent_global,
                "feats":      crop_feats,
                "painting":   np.zeros(crop_feats.shape[:2], dtype=bool),
                "count":      0,
                "bbox":       (y0, y1, x0, x1),
            })

    return _render(step_state, _queue_status(step_state), new_mask,
                   prev_visible=current_visible)


def do_reset() -> tuple:
    upd = gr.update(maximum=0, value=0)
    return (None, None, None,
            "Reset — upload an image and press 'Next Mask'.",
            None, None, None, None, None,
            upd, upd, upd,
            gr.update(choices=[], value=[]))


# ── navigation callbacks ───────────────────────────────────────────────────────

def nav_l1(image_rgb, found, idx: int):
    if image_rgb is None or not found:
        return None
    return visualize_level1_selected(image_rgb, found, int(idx))


def nav_l2_parent(image_rgb, found, parent_idx: int):
    if image_rgb is None or not found:
        return None, None, gr.update(maximum=0, value=0)
    l1  = masks_at_level(found, 1)
    idx = int(parent_idx) % max(1, len(l1))
    n   = len(children_of(found, 2, idx))
    return (
        visualize_level1_selected(image_rgb, found, idx),
        visualize_children_crop(image_rgb, found, idx),
        gr.update(maximum=max(0, n - 1), value=0),
    )


def nav_l2_child(image_rgb, found, parent_idx: int, child_idx: int):
    if image_rgb is None or not found:
        return None
    return visualize_child_single(image_rgb, found, int(parent_idx), int(child_idx))


def rerender_overview(image_rgb, found, selected):
    if image_rgb is None or not found:
        return None
    visible = {int(v) for v in (selected or [])}
    return visualize_filtered(image_rgb, found, visible)


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_app(image_dir: str = _DEFAULT_IMAGE_DIR) -> gr.Blocks:
    with gr.Blocks(title="Hierarchical DINOv3 Segmentation") as demo:

        gr.Markdown("## Hierarchical DINOv3 Segmentation")
        gr.Markdown(
            "Each **⊕ Next Mask** extracts one mask via spectral bipartitioning on DINOv3 features. "
            "Level-1 first (whole image), then level-2 crops per parent (BFS).  "
            "**🔄 Reset** to change hyperparameters."
        )

        s_step  = gr.State(None)
        s_image = gr.State(None)
        s_seg   = gr.State(None)

        with gr.Row():

            # ── left: controls ─────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):

                image_in = gr.Image(label="Input image", type="pil")
                with gr.Row():
                    btn_rand  = gr.Button("🎲 Random ImageNet", variant="secondary", scale=1)
                    btn_next  = gr.Button("⊕ Next Mask",        variant="primary",   scale=2)
                    btn_reset = gr.Button("🔄 Reset",           variant="stop",      scale=1)
                status_box = gr.Textbox(
                    label="Status / queue", interactive=False,
                    value="Upload an image and press 'Next Mask'.",
                )

                with gr.Accordion("Thresholds", open=True):
                    gr.Markdown(
                        "*Sliders shown/hidden as **max_j** changes.  "
                        "Baked in at first press — Reset to change.*"
                    )
                    tau_sliders: list[gr.Slider] = []
                    for i in range(MAX_LEVELS):
                        s = gr.Slider(
                            0.0, 1.0,
                            value=_DEFAULT_TAUS[i],
                            step=0.01,
                            label=f"tau Level {i + 1}{'  (easier)' if i == 0 else '  (harder)' if i == 1 else ''}",
                            visible=(i < DEFAULT_MAX_J),
                        )
                        tau_sliders.append(s)
                    adaptive = gr.Checkbox(
                        value=True,
                        label="Adaptive  (tau = quantile of similarity distribution)",
                    )

                with gr.Accordion("Structure", open=True):
                    max_k = gr.Slider(1, 8,          value=DEFAULT_MAX_K,  step=1,
                                      label="max_k  — masks extracted per region")
                    max_j = gr.Slider(1, MAX_LEVELS, value=DEFAULT_MAX_J,  step=1,
                                      label="max_j  — hierarchy depth")
                    kept_thresh = gr.Slider(0.0, 1.0, value=0.9, step=0.05,
                                            label="kept_thresh  — min fraction of child mask covered by parent  (level 2+ only)")
                    nms_iou     = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                            label="nms_iou  — max IoU between sibling masks before one is dropped")

                with gr.Accordion("Similarity & features", open=False):
                    mode           = gr.Radio(["rbf", "cosine"], value="rbf",
                                              label="Affinity mode")
                    sigma_str      = gr.Textbox(value="", label="RBF σ  (empty = auto-median)",
                                                placeholder="e.g. 3.5")
                    use_dino_crops = gr.Checkbox(value=True,
                                                 label="Re-run DINOv3 on each crop  "
                                                       "(uncheck = bilinear interpolation)")
                    use_largest_cc = gr.Checkbox(value=True,
                                                 label="Largest connected component only")

                with gr.Accordion("Advanced", open=False):
                    backbone_id   = gr.Textbox(value=DEFAULT_BACKBONE_ID, label="DINOv3 model ID")
                    image_dir_box = gr.Textbox(value=image_dir, label="ImageNet images directory")

            # ── right: results ──────────────────────────────────────────────
            with gr.Column(scale=2):
                with gr.Tabs():

                    with gr.Tab("Overview"):
                        overview_img = gr.Image(type="numpy", interactive=False,
                                                label="Found masks")
                        mask_select  = gr.CheckboxGroup(
                            choices=[], value=[],
                            label="Visible masks  (uncheck to hide from overview)",
                        )

                    with gr.Tab("Level 1"):
                        l1_idx = gr.Slider(0, 0, value=0, step=1,
                                           label="Level-1 mask index")
                        l1_img = gr.Image(type="numpy", interactive=False,
                                          label="Selected level-1 mask (white contour)")

                    with gr.Tab("Level 2"):
                        l2_parent = gr.Slider(0, 0, value=0, step=1,
                                              label="Parent (level-1 index)")
                        with gr.Row():
                            l2_parent_img = gr.Image(type="numpy", interactive=False, scale=1,
                                                     label="Parent highlighted")
                            l2_crop_img   = gr.Image(type="numpy", interactive=False, scale=1,
                                                     label="All children overlaid (crop)")
                        l2_child      = gr.Slider(0, 0, value=0, step=1,
                                                  label="Child index")
                        l2_single_img = gr.Image(type="numpy", interactive=False,
                                                 label="Single child mask (crop)")

        # ── event wiring ──────────────────────────────────────────────────────

        _next_inputs = [
            s_step, image_in, backbone_id,
            *tau_sliders,   # tau1, tau2, tau3, tau4
            max_k, max_j, mode, sigma_str, adaptive,
            use_dino_crops, use_largest_cc,
            kept_thresh, nms_iou,
            mask_select,
        ]
        _all_outputs = [
            s_step, s_image, s_seg, status_box,
            overview_img,
            l1_img,
            l2_parent_img, l2_crop_img, l2_single_img,
            l1_idx, l2_parent, l2_child,
            mask_select,
        ]

        btn_next.click(fn=do_next,   inputs=_next_inputs, outputs=_all_outputs)
        btn_reset.click(fn=do_reset, inputs=[],           outputs=_all_outputs)

        # Show/hide tau sliders when max_j changes
        max_j.change(
            fn=lambda j: [gr.update(visible=(i < int(j))) for i in range(MAX_LEVELS)],
            inputs=[max_j],
            outputs=tau_sliders,
        )

        # Level-1 navigation
        l1_idx.change(fn=nav_l1, inputs=[s_image, s_seg, l1_idx], outputs=[l1_img])

        # Level-2 parent navigation
        l2_parent.change(
            fn=nav_l2_parent,
            inputs=[s_image, s_seg, l2_parent],
            outputs=[l2_parent_img, l2_crop_img, l2_child],
        ).then(
            fn=nav_l2_child,
            inputs=[s_image, s_seg, l2_parent, l2_child],
            outputs=[l2_single_img],
        )

        # Level-2 child navigation
        l2_child.change(
            fn=nav_l2_child,
            inputs=[s_image, s_seg, l2_parent, l2_child],
            outputs=[l2_single_img],
        )

        # Visibility checkboxes → rerender overview without re-running segmentation
        mask_select.change(
            fn=rerender_overview,
            inputs=[s_image, s_seg, mask_select],
            outputs=[overview_img],
        )

        # Random ImageNet
        btn_rand.click(fn=lambda d: _random_image(d), inputs=[image_dir_box], outputs=[image_in])

    return demo


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=_DEFAULT_IMAGE_DIR)
    parser.add_argument("--port",  type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    build_app(args.image_dir).launch(server_port=args.port, share=args.share)
