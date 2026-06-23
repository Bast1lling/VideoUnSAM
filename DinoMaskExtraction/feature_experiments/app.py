"""Gradio app for foundational DINOv3 feature experiments.

Experiment 1 — **feature-edge map + graph segmentation**. Load a random ImageNet
image, read DINOv3 patch features at the chosen layer, optionally preprocess
them (resolution / PCA reduction / whitening / Gaussian smoothing), and show the
Sobel-like feature-edge map (neighbour cosine/euclidean distance) as a greyscale
boundary image. The same neighbour distances drive a graph cut: every patch
starts linked to its neighbours, connections with distance above the threshold
are cut (the boundaries), and the surviving connected components are coloured as
masks (raw + overlaid on the image).

Two cost tiers of live controls:

* **Re-extract** (resolution, apply-LayerNorm) — these change what the backbone
  returns, so the image is pushed through DINOv3 again.
* **Recompute** (PCA, whiten, smoothing, metric, connectivity, display) — cheap
  ops on the cached per-layer grids; no backbone call.

Run::

    uv run python -m feature_experiments.app
    uv run python -m feature_experiments.app --config feature_experiments/config.yaml
"""
from __future__ import annotations

import argparse
import random

import gradio as gr
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from backbone import HookedViTFeatV3, pca_to_rgb
from dataset import ImageNet1k
from mask_extraction import color_labels, overlay_masks, reduce_features
from unsup_seg.pca import prep_image

from .edges import feature_edge_map, graph_segment, grow_and_merge, to_greyscale

_RESAMPLE = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR}
# patch grid is res/16 per side, so these are the feature-resolution choices
RESOLUTIONS = [224, 336, 448, 512, 672, 768, 1024]


class Inspector:
    """Holds the dataset and a lazily-built DINOv3 extractor + per-image cache."""

    def __init__(self, cfg: dict, config_path: str = "feature_experiments/config.yaml"):
        self.cfg = cfg
        self.config_path = config_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.images = ImageNet1k(split=cfg["split"])
        self._extractor: HookedViTFeatV3 | None = None
        # cache of the last extracted image so cheap control tweaks are instant
        self._cur_pil: Image.Image | None = None
        self._cur_grids: dict[str, torch.Tensor] | None = None
        self._cur_idx: int | None = None
        self._cur_hw: tuple[int, int] | None = None
        print(f"feature_experiments: {len(self.images)} images in split '{cfg['split']}'")

    def _ext(self) -> HookedViTFeatV3:
        if self._extractor is None:
            self._extractor = HookedViTFeatV3(device=self.device)
        return self._extractor

    def make_saver(self, keys: list[str]):
        """Return a Gradio handler that writes the given control values to the YAML.

        Non-control entries already in the config (e.g. ``split``) are preserved.
        """
        def save(*values) -> str:
            self.cfg.update({k: v for k, v in zip(keys, values)})
            with open(self.config_path, "w") as fh:
                yaml.safe_dump(self.cfg, fh, sort_keys=False, default_flow_style=False)
            return f"✅ Saved {len(keys)} settings to `{self.config_path}` (reloads on restart)."
        return save

    # ------------------------------------------------------------------ logic
    def _extract(self, idx: int, res: int, apply_norm: bool) -> None:
        """Sample/re-extract image ``idx`` at ``res``; cache prepped PIL + grids."""
        pil, _ = self.images[idx]
        pil = prep_image(pil, int(res))
        grids, fh, fw = self._ext().extract(pil, apply_norm=bool(apply_norm))
        self._cur_pil, self._cur_grids, self._cur_idx, self._cur_hw = pil, grids, idx, (fh, fw)

    def _processed_grid(self, layer, pca, whiten, smooth, l2_normalize) -> torch.Tensor:
        """Apply PCA / whitening / smoothing / L2-norm to the cached layer grid."""
        grid = reduce_features(self._cur_grids[layer], int(pca), float(smooth), bool(whiten))
        if l2_normalize:
            grid = F.normalize(grid, dim=-1)
        return grid

    def _to_canvas(self, img: Image.Image, res: int, upscale: str) -> Image.Image:
        return img.resize((int(res), int(res)), _RESAMPLE[upscale])

    def _mask_pair(self, labels, res, seed, alpha):
        """Colour a label map + overlay it on the current image (bg = 0)."""
        masks = self._to_canvas(
            Image.fromarray(color_labels(labels, seed=int(seed), bg_label=0)),
            res, "nearest")
        overlay = overlay_masks(self._cur_pil, labels, alpha=float(alpha),
                                seed=int(seed), bg_label=0)
        return masks, overlay

    def render(self, res, layer, pca, whiten, smooth, l2_normalize,
               metric, connectivity, radii, display_q, sharpen, upscale,
               seg_by, seg_q, seg_thr, seg_min, seg_seed, seg_alpha,
               merge_mode, merge_by, merge_thr, merge_q, seam_thickness, apply_norm):
        """Re-render the cached image with the given controls (no new sampling)."""
        blank = (None,) * 8
        if self._cur_grids is None:
            return (None, *blank, "Press **Random image** to begin.")
        if not radii:
            return (self._cur_pil, *blank, "Select at least one neighbour radius.")
        grid = self._processed_grid(layer, pca, whiten, smooth, l2_normalize)
        radii_i = [int(r) for r in radii]

        pca_rgb = self._to_canvas(
            Image.fromarray(pca_to_rgb(grid).numpy(), mode="RGB"), res, "nearest")
        edge = feature_edge_map(grid, metric=metric, connectivity=int(connectivity),
                                radii=radii_i)
        grey = Image.fromarray(
            to_greyscale(edge, q=float(display_q), gamma=float(sharpen)), mode="L")
        edge_img = self._to_canvas(grey, res, upscale)

        # stage 1 — cut edges above the threshold, label connected components
        thr = float(seg_thr) if seg_by == "absolute" else None
        labels, cut, n_masks, n_filtered = graph_segment(
            grid, metric=metric, connectivity=int(connectivity), radii=radii_i,
            threshold=thr, quantile=float(seg_q), min_size=int(seg_min))
        # stage 2 — grow cores into leftovers; stage 3 — merge similar masks
        grown, merged, n_grown, n_merged, merge_cut = grow_and_merge(
            grid, labels, metric=metric, connectivity=int(connectivity),
            merge_threshold=float(merge_thr), merge_mode=merge_mode,
            seam_thickness=int(seam_thickness),
            merge_by=merge_by, merge_quantile=float(merge_q))

        cut_masks, cut_ov = self._mask_pair(labels, res, seg_seed, seg_alpha)
        grow_masks, grow_ov = self._mask_pair(grown, res, seg_seed, seg_alpha)
        merge_masks, merge_ov = self._mask_pair(merged, res, seg_seed, seg_alpha)

        fh, fw = self._cur_hw
        radii_txt = "+".join(str(r) for r in sorted(radii_i))
        cut_txt = f"thr={seg_thr}" if seg_by == "absolute" else f"q={seg_q}→{cut:.3f}"
        filt_txt = f" (−{n_filtered} small)" if int(seg_min) > 0 and n_filtered else ""
        caption = (f"image index {self._cur_idx} | {int(res)}px → grid {fh}×{fw} | "
                   f"layer `{layer}` → {grid.shape[-1]}d"
                   + (f" PCA{int(pca)}" + ("(whiten)" if whiten else "") if int(pca) > 0 else "")
                   + (f" · σ{smooth}" if float(smooth) > 0 else "")
                   + f" | {metric} · {int(connectivity)}-neighbour · radius {radii_txt}"
                   + f" | cut {cut_txt} · min={int(seg_min)}"
                   + (f" | merge: {merge_mode}"
                      + (f" seam={int(seam_thickness)}" if merge_mode == "boundary" else ""))
                   + (f" {'q=' + str(merge_q) + '→' if merge_by == 'quantile' else '≥'}"
                      f"{merge_cut:.3f}")
                   + f" | **stages: {n_masks}{filt_txt} → grow {n_grown} → "
                   + f"merge {n_merged}**")
        return (self._cur_pil, pca_rgb, edge_img, cut_masks, cut_ov,
                grow_masks, grow_ov, merge_masks, merge_ov, caption)

    def run_random(self, *args):
        self._extract(random.randrange(len(self.images)), args[0], args[-1])
        return self.render(*args)

    def run_reextract(self, *args):
        """Re-run the backbone on the current image (resolution / apply_norm change)."""
        if self._cur_idx is None:
            return self.run_random(*args)
        self._extract(self._cur_idx, args[0], args[-1])
        return self.render(*args)


def build_ui(insp: Inspector) -> gr.Blocks:
    cfg = insp.cfg
    layers = ["final_norm"] + [f"block_{i:02d}" for i in range(23, -1, -1)] + ["embeddings"]
    res_default = cfg["res"] if cfg["res"] in RESOLUTIONS else 512
    with gr.Blocks(title="DINOv3 Feature Experiments") as demo:
        gr.Markdown("## DINOv3 feature-edge map\n"
                    "Sobel-like boundary detector over patch features: per-patch "
                    "mean neighbour distance.\n`original | PCA-RGB features | "
                    "feature-edge (greyscale)`")

        with gr.Accordion("Feature source (re-extracts on change)", open=True):
            with gr.Row():
                res = gr.Dropdown(RESOLUTIONS, value=res_default,
                                  label="feature resolution (px, /16)")
                layer = gr.Dropdown(layers, value=cfg["layer"], label="feature layer")
                apply_norm = gr.Checkbox(value=bool(cfg["apply_norm"]),
                                         label="apply final LayerNorm")
        with gr.Accordion("Feature preprocessing", open=True):
            with gr.Row():
                pca = gr.Slider(0, 512, value=int(cfg["pca"]), step=1,
                                label="PCA components (0 = full feature dim)")
                whiten = gr.Checkbox(value=bool(cfg["whiten"]), label="whiten PCs")
                smooth = gr.Slider(0.0, 3.0, value=float(cfg["smooth"]), step=0.1,
                                   label="spatial smoothing σ (0 = off)")
                l2_normalize = gr.Checkbox(value=bool(cfg["l2_normalize"]),
                                           label="L2-normalise features")
        with gr.Accordion("Edge map & display", open=True):
            with gr.Row():
                metric = gr.Radio(["cosine", "euclidean"], value=cfg["metric"], label="metric")
                connectivity = gr.Radio([4, 8], value=int(cfg["connectivity"]),
                                        label="connectivity")
                radii = gr.CheckboxGroup(
                    [1, 2, 3, 4, 5, 6], value=[int(r) for r in cfg["radii"]],
                    label="neighbour radii (patches)",
                    info="1 = adjacent; pick several to average over scales")
            with gr.Row():
                sharpen = gr.Slider(0.2, 5.0, value=float(cfg["sharpen"]), step=0.1,
                                    label="edge sharpness γ (>1 sharpens, <1 softens)",
                                    info="contrast curve on the edge map (display only)")
                upscale = gr.Radio(["nearest", "bilinear"], value=cfg["upscale"],
                                   label="display upscaling")
                display_q = gr.Slider(0.0, 0.2, value=float(cfg["display_q"]), step=0.01,
                                      label="display clip quantile")
        with gr.Accordion("Graph segmentation (cut edges → connected-component masks)",
                          open=True):
            with gr.Row():
                seg_by = gr.Radio(["quantile", "absolute"], value=cfg["seg_by"],
                                  label="cut threshold by",
                                  info="cut a connection when its feature distance is above this")
                seg_q = gr.Slider(0.0, 1.0, value=float(cfg["seg_q"]), step=0.01,
                                  label="cut quantile (cut strongest 1-q of edges)")
                seg_thr = gr.Number(value=float(cfg["seg_thr"]),
                                    label="absolute cut distance")
            with gr.Row():
                seg_min = gr.Slider(0, 50, value=int(cfg["seg_min"]), step=1,
                                    label="min mask size — final filter, drops "
                                          "components with fewer patches (0 = keep all)")
                seg_seed = gr.Number(value=int(cfg["seg_seed"]), precision=0,
                                     label="colour seed")
                seg_alpha = gr.Slider(0.0, 1.0, value=float(cfg["seg_alpha"]), step=0.05,
                                      label="overlay alpha")
        with gr.Accordion("Grow & merge (greedy_multistage stages 2–3)", open=True):
            gr.Markdown("**Stage 2** grows the stage-1 cores into the leftover "
                        "(filtered-out) patches by feature similarity. **Stage 3** "
                        "merges masks whose aggregated prototypes are similar — the "
                        "top-level hierarchy. Raise *min mask size* above to create "
                        "leftovers for stage 2 to distribute.")
            with gr.Row():
                merge_mode = gr.Radio(
                    ["boundary", "prototype"], value=cfg["merge_mode"],
                    label="stage-3 merge",
                    info="boundary = greedy_multistage2 (adjacent masks, shared-seam "
                         "features); prototype = greedy_multistage (whole-mask means)")
                seam_thickness = gr.Slider(
                    1, 8, value=int(cfg["seam_thickness"]), step=1,
                    label="seam thickness (boundary mode; patches each side)")
            with gr.Row():
                merge_by = gr.Radio(
                    ["quantile", "absolute"], value=cfg["merge_by"],
                    label="merge threshold by",
                    info="quantile adapts to this image's pair-similarity spread")
                merge_q = gr.Slider(0.0, 1.0, value=float(cfg["merge_q"]), step=0.01,
                                    label="merge quantile (merge top 1-q most-similar pairs)")
                merge_thr = gr.Number(value=float(cfg["merge_thr"]),
                                      label="absolute merge similarity (higher = fewer merges)")

        with gr.Row():
            random_btn = gr.Button("Random image", variant="primary", scale=3)
            save_btn = gr.Button("💾 Save settings", variant="secondary", scale=1)
        caption = gr.Markdown()
        save_status = gr.Markdown()
        with gr.Row():
            orig = gr.Image(label="original", type="pil")
            pca_view = gr.Image(label="PCA-RGB features", type="pil")
            edge = gr.Image(label="feature-edge map", type="pil")
        with gr.Row():
            masks = gr.Image(label="stage 1 — cut masks", type="pil")
            overlay = gr.Image(label="stage 1 — overlay", type="pil")
        with gr.Row():
            grow_masks = gr.Image(label="stage 2 — grown masks", type="pil")
            grow_overlay = gr.Image(label="stage 2 — overlay", type="pil")
        with gr.Row():
            merge_masks = gr.Image(label="stage 3 — merged (final) masks", type="pil")
            merge_overlay = gr.Image(label="stage 3 — overlay", type="pil")

        live = [res, layer, pca, whiten, smooth, l2_normalize,
                metric, connectivity, radii, display_q, sharpen, upscale,
                seg_by, seg_q, seg_thr, seg_min, seg_seed, seg_alpha,
                merge_mode, merge_by, merge_thr, merge_q, seam_thickness, apply_norm]
        outs = [orig, pca_view, edge, masks, overlay,
                grow_masks, grow_overlay, merge_masks, merge_overlay, caption]
        random_btn.click(insp.run_random, live, outs)

        # save every control's value back to the YAML config (split is preserved)
        persist = {
            "res": res, "layer": layer, "apply_norm": apply_norm,
            "pca": pca, "whiten": whiten, "smooth": smooth, "l2_normalize": l2_normalize,
            "metric": metric, "connectivity": connectivity, "radii": radii,
            "display_q": display_q, "sharpen": sharpen, "upscale": upscale,
            "seg_by": seg_by, "seg_q": seg_q, "seg_thr": seg_thr, "seg_min": seg_min,
            "seg_seed": seg_seed, "seg_alpha": seg_alpha,
            "merge_mode": merge_mode, "merge_by": merge_by, "merge_thr": merge_thr,
            "merge_q": merge_q, "seam_thickness": seam_thickness,
        }
        save_btn.click(insp.make_saver(list(persist)), list(persist.values()), save_status)
        # resolution / apply_norm change what the backbone returns -> re-extract
        for ctrl in (res, apply_norm):
            ctrl.change(insp.run_reextract, live, outs)
        # everything else is a cheap recompute on the cached feature grids
        for ctrl in (layer, pca, whiten, smooth, l2_normalize,
                     metric, connectivity, radii, display_q, sharpen, upscale,
                     seg_by, seg_q, seg_thr, seg_min, seg_seed, seg_alpha,
                     merge_mode, merge_by, merge_thr, merge_q, seam_thickness):
            ctrl.change(insp.render, live, outs)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="feature_experiments/config.yaml")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    insp = Inspector(cfg, config_path=args.config)
    build_ui(insp).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
