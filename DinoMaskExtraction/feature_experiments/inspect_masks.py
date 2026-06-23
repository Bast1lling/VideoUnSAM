"""Quick Gradio viewer for masks written by ``feature_experiments.extract``.

Loads already-saved label-map PNGs (no model, no GPU), finds each one's source
ImageNet image, and shows ``original | coloured masks | overlay`` so you can
sanity-check that the batch run used the right parameters. The extraction
``config.yaml`` is shown verbatim alongside.

Run::

    uv run python -m feature_experiments.inspect_masks
    uv run python -m feature_experiments.inspect_masks --masks /path/to/masks --split train
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image

from mask_extraction import color_labels, overlay_masks
from unsup_seg.pca import prep_image

_IMG_EXTS = (".JPEG", ".jpeg", ".jpg", ".JPG", ".png", ".PNG")


class MaskBrowser:
    """Samples saved masks and pairs them with their source images for display."""

    def __init__(self, masks_root: str, images_root: str, config_path: str):
        self.masks_root = Path(masks_root)
        self.images_root = Path(images_root)
        if not self.masks_root.is_dir():
            raise FileNotFoundError(f"masks dir not found: {self.masks_root}")
        self.cfg = yaml.safe_load(open(config_path))
        self.res = int(self.cfg["res"])
        print(f"masks: {self.masks_root} | images: {self.images_root} | res {self.res}")

    # ---------------------------------------------------------------- sampling
    def _random_mask(self) -> Path | None:
        """Descend random sub-dirs to a leaf, then pick a random PNG (no full scan)."""
        for _ in range(20):
            d = self.masks_root
            for _ in range(8):                          # walk down to a leaf dir
                subdirs = [e.path for e in os.scandir(d) if e.is_dir()]
                if not subdirs:
                    break
                d = Path(random.choice(subdirs))
            pngs = [e.name for e in os.scandir(d) if e.name.lower().endswith(".png")]
            if pngs:
                return d / random.choice(pngs)
        return None

    def _source_image(self, mask_path: Path) -> Path | None:
        """Map ``<masks>/split/class/stem.png`` back to the source image file."""
        rel = mask_path.relative_to(self.masks_root)    # split/class/stem.png
        cls_dir = self.images_root / rel.parent
        for ext in _IMG_EXTS:
            cand = cls_dir / (rel.stem + ext)
            if cand.exists():
                return cand
        return None

    # ------------------------------------------------------------------ render
    def show(self, mask_path: Path, alpha: float, seed: int):
        labels = torch.from_numpy(np.array(Image.open(mask_path)).astype(np.int64))
        src = self._source_image(mask_path)
        if src is not None:
            pil = prep_image(Image.open(src), self.res)
            src_note = f"src `{src.name}`"
        else:                                           # show masks alone if image is missing
            pil = Image.new("RGB", (self.res, self.res), (32, 32, 32))
            src_note = "⚠️ source image not found"

        masks_img = Image.fromarray(color_labels(labels, seed=int(seed), bg_label=0)
                                    ).resize((self.res, self.res), Image.NEAREST)
        overlay = overlay_masks(pil, labels, alpha=float(alpha), seed=int(seed), bg_label=0)

        h, w = labels.shape
        n = int(labels.max())
        bg = float((labels == 0).float().mean()) * 100
        caption = (f"`{mask_path.relative_to(self.masks_root)}` | grid {h}×{w} | "
                   f"**{n} masks** | bg {bg:.1f}% | {src_note}")
        return pil, masks_img, overlay, caption

    def random(self, alpha: float, seed: int):
        mp = self._random_mask()
        if mp is None:
            return None, None, None, f"No `.png` masks found under {self.masks_root}"
        return self.show(mp, alpha, seed)

    def by_path(self, path: str, alpha: float, seed: int):
        if not path:
            return self.random(alpha, seed)
        p = Path(path)
        if not p.is_absolute():
            p = self.masks_root / path
        if not p.exists():
            return None, None, None, f"not found: `{p}`"
        return self.show(p, alpha, seed)


def build_ui(browser: MaskBrowser) -> gr.Blocks:
    params = yaml.safe_dump(browser.cfg, sort_keys=False, default_flow_style=False)
    with gr.Blocks(title="Saved Mask Inspector") as demo:
        gr.Markdown("## Saved mask inspector\n"
                    "`original | coloured masks | overlay` — loaded from disk "
                    "(masks written by `feature_experiments.extract`).")
        with gr.Accordion("Extraction config (config.yaml)", open=False):
            gr.Code(params, language="yaml")
        with gr.Row():
            alpha = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="overlay alpha")
            seed = gr.Number(value=0, precision=0, label="colour seed")
        with gr.Row():
            random_btn = gr.Button("Random saved mask", variant="primary", scale=3)
            path_box = gr.Textbox(label="…or a specific mask path "
                                        "(absolute or relative to masks root)", scale=3)
            show_btn = gr.Button("Show", scale=1)
        caption = gr.Markdown()
        with gr.Row():
            orig = gr.Image(label="original (prepped)", type="pil")
            masks = gr.Image(label="masks", type="pil")
            overlay = gr.Image(label="overlay", type="pil")

        outs = [orig, masks, overlay, caption]
        random_btn.click(browser.random, [alpha, seed], outs)
        show_btn.click(browser.by_path, [path_box, alpha, seed], outs)
        # re-render the current view when alpha/seed change is handled by re-clicking
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", default="/home/sebastian/data/imagenet1k/masks")
    ap.add_argument("--images", default="/home/sebastian/data/imagenet1k/images")
    ap.add_argument("--config", default="feature_experiments/config.yaml")
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    browser = MaskBrowser(args.masks, args.images, args.config)
    build_ui(browser).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
