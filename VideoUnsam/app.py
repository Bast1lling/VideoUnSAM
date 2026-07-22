"""Gradio explorer for the UnSAMv2 granular-mask training data.

    python VideoUnsam/app.py [--image-dir DIR] [--port 7860]

Works before the SA-1B images are downloaded: masks are drawn on a grey
canvas, and the image directory can be pointed at the extracted shard from
inside the UI once it is ready.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import gradio as gr
import numpy as np
from matplotlib import colormaps

from unsamv2_data import (
    DEFAULT_ANN_DIR,
    Annotation,
    Sample,
    decode_all,
    display_size,
    find_image_dir,
    list_image_ids,
    load_image,
    load_sample,
)

MAX_SIDE = 900          # display resolution; masks are decoded to match
GALLERY_LIMIT = 48      # per-mask crops rendered in the gallery tab

ANN_DIR = str(DEFAULT_ANN_DIR)
IMAGE_IDS: list[int] = []


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _palette_color(idx: int) -> np.ndarray:
    """Distinct, evenly spread hue per mask index."""
    h = (idx * 0.6180339887) % 1.0
    s = 0.65 + 0.25 * ((idx // 3) % 2)
    v = 0.85 + 0.15 * ((idx // 5) % 2)
    return np.array(colorsys.hsv_to_rgb(h, s, v)) * 255


def _granularity_color(g: float) -> np.ndarray:
    return np.array(colormaps["turbo"](float(np.clip(g, 0, 1)))[:3]) * 255


def _boundary(mask: np.ndarray) -> np.ndarray:
    """1px inner boundary, via numpy shifts (no scipy/cv2 dependency)."""
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return mask & ~eroded


def _overlay(
    base: np.ndarray,
    masks: list[np.ndarray],
    colors: list[np.ndarray],
    alpha: float,
    outline: bool,
) -> np.ndarray:
    """Alpha-blend masks onto the base image, largest first so small masks win."""
    canvas = base.astype(np.float32)
    order = sorted(range(len(masks)), key=lambda i: masks[i].sum(), reverse=True)
    for i in order:
        m, c = masks[i], colors[i]
        canvas[m] = (1 - alpha) * canvas[m] + alpha * c
    if outline:
        for i in order:
            b = _boundary(masks[i])
            canvas[b] = np.minimum(colors[i] * 1.25, 255)
    return canvas.clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

def _select(
    sample: Sample,
    g_min: float,
    g_max: float,
    kinds: list[str],
    instance: str,
    min_area_pct: float,
    max_area_pct: float,
) -> list[int]:
    """Indices of annotations passing the current filters."""
    total = sample.width * sample.height
    keep = []
    for i, a in enumerate(sample.annotations):
        if not (g_min <= a.granularity <= g_max):
            continue
        if kinds and a.kind not in kinds:
            continue
        if instance != "all" and str(a.instance_id) != instance:
            continue
        pct = 100.0 * a.area / total
        if not (min_area_pct <= pct <= max_area_pct):
            continue
        keep.append(i)
    return keep


def _stats_md(sample: Sample, kept: list[int], image_found: bool, image_dir) -> str:
    anns = sample.annotations
    gs = [anns[i].granularity for i in kept]
    total = sample.width * sample.height
    stats = sample.stats or {}

    img_line = (
        f"image found in `{image_dir}`"
        if image_found
        else "**image not found** — masks drawn on grey canvas"
    )
    lines = [
        f"### `{sample.file_name}` — {sample.width}×{sample.height}",
        f"{img_line}",
        "",
        f"**{len(kept)} / {len(anns)} masks shown**",
        f"- granularity of shown: {min(gs):.2f} – {max(gs):.2f}" if gs else "- (nothing selected)",
        f"- instances: {stats.get('total_instances', '?')} · parts: {stats.get('total_parts', '?')}",
        f"- area of shown: {100 * sum(anns[i].area for i in kept) / total:.1f}% of frame (with overlap)",
    ]
    mapping = stats.get("instance_part_mapping")
    if mapping:
        pretty = ", ".join(f"inst {k} → {v} parts" for k, v in mapping.items())
        lines.append(f"- decomposition: {pretty}")
    return "\n".join(lines)


def _mask_rows(sample: Sample, kept: list[int]) -> list[list]:
    total = sample.width * sample.height
    rows = []
    for i in kept:
        a = sample.annotations[i]
        rows.append(
            [
                a.id,
                round(a.granularity, 3),
                a.instance_id,
                a.kind,
                a.is_divide,
                round(100 * a.area / total, 3),
                "{}, {}, {}, {}".format(*(round(v) for v in a.bbox)),
            ]
        )
    return sorted(rows, key=lambda r: (-r[1], r[5]))


def _crops(sample: Sample, masks: list[np.ndarray], base: np.ndarray, kept: list[int]):
    """Per-mask thumbnails: bbox crop with the mask tinted."""
    order = sorted(kept, key=lambda i: -sample.annotations[i].granularity)[:GALLERY_LIMIT]
    h, w = base.shape[:2]
    sx, sy = w / sample.width, h / sample.height
    out = []
    for i in order:
        a = sample.annotations[i]
        x, y, bw, bh = a.bbox
        x0, y0 = max(0, int(x * sx) - 4), max(0, int(y * sy) - 4)
        x1, y1 = min(w, int((x + bw) * sx) + 4), min(h, int((y + bh) * sy) + 4)
        if x1 <= x0 or y1 <= y0:
            continue
        sub_base = base[y0:y1, x0:x1]
        sub_mask = masks[i][y0:y1, x0:x1]
        tile = sub_base.astype(np.float32)
        tile[~sub_mask] *= 0.25
        tile[sub_mask] = 0.55 * tile[sub_mask] + 0.45 * _granularity_color(a.granularity)
        out.append((tile.clip(0, 255).astype(np.uint8), f"#{a.id} · g={a.granularity:.2f} · {a.kind}"))
    return out


# --------------------------------------------------------------------------- #
# callbacks
# --------------------------------------------------------------------------- #

def render(idx, g_min, g_max, kinds, instance, min_area, max_area, alpha, color_by,
           outline, image_dir_text, want_gallery):
    image_id = IMAGE_IDS[int(idx)]
    sample = load_sample(ANN_DIR, image_id)
    masks = decode_all(ANN_DIR, image_id, MAX_SIDE)

    image_dir = find_image_dir(image_dir_text or None)
    base, found = load_image(image_dir, sample, MAX_SIDE)

    if g_min > g_max:
        g_min, g_max = g_max, g_min
    kept = _select(sample, g_min, g_max, kinds, instance, min_area, max_area)

    if color_by == "granularity":
        colors = [_granularity_color(sample.annotations[i].granularity) for i in kept]
    elif color_by == "instance":
        colors = [_palette_color(sample.annotations[i].instance_id) for i in kept]
    else:
        colors = [_palette_color(i) for i in kept]

    view = _overlay(base, [masks[i] for i in kept], colors, alpha, outline)
    gallery = _crops(sample, masks, base, kept) if want_gallery else []

    instances = ["all"] + [str(i) for i in sorted({a.instance_id for a in sample.annotations})]
    return (
        view,
        _stats_md(sample, kept, found, image_dir),
        _mask_rows(sample, kept),
        gallery,
        gr.update(choices=instances, value=instance if instance in instances else "all"),
        f"{image_id}",
    )


def probe(idx, g_min, g_max, kinds, instance, min_area, max_area, evt: gr.SelectData):
    """Report the smallest visible mask under the clicked pixel."""
    if evt.index is None:
        return "click inside the image"
    x, y = evt.index
    image_id = IMAGE_IDS[int(idx)]
    sample = load_sample(ANN_DIR, image_id)
    masks = decode_all(ANN_DIR, image_id, MAX_SIDE)
    if g_min > g_max:
        g_min, g_max = g_max, g_min
    kept = _select(sample, g_min, g_max, kinds, instance, min_area, max_area)

    hits = [i for i in kept if 0 <= y < masks[i].shape[0] and 0 <= x < masks[i].shape[1] and masks[i][y, x]]
    if not hits:
        return f"({x}, {y}) — no visible mask here"
    hits.sort(key=lambda i: sample.annotations[i].area)
    head = sample.annotations[hits[0]]
    others = ", ".join(f"#{sample.annotations[i].id} (g={sample.annotations[i].granularity:.2f})" for i in hits[1:6])
    txt = (
        f"**({x}, {y})** → smallest: **#{head.id}** · g={head.granularity:.3f} · "
        f"instance {head.instance_id} · {head.kind} · divide={head.is_divide} · "
        f"area={100 * head.area / (sample.width * sample.height):.3f}%"
    )
    if others:
        txt += f"\n\nalso covered by {len(hits) - 1}: {others}"
    return txt


def jump(idx, delta):
    return int(np.clip(int(idx) + delta, 0, len(IMAGE_IDS) - 1))


def goto_id(text, idx):
    try:
        wanted = int(str(text).strip().replace("sa_", "").replace(".json", ""))
    except ValueError:
        return idx
    return IMAGE_IDS.index(wanted) if wanted in IMAGE_IDS else idx


# --------------------------------------------------------------------------- #
# ui
# --------------------------------------------------------------------------- #

def build(image_dir_default: str) -> gr.Blocks:
    with gr.Blocks(title="UnSAMv2 mask explorer", fill_width=True) as demo:
        gr.Markdown(
            f"## UnSAMv2 mask explorer — {len(IMAGE_IDS)} images from `{ANN_DIR}`\n"
            "Drag the **granularity** range to sweep from whole objects (1.0) to fine parts (0.1). "
            "Click the image to identify the mask under the cursor."
        )

        with gr.Row():
            with gr.Column(scale=1):
                idx = gr.Slider(0, max(0, len(IMAGE_IDS) - 1), value=0, step=1, label="image index")
                with gr.Row():
                    prev_btn = gr.Button("◀ prev")
                    next_btn = gr.Button("next ▶")
                    rand_btn = gr.Button("🎲 random")
                with gr.Row():
                    id_box = gr.Textbox(label="jump to image id", scale=3)
                    go_btn = gr.Button("go", scale=1)

                gr.Markdown("**Granularity**")
                g_min = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="min")
                g_max = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="max")

                kinds = gr.CheckboxGroup(
                    ["instance", "part", "part+conquer"],
                    value=["instance", "part", "part+conquer"],
                    label="mask kind",
                )
                instance = gr.Dropdown(["all"], value="all", label="instance id")

                gr.Markdown("**Area filter (% of frame)**")
                min_area = gr.Slider(0.0, 100.0, value=0.0, step=0.01, label="min")
                max_area = gr.Slider(0.0, 100.0, value=100.0, step=0.5, label="max")

                gr.Markdown("**Appearance**")
                color_by = gr.Radio(["granularity", "instance", "index"], value="granularity", label="colour by")
                alpha = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="opacity")
                outline = gr.Checkbox(value=True, label="draw outlines")
                image_dir_text = gr.Textbox(
                    value=image_dir_default,
                    label="image directory (set once the SA-1B shard is extracted)",
                )
                want_gallery = gr.Checkbox(value=False, label="build per-mask gallery (slower)")

            with gr.Column(scale=3):
                view = gr.Image(label="masks", type="numpy", height=760)
                probe_md = gr.Markdown("click the image to inspect a mask")
                stats_md = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("mask table"):
                        table = gr.Dataframe(
                            headers=["id", "granularity", "instance", "kind", "is_divide", "area %", "bbox"],
                            row_count=(0, "dynamic"),
                            wrap=True,
                        )
                    with gr.Tab("per-mask crops"):
                        gallery = gr.Gallery(columns=6, height=520, object_fit="contain")

        controls = [idx, g_min, g_max, kinds, instance, min_area, max_area,
                    alpha, color_by, outline, image_dir_text, want_gallery]
        outputs = [view, stats_md, table, gallery, instance, id_box]

        for c in controls:
            c.change(render, controls, outputs, show_progress="minimal")

        prev_btn.click(lambda i: jump(i, -1), idx, idx)
        next_btn.click(lambda i: jump(i, +1), idx, idx)
        rand_btn.click(lambda: int(np.random.randint(len(IMAGE_IDS))), None, idx)
        go_btn.click(goto_id, [id_box, idx], idx)
        id_box.submit(goto_id, [id_box, idx], idx)

        view.select(probe, [idx, g_min, g_max, kinds, instance, min_area, max_area], probe_md)
        demo.load(render, controls, outputs)

    return demo


def main() -> None:
    global ANN_DIR, IMAGE_IDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-dir", default=str(DEFAULT_ANN_DIR), help="directory of sa_*.json mask files")
    parser.add_argument("--image-dir", default=None, help="directory of extracted SA-1B sa_*.jpg files")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    ANN_DIR = str(Path(args.ann_dir).expanduser())
    IMAGE_IDS = list_image_ids(ANN_DIR)
    if not IMAGE_IDS:
        raise SystemExit(f"no sa_*.json files found in {ANN_DIR}")

    found = find_image_dir(args.image_dir)
    print(f"{len(IMAGE_IDS)} annotated images in {ANN_DIR}")
    print(f"images: {found if found else 'not found yet — masks will render on a grey canvas'}")

    build(args.image_dir or (str(found) if found else "")).launch(
        server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
