"""
Gradio app to inspect DINOv3 features and prototype mask extraction.

A shared image (random ImageNet sample) is captured once via a hooked forward
pass that records the patch representation at every transformer block. Two tabs
then operate on it:

* **Feature inspector** -- PCA-to-RGB of each hooked location, and optionally of
  a *channel-concatenation* of several layers (the DINOv3 dense-decoder recipe).
* **Mask extraction** -- run a ``mask_extraction`` algorithm on chosen features /
  similarity metric / hyper-params and view the masks as colourful overlays.

Run::

    uv run python app.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

import gradio as gr

from dataset import ImageNet1k
from backbone import HookedViTFeatV3, pca_to_rgb
from mask_extraction import (
    EXTRACTORS, SIMILARITIES, color_labels, get_extractor, label_triple,
    overlay_masks, reduce_features, refine_crf,
)
from mask_extraction.pipeline import HierPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTIONS = [224, 336, 448, 512, 672, 768, 1024]
DEFAULT_RES = 448

# Stage 1 (coarse) greedy_multistage2 — algorithm knobs are LOCKED to the tuned
# values; only its feature grid and the whole of stage 2 are exposed in the UI.
COARSE_SIM = "cosine"
COARSE_PARAMS: dict = dict(
    connectivity=8,
    threshold=None,
    quantile=0.65,
    min_mask_size=10,
    consistency_filter=True,
    consistency_min=0.5,
    merge_threshold=0.7,
    seam_thickness=3,
)
COARSE_DESC = ("cosine · conn=8 · q=0.65 · min=10 · consistency<0.5 · "
               "merge≥0.7 · seam=3")

print("Loading dataset index ...")
DATASETS = {
    "train": ImageNet1k(split="train"),
    "validation": ImageNet1k(split="validation"),
}
print("Loading model ...")
EXTRACTOR = HookedViTFeatV3(device=DEVICE)
HOOK_NAMES = EXTRACTOR.hook_names
# a sensible default subset spanning the depth of the network
DEFAULT_HOOKS = ["block_00", "block_05", "block_11", "block_17", "block_23", "final_norm"]
# ~quartile-depth layers: the ViT-L analogue of the paper's [10,20,30,40] on 7B
CONCAT_DEFAULT = ["block_05", "block_11", "block_17", "block_23"]


# --------------------------------------------------------------------------- #
#  Persistent UI settings (app_config.yaml)                                   #
# --------------------------------------------------------------------------- #
CONFIG_PATH = Path(__file__).with_name("app_config.yaml")

# Every persisted control -> its factory default. The YAML stores exactly these
# keys; the order here is also the order the Save button feeds values back in.
DEFAULTS: dict = {
    "split": "train",
    "res": DEFAULT_RES,
    "ins_hooks": DEFAULT_HOOKS,
    "ins_apply_norm": False,
    "ins_combine": True,
    "ins_norm_combo": True,
    "msk_layers": ["final_norm"],
    "msk_apply_norm": False,
    "msk_norm_combo": True,
    "msk_algo": "greedy_graph",
    "msk_sim": "cosine",
    "msk_conn": 4,
    "msk_gamma": 0.0,
    "msk_seed": 0,
    "msk_pca": 0,
    "msk_whiten": False,
    "msk_smooth": 0.0,
    "msk_refine": False,
    "msk_ref_iters": 5,
    "msk_ref_radius": 4,
    "msk_ref_weight": 1.5,
    "msk_use_thr": False,
    "msk_thr": 0.5,
    "msk_q": 0.6,
    "msk_min": 0,
    "msk_cons_filter": False,
    "msk_cons_min": 0.5,
    "msk_merge": 0.85,
    "msk_seam": 1,
    "msk_seed_q": 0.8,
    "msk_seed_min": 2,
    "msk_mask_min": 0,
    "msk_proto": "mean",
    "msk_delta": 0.15,
    "msk_norm_sim": True,
    "msk_max_masks": 0,
    "msk_tau": 0.15,
    "msk_nmasks": 3,
    "msk_area": 0.01,
    # ---- tab 3: coarse ▸ fine pipeline (greedy_multistage2 twice)
    "pl_seed": 0,
    # stage 1 (coarse) features — algorithm params are locked (see COARSE_PARAMS)
    "pl_co_res": 512,
    "pl_co_layers": ["final_norm"],
    "pl_co_apply_norm": True,
    "pl_co_norm_combo": True,
    "pl_co_pca": 256,
    "pl_co_smooth": 0.5,
    "pl_co_whiten": False,
    # stage 2 (fine) features — runs at the global load resolution (the high one)
    "pl_fi_layers": ["final_norm"],
    "pl_fi_apply_norm": True,
    "pl_fi_norm_combo": True,
    "pl_fi_pca": 0,
    "pl_fi_smooth": 0.0,
    "pl_fi_whiten": False,
    # stage 2 (fine) greedy_multistage2 params
    "pl_fi_sim": "cosine",
    "pl_fi_gamma": 0.0,
    "pl_fi_conn": 8,
    "pl_fi_use_thr": False,
    "pl_fi_thr": 0.5,
    "pl_fi_q": 0.7,
    "pl_fi_min": 4,
    "pl_fi_cf": False,
    "pl_fi_cmin": 0.5,
    "pl_fi_merge": 0.85,
    "pl_fi_seam": 1,
    # final cross-mask merge (adjacent + feature-similar)
    "pl_final_merge_on": True,
    "pl_final_merge": 0.6,
    # visualisation
    "pl_detail_max": 6,
    "pl_crop_pad": 8,
}
CONFIG_KEYS = list(DEFAULTS)


def _write_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))


def load_config() -> dict:
    """Return defaults overlaid with ``app_config.yaml`` (creating it if absent)."""
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            saved = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            cfg.update({k: v for k, v in saved.items() if k in DEFAULTS})
            print(f"Loaded UI settings from {CONFIG_PATH.name}")
        except Exception as exc:                       # pragma: no cover
            print(f"  warning: could not read {CONFIG_PATH.name} ({exc}); using defaults")
    else:
        _write_config(cfg)
        print(f"Wrote default UI settings to {CONFIG_PATH.name}")
    return cfg


def save_config(*values) -> str:
    """Persist the current control values (Save button handler)."""
    cfg = dict(zip(CONFIG_KEYS, values))
    _write_config(cfg)
    return f"✅ Saved {len(cfg)} settings to `{CONFIG_PATH.name}` — will reload on restart."


CFG = load_config()


# --------------------------------------------------------------------------- #
#  Shared helpers                                                             #
# --------------------------------------------------------------------------- #
def _prep_image(pil: Image.Image, res: int) -> Image.Image:
    """Resize so the shorter side == res, then centre-crop to res x res."""
    pil = pil.convert("RGB")
    w, h = pil.size
    scale = res / min(w, h)
    pil = pil.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = pil.size
    left, top = (w - res) // 2, (h - res) // 2
    return pil.crop((left, top, left + res, top + res))


def _as_pil(image) -> Image.Image:
    return Image.fromarray(image) if isinstance(image, np.ndarray) else image


def build_feature_grid(grids: dict, names: list[str], normalize: bool) -> torch.Tensor:
    """Channel-concatenate the chosen layers into one ``(H, W, sum C)`` grid.

    When ``normalize`` is set, each layer is L2-normalised per patch *before*
    concatenation so that a high-magnitude layer (e.g. ``block_23``) cannot
    dominate the combined representation (see DINOV3_ARCHITECTURE.md).
    """
    parts = []
    for n in names:
        g = grids[n].float()
        if normalize:
            g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        parts.append(g)
    return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


def load_random(split: str, res: int):
    ds = DATASETS[split]
    idx = random.randrange(len(ds))
    pil, label = ds[idx]
    img = _prep_image(pil, res)
    info = f"**{split}** index {idx} | class folder {label:04d} | {res}×{res}"
    return img, info


# --------------------------------------------------------------------------- #
#  Tab 1: feature inspector                                                   #
# --------------------------------------------------------------------------- #
def inspect(image, selected: list[str], res: int, apply_norm: bool,
            combine: bool, normalize_combo: bool):
    if image is None:
        return [], "Load an image first."
    pil = _prep_image(_as_pil(image), res)
    grids, feat_h, feat_w = EXTRACTOR.extract(pil, apply_norm=apply_norm)
    chosen = [n for n in HOOK_NAMES if n in selected and n in grids]

    gallery = []
    if combine and len(chosen) >= 2:
        combo = build_feature_grid(grids, chosen, normalize_combo)
        rgb = pca_to_rgb(combo).numpy()
        tag = ("concat[" + "+".join(chosen) + f"] {combo.shape[-1]}d"
               + (" L2-norm" if normalize_combo else ""))
        gallery.append((Image.fromarray(rgb).resize((res, res), Image.NEAREST), tag))

    for name in chosen:
        rgb = pca_to_rgb(grids[name]).numpy()
        gallery.append((Image.fromarray(rgb).resize((res, res), Image.NEAREST), name))

    norm_note = "final LayerNorm applied" if apply_norm else "raw block outputs"
    combo_note = (f" | + concat of {len(chosen)} layers" if combine and len(chosen) >= 2
                  else "")
    status = (f"patch grid {feat_h}×{feat_w} ({feat_h * feat_w} tokens) | "
              f"{norm_note}{combo_note} | {len(gallery)} panels")
    return gallery, status


def load_and_inspect(split, selected, res, apply_norm, combine, normalize_combo):
    img, info = load_random(split, res)
    gallery, status = inspect(img, selected, res, apply_norm, combine, normalize_combo)
    return img, info, gallery, status


# --------------------------------------------------------------------------- #
#  Tab 2: mask extraction                                                     #
# --------------------------------------------------------------------------- #
def run_masks(image, res: int, layers: list[str], apply_norm: bool, normalize_combo: bool,
              pca_components: float, smooth_sigma: float, whiten: bool,
              algo: str, similarity: str, gamma: float, connectivity: int, seed: float,
              # greedy_graph params
              use_threshold: bool, threshold: float, quantile: float, min_size: float,
              # greedy_multistage params
              consistency_filter: bool, consistency_min: float, merge_threshold: float,
              seam_thickness: float,
              # frontier_grow / sequential_grow params
              seed_quantile: float, seed_min_size: float, mask_min_size: float, prototype: str,
              boundary_delta: float, normalize_sim: bool, max_masks: float,
              # maskcut params
              tau: float, n_masks: float, min_area_ratio: float,
              # post-processing
              refine: bool, refine_iters: float, refine_radius: float, refine_weight: float):
    if image is None:
        return [], "Load an image first."
    chosen = [n for n in HOOK_NAMES if n in layers]
    if not chosen:
        return [], "Select at least one feature layer."

    pil = _prep_image(_as_pil(image), res)
    grids, feat_h, feat_w = EXTRACTOR.extract(pil, apply_norm=apply_norm)
    grid = build_feature_grid(grids, chosen, normalize_combo)
    grid = reduce_features(grid, int(pca_components), float(smooth_sigma), bool(whiten))

    sim_kwargs = {"gamma": gamma} if (similarity == "rbf" and gamma and gamma > 0) else None
    common = dict(similarity=similarity, connectivity=int(connectivity), sim_kwargs=sim_kwargs)

    feat_desc = "+".join(chosen) + (f" (concat {len(chosen)} layers)" if len(chosen) > 1 else "")
    pre = []
    if int(pca_components) > 0:
        pre.append(f"PCA{int(pca_components)}" + ("(whiten)" if whiten else ""))
    if float(smooth_sigma) > 0:
        pre.append(f"smooth σ={smooth_sigma}")
    pre_desc = (" | pre: " + "+".join(pre)) if pre else ""
    pca = Image.fromarray(pca_to_rgb(grid).numpy()).resize((res, res), Image.NEAREST)

    # ----- greedy_multistage: show the per-stage progression, not just the final
    if algo in ("greedy_multistage", "greedy_multistage2"):
        extra = ({"seam_thickness": int(seam_thickness)}
                 if algo == "greedy_multistage2" else {})
        seg = get_extractor(algo, **common,
                            threshold=(float(threshold) if use_threshold else None),
                            quantile=float(quantile), min_mask_size=int(min_size),
                            consistency_filter=bool(consistency_filter),
                            consistency_min=float(consistency_min),
                            merge_threshold=float(merge_threshold), **extra)
        stages = seg.extract_stages(grid)
        gallery = []
        counts = []
        for title, lab in stages:
            counts.append(int((lab.unique() > 0).sum()))
            ov = overlay_masks(pil, lab, alpha=0.55, seed=int(seed), bg_label=0)
            gallery.append((ov, title))
        final = stages[-1][1]
        if refine:
            final = refine_crf(final, pil, n_iter=int(refine_iters),
                               radius=int(refine_radius), weight=float(refine_weight))
            gallery.append((overlay_masks(pil, final, alpha=0.55, seed=int(seed), bg_label=0),
                            "final — CRF-refined"))
        gallery.append((Image.fromarray(color_labels(final, seed=int(seed), bg_label=0)
                                        ).resize((res, res), Image.NEAREST), "final masks"))
        gallery.append((pca, "PCA of features"))
        cut = f"thr={threshold}" if use_threshold else f"quantile={quantile}"
        cons = f" cons<{consistency_min}" if consistency_filter else ""
        seam = f" seam={int(seam_thickness)}" if algo == "greedy_multistage2" else ""
        post_desc = f" | refined (CRF ×{int(refine_iters)})" if refine else ""
        status = (f"features: {feat_desc} → {grid.shape[-1]}d{pre_desc} | "
                  f"{algo} / {similarity} | {cut} min_size={int(min_size)}{cons} "
                  f"merge≥{merge_threshold}{seam} | conn={connectivity} | grid {feat_h}×{feat_w} | "
                  f"stages: {counts[0]}→{counts[1]}→{counts[2]} masks{post_desc}")
        return gallery, status

    if algo == "greedy_graph":
        bg = 0 if int(min_size) > 0 else None
        seg = get_extractor(algo, **common,
                            threshold=(float(threshold) if use_threshold else None),
                            quantile=float(quantile), min_size=int(min_size))
        cut = f"thr={threshold}" if use_threshold else f"quantile={quantile}"
        param_desc = f"{cut} min_size={int(min_size)}"
    elif algo in ("frontier_grow", "sequential_grow"):
        bg = 0                                  # label 0 = unreached background
        kw = dict(seed_quantile=float(seed_quantile), seed_min_size=int(seed_min_size),
                  min_mask_size=int(mask_min_size), prototype=prototype,
                  boundary_delta=float(boundary_delta), normalize_sim=bool(normalize_sim))
        param_desc = (f"seed_q={seed_quantile} seed_min={int(seed_min_size)} "
                      f"mask_min={int(mask_min_size)} proto={prototype} "
                      f"delta={boundary_delta} norm_sim={normalize_sim}")
        if algo == "sequential_grow":
            kw["max_masks"] = int(max_masks) if int(max_masks) > 0 else None
            param_desc += f" max_masks={kw['max_masks']}"
        seg = get_extractor(algo, **common, **kw)
    elif algo == "maskcut":
        bg = 0                                  # label 0 = background
        seg = get_extractor(algo, **common, tau=float(tau), n_masks=int(n_masks),
                            min_area_ratio=float(min_area_ratio))
        param_desc = f"tau={tau} n_masks={int(n_masks)} min_area={min_area_ratio}"
    else:                                       # generic fallback
        bg = None
        seg = get_extractor(algo, **common)
        param_desc = ""

    labels = seg(grid)
    uniq = labels.unique()
    n_masks = int((uniq > 0).sum()) if bg == 0 else int(uniq.numel())

    if refine:
        labels = refine_crf(labels, pil, n_iter=int(refine_iters),
                            radius=int(refine_radius), weight=float(refine_weight))
    overlay = overlay_masks(pil, labels, alpha=0.55, seed=int(seed), bg_label=bg)
    masks = Image.fromarray(color_labels(labels, seed=int(seed), bg_label=bg)
                            ).resize((res, res), Image.NEAREST)

    post_desc = f" | refined (CRF ×{int(refine_iters)})" if refine else ""
    status = (f"features: {feat_desc} → {grid.shape[-1]}d{pre_desc} | "
              f"{algo} / {similarity} | {param_desc} | conn={connectivity} | "
              f"grid {feat_h}×{feat_w} → **{n_masks} masks**{post_desc}")
    gallery = [(overlay, "mask overlay"), (masks, "masks"), (pca, "PCA of features")]
    return gallery, status


# --------------------------------------------------------------------------- #
#  Tab 3: coarse ▸ fine pipeline (greedy_multistage2 twice)                   #
# --------------------------------------------------------------------------- #
def _stage_features(pil, layers_sel, apply_norm, norm_combo, pca, smooth, whiten):
    """Build one stage's feature grid; returns ``(grid, fh, fw, chosen_layers)``."""
    grids, fh, fw = EXTRACTOR.extract(pil, apply_norm=bool(apply_norm))
    chosen = [n for n in HOOK_NAMES if n in layers_sel]
    grid = build_feature_grid(grids, chosen, bool(norm_combo))
    grid = reduce_features(grid, int(pca), float(smooth), bool(whiten))
    return grid, fh, fw, chosen


def _feat_desc(chosen, pca, smooth):
    d = "+".join(chosen) + (f" (concat {len(chosen)})" if len(chosen) > 1 else "")
    extra = []
    if int(pca) > 0:
        extra.append(f"PCA{int(pca)}")
    if float(smooth) > 0:
        extra.append(f"σ{smooth}")
    return d + (" · " + "+".join(extra) if extra else "")


def run_pipeline(image, res, seed,
                 # stage 1 (coarse) features — coarse algorithm params are locked
                 co_res, co_layers, co_apply_norm, co_norm_combo, co_pca, co_smooth, co_whiten,
                 # stage 2 (fine) features
                 fi_layers, fi_apply_norm, fi_norm_combo, fi_pca, fi_smooth, fi_whiten,
                 # stage 2 (fine) greedy_multistage2 params
                 fi_sim, fi_gamma, fi_conn, fi_use_thr, fi_thr, fi_q, fi_min,
                 fi_cf, fi_cmin, fi_merge, fi_seam,
                 # final cross-mask merge + visualisation
                 final_merge_on, final_merge, detail_max, crop_pad):
    if image is None:
        return [], "Load an image first."
    if not co_layers or not fi_layers:
        return [], "Select at least one feature layer for **both** stages."

    # stage 2 (fine) runs at the global load resolution (the high one); stage 1
    # (coarse) downsamples to its own, usually lower, resolution
    pil_co = _prep_image(_as_pil(image), int(co_res))
    pil_fi = _prep_image(_as_pil(image), int(res))
    coarse_grid, fhc, fwc, co_chosen = _stage_features(
        pil_co, co_layers, co_apply_norm, co_norm_combo, co_pca, co_smooth, co_whiten)
    fine_grid, fhf, fwf, fi_chosen = _stage_features(
        pil_fi, fi_layers, fi_apply_norm, fi_norm_combo, fi_pca, fi_smooth, fi_whiten)

    fi_kwargs = ({"gamma": fi_gamma}
                 if (fi_sim == "rbf" and fi_gamma and fi_gamma > 0) else None)
    divide = get_extractor("greedy_multistage2", similarity=COARSE_SIM,
                           sim_kwargs=None, **COARSE_PARAMS)
    parts = get_extractor(
        "greedy_multistage2", similarity=fi_sim, connectivity=int(fi_conn),
        sim_kwargs=fi_kwargs,
        threshold=(float(fi_thr) if fi_use_thr else None), quantile=float(fi_q),
        min_mask_size=int(fi_min), consistency_filter=bool(fi_cf),
        consistency_min=float(fi_cmin), merge_threshold=float(fi_merge),
        seam_thickness=int(fi_seam))
    merge_thr = float(final_merge) if final_merge_on else None
    result = HierPipeline(divide, parts, final_merge_threshold=merge_thr).run(
        coarse_grid, fine_grid)

    seed = int(seed)
    coarse, final = result["divide"], result["final"]
    disp = pil_fi                                  # display on the higher-res canvas
    gallery = []
    # ---- top row: coarse masks (stage 1) as a triple
    cm, c_img, cov = label_triple(disp, coarse, seed=seed)
    gallery += [(cm, f"COARSE masks ({result['n_divide']})"),
                (c_img, "original"), (cov, "COARSE overlay")]
    # ---- second row: final fine parts (stage 2) as a triple
    fm, f_img, fov = label_triple(disp, final, seed=seed)
    gallery += [(fm, f"FINAL masks ({result['n_parts']})"),
                (f_img, "original"), (fov, "FINAL overlay")]
    # ---- one zoomed triple per coarse mask (its fine parts, cropped)
    detail = result["conquer"]
    if int(detail_max) > 0:
        detail = detail[:int(detail_max)]
    for entry in detail:
        triple = label_triple(disp, final, region=entry["region"], seed=seed,
                              pad=int(crop_pad))
        if triple is None:
            continue
        did = entry["divide_id"]
        npart = len([p for p in entry["parts"].unique().tolist() if p != 0]) or 1
        m, o_img, ov = triple
        gallery += [(m, f"coarse {did}: {npart} parts"),
                    (o_img, f"coarse {did}: crop"), (ov, f"coarse {did}: overlay")]

    fmerge = (f" | final merge≥{final_merge} ({result['n_premerge']}→{result['n_parts']})"
              if final_merge_on else "")
    status = (
        f"**coarse ▸ fine** (greedy_multistage2 ×2) | "
        f"coarse grid {fhc}×{fwc} @ {int(co_res)}px → {coarse_grid.shape[-1]}d "
        f"({_feat_desc(co_chosen, co_pca, co_smooth)}) [locked: {COARSE_DESC}] | "
        f"fine grid {fhf}×{fwf} @ {int(res)}px → {fine_grid.shape[-1]}d "
        f"({_feat_desc(fi_chosen, fi_pca, fi_smooth)}) "
        f"[{fi_sim} conn={int(fi_conn)} "
        f"{('thr=' + str(fi_thr)) if fi_use_thr else ('q=' + str(fi_q))} "
        f"min={int(fi_min)} merge≥{fi_merge} seam={int(fi_seam)}]{fmerge} | "
        f"**{result['n_divide']} coarse → {result['n_parts']} parts**")
    return gallery, status


# --------------------------------------------------------------------------- #
#  UI                                                                         #
# --------------------------------------------------------------------------- #
with gr.Blocks(title="DINOv3 Feature & Mask Lab") as demo:
    gr.Markdown(
        "# DINOv3 Feature & Mask Lab\n"
        "Inspect DINOv3 ViT-L/16 patch features and prototype mask-extraction "
        "algorithms on the same image."
    )

    with gr.Row():
        with gr.Column(scale=1):
            split = gr.Radio(["train", "validation"], value=CFG["split"], label="Split")
            res = gr.Dropdown(RESOLUTIONS, value=CFG["res"], label="Resolution (px, /16)")
            load_btn = gr.Button("Load random image", variant="primary")
            image = gr.Image(label="Current image", type="pil", height=300)
            info = gr.Markdown()
            save_btn = gr.Button("💾 Save current settings", variant="secondary")
            save_status = gr.Markdown()

        with gr.Column(scale=2):
            with gr.Tabs():
                # ---------------------------------------------- inspector tab
                with gr.Tab("Feature inspector"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            ins_hooks = gr.CheckboxGroup(
                                HOOK_NAMES, value=CFG["ins_hooks"],
                                label="Hook locations to visualise")
                            ins_apply_norm = gr.Checkbox(
                                value=CFG["ins_apply_norm"],
                                label="Apply final LayerNorm to intermediate layers",
                                info="DINOv3 dense-decoder recipe (final_norm left as-is)")
                            ins_combine = gr.Checkbox(
                                value=CFG["ins_combine"],
                                label="Also show channel-concat of selected layers")
                            ins_norm_combo = gr.Checkbox(
                                value=CFG["ins_norm_combo"],
                                label="L2-normalise each layer before concat",
                                info="Stops a high-magnitude layer dominating the PCA")
                            ins_btn = gr.Button("Extract / refresh", variant="primary")
                        with gr.Column(scale=2):
                            ins_gallery = gr.Gallery(
                                label="PCA-RGB features", columns=3, height=700)
                            ins_status = gr.Markdown()

                # ----------------------------------------------- mask tab
                with gr.Tab("Mask extraction"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            msk_layers = gr.CheckboxGroup(
                                HOOK_NAMES, value=CFG["msk_layers"],
                                label="Feature layer(s) — multiple = channel-concat")
                            msk_apply_norm = gr.Checkbox(
                                value=CFG["msk_apply_norm"], label="Apply final LayerNorm to layers")
                            msk_norm_combo = gr.Checkbox(
                                value=CFG["msk_norm_combo"],
                                label="L2-normalise each layer before concat")
                            msk_algo = gr.Dropdown(
                                sorted(EXTRACTORS), value=CFG["msk_algo"], label="Algorithm")
                            msk_sim = gr.Dropdown(
                                sorted(SIMILARITIES), value=CFG["msk_sim"],
                                label="Similarity metric")
                            msk_conn = gr.Radio([4, 8], value=CFG["msk_conn"], label="connectivity")
                            msk_gamma = gr.Number(value=CFG["msk_gamma"], label="rbf gamma (0 = auto)")
                            msk_seed = gr.Number(value=CFG["msk_seed"], precision=0,
                                                 label="colour seed")

                            with gr.Accordion("Feature preprocessing (denoising)", open=True):
                                msk_pca = gr.Slider(
                                    0, 512, value=CFG["msk_pca"], step=1,
                                    label="PCA components (0 = full features)",
                                    info="Keep top-k PCs — the low-rank structure PCA-RGB shows")
                                msk_whiten = gr.Checkbox(
                                    value=CFG["msk_whiten"],
                                    label="whiten PCs (equal-weight components)")
                                msk_smooth = gr.Slider(
                                    0.0, 2.0, value=CFG["msk_smooth"], step=0.1,
                                    label="spatial smoothing σ (0 = off)",
                                    info="Gaussian low-pass over the patch grid")

                            with gr.Accordion("Mask post-processing (CRF refinement)",
                                              open=True):
                                msk_refine = gr.Checkbox(
                                    value=CFG["msk_refine"],
                                    label="Refine masks (edge-aware, snaps to image)")
                                msk_ref_iters = gr.Slider(
                                    0, 10, value=CFG["msk_ref_iters"], step=1,
                                    label="CRF iterations")
                                msk_ref_radius = gr.Slider(
                                    1, 12, value=CFG["msk_ref_radius"], step=1,
                                    label="guided-filter radius")
                                msk_ref_weight = gr.Slider(
                                    0.0, 4.0, value=CFG["msk_ref_weight"], step=0.1,
                                    label="pairwise weight")

                            with gr.Accordion("greedy_graph / greedy_multistage cut", open=True,
                                              visible=CFG["msk_algo"] in ("greedy_graph",
                                                                          "greedy_multistage",
                                                                          "greedy_multistage2")
                                              ) as greedy_acc:
                                msk_use_thr = gr.Checkbox(
                                    value=CFG["msk_use_thr"],
                                    label="Use absolute threshold (else quantile)")
                                msk_thr = gr.Number(value=CFG["msk_thr"], label="threshold")
                                msk_q = gr.Slider(0.0, 1.0, value=CFG["msk_q"], step=0.01,
                                                  label="quantile cut-off")
                                msk_min = gr.Number(value=CFG["msk_min"], precision=0,
                                                    label="min mask size (patches)")

                            with gr.Accordion("greedy_multistage parameters", open=True,
                                              visible=CFG["msk_algo"] in ("greedy_multistage",
                                                                          "greedy_multistage2")
                                              ) as multistage_acc:
                                msk_cons_filter = gr.Checkbox(
                                    value=CFG["msk_cons_filter"],
                                    label="Stage 1: filter internally inconsistent cores")
                                msk_cons_min = gr.Slider(
                                    -1.0, 1.0, value=CFG["msk_cons_min"], step=0.01,
                                    label="consistency min (mean sim to own prototype)")
                                msk_merge = gr.Slider(
                                    -1.0, 1.0, value=CFG["msk_merge"], step=0.01,
                                    label="Stage 3: merge threshold (prototype similarity)")
                                msk_seam = gr.Slider(
                                    1, 8, value=CFG["msk_seam"], step=1,
                                    label="Stage 3 (multistage2 only): seam thickness "
                                          "(band width sampled at the border, in patches)")

                            with gr.Accordion("frontier_grow / sequential_grow parameters",
                                              open=True,
                                              visible=CFG["msk_algo"] in ("frontier_grow",
                                                                          "sequential_grow")
                                              ) as frontier_acc:
                                msk_seed_q = gr.Slider(0.0, 1.0, value=CFG["msk_seed_q"], step=0.01,
                                                       label="seed quantile (rough guesses)")
                                msk_seed_min = gr.Number(value=CFG["msk_seed_min"], precision=0,
                                                         label="seed min size (patches)")
                                msk_mask_min = gr.Number(value=CFG["msk_mask_min"], precision=0,
                                                         label="final min mask size (patches)")
                                msk_proto = gr.Radio(["mean", "max"], value=CFG["msk_proto"],
                                                     label="prototype aggregation")
                                msk_delta = gr.Slider(0.0, 1.0, value=CFG["msk_delta"], step=0.01,
                                                      label="boundary delta (max sim drop)")
                                msk_norm_sim = gr.Checkbox(
                                    value=CFG["msk_norm_sim"],
                                    label="normalise similarity map (delta = fraction)")
                                msk_max_masks = gr.Number(
                                    value=CFG["msk_max_masks"], precision=0,
                                    label="max masks (sequential_grow; 0 = unlimited)")

                            with gr.Accordion("maskcut parameters", open=True,
                                              visible=CFG["msk_algo"] == "maskcut"
                                              ) as maskcut_acc:
                                msk_tau = gr.Slider(0.0, 0.6, value=CFG["msk_tau"], step=0.01,
                                                    label="tau (affinity threshold)")
                                msk_nmasks = gr.Number(value=CFG["msk_nmasks"], precision=0,
                                                       label="n_masks (NCut iterations)")
                                msk_area = gr.Slider(0.0, 0.2, value=CFG["msk_area"], step=0.005,
                                                     label="min area ratio (stop below)")

                            msk_btn = gr.Button("Run mask extraction", variant="primary")
                        with gr.Column(scale=2):
                            msk_gallery = gr.Gallery(
                                label="Masks", columns=3, height=700)
                            msk_status = gr.Markdown()

                # ------------------------------------------- pipeline tab
                with gr.Tab("Pipelines"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(
                                "**Coarse ▸ fine** — greedy_multistage2 runs twice. "
                                "Stage 1 finds coarse object masks; stage 2 re-runs "
                                "*inside each* coarse mask to split it into parts. The "
                                "two stages can use **different features**.\n\n"
                                f"Stage-1 algorithm params are **locked**: {COARSE_DESC}. "
                                "Only its features (below) and all of stage 2 are tunable.")
                            pl_seed = gr.Number(value=CFG["pl_seed"], precision=0,
                                                label="colour seed")

                            with gr.Accordion("STAGE 1 (coarse) — features", open=True):
                                pl_co_res = gr.Dropdown(
                                    RESOLUTIONS, value=CFG["pl_co_res"],
                                    label="coarse input resolution (px, /16) — downsample",
                                    info="stage 1 runs here; keep ≤ the global load "
                                         "resolution. Lower = coarser/faster object masks")
                                pl_co_layers = gr.CheckboxGroup(
                                    HOOK_NAMES, value=CFG["pl_co_layers"],
                                    label="Feature layer(s) — multiple = channel-concat")
                                pl_co_apply_norm = gr.Checkbox(
                                    value=CFG["pl_co_apply_norm"],
                                    label="Apply final LayerNorm to layers")
                                pl_co_norm_combo = gr.Checkbox(
                                    value=CFG["pl_co_norm_combo"],
                                    label="L2-normalise each layer before concat")
                                pl_co_pca = gr.Slider(
                                    0, 512, value=CFG["pl_co_pca"], step=1,
                                    label="PCA components (0 = full features)")
                                pl_co_smooth = gr.Slider(
                                    0.0, 2.0, value=CFG["pl_co_smooth"], step=0.1,
                                    label="spatial smoothing σ (0 = off)")
                                pl_co_whiten = gr.Checkbox(
                                    value=CFG["pl_co_whiten"], label="whiten PCs")

                            with gr.Accordion("STAGE 2 (fine) — features "
                                              "(runs at the global load resolution)",
                                              open=True):
                                pl_fi_layers = gr.CheckboxGroup(
                                    HOOK_NAMES, value=CFG["pl_fi_layers"],
                                    label="Feature layer(s) — multiple = channel-concat")
                                pl_fi_apply_norm = gr.Checkbox(
                                    value=CFG["pl_fi_apply_norm"],
                                    label="Apply final LayerNorm to layers")
                                pl_fi_norm_combo = gr.Checkbox(
                                    value=CFG["pl_fi_norm_combo"],
                                    label="L2-normalise each layer before concat")
                                pl_fi_pca = gr.Slider(
                                    0, 512, value=CFG["pl_fi_pca"], step=1,
                                    label="PCA components (0 = full features, e.g. 1024)")
                                pl_fi_smooth = gr.Slider(
                                    0.0, 2.0, value=CFG["pl_fi_smooth"], step=0.1,
                                    label="spatial smoothing σ (0 = off)")
                                pl_fi_whiten = gr.Checkbox(
                                    value=CFG["pl_fi_whiten"], label="whiten PCs")

                            with gr.Accordion("STAGE 2 (fine) — greedy_multistage2 params",
                                              open=True):
                                pl_fi_sim = gr.Dropdown(
                                    sorted(SIMILARITIES), value=CFG["pl_fi_sim"],
                                    label="Similarity metric")
                                pl_fi_gamma = gr.Number(
                                    value=CFG["pl_fi_gamma"], label="rbf gamma (0 = auto)")
                                pl_fi_conn = gr.Radio(
                                    [4, 8], value=CFG["pl_fi_conn"], label="connectivity")
                                pl_fi_use_thr = gr.Checkbox(
                                    value=CFG["pl_fi_use_thr"],
                                    label="Use absolute threshold (else quantile)")
                                pl_fi_thr = gr.Number(value=CFG["pl_fi_thr"], label="threshold")
                                pl_fi_q = gr.Slider(0.0, 1.0, value=CFG["pl_fi_q"], step=0.01,
                                                    label="quantile cut-off (per mask)")
                                pl_fi_min = gr.Number(value=CFG["pl_fi_min"], precision=0,
                                                      label="min part size (patches)")
                                pl_fi_cf = gr.Checkbox(
                                    value=CFG["pl_fi_cf"], label="filter inconsistent cores")
                                pl_fi_cmin = gr.Slider(-1.0, 1.0, value=CFG["pl_fi_cmin"],
                                                       step=0.01, label="consistency min")
                                pl_fi_merge = gr.Slider(-1.0, 1.0, value=CFG["pl_fi_merge"],
                                                        step=0.01,
                                                        label="Stage 3: merge threshold "
                                                              "(boundary-seam similarity)")
                                pl_fi_seam = gr.Slider(
                                    1, 8, value=CFG["pl_fi_seam"], step=1,
                                    label="Stage 3: seam thickness (patches)")

                            with gr.Accordion("Final merge (cross-mask cleanup)", open=True):
                                gr.Markdown(
                                    "Merge **adjacent** final masks whose patch features "
                                    "match — rejoins slivers a coarse mask stole from a "
                                    "neighbour and that stage 2 re-split off.")
                                pl_final_merge_on = gr.Checkbox(
                                    value=CFG["pl_final_merge_on"],
                                    label="merge adjacent, feature-similar final masks")
                                pl_final_merge = gr.Slider(
                                    -1.0, 1.0, value=CFG["pl_final_merge"], step=0.01,
                                    label="merge threshold (whole-mask prototype similarity)")

                            with gr.Accordion("Visualisation", open=True):
                                pl_detail_max = gr.Number(
                                    value=CFG["pl_detail_max"], precision=0,
                                    label="max coarse masks to zoom (0 = all)")
                                pl_crop_pad = gr.Slider(
                                    0, 48, value=CFG["pl_crop_pad"], step=1,
                                    label="crop padding around each coarse mask (px)")

                            pl_btn = gr.Button("Run pipeline", variant="primary")
                        with gr.Column(scale=2):
                            pl_gallery = gr.Gallery(
                                label="coarse ▸ final ▸ per-mask zoom (rows of 3)",
                                columns=3, height=750)
                            pl_status = gr.Markdown()

    # ------------------------------------------------------------- wiring
    ins_inputs = [image, ins_hooks, res, ins_apply_norm, ins_combine, ins_norm_combo]
    load_btn.click(load_and_inspect,
                   [split, ins_hooks, res, ins_apply_norm, ins_combine, ins_norm_combo],
                   [image, info, ins_gallery, ins_status])
    ins_btn.click(inspect, ins_inputs, [ins_gallery, ins_status])

    def _toggle_algo(algo):
        return (gr.update(visible=algo in ("greedy_graph", "greedy_multistage",
                                           "greedy_multistage2")),
                gr.update(visible=algo in ("greedy_multistage", "greedy_multistage2")),
                gr.update(visible=algo in ("frontier_grow", "sequential_grow")),
                gr.update(visible=algo == "maskcut"))
    msk_algo.change(_toggle_algo, msk_algo,
                    [greedy_acc, multistage_acc, frontier_acc, maskcut_acc])

    msk_inputs = [image, res, msk_layers, msk_apply_norm, msk_norm_combo,
                  msk_pca, msk_smooth, msk_whiten, msk_algo,
                  msk_sim, msk_gamma, msk_conn, msk_seed,
                  msk_use_thr, msk_thr, msk_q, msk_min,
                  msk_cons_filter, msk_cons_min, msk_merge, msk_seam,
                  msk_seed_q, msk_seed_min, msk_mask_min, msk_proto, msk_delta,
                  msk_norm_sim, msk_max_masks,
                  msk_tau, msk_nmasks, msk_area,
                  msk_refine, msk_ref_iters, msk_ref_radius, msk_ref_weight]
    msk_btn.click(run_masks, msk_inputs, [msk_gallery, msk_status])

    pl_inputs = [image, res, pl_seed,
                 pl_co_res, pl_co_layers, pl_co_apply_norm, pl_co_norm_combo,
                 pl_co_pca, pl_co_smooth, pl_co_whiten,
                 pl_fi_layers, pl_fi_apply_norm, pl_fi_norm_combo,
                 pl_fi_pca, pl_fi_smooth, pl_fi_whiten,
                 pl_fi_sim, pl_fi_gamma, pl_fi_conn, pl_fi_use_thr, pl_fi_thr,
                 pl_fi_q, pl_fi_min, pl_fi_cf, pl_fi_cmin, pl_fi_merge, pl_fi_seam,
                 pl_final_merge_on, pl_final_merge, pl_detail_max, pl_crop_pad]
    pl_btn.click(run_pipeline, pl_inputs, [pl_gallery, pl_status])

    # persist every control to app_config.yaml (order must match CONFIG_KEYS)
    CONFIG_COMPONENTS = {
        "split": split, "res": res,
        "ins_hooks": ins_hooks, "ins_apply_norm": ins_apply_norm,
        "ins_combine": ins_combine, "ins_norm_combo": ins_norm_combo,
        "msk_layers": msk_layers, "msk_apply_norm": msk_apply_norm,
        "msk_norm_combo": msk_norm_combo, "msk_algo": msk_algo, "msk_sim": msk_sim,
        "msk_conn": msk_conn, "msk_gamma": msk_gamma, "msk_seed": msk_seed,
        "msk_pca": msk_pca, "msk_whiten": msk_whiten, "msk_smooth": msk_smooth,
        "msk_refine": msk_refine, "msk_ref_iters": msk_ref_iters,
        "msk_ref_radius": msk_ref_radius, "msk_ref_weight": msk_ref_weight,
        "msk_use_thr": msk_use_thr, "msk_thr": msk_thr, "msk_q": msk_q, "msk_min": msk_min,
        "msk_cons_filter": msk_cons_filter, "msk_cons_min": msk_cons_min, "msk_merge": msk_merge,
        "msk_seam": msk_seam,
        "msk_seed_q": msk_seed_q, "msk_seed_min": msk_seed_min, "msk_mask_min": msk_mask_min,
        "msk_proto": msk_proto, "msk_delta": msk_delta, "msk_norm_sim": msk_norm_sim,
        "msk_max_masks": msk_max_masks,
        "msk_tau": msk_tau, "msk_nmasks": msk_nmasks, "msk_area": msk_area,
        "pl_seed": pl_seed,
        "pl_co_res": pl_co_res,
        "pl_co_layers": pl_co_layers, "pl_co_apply_norm": pl_co_apply_norm,
        "pl_co_norm_combo": pl_co_norm_combo, "pl_co_pca": pl_co_pca,
        "pl_co_smooth": pl_co_smooth, "pl_co_whiten": pl_co_whiten,
        "pl_fi_layers": pl_fi_layers, "pl_fi_apply_norm": pl_fi_apply_norm,
        "pl_fi_norm_combo": pl_fi_norm_combo, "pl_fi_pca": pl_fi_pca,
        "pl_fi_smooth": pl_fi_smooth, "pl_fi_whiten": pl_fi_whiten,
        "pl_fi_sim": pl_fi_sim, "pl_fi_gamma": pl_fi_gamma, "pl_fi_conn": pl_fi_conn,
        "pl_fi_use_thr": pl_fi_use_thr, "pl_fi_thr": pl_fi_thr, "pl_fi_q": pl_fi_q,
        "pl_fi_min": pl_fi_min, "pl_fi_cf": pl_fi_cf, "pl_fi_cmin": pl_fi_cmin,
        "pl_fi_merge": pl_fi_merge, "pl_fi_seam": pl_fi_seam,
        "pl_final_merge_on": pl_final_merge_on, "pl_final_merge": pl_final_merge,
        "pl_detail_max": pl_detail_max, "pl_crop_pad": pl_crop_pad,
    }
    assert list(CONFIG_COMPONENTS) == CONFIG_KEYS, "config component/key mismatch"
    save_btn.click(save_config, [CONFIG_COMPONENTS[k] for k in CONFIG_KEYS], save_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
