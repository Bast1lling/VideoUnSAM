"""
Run a mask-extraction algorithm on a DINOv3 feature grid and save a panel.

Examples
--------
    uv run python -m mask_extraction.run                       # random val image
    uv run python -m mask_extraction.run --index 12 --layer block_17
    uv run python -m mask_extraction.run --similarity dot --quantile 0.7
    uv run python -m mask_extraction.run --similarity rbf --gamma 0.01 --min-size 4

The output PNG is a side-by-side of: input | PCA-RGB features | masks | overlay.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import torch
from PIL import Image

# make the project root importable whether run as `-m mask_extraction.run`
# or as `python mask_extraction/run.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backbone import HookedViTFeatV3, pca_to_rgb          # noqa: E402
from dataset import ImageNet1k                            # noqa: E402
from mask_extraction import (                             # noqa: E402
    EXTRACTORS, SIMILARITIES, color_labels, get_extractor, hstack, overlay_masks,
    reduce_features, refine_crf,
)


def prep_image(pil: Image.Image, res: int) -> Image.Image:
    pil = pil.convert("RGB")
    w, h = pil.size
    s = res / min(w, h)
    pil = pil.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = pil.size
    left, top = (w - res) // 2, (h - res) // 2
    return pil.crop((left, top, left + res, top + res))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--index", type=int, default=None, help="dataset index (random if unset)")
    ap.add_argument("--res", type=int, default=448, help="square input size (divisible by 16)")
    ap.add_argument("--layer", default="final_norm", help="hook layer to segment")
    ap.add_argument("--apply-norm", action="store_true",
                    help="apply final LayerNorm to the chosen intermediate layer")
    ap.add_argument("--algo", default="greedy_graph", choices=sorted(EXTRACTORS))
    ap.add_argument("--similarity", default="cosine", choices=sorted(SIMILARITIES))
    ap.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    ap.add_argument("--gamma", type=float, default=None, help="rbf kernel bandwidth")
    ap.add_argument("--pca-components", type=int, default=0,
                    help="reduce features to top-k PCs before extraction (0 = off)")
    ap.add_argument("--whiten", action="store_true", help="whiten the kept PCs")
    ap.add_argument("--smooth-sigma", type=float, default=0.0,
                    help="Gaussian spatial smoothing of the feature grid (0 = off)")
    ap.add_argument("--seed", type=int, default=0, help="colour palette seed")
    ap.add_argument("--out", default=None, help="output PNG path")
    # greedy_graph
    g = ap.add_argument_group("greedy_graph")
    g.add_argument("--threshold", type=float, default=None,
                   help="absolute similarity cut-off (overrides --quantile)")
    g.add_argument("--quantile", type=float, default=0.5,
                   help="edge-similarity quantile used as cut-off")
    g.add_argument("--min-size", type=int, default=0, help="drop masks smaller than this")
    # greedy_multistage
    ms = ap.add_argument_group("greedy_multistage")
    ms.add_argument("--consistency-filter", action="store_true",
                    help="stage 1: drop internally inconsistent cores")
    ms.add_argument("--consistency-min", type=float, default=0.5,
                    help="min mean similarity of a core's patches to their prototype")
    ms.add_argument("--merge-threshold", type=float, default=0.85,
                    help="stage 3: merge masks whose prototypes are at least this similar")
    # frontier_grow
    f = ap.add_argument_group("frontier_grow")
    f.add_argument("--seed-quantile", type=float, default=0.8,
                   help="quantile for the rough seed guesses")
    f.add_argument("--seed-min-size", type=int, default=2, help="discard seeds smaller than this")
    f.add_argument("--min-mask-size", type=int, default=0,
                   help="discard final grown masks smaller than this (0 = off)")
    f.add_argument("--prototype", default="mean", choices=["mean", "max"],
                   help="how to summarise a seed's features")
    f.add_argument("--boundary-delta", type=float, default=0.15,
                   help="max similarity drop per step before marking a boundary")
    f.add_argument("--no-normalize-sim", action="store_true",
                   help="do not min-max normalise the prototype-similarity map")
    f.add_argument("--max-masks", type=int, default=0,
                   help="sequential_grow: stop after N masks (0 = unlimited)")
    # maskcut
    mc = ap.add_argument_group("maskcut")
    mc.add_argument("--tau", type=float, default=0.15, help="affinity binarisation threshold")
    mc.add_argument("--n-masks", type=int, default=3, help="number of NCut iterations / objects")
    mc.add_argument("--min-area-ratio", type=float, default=0.01,
                    help="stop when a cut covers less than this fraction of patches")
    # post-processing
    pp = ap.add_argument_group("post-processing")
    pp.add_argument("--refine", action="store_true",
                    help="edge-aware CRF refinement of the masks (snaps to image)")
    pp.add_argument("--refine-iters", type=int, default=5, help="CRF mean-field iterations")
    pp.add_argument("--refine-radius", type=int, default=4, help="guided-filter radius")
    pp.add_argument("--refine-weight", type=float, default=1.5, help="pairwise term weight")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ImageNet1k(split=args.split)
    idx = args.index if args.index is not None else random.randrange(len(ds))
    pil, label = ds[idx]
    img = prep_image(pil, args.res)

    ext = HookedViTFeatV3(device=device)
    grids, fh, fw = ext.extract(img, apply_norm=args.apply_norm)
    if args.layer not in grids:
        raise SystemExit(f"unknown layer '{args.layer}'. Available: {ext.hook_names}")
    grid = grids[args.layer]                              # (fh, fw, C)
    grid = reduce_features(grid, args.pca_components, args.smooth_sigma, args.whiten)

    sim_kwargs = {"gamma": args.gamma} if args.gamma is not None else None
    common = dict(similarity=args.similarity, connectivity=args.connectivity,
                  sim_kwargs=sim_kwargs)
    if args.algo == "greedy_graph":
        bg = 0 if args.min_size > 0 else None
        seg = get_extractor(args.algo, **common, threshold=args.threshold,
                            quantile=args.quantile, min_size=args.min_size)
        cut = f"thr={args.threshold}" if args.threshold is not None else f"q={args.quantile}"
        param_desc = f"{cut} min_size={args.min_size}"
    elif args.algo in ("greedy_multistage", "greedy_multistage2"):
        bg = 0
        seg = get_extractor(args.algo, **common, threshold=args.threshold,
                            quantile=args.quantile, min_mask_size=args.min_size,
                            consistency_filter=args.consistency_filter,
                            consistency_min=args.consistency_min,
                            merge_threshold=args.merge_threshold)
        cut = f"thr={args.threshold}" if args.threshold is not None else f"q={args.quantile}"
        cons = f" cons<{args.consistency_min}" if args.consistency_filter else ""
        param_desc = f"{cut} min_size={args.min_size}{cons} merge>={args.merge_threshold}"
    elif args.algo in ("frontier_grow", "sequential_grow"):
        bg = 0
        kw = dict(seed_quantile=args.seed_quantile, seed_min_size=args.seed_min_size,
                  min_mask_size=args.min_mask_size, prototype=args.prototype,
                  boundary_delta=args.boundary_delta, normalize_sim=not args.no_normalize_sim)
        param_desc = (f"seed_q={args.seed_quantile} seed_min={args.seed_min_size} "
                      f"mask_min={args.min_mask_size} proto={args.prototype} "
                      f"delta={args.boundary_delta}")
        if args.algo == "sequential_grow":
            kw["max_masks"] = args.max_masks if args.max_masks > 0 else None
            param_desc += f" max_masks={kw['max_masks']}"
        seg = get_extractor(args.algo, **common, **kw)
    elif args.algo == "maskcut":
        bg = 0
        seg = get_extractor(args.algo, **common, tau=args.tau, n_masks=args.n_masks,
                            min_area_ratio=args.min_area_ratio)
        param_desc = f"tau={args.tau} n_masks={args.n_masks} min_area={args.min_area_ratio}"
    else:
        bg = None
        seg = get_extractor(args.algo, **common)
        param_desc = ""

    labels = seg(grid)                                    # (fh, fw)
    uniq = labels.unique()
    n_masks = int((uniq > 0).sum()) if bg == 0 else int(uniq.numel())

    if args.refine:
        labels = refine_crf(labels, img, n_iter=args.refine_iters,
                            radius=args.refine_radius, weight=args.refine_weight)

    print(f"image {args.split}[{idx}] class {label:04d} | layer={args.layer} "
          f"grid {fh}x{fw} | {args.algo}/{args.similarity} {param_desc} "
          f"conn={args.connectivity} -> {n_masks} masks"
          f"{' | CRF-refined' if args.refine else ''}")

    pca = Image.fromarray(pca_to_rgb(grid).numpy()).resize((args.res, args.res), Image.NEAREST)
    masks = Image.fromarray(color_labels(labels, seed=args.seed, bg_label=bg)
                            ).resize((args.res, args.res), Image.NEAREST)
    over = overlay_masks(img, labels, alpha=0.55, seed=args.seed, bg_label=bg)
    panel = hstack([img, pca, masks, over])

    out = args.out or f"masks_{args.split}{idx}_{args.layer}_{args.similarity}.png"
    panel.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
