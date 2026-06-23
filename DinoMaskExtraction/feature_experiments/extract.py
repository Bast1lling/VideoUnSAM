"""Batch mask extraction over ImageNet using the tuned feature_experiments pipeline.

Runs the exact algorithm the Gradio app prototypes — DINOv3 features →
PCA/preprocess → graph-cut connected components (stage 1) → grow leftovers
(stage 2) → merge (stage 3) — over a whole ImageNet split, reading every
parameter from ``config.yaml``, and writes one label-map PNG per image.

Design — saturate **CPU and GPU at once** with a producer/consumer pipeline:

* the **GPU** process runs the DINOv3 forward pass + PCA reduction back-to-back,
  fed by a ``DataLoader`` whose worker processes decode/resize/normalise images
  in parallel (CPU);
* each reduced feature grid (small, post-PCA) is handed to a pool of **CPU
  worker processes** that run the pure-Python segmentation (graph cut, region
  growing, seam merge) and save the result.

A bounded queue provides back-pressure so the GPU never races too far ahead of
the CPU workers. The GPU stays busy extracting while the CPU stays busy
segmenting — the two stages overlap instead of blocking each other.

Output layout mirrors the dataset::

    <out>/<split>/<class>/<stem>.png        # uint16 label map, 0 = background

Run::

    uv run python -m feature_experiments.extract                       # whole split
    uv run python -m feature_experiments.extract --limit 200           # quick test
    uv run python -m feature_experiments.extract --seg-workers 12 --batch-size 2
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from feature_experiments.edges import graph_segment, grow_and_merge

# Preprocessing replicated locally (identical to backbone._to_tensor /
# unsup_seg.pca.prep_image) so DataLoader/segmentation workers stay lightweight
# and never import transformers.
_PATCH = 16
_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def prep_image(pil: Image.Image, res: int) -> Image.Image:
    """Resize short side -> ``res`` (BICUBIC) + centre-crop to a square /16."""
    res = (res // _PATCH) * _PATCH
    pil = pil.convert("RGB")
    w, h = pil.size
    s = res / min(w, h)
    pil = pil.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = pil.size
    left, top = (w - res) // 2, (h - res) // 2
    return pil.crop((left, top, left + res, top + res))


# --------------------------------------------------------------------------- #
#  Tuned parameters (read from config.yaml)                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    res: int
    layer: str
    apply_norm: bool
    pca: int
    whiten: bool
    smooth: float
    l2_normalize: bool
    metric: str
    connectivity: int
    radii: list
    seg_by: str
    seg_q: float
    seg_thr: float
    seg_min: int
    merge_mode: str
    merge_by: str
    merge_thr: float
    merge_q: float
    seam_thickness: int

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Params":
        return cls(
            res=int(cfg["res"]), layer=str(cfg["layer"]),
            apply_norm=bool(cfg["apply_norm"]), pca=int(cfg["pca"]),
            whiten=bool(cfg["whiten"]), smooth=float(cfg["smooth"]),
            l2_normalize=bool(cfg["l2_normalize"]), metric=str(cfg["metric"]),
            connectivity=int(cfg["connectivity"]), radii=[int(r) for r in cfg["radii"]],
            seg_by=str(cfg["seg_by"]), seg_q=float(cfg["seg_q"]),
            seg_thr=float(cfg["seg_thr"]), seg_min=int(cfg["seg_min"]),
            merge_mode=str(cfg["merge_mode"]), merge_by=str(cfg["merge_by"]),
            merge_thr=float(cfg["merge_thr"]), merge_q=float(cfg["merge_q"]),
            seam_thickness=int(cfg["seam_thickness"]),
        )


def out_path_for(out_root: Path, split: str, img_path: str) -> Path:
    """``<out>/<split>/<class>/<stem>.png`` mirroring the dataset tree."""
    p = Path(img_path)
    return out_root / split / p.parent.name / f"{p.stem}.png"


# --------------------------------------------------------------------------- #
#  GPU side: dataset of prepped tensors + output paths                        #
# --------------------------------------------------------------------------- #
class PrepDataset(Dataset):
    """Yields ``(normalised_tensor, out_path)`` for each pending image.

    Holds only a flat ``(img_path, out_path)`` list, so DataLoader workers never
    touch the heavy modules (backbone/transformers).
    """

    def __init__(self, items: list[tuple[str, str]], res: int):
        self.items = items
        self.res = res

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img_path, out_path = self.items[i]
        try:
            pil = prep_image(Image.open(img_path), self.res)
            return _to_tensor(pil), out_path
        except Exception as exc:                       # unreadable/corrupt image
            print(f"  [load] skip {img_path}: {exc}", flush=True)
            return torch.zeros(3, self.res, self.res), None


def _collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# --------------------------------------------------------------------------- #
#  CPU side: segmentation worker                                              #
# --------------------------------------------------------------------------- #
def _save_labels(labels: torch.Tensor, out_path: str) -> None:
    arr = labels.cpu().numpy().astype(np.uint16)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="I;16").save(out_path)


def seg_worker(task_q: "mp.Queue", params: Params, done) -> None:
    """Consume ``(out_path, grid_np)`` tasks, segment, and save the label map."""
    torch.set_num_threads(1)                           # avoid CPU oversubscription
    thr = params.seg_thr if params.seg_by == "absolute" else None
    while True:
        item = task_q.get()
        if item is None:
            break
        out_path, grid_np = item
        try:
            grid = torch.from_numpy(grid_np)
            labels, _, _, _ = graph_segment(
                grid, metric=params.metric, connectivity=params.connectivity,
                radii=params.radii, threshold=thr, quantile=params.seg_q,
                min_size=params.seg_min)
            _, merged, _, _, _ = grow_and_merge(
                grid, labels, metric=params.metric, connectivity=params.connectivity,
                merge_threshold=params.merge_thr, merge_mode=params.merge_mode,
                seam_thickness=params.seam_thickness, merge_by=params.merge_by,
                merge_quantile=params.merge_q)
            _save_labels(merged, out_path)
        except Exception as exc:                        # one bad image must not kill the worker
            print(f"  [seg] fail {out_path}: {exc}", flush=True)
        with done.get_lock():
            done.value += 1


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
def build_todo(samples, out_root: Path, split: str, overwrite: bool, limit: int | None):
    items = []
    for img_path, _label in samples:
        op = out_path_for(out_root, split, img_path)
        if overwrite or not op.exists():
            items.append((img_path, str(op)))
        if limit and len(items) >= limit:
            break
    return items


def main() -> None:
    cpu = os.cpu_count() or 8
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="feature_experiments/config.yaml")
    ap.add_argument("--out", default="/home/sebastian/data/imagenet1k/masks",
                    help="output root for label-map PNGs")
    ap.add_argument("--split", default=None, help="override config split")
    ap.add_argument("--batch-size", type=int, default=2, help="GPU forward batch (1024px is heavy)")
    ap.add_argument("--load-workers", type=int, default=max(2, cpu // 4),
                    help="DataLoader processes (image decode/resize)")
    ap.add_argument("--seg-workers", type=int, default=None,
                    help="CPU segmentation processes (default: fill remaining cores)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N pending images")
    ap.add_argument("--overwrite", action="store_true", help="re-extract even if output exists")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    params = Params.from_cfg(cfg)
    split = args.split or str(cfg["split"])
    seg_workers = args.seg_workers or max(1, cpu - args.load_workers - 1)
    out_root = Path(args.out)

    # heavy imports kept out of the worker import path
    from backbone import HookedViTFeatV3
    from dataset import ImageNet1k
    from mask_extraction import reduce_features
    import torch.nn.functional as F

    ds = ImageNet1k(split=split)
    todo = build_todo(ds.samples, out_root, split, args.overwrite, args.limit)
    print(f"split={split} | {len(ds.samples)} images | {len(todo)} pending → {out_root}")
    print(f"params: {params.res}px · {params.layer} · PCA{params.pca} · {params.metric} "
          f"conn{params.connectivity} · seg({params.seg_by} "
          f"{params.seg_q if params.seg_by=='quantile' else params.seg_thr}, min={params.seg_min}) "
          f"· merge({params.merge_mode}/{params.merge_by} "
          f"{params.merge_q if params.merge_by=='quantile' else params.merge_thr}, "
          f"seam={params.seam_thickness})")
    if not todo:
        print("nothing to do.")
        return
    print(f"workers: {args.load_workers} loader + {seg_workers} segmentation "
          f"(of {cpu} cores) | batch {args.batch_size}")

    # start CPU segmentation pool (spawn: safe alongside CUDA in this process)
    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=seg_workers * 4)
    done = ctx.Value("i", 0)
    workers = [ctx.Process(target=seg_worker, args=(task_q, params, done), daemon=True)
               for _ in range(seg_workers)]
    for w in workers:
        w.start()

    loader = DataLoader(
        PrepDataset(todo, params.res), batch_size=args.batch_size,
        num_workers=args.load_workers, collate_fn=_collate, pin_memory=True,
        persistent_workers=args.load_workers > 0,
        multiprocessing_context=ctx if args.load_workers > 0 else None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = HookedViTFeatV3(device=device)
    fh = fw = params.res // _PATCH

    n = len(todo)
    skipped = [0]                                       # decode failures (never queued)
    # progress is driven by the workers' shared counter; a monitor thread keeps the
    # bar moving smoothly even while the main loop blocks on back-pressure or join.
    pbar = tqdm(total=n, unit="img", smoothing=0.05, dynamic_ncols=True,
                desc=f"masks[{split}]")
    stop = threading.Event()

    def _monitor():
        while not stop.is_set():
            pbar.n = min(done.value + skipped[0], n)
            pbar.refresh()
            time.sleep(0.3)

    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()

    t0 = time.time()
    for batch, out_paths in loader:
        grids = ext.extract_tensor(batch, fh, fw, apply_norm=params.apply_norm,
                                   names=[params.layer], to_cpu=False)
        g = grids[params.layer]                         # (B, fh, fw, C) on GPU
        for b, op in enumerate(out_paths):
            if op is None:                              # failed decode
                skipped[0] += 1
                continue
            grid = reduce_features(g[b], params.pca, params.smooth, params.whiten)
            if params.l2_normalize:
                grid = F.normalize(grid, dim=-1)
            task_q.put((op, grid.detach().cpu().numpy().astype(np.float32)))

    # drain: signal workers and wait for every queued task to be saved
    for _ in workers:
        task_q.put(None)
    for w in workers:
        w.join()

    stop.set()
    mon.join()
    pbar.n = done.value + skipped[0]
    pbar.refresh()
    pbar.close()
    el = time.time() - t0
    print(f"done: {done.value} masks ({skipped[0]} skipped) in {el/60:.1f} min "
          f"({done.value/max(el,1e-6):.1f} img/s) → {out_root/split}")


if __name__ == "__main__":
    main()
