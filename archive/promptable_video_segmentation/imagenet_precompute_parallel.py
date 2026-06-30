"""
imagenet_precompute_parallel.py  —  High-throughput chunk-parallel precompute

Double-buffered pipeline per chunk
────────────────────────────────────────────────────────────────────────────
  CPU (thread pool)   ── LOAD chunk N+1 ───────────────────────────────────▶
  GPU (main thread)   ── CutLER batches ── DINOv3 mini-batches ────────────▶
  CPU (process pool)  ── iterative-merge (submitted per DINOv3 mini-batch) ─▶
                      ── NMS + save chunk N-1 ────────────────────────────▶

While the GPU runs CutLER+DINOv3 on chunk N:
  • a background thread pool is already loading chunk N+1 from disk
  • a process pool is running iterative-merge for chunk N's DINOv3 results
    as each mini-batch finishes (overlapping with subsequent DINOv3 batches)
  • the same process pool is saving/compressing chunk N-1 results

Chunk ranges for resumability
────────────────────────────────────────────────────────────────────────────
  Chunk indices are determined by sorting all images in the split and
  slicing by CHUNK_SIZE.  Use --chunk-start / --chunk-end to process a
  sub-range so long runs can be split across days:

    python imagenet_precompute_parallel.py --split train --chunk-start 0   --chunk-end 100
    python imagenet_precompute_parallel.py --split train --chunk-start 100 --chunk-end 200

  Re-running the same range safely skips images whose .npz already exists
  (controlled by SKIP_EXISTING).

Usage
────────────────────────────────────────────────────────────────────────────
  python imagenet_precompute_parallel.py [options]

  --method       v3 | original                 (default: v3)
                 v3       = DINOv3 ViT-L/16 backbone, tuned hyper-params
                 original = DINOv1 ViT-B/8 key features, original hyper-params
                            (from divide_and_conquer/divide_conquer.py)
  --masks-dir    DIR  output dir for .npz masks (default: built-in MASKS_DIR;
                      use a per-method dir to avoid mixing the two outputs)
  --split        train | validation | all     (default: all)
  --chunk-start  N  first chunk index to process  (default: 0)
  --chunk-end    N  exclusive end chunk index      (default: all chunks)
  --chunk-size   N  images per chunk               (default: 256)
  --cutler-batch N  images per CutLER GPU call     (default: 8)
  --dino-batch   N  crops per DINOv3 GPU call      (default: 16)
  --io-workers   N  threads for image loading      (default: 12)
  --post-workers N  processes for CPU post-proc    (default: ncpu-2)

# unchanged default (DINOv3)
python promptable_video_segmentation/imagenet_precompute_parallel.py --masks-dir "/home/sebastian/data/imagenet1k/masksV3" --split validation --chunk-start 0   --chunk-end 100

# original UnSAM algorithm + original hyperparams
python promptable_video_segmentation/imagenet_precompute_parallel.py --method original --masks-dir "/home/sebastian/data/imagenet1k/masksV1" --split validation --chunk-start 0   --chunk-end 100
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

from tqdm import tqdm

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as tvt

# ── path setup — runs in main process AND in spawned/forked workers ───────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
_DAC  = os.path.join(_REPO, "divide_and_conquer")
for _p in (_REPO, _DAC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from promptable_video_segmentation.masks import (
    NMS,
    coverage,
    iterative_merge,
    resize_mask,
    smallest_square_containing_mask,
)


# ── configuration ─────────────────────────────────────────────────────────────
IMAGE_DIR  = "/home/sebastian/data/imagenet1k/images"
MASKS_DIR  = "/home/sebastian/data/imagenet1k/masks"

SPLITS = ["train", "validation"]

CUTLER_CONFIG  = os.path.join(_DAC, "model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml")
CUTLER_WEIGHTS = os.path.join(_DAC, "model_zoo/ckpts/cutler_cascade_final.pth")

DEVICE            = "cuda"
CONFIDENCE_THRESH = 0.1
IMAGE_EXTENSIONS  = (".jpg", ".jpeg", ".png", ".webp")

# ── method presets ────────────────────────────────────────────────────────────
# Two interchangeable divide-and-conquer configurations.  Both run the same
# CutLER divide phase and the same cosine spectral-merge conquer phase; they
# differ only in the feature backbone and the conquer hyper-parameters.
#   • "v3"       — DINOv3 ViT-L/16 backbone with tuned hyper-parameters.
#   • "original" — DINOv1 ViT-B/8 key features with the original hyper-parameters
#                  from divide_and_conquer/divide_conquer.py.
METHOD_PRESETS = {
    "v3": {
        "backbone_kind":     "dinov3",
        "backbone_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "local_size":  512,
        "kept_thresh": 0.8,
        "nms_iou":     0.8,
        "nms_step":    5,
        "thetas":      [0.73, 0.62, 0.51, 0.4, 0.3, 0.2],
    },
    "original": {
        "backbone_kind": "dinov1",
        "backbone_url":  "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        "backbone_arch": "base",
        "vit_feat":      "k",
        "patch_size":    8,
        "feat_dim":      768,
        "local_size":  256,
        "kept_thresh": 0.9,
        "nms_iou":     0.9,
        "nms_step":    5,
        "thetas":      [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    },
}
DEFAULT_METHOD = "v3"

SKIP_EXISTING = True
LOG_EVERY     = 10    # log every N chunks

# Defaults (overridable via CLI)
CHUNK_SIZE   = 256
CUTLER_BATCH = 16
DINO_BATCH   = 32
IO_WORKERS   = 12
POST_WORKERS = max(1, (os.cpu_count() or 4) - 2)
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation matching backbone._to_tensor
_to_tensor = tvt.Compose([
    tvt.ToTensor(),
    tvt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# ── Worker functions (module-level so multiprocessing can pickle them) ────────

def _load_one(entry: tuple[str, str]) -> tuple[str, str, np.ndarray | None]:
    """Load one image. Returns None for bgr if the output already exists."""
    img_path, out_path = entry
    if SKIP_EXISTING and os.path.exists(out_path):
        return img_path, out_path, None
    bgr = cv2.imread(img_path)
    return img_path, out_path, bgr


def _conquer_worker(args: tuple) -> list[np.ndarray]:
    """
    CPU process worker: iterative-merge → coverage filter → NMS for one crop.

    args = (feat_np, ymin, ymax, xmin, xmax, dmask_np,
            thetas, kept_thresh, nms_iou, nms_step)
      feat_np  : (feat_h, feat_w, feat_dim) float32 numpy — backbone features
      dmask_np : (H_img, W_img) uint8 — the CutLER divide-mask for this crop
      thetas, kept_thresh, nms_iou, nms_step — conquer hyper-parameters (passed
        explicitly so they survive process forking regardless of method).

    Returns a list of (H_img, W_img) uint8 candidate conquer-masks.
    """
    (feat_np, ymin, ymax, xmin, xmax, dmask_np,
     thetas, kept_thresh, nms_iou, nms_step) = args
    import torch as _torch
    feat = _torch.from_numpy(feat_np)
    candidates: list[np.ndarray] = []
    for layer in iterative_merge(feat, thetas):
        if layer.shape[0] == 0:
            continue
        for i in range(layer.shape[0]):
            mask     = resize_mask(layer[i], [xmax - xmin, ymax - ymin])
            mask_bin = (mask > 0.5 * 255).astype(np.int32)
            if coverage(mask_bin, dmask_np[ymin:ymax, xmin:xmax]) <= kept_thresh:
                continue
            full = np.zeros(dmask_np.shape, dtype=np.uint8)
            full[ymin:ymax, xmin:xmax] = mask_bin
            candidates.append(full)
    return NMS(candidates, nms_iou, nms_step)


def _save_worker(args: tuple) -> str:
    """CPU process worker: assemble divide+conquer masks and write .npz."""
    out_path, img_path, divide_masks, conquer_per_divide = args
    all_masks:   list[np.ndarray] = []
    mask_types:  list[int]        = []
    divide_idxs: list[int]        = []

    for d_idx, dmask in enumerate(divide_masks):
        all_masks.append(dmask.astype(np.uint8))
        mask_types.append(0)
        divide_idxs.append(-1)
        for cmask in conquer_per_divide[d_idx]:
            all_masks.append(cmask.astype(np.uint8))
            mask_types.append(1)
            divide_idxs.append(d_idx)

    if not all_masks:
        return "skip"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        image_path = np.bytes_(img_path),
        masks      = np.stack(all_masks, axis=0),
        mask_type  = np.array(mask_types,  dtype=np.int8),
        divide_idx = np.array(divide_idxs, dtype=np.int16),
    )
    return "done"


# ── Model loading ─────────────────────────────────────────────────────────────

def _make_cutler_predictor(conf: float):
    from detectron2.config import get_cfg
    from engine.defaults import DefaultPredictor  # CutLER's own DefaultPredictor

    cfg = get_cfg()
    cfg.DATALOADER.COPY_PASTE               = False
    cfg.DATALOADER.COPY_PASTE_RATE          = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO     = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO     = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM    = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE     = False
    cfg.MODEL.ROI_HEADS.USE_DROPLOSS        = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS     = False
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE   = False
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
    cfg.SOLVER.BASE_LR_MULTIPLIER           = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES     = []
    cfg.TEST.NO_SEGM                        = False
    cfg.merge_from_file(CUTLER_CONFIG)
    cfg.merge_from_list(["MODEL.WEIGHTS", CUTLER_WEIGHTS, "MODEL.DEVICE", DEVICE])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST   = conf
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST   = conf
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf
    cfg.freeze()
    return DefaultPredictor(cfg)


def _make_backbone(preset: dict):
    """Build the feature backbone for the chosen method.

    Both variants expose ``.patch_size`` and ``.feat_dim`` and return features
    shaped (B, feat_dim, feat_num²) from a normalised NCHW float tensor, so the
    rest of the pipeline is backbone-agnostic.
    """
    if preset["backbone_kind"] == "dinov1":
        import dino  # divide_and_conquer/dino.py — _DAC is on sys.path
        backbone = dino.ViTFeat(
            preset["backbone_url"], preset["feat_dim"],
            preset["backbone_arch"], preset["vit_feat"], preset["patch_size"],
        )
        backbone.eval()
        backbone.cuda().to(torch.float16)
        return backbone

    from promptable_video_segmentation.backbone import ViTFeatV3
    return ViTFeatV3(model_id=preset["backbone_model_id"]).eval()


# ── GPU inference ─────────────────────────────────────────────────────────────

def _cutler_forward(
    predictor,
    images_bgr: list[np.ndarray],
) -> list[list[np.ndarray]]:
    """
    Batch CutLER forward pass.
    Returns one list of (H, W) bool divide-masks per input image.
    """
    inputs = []
    for bgr in images_bgr:
        h, w = bgr.shape[:2]
        img  = bgr[:, :, ::-1] if predictor.input_format == "RGB" else bgr
        img  = np.ascontiguousarray(img)
        image_t = torch.as_tensor(
            predictor.aug.get_transform(img).apply_image(img)
            .astype("float32").transpose(2, 0, 1)
        )
        inputs.append({"image": image_t, "height": h, "width": w})

    with torch.no_grad():
        outputs = predictor.model(inputs)

    results = []
    for out in outputs:
        masks_t = out["instances"].get("pred_masks")  # (N, H, W) bool
        results.append([masks_t[i].cpu().numpy() for i in range(masks_t.shape[0])])
    return results


def _dino_forward(
    backbone,
    crops_pil: list,
    local_size: int,
) -> np.ndarray:
    """
    Batch backbone forward pass on a list of local_size×local_size PIL crops.
    Returns (B, feat_num, feat_num, feat_dim) float32 numpy array.
    """
    feat_num = local_size // backbone.patch_size
    dtype    = next(backbone.parameters()).dtype
    tensors  = torch.stack([_to_tensor(c) for c in crops_pil]).to(dtype).cuda()

    with torch.no_grad():
        feats = backbone(tensors)  # (B, feat_dim, feat_num²)

    B = len(crops_pil)
    return (
        feats.cpu().float().numpy()
             .reshape(B, backbone.feat_dim, feat_num, feat_num)
             .transpose(0, 2, 3, 1)   # → (B, feat_num, feat_num, feat_dim)
    )


# ── Chunk loading (runs in a background thread) ───────────────────────────────

def _load_chunk(
    entries: list[tuple[str, str]],
    io_workers: int,
) -> list[tuple[str, str, np.ndarray | None]]:
    """Load all images for a chunk using parallel threads."""
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        return list(pool.map(_load_one, entries))


# ── Per-chunk GPU processing ──────────────────────────────────────────────────

def _process_chunk_gpu(
    predictor,
    backbone,
    cpu_pool:      ProcessPoolExecutor,
    chunk_loaded:  list[tuple[str, str, np.ndarray | None]],
    cutler_batch:  int,
    dino_batch:    int,
    local_size:    int,
    thetas:        list[float],
    kept_thresh:   float,
    nms_iou:       float,
    nms_step:      int,
) -> tuple[list[Future], int, int]:
    """
    Run CutLER + DINOv3 on one loaded chunk, dispatch CPU work to cpu_pool.

    Returns
    -------
    save_futures : Future objects for per-image save jobs (not yet collected)
    n_no_masks   : images where CutLER found no instances (counted as skipped)
    n_skipped    : images skipped because output already existed
    """
    to_process = [(img, out, bgr) for img, out, bgr in chunk_loaded if bgr is not None]
    n_skipped  = sum(1 for *_, bgr in chunk_loaded if bgr is None)

    if not to_process:
        return [], 0, n_skipped

    img_paths = [x[0] for x in to_process]
    out_paths = [x[1] for x in to_process]
    bgrs      = [x[2] for x in to_process]

    # ── Stage 1: CutLER (GPU, in sub-batches) ─────────────────────────────────
    divide_masks_list: list[list[np.ndarray]] = []
    rgbs: list[np.ndarray] = []

    cutler_bar = tqdm(
        range(0, len(bgrs), cutler_batch),
        desc="  CutLER", unit="batch",
        total=(len(bgrs) + cutler_batch - 1) // cutler_batch,
        position=1, leave=False, dynamic_ncols=True,
    )
    n_divide_total = 0
    for start in cutler_bar:
        batch_bgrs  = bgrs[start:start + cutler_batch]
        batch_masks = _cutler_forward(predictor, batch_bgrs)
        divide_masks_list.extend(batch_masks)
        rgbs.extend(cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in batch_bgrs)
        n_divide_total += sum(len(m) for m in batch_masks)
        cutler_bar.set_postfix(imgs=start + len(batch_bgrs), masks=n_divide_total)
    cutler_bar.close()

    # ── Stage 2: collect crops ────────────────────────────────────────────────
    # Each entry: (img_idx, mask_idx, crop_pil, ymin, ymax, xmin, xmax, dmask_uint8)
    crop_jobs: list[tuple] = []
    for img_idx, (rgb, divide_masks) in enumerate(zip(rgbs, divide_masks_list)):
        for mask_idx, dmask in enumerate(divide_masks):
            ymin, ymax, xmin, xmax = smallest_square_containing_mask(dmask)
            if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
                continue
            local = rgb[ymin:ymax, xmin:xmax]
            crop  = Image.fromarray(local).resize([local_size, local_size])
            crop_jobs.append(
                (img_idx, mask_idx, crop, ymin, ymax, xmin, xmax, dmask.astype(np.uint8))
            )

    # ── Stage 2: DINOv3 mini-batches + concurrent iterative-merge ────────────
    # Submit _conquer_worker to cpu_pool immediately after each mini-batch so
    # process-pool CPU work overlaps with the next DINOv3 GPU mini-batch.
    conquer_futures: dict[tuple[int, int], Future] = {}

    dino_bar = tqdm(
        range(0, len(crop_jobs), dino_batch),
        desc="  DINOv3", unit="batch",
        total=(len(crop_jobs) + dino_batch - 1) // dino_batch if crop_jobs else 0,
        position=1, leave=False, dynamic_ncols=True,
    )
    for start in dino_bar:
        mini  = crop_jobs[start:start + dino_batch]
        feats = _dino_forward(backbone, [m[2] for m in mini], local_size)  # GPU

        for i, (img_idx, mask_idx, _, ymin, ymax, xmin, xmax, dmask_u8) in enumerate(mini):
            conquer_futures[(img_idx, mask_idx)] = cpu_pool.submit(
                _conquer_worker,
                (feats[i], ymin, ymax, xmin, xmax, dmask_u8,
                 thetas, kept_thresh, nms_iou, nms_step),
            )
        dino_bar.set_postfix(crops=start + len(mini), submitted=len(conquer_futures))
    dino_bar.close()

    # ── Collect iterative-merge results ───────────────────────────────────────
    # Some pool tasks may already be done (ran during DINOv3); the rest we wait on.
    conquer_results: dict[tuple[int, int], list[np.ndarray]] = {}
    merge_bar = tqdm(
        conquer_futures.items(),
        desc="  merge", unit="crop",
        total=len(conquer_futures),
        position=1, leave=False, dynamic_ncols=True,
    )
    for key, fut in merge_bar:
        conquer_results[key] = fut.result()
    merge_bar.close()

    # ── Stage 3: dispatch save jobs ───────────────────────────────────────────
    save_futures: list[Future] = []
    n_no_masks = 0

    for img_idx in range(len(img_paths)):
        divide_masks = divide_masks_list[img_idx]
        if not divide_masks:
            n_no_masks += 1
            continue
        conquer_per_divide = [
            conquer_results.get((img_idx, m_idx), [])
            for m_idx in range(len(divide_masks))
        ]
        save_futures.append(cpu_pool.submit(
            _save_worker,
            (out_paths[img_idx], img_paths[img_idx], divide_masks, conquer_per_divide),
        ))

    return save_futures, n_no_masks, n_skipped


# ── Split-level driver ────────────────────────────────────────────────────────

def _collect_entries(split: str) -> list[tuple[str, str]]:
    """Return sorted list of (img_path, out_path) for all images in a split."""
    split_img_dir  = os.path.join(IMAGE_DIR, split)
    split_mask_dir = os.path.join(MASKS_DIR, split)
    entries: list[tuple[str, str]] = []
    for dirpath, _, files in os.walk(split_img_dir):
        for fname in sorted(files):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                abs_path = os.path.join(dirpath, fname)
                rel_stem = os.path.relpath(os.path.splitext(abs_path)[0], split_img_dir)
                out_path = os.path.join(split_mask_dir, rel_stem + ".npz")
                entries.append((abs_path, out_path))
    entries.sort()
    return entries


def run_precompute(
    splits:       list[str],
    preset:       dict,
    chunk_start:  int,
    chunk_end:    int | None,
    chunk_size:   int,
    cutler_batch: int,
    dino_batch:   int,
    io_workers:   int,
    post_workers: int,
) -> None:
    print(f"Loading CutLER…")
    predictor = _make_cutler_predictor(CONFIDENCE_THRESH)
    print(f"Loading {preset['backbone_kind']} backbone…")
    backbone  = _make_backbone(preset)

    # Create process pool BEFORE any CUDA tensors are allocated to avoid
    # fork-after-CUDA issues (workers do pure CPU work, so fork is safe here).
    mp_ctx = __import__("multiprocessing").get_context("fork")
    cpu_pool = ProcessPoolExecutor(max_workers=post_workers, mp_context=mp_ctx)

    # Single background thread for chunk pre-loading
    prefetch_ex = ThreadPoolExecutor(max_workers=1)

    try:
        for split in splits:
            _run_split(
                split, predictor, backbone, preset, cpu_pool, prefetch_ex,
                chunk_start, chunk_end, chunk_size, cutler_batch,
                dino_batch, io_workers,
            )
    finally:
        cpu_pool.shutdown(wait=True)
        prefetch_ex.shutdown(wait=False)


def _run_split(
    split:        str,
    predictor,
    backbone,
    preset:       dict,
    cpu_pool:     ProcessPoolExecutor,
    prefetch_ex:  ThreadPoolExecutor,
    chunk_start:  int,
    chunk_end:    int | None,
    chunk_size:   int,
    cutler_batch: int,
    dino_batch:   int,
    io_workers:   int,
) -> None:
    split_img_dir = os.path.join(IMAGE_DIR, split)
    if not os.path.isdir(split_img_dir):
        tqdm.write(f"[{split}] image dir not found ({split_img_dir}), skipping.")
        return

    all_entries = _collect_entries(split)
    n_total     = len(all_entries)
    n_chunks    = (n_total + chunk_size - 1) // chunk_size

    c_end = min(chunk_end, n_chunks) if chunk_end is not None else n_chunks
    c_end = max(c_end, chunk_start)

    tqdm.write(
        f"\n[{split}]  images={n_total:,}  total_chunks={n_chunks}"
        f"  processing chunks [{chunk_start}, {c_end})"
        f"  ({(c_end - chunk_start) * chunk_size:,} images max)"
    )

    # Pre-load the first chunk in the background
    first_entries = all_entries[chunk_start * chunk_size : (chunk_start + 1) * chunk_size]
    prefetch_fut: Future = prefetch_ex.submit(_load_chunk, first_entries, io_workers)

    # Saves from the previous chunk; collected at start of each new chunk to
    # bound memory and give the pool time to finish while the GPU runs.
    prev_save_futures: list[Future] = []

    total_done = total_skipped = total_failed = 0

    chunk_bar = tqdm(
        range(chunk_start, c_end),
        desc=f"[{split}]", unit="chunk",
        position=0, leave=True, dynamic_ncols=True,
    )

    for c_idx in chunk_bar:
        # ── Wait for this chunk's images to finish loading ─────────────────
        loaded_chunk = prefetch_fut.result()

        # ── Kick off loading the NEXT chunk immediately ────────────────────
        if c_idx + 1 < c_end:
            next_entries = all_entries[(c_idx + 1) * chunk_size : (c_idx + 2) * chunk_size]
            prefetch_fut = prefetch_ex.submit(_load_chunk, next_entries, io_workers)

        # ── Collect saves from the PREVIOUS chunk ─────────────────────────
        # (these ran concurrently while the GPU processed the current chunk)
        if prev_save_futures:
            save_bar = tqdm(
                prev_save_futures,
                desc="  saves", unit="img",
                position=1, leave=False, dynamic_ncols=True,
            )
            for fut in save_bar:
                try:
                    if fut.result() == "done":
                        total_done += 1
                except Exception as exc:
                    tqdm.write(f"  [save error] {exc}")
                    total_failed += 1
            save_bar.close()
            prev_save_futures = []

        chunk_bar.set_postfix(done=total_done, skip=total_skipped, fail=total_failed)

        # ── GPU: CutLER + backbone + dispatch iterative-merge ─────────────
        save_futures, n_no_masks, n_skipped = _process_chunk_gpu(
            predictor, backbone, cpu_pool, loaded_chunk,
            cutler_batch, dino_batch,
            preset["local_size"], preset["thetas"], preset["kept_thresh"],
            preset["nms_iou"], preset["nms_step"],
        )
        total_skipped += n_skipped + n_no_masks
        prev_save_futures = save_futures
        chunk_bar.set_postfix(done=total_done, skip=total_skipped, fail=total_failed)

    chunk_bar.close()

    # ── Collect saves from the last chunk ─────────────────────────────────
    if prev_save_futures:
        save_bar = tqdm(
            prev_save_futures,
            desc="  final saves", unit="img",
            position=1, leave=False, dynamic_ncols=True,
        )
        for fut in save_bar:
            try:
                if fut.result() == "done":
                    total_done += 1
            except Exception as exc:
                tqdm.write(f"  [save error] {exc}")
                total_failed += 1
        save_bar.close()

    tqdm.write(
        f"[{split}] finished chunks [{chunk_start}, {c_end})."
        f"  done={total_done:,}  skipped={total_skipped:,}  failed={total_failed:,}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global MASKS_DIR
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method",
        choices=list(METHOD_PRESETS), default=DEFAULT_METHOD,
        help=f"Divide-and-conquer variant: 'v3' (DINOv3, tuned) or 'original' "
             f"(DINOv1 ViT-B/8, original hyper-params). Default: {DEFAULT_METHOD}")
    parser.add_argument("--masks-dir", type=str, default=MASKS_DIR,
        help=f"Output directory for .npz masks (default: {MASKS_DIR}). "
             f"Use a method-specific dir to avoid mixing outputs from the two methods.")
    parser.add_argument("--split",
        choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--chunk-start", type=int, default=0,
        help="First chunk index to process (default: 0)")
    parser.add_argument("--chunk-end", type=int, default=None,
        help="Exclusive end chunk index (default: all chunks)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Images per chunk (default: {CHUNK_SIZE})")
    parser.add_argument("--cutler-batch", type=int, default=CUTLER_BATCH,
        help=f"Images per CutLER GPU call (default: {CUTLER_BATCH})")
    parser.add_argument("--dino-batch", type=int, default=DINO_BATCH,
        help=f"Crops per DINOv3 GPU call (default: {DINO_BATCH})")
    parser.add_argument("--io-workers", type=int, default=IO_WORKERS,
        help=f"Threads for image loading (default: {IO_WORKERS})")
    parser.add_argument("--post-workers", type=int, default=POST_WORKERS,
        help=f"Processes for CPU post-processing (default: {POST_WORKERS})")
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]
    preset = METHOD_PRESETS[args.method]

    MASKS_DIR = args.masks_dir

    print(
        f"Method: {args.method}  (backbone={preset['backbone_kind']}, "
        f"local_size={preset['local_size']}, kept_thresh={preset['kept_thresh']}, "
        f"nms_iou={preset['nms_iou']}, thetas={preset['thetas']})\n"
        f"Masks output dir: {MASKS_DIR}"
    )

    run_precompute(
        splits       = splits,
        preset       = preset,
        chunk_start  = args.chunk_start,
        chunk_end    = args.chunk_end,
        chunk_size   = args.chunk_size,
        cutler_batch = args.cutler_batch,
        dino_batch   = args.dino_batch,
        io_workers   = args.io_workers,
        post_workers = args.post_workers,
    )


if __name__ == "__main__":
    main()
