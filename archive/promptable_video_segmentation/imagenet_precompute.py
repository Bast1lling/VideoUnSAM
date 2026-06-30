"""
Download ImageNet-1k from HuggingFace and precompute Divide-and-Conquer masks.

Prerequisites
─────────────
  pip install datasets huggingface_hub
  huggingface-cli login          # one-time; account must have accepted imagenet-1k terms

Phases
──────
  download    — fetch train+val splits, save as JPEG files on disk
  precompute  — run CutLER divide + DINOv3 conquer on every image, save .npz

Both phases are fully resumable: re-run and already-saved files are skipped.

Parallelism
───────────
  download   — N worker threads encode and save JPEGs concurrently
  precompute — reader thread prefetches images from disk; GPU inference runs on
               the main thread; a writer thread compresses and saves .npz files
               asynchronously, overlapping I/O with GPU work

Usage
─────
  python imagenet_precompute.py                     # both phases in sequence
  python imagenet_precompute.py --phase download
  python imagenet_precompute.py --phase precompute
  python imagenet_precompute.py --phase precompute --split validation
  python imagenet_precompute.py --workers 16        # override thread count
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import PIL.Image as Image

# ── resolve repo root ─────────────────────────────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
_DAC  = os.path.join(_REPO, "divide_and_conquer")
sys.path.insert(0, _REPO)
# CutLER has its own registries (ROI_HEADS, META_ARCH) separate from Detectron2's.
# Its engine/defaults.py must be importable so the custom DefaultPredictor can
# call CutLER's build_model, which routes through the correct registries.
sys.path.insert(0, _DAC)

from promptable_video_segmentation.masks import (
    NMS,
    coverage,
    iterative_merge,
    resize_mask,
    smallest_square_containing_mask,
)
from promptable_video_segmentation.backbone import ViTFeatV3, extract_feature_matrix


# ── configuration ─────────────────────────────────────────────────────────────
# Paths
IMAGE_DIR  = "/home/sebastian/data/imagenet1k/images"   # images saved here by download phase
MASKS_DIR  = "/home/sebastian/data/imagenet1k/masks"    # .npz files saved here by precompute phase
HF_CACHE   = None                           # HuggingFace cache dir (None = default)

# Splits to process (download and precompute both respect this list)
SPLITS = ["train", "validation"]

# CutLER
CUTLER_CONFIG  = os.path.join(
    _REPO, "divide_and_conquer",
    "model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
)
CUTLER_WEIGHTS = os.path.join(
    _REPO, "divide_and_conquer",
    "model_zoo/ckpts/cutler_cascade_final.pth",
)

# Inference
BACKBONE_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEVICE            = "cuda"
CONFIDENCE_THRESH = 0.1
IMAGE_EXTENSIONS  = (".jpg", ".jpeg", ".png", ".webp", ".jpeg")

# Conquer hyper-parameters (mirror precompute_masks.py defaults)
LOCAL_SIZE   = 512
KEPT_THRESH  = 0.8
NMS_IOU      = 0.8
NMS_STEP     = 5
THETAS       = [0.73, 0.62, 0.51, 0.4, 0.3, 0.2]

SKIP_EXISTING   = True    # skip images whose .npz / JPEG already exists
LOG_EVERY       = 500     # print a summary line every N images
NUM_WORKERS     = 12 #min(8, os.cpu_count() or 4)   # default I/O thread count
PREFETCH_BUFFER = 16      # max images queued ahead of GPU inference
# ─────────────────────────────────────────────────────────────────────────────


# ── progress bar helper ───────────────────────────────────────────────────────

def _make_pbar(total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc)
    except ImportError:
        class _Noop:
            def update(self, _n: int = 1) -> None: pass
            def close(self) -> None: pass
        return _Noop()


# ── Phase 1 — download ────────────────────────────────────────────────────────

def _save_image(global_idx: int, item: dict, split_dir: str) -> str:
    """Encode one PIL image to JPEG on disk. Returns 'done', 'skip', or 'fail:<msg>'."""
    label: int       = item["label"]
    img: Image.Image = item["image"]

    class_dir = os.path.join(split_dir, f"{label:04d}")
    os.makedirs(class_dir, exist_ok=True)
    out_path = os.path.join(class_dir, f"{global_idx:08d}.JPEG")

    if SKIP_EXISTING and os.path.exists(out_path):
        return "skip"
    try:
        img.convert("RGB").save(out_path, "JPEG", quality=95)
        return "done"
    except Exception as exc:
        return f"fail:{exc}"


def download_split(split: str, workers: int = NUM_WORKERS) -> None:
    """Download one HuggingFace imagenet-1k split and write JPEG files to disk.

    Uses a thread pool so JPEG encoding and disk writes happen in parallel with
    HuggingFace dataset iteration.  A sliding window bounds in-flight futures so
    memory stays constant regardless of split size.
    """
    from datasets import load_dataset  # type: ignore

    split_dir = os.path.join(IMAGE_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

    print(f"\n[download] split={split}  target={split_dir}  workers={workers}")
    print("Loading dataset metadata from HuggingFace (may take a few minutes)…")

    ds = load_dataset(
        "imagenet-1k",
        split=split,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    total = len(ds)
    print(f"  {total:,} examples")

    n_done = n_skipped = n_failed = 0
    WINDOW = workers * 8
    pending: deque[tuple[int, object]] = deque()
    pbar = _make_pbar(total, split)

    def _drain_one(idx: int, fut) -> None:
        nonlocal n_done, n_skipped, n_failed
        r = fut.result()
        if r == "done":
            n_done += 1
        elif r == "skip":
            n_skipped += 1
        else:
            n_failed += 1
            print(f"  [{idx}] save error: {r[5:]}")
        pbar.update(1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for global_idx, item in enumerate(ds):
            # Drain oldest future when the window is full to bound memory.
            while len(pending) >= WINDOW:
                _drain_one(*pending.popleft())

            if global_idx % LOG_EVERY == 0 and global_idx > 0:
                print(
                    f"  {global_idx:>8,}/{total:,}  "
                    f"done={n_done}  skipped={n_skipped}  failed={n_failed}"
                )

            pending.append((global_idx, pool.submit(_save_image, global_idx, item, split_dir)))

        while pending:
            _drain_one(*pending.popleft())

    pbar.close()
    print(
        f"[download] {split} done.  "
        f"done={n_done}  skipped={n_skipped}  failed={n_failed}"
    )


def run_download(splits: list[str], workers: int = NUM_WORKERS) -> None:
    for split in splits:
        download_split(split, workers)


# ── CutLER predictor ──────────────────────────────────────────────────────────

def _make_cutler_predictor(config_file: str, weights: str, device: str, conf: float):
    from detectron2.config import get_cfg
    # CutLER's DefaultPredictor calls its own build_model, which uses the CutLER-local
    # ROI_HEADS / META_ARCH registries where CustomCascadeROIHeads is registered.
    # detectron2.engine.DefaultPredictor uses detectron2's registries and will not find it.
    from engine.defaults import DefaultPredictor

    cfg = get_cfg()
    cfg.DATALOADER.COPY_PASTE              = False
    cfg.DATALOADER.COPY_PASTE_RATE         = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO    = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO    = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM   = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE    = False
    cfg.MODEL.ROI_HEADS.USE_DROPLOSS       = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS    = False
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE  = False
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
    cfg.SOLVER.BASE_LR_MULTIPLIER          = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES    = []
    cfg.TEST.NO_SEGM                       = False

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights, "MODEL.DEVICE", device])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST  = conf
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST  = conf
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf
    cfg.freeze()
    return DefaultPredictor(cfg)


# ── per-image GPU inference ───────────────────────────────────────────────────

def _run_divide(predictor, image_bgr: np.ndarray) -> list[np.ndarray]:
    preds   = predictor(image_bgr)
    masks_t = preds["instances"].get("pred_masks")
    return [masks_t[i].cpu().numpy() for i in range(masks_t.shape[0])]


def _run_conquer(
    backbone: ViTFeatV3,
    image_rgb: np.ndarray,
    divide_mask: np.ndarray,
) -> list[np.ndarray]:
    ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
    if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
        return []

    local_rgb = image_rgb[ymin:ymax, xmin:xmax]
    resized   = Image.fromarray(local_rgb).resize([LOCAL_SIZE, LOCAL_SIZE])
    feat_num  = LOCAL_SIZE // backbone.patch_size
    feat      = extract_feature_matrix(backbone, resized, backbone.feat_dim, feat_num)

    candidates: list[np.ndarray] = []
    for layer in iterative_merge(feat, THETAS):
        if layer.shape[0] == 0:
            continue
        for i in range(layer.shape[0]):
            mask     = resize_mask(layer[i], [xmax - xmin, ymax - ymin])
            mask_bin = (mask > 0.5 * 255).astype(int)
            if coverage(mask_bin, divide_mask[ymin:ymax, xmin:xmax]) <= KEPT_THRESH:
                continue
            full = np.zeros_like(divide_mask)
            full[ymin:ymax, xmin:xmax] = mask_bin
            candidates.append(full)

    return NMS(candidates, NMS_IOU, NMS_STEP)


# ── Phase 2 — precompute ──────────────────────────────────────────────────────

def _reader_worker(
    entries: list[tuple[str, str, str]],
    read_q: queue.Queue,
) -> None:
    """Background thread: read images from disk and push decoded arrays to read_q.

    Pushes tuples of ("ok"|"skip"|"fail", img_path, out_path, bgr, rgb).
    A None sentinel is pushed when all entries are exhausted.
    """
    for img_path, _, out_path in entries:
        if SKIP_EXISTING and os.path.exists(out_path):
            read_q.put(("skip", img_path, out_path, None, None))
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            read_q.put(("fail", img_path, out_path, None, None))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        read_q.put(("ok", img_path, out_path, bgr, rgb))
    read_q.put(None)


def _writer_worker(write_q: queue.Queue) -> None:
    """Background thread: compress and write .npz files from write_q.

    Overlaps zlib compression with GPU inference on the main thread.
    Exits when it receives a None sentinel.
    """
    while True:
        item = write_q.get()
        if item is None:
            break
        out_path, img_path, all_masks, mask_types, divide_idxs = item
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            image_path = np.bytes_(img_path),
            masks      = np.stack(all_masks, axis=0),
            mask_type  = np.array(mask_types, dtype=np.int8),
            divide_idx = np.array(divide_idxs, dtype=np.int16),
        )


def precompute_split(
    predictor,
    backbone: ViTFeatV3,
    split: str,
) -> None:
    """Run divide+conquer inference on all images for one split.

    Pipeline:
      reader thread  →  [read_q, size=PREFETCH_BUFFER]  →  GPU (main thread)
                                                         →  [write_q]  →  writer thread

    The reader prefetches decoded images so the GPU is never stalled waiting
    for disk I/O.  The writer compresses and saves .npz files asynchronously
    so the GPU is not blocked by zlib compression.
    """
    split_img_dir  = os.path.join(IMAGE_DIR, split)
    split_mask_dir = os.path.join(MASKS_DIR, split)

    if not os.path.isdir(split_img_dir):
        print(f"[precompute] {split}: image dir not found ({split_img_dir}), skipping.")
        return

    img_entries: list[tuple[str, str, str]] = []
    for dirpath, _dirs, files in os.walk(split_img_dir):
        for fname in sorted(files):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                abs_path = os.path.join(dirpath, fname)
                rel_stem = os.path.relpath(
                    os.path.splitext(abs_path)[0], split_img_dir
                )
                out_path = os.path.join(split_mask_dir, rel_stem + ".npz")
                img_entries.append((abs_path, rel_stem, out_path))

    total = len(img_entries)
    print(
        f"\n[precompute] split={split}  images={total:,}  out={split_mask_dir}"
        f"  prefetch={PREFETCH_BUFFER}"
    )

    read_q  = queue.Queue(maxsize=PREFETCH_BUFFER)
    write_q: queue.Queue = queue.Queue()

    reader = threading.Thread(target=_reader_worker, args=(img_entries, read_q), daemon=True)
    writer = threading.Thread(target=_writer_worker, args=(write_q,), daemon=True)
    reader.start()
    writer.start()

    n_done = n_skipped = n_failed = 0
    pbar = _make_pbar(total, split)

    for i in range(total):
        packet = read_q.get()
        if packet is None:
            break

        status, img_path, out_path, bgr, rgb = packet

        if status == "skip":
            n_skipped += 1
            pbar.update(1)
            continue

        if status == "fail":
            n_failed += 1
            print(f"  FAIL (read) [{i+1}/{total}] {img_path}")
            pbar.update(1)
            continue

        # ── GPU inference on main thread ──────────────────────────────────
        try:
            divide_masks = _run_divide(predictor, bgr)
        except Exception:
            traceback.print_exc()
            n_failed += 1
            pbar.update(1)
            continue

        if not divide_masks:
            n_skipped += 1
            pbar.update(1)
            continue

        all_masks:   list[np.ndarray] = []
        mask_types:  list[int]        = []
        divide_idxs: list[int]        = []

        for d_idx, dmask in enumerate(divide_masks):
            all_masks.append(dmask.astype(np.uint8))
            mask_types.append(0)
            divide_idxs.append(-1)

            try:
                c_masks = _run_conquer(backbone, rgb, dmask)
            except Exception:
                c_masks = []

            for cmask in c_masks:
                all_masks.append(cmask.astype(np.uint8))
                mask_types.append(1)
                divide_idxs.append(d_idx)

        # Hand off to the writer thread; GPU can immediately start the next image.
        write_q.put((out_path, img_path, all_masks, mask_types, divide_idxs))
        n_done += 1
        # ─────────────────────────────────────────────────────────────────

        pbar.update(1)
        if (i + 1) % LOG_EVERY == 0:
            print(
                f"  [{i+1:>8,}/{total:,}]  "
                f"done={n_done}  skipped={n_skipped}  failed={n_failed}"
            )

    # Signal writer to flush and exit, then wait for both threads.
    write_q.put(None)
    writer.join()
    reader.join()
    pbar.close()

    print(
        f"[precompute] {split} done.  "
        f"done={n_done}  skipped={n_skipped}  failed={n_failed}"
    )


def run_precompute(splits: list[str]) -> None:
    print("Loading CutLER…")
    predictor = _make_cutler_predictor(CUTLER_CONFIG, CUTLER_WEIGHTS, DEVICE, CONFIDENCE_THRESH)

    print("Loading DINOv3 backbone…")
    backbone = ViTFeatV3(model_id=BACKBONE_MODEL_ID).eval()

    for split in splits:
        precompute_split(predictor, backbone, split)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--phase",
        choices=["download", "precompute", "all"],
        default="precompute",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "all"],
        default="all",
        help="Which split(s) to process (default: all = train + validation)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"I/O thread count (default: {NUM_WORKERS} = min(8, cpu_count))",
    )
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]

    if args.phase in ("download", "all"):
        run_download(splits, args.workers)

    if args.phase in ("precompute", "all"):
        run_precompute(splits)


if __name__ == "__main__":
    main()
