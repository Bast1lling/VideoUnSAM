"""Generate a temporal pseudo-mask dataset for KV memory-bank training (label-free).

For each of the 26 train-split ("dirty") clips in davis_split.json, samples one
T=6 window (stride 2 -> frames 0,2,4,6,8,10) and runs the same CuVLER+conquer
pipeline that produced datasets/davis_train (divide_conquerV3.py --divide-method
cuvler, invoked as a subprocess so its detectron2 import chain stays isolated).

Cross-frame track IDs are assigned label-free: every frame-0 (window position 0)
mask gets a new global track_id; for later positions, each active track's mask is
OT-propagated (video/propagation/sinkhorn_ot, direct/unsharpened single hop, per
propagation-ot-vs-alternatives) to the next frame and greedily matched to that
frame's CuVLER+conquer proposals by IoU >= --iou-thresh. A track ends when no
proposal matches; unmatched proposals get track_id = -1.

Output: <out-dir>/images/sa_<clip>_<frame_idx>.jpg +
        <out-dir>/annotations/train.json (COCO-style; each image carries
        clip/frame_idx/video_id/window_pos, each annotation carries track_id).

    python -m video.scripts.generate_video_pseudo_dataset --out-dir datasets/davis_video_pseudo
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F
from pycocotools import mask as mask_util

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from video.loaders import davis  # noqa: E402
from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.propagation.sinkhorn_ot import propagate_patch, _mask_to_patch_indicator  # noqa: E402

_SPLIT = _REPO / "video" / "decoder" / "davis_split.json"
_VENV_PY = _REPO / ".venv" / "bin" / "python"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="datasets/davis_video_pseudo")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--iou-thresh", type=float, default=0.3)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--tmp-dir", default="/tmp/davis_video_pseudo_raw")
    ap.add_argument("--skip-divide-conquer", action="store_true",
                     help="reuse an existing tmp-dir/out from a previous run (skip the subprocess)")
    args = ap.parse_args()

    clips = json.load(open(_SPLIT))["dirty"]
    if args.limit_clips:
        clips = clips[:args.limit_clips]

    offsets = [i * args.stride for i in range(args.window)]
    windows = []
    for clip in clips:
        n = davis.num_frames(clip)
        if n > offsets[-1]:
            windows.append((clip, offsets))
        else:
            print(f"[skip] {clip}: only {n} frames, need > {offsets[-1]}")

    print(f"[plan] {len(windows)} windows x {args.window} frames = {len(windows) * args.window} images")

    tmp_in = Path(args.tmp_dir) / "in"
    tmp_out = Path(args.tmp_dir) / "out"

    index = []  # global_idx -> (clip, frame_idx, window_idx, pos)
    for wi, (clip, idxs) in enumerate(windows):
        for pos, fidx in enumerate(idxs):
            index.append((clip, fidx, wi, pos))
    n = len(index)

    if not args.skip_divide_conquer:
        if tmp_in.exists():
            shutil.rmtree(tmp_in)
        if tmp_out.exists():
            shutil.rmtree(tmp_out)
        tmp_in.mkdir(parents=True)
        tmp_out.mkdir(parents=True)

        for gid, (clip, fidx, wi, pos) in enumerate(index):
            frame = davis.load_frame(clip, fidx)
            cv2.imwrite(str(tmp_in / f"sa_{gid}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[dump] {n} frames -> {tmp_in}")

        cmd = [
            str(_VENV_PY), "divide_conquerV3.py",
            "--input-dir", str(tmp_in.resolve()),
            "--output-dir", str(tmp_out.resolve()),
            "--preprocess", "True",
            "--divide-method", "cuvler",
            "--end-id", str(n + 1),  # off-by-one in divide_conquerV3's preprocess loop
        ]
        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(_REPO / "divide_and_conquer"), check=True)

    # --- Track matching + final COCO assembly -------------------------------
    extractor = DenseDINOv3()

    out_images_dir = Path(args.out_dir) / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    images_out, annotations_out = [], []
    img_id, ann_id, next_track_id = 1, 1, 1

    for wi, (clip, idxs) in enumerate(windows):
        active: dict[int, np.ndarray] = {}
        prev_feats = None
        n_anns_window, n_tracked_window = 0, 0
        for pos, fidx in enumerate(idxs):
            gid = sum(len(w[1]) for w in windows[:wi]) + pos
            d = json.load(open(tmp_out / f"sa_{gid}.json"))
            H, W = d["image"]["height"], d["image"]["width"]
            anns = d["annotations"]
            masks = [mask_util.decode(a["segmentation"]) for a in anns]

            cur_feats = extractor.extract(davis.load_frame(clip, fidx))
            track_assign = [-1] * len(anns)

            if pos == 0:
                for i in range(len(anns)):
                    track_assign[i] = next_track_id
                    active[next_track_id] = masks[i]
                    next_track_id += 1
            elif active:
                proposals = []  # (tid, cand_idx, iou)
                for tid, prev_mask in active.items():
                    try:
                        m_a = _mask_to_patch_indicator(prev_mask, prev_feats["grid_h"], prev_feats["grid_w"])
                    except ValueError:
                        continue
                    m_b = propagate_patch(prev_feats["feats"], cur_feats["feats"], m_a, blur=0.05)
                    heat = m_b.reshape(cur_feats["grid_h"], cur_feats["grid_w"]).cpu()
                    heat_up = F.interpolate(heat[None, None], size=(H, W),
                                             mode="bilinear", align_corners=False)[0, 0].numpy()
                    pred = (heat_up > 0.5 * heat_up.max()).astype(np.uint8)
                    for ci, cm in enumerate(masks):
                        s = iou(pred, cm)
                        if s >= args.iou_thresh:
                            proposals.append((tid, ci, s))
                proposals.sort(key=lambda x: -x[2])
                used_tracks, used_cands, new_active = set(), set(), {}
                for tid, ci, s in proposals:
                    if tid in used_tracks or ci in used_cands:
                        continue
                    used_tracks.add(tid); used_cands.add(ci)
                    track_assign[ci] = tid
                    new_active[tid] = masks[ci]
                active = new_active

            out_name = f"sa_{clip}_{fidx}.jpg"
            shutil.copy(tmp_in / f"sa_{gid}.jpg", out_images_dir / out_name)
            images_out.append({
                "id": img_id, "file_name": out_name, "width": W, "height": H,
                "clip": clip, "frame_idx": fidx, "video_id": wi, "window_pos": pos,
            })
            for i, a in enumerate(anns):
                annotations_out.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                    "area": a["area"], "bbox": a["bbox"], "segmentation": a["segmentation"],
                    "width": W, "height": H, "track_id": track_assign[i],
                })
                ann_id += 1
                n_anns_window += 1
                if track_assign[i] != -1:
                    n_tracked_window += 1
            img_id += 1
            prev_feats = cur_feats

        print(f"  [{clip}] window {wi}: {len(idxs)} frames, {n_anns_window} masks, "
              f"{n_tracked_window} tracked, {len(active)} tracks alive at end")

    coco = {
        "info": {"description": "VideoUnSAM temporal pseudo-masks (CuVLER+conquer, OT-tracked)"},
        "licenses": [{"id": 1, "name": "", "url": ""}],
        "categories": [{"id": 1, "name": "fg", "supercategory": "fg"}],
        "images": images_out,
        "annotations": annotations_out,
    }
    out_path = Path(args.out_dir) / "annotations" / "train.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out_path, "w"))

    n_tracks = next_track_id - 1
    n_tracked_anns = sum(1 for a in annotations_out if a["track_id"] != -1)
    print(f"[done] {len(images_out)} images, {len(annotations_out)} annotations, "
          f"{n_tracks} tracks started, {n_tracked_anns} tracked annotations -> {out_path}")


if __name__ == "__main__":
    main()
