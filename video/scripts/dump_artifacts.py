"""Dump per-frame pipeline artifacts so boundary refinement can be developed offline.

Runs the SAME OT chain as video/scripts/eval_davis2016.py (CuVLER+conquer seed →
Sinkhorn OT → periodic reseed) but instead of scoring, writes every intermediate a
refinement experiment needs to disk. The dump is a few hundred MB for all of DAVIS
2016 val, so it can be scp'd to a laptop and iterated on with no GPU.

WHY THIS IS FAITHFUL
--------------------
In eval_clip the OT chain carries `patch = heat` forward, and the *display* mask
(what J&F is computed on) is produced downstream by CRF/threshold. So any refinement
that only changes the display stage is EXACTLY reproducible from this dump — no GPU,
no re-running the chain.

CAVEAT: `ot_mask` (the thresholded binary) does feed back into the chain at reseed
frames, via pick_proposal. So changes to --thresh / --guided / --cc-filter alter the
chain itself and are NOT reproducible offline; those need a real cluster re-run. The
harness prints a warning when you touch a chain-altering parameter.

Per frame it writes:
    frames/<clip>/<fidx>.jpg      RGB frame
    npz/<clip>/<fidx>.npz         heat [gh,gw] f16, pca3 [gh,gw,3] u8,
                                  ot_mask + gt (bit-packed), shape
    props/<clip>/<fidx>.pkl       CuVLER+conquer proposals as COCO RLEs

Usage (on the cluster, inside a Slurm job):
    python -m video.scripts.dump_artifacts \
        --davis-root /storage/slurm/$USER/davis/DAVIS \
        --out-dir /storage/slurm/$USER/dumps/davis2016_default \
        --clips blackswan,car-roundabout,car-shadow,drift-straight,dog,goat

    # everything (20 clips, ~35 min on one 24GB GPU, ~250 MB)
    python -m video.scripts.dump_artifacts --out-dir dumps/davis2016_default
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))

from pycocotools import mask as mask_util  # noqa: E402

from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.loaders import davis  # noqa: E402
from video.divide.cuvler_divide import CuVLERDivider  # noqa: E402
from video.divide.conquer import load_backbone, run_conquer  # noqa: E402
from video.propagation.sinkhorn_ot import (  # noqa: E402
    propagate_patch, _mask_to_patch_indicator,
)

# NOTE: the helpers below are deliberately duplicated from
# video/scripts/eval_davis2016.py rather than imported. Importing that module
# transitively drags in segment_anything (via video.decoder.train_sam_decoder)
# and pydensecrf (via video.refine.dense_crf), neither of which this script
# needs — and both of which are awkward to install. Keep them in sync; they are
# short and stable. Verified identical to eval_davis2016 as of commit 721977e.

DAVIS_2016_VAL = [
    "blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout",
    "car-shadow", "cows", "dance-twirl", "dog", "drift-chicane",
    "drift-straight", "goat", "horsejump-high", "kite-surf", "libby",
    "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox",
]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def mask_to_patch(mask: np.ndarray, gh: int = 64, gw: int = 64) -> torch.Tensor:
    try:
        return _mask_to_patch_indicator(mask, gh, gw)
    except ValueError:
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
        if pooled.sum() <= 0:
            pooled = pooled + 1.0 / (gh * gw)
        return pooled.flatten()


def pick_proposal(masks: list[np.ndarray], ref: np.ndarray,
                  click_xy: tuple | None = None) -> np.ndarray | None:
    if not masks:
        return None
    if click_xy is not None:
        cx, cy = click_xy
        containing = [m for m in masks if m[int(cy), int(cx)] > 0]
        if containing:
            return max(containing, key=lambda m: iou(m, ref))
    return max(masks, key=lambda m: iou(m, ref))


def largest_cc_near(mask: np.ndarray, prev_centroid: tuple[float, float] | None) -> np.ndarray:
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return mask
    comp_ids = range(1, n_labels)
    if prev_centroid is None:
        best = max(comp_ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    else:
        px, py = prev_centroid
        best = min(comp_ids, key=lambda i: (centroids[i][0] - px) ** 2 + (centroids[i][1] - py) ** 2)
    return (labels == best).astype(np.uint8)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _sample_center_click(mask: np.ndarray) -> tuple[float, float] | None:
    """Distance-transform peak — same click simulation as eval_davis2016."""
    if mask.sum() == 0:
        return None
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    fy, fx = np.unravel_index(int(dt.argmax()), dt.shape)
    return float(fx), float(fy)


def _pca3(feats_hwd: np.ndarray) -> np.ndarray:
    """[h,w,D] float → [h,w,3] uint8. Same reduction dense_crf.py uses for its
    feature-bilateral kernel, precomputed here so the laptop never needs DINOv3."""
    h, w, d = feats_hwd.shape
    flat = feats_hwd.reshape(-1, d).astype(np.float32)
    centered = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:3].T
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    proj = (proj - lo) / np.clip(hi - lo, 1e-6, None) * 255.0
    return proj.reshape(h, w, 3).astype(np.uint8)


def _encode_props(masks: list[np.ndarray]) -> list[dict]:
    out = []
    for m in masks:
        if m is None or m.sum() == 0:
            continue
        rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
        out.append(rle)
    return out


def dump_clip(clip: str, dino: DenseDINOv3, divider: CuVLERDivider,
              conquer_backbone, args, out: Path) -> dict:
    n = davis.num_frames(clip)
    if args.max_frames:
        n = min(n, args.max_frames)
    inst_ids = davis.instance_ids(clip, 0)
    if not inst_ids:
        return {"clip": clip, "skipped": "no instances"}
    inst = args.instance_id if args.instance_id > 0 else inst_ids[0]

    (out / "frames" / clip).mkdir(parents=True, exist_ok=True)
    (out / "npz" / clip).mkdir(parents=True, exist_ok=True)
    (out / "props" / clip).mkdir(parents=True, exist_ok=True)

    frame0 = davis.load_frame(clip, 0)
    H, W = frame0.shape[:2]
    gt0 = davis.load_mask(clip, 0, instance_id=inst)
    click_xy = _sample_center_click(gt0)

    # ── Frame 0 seed ────────────────────────────────────────────────────────
    props0 = divider.predict(frame0)
    if conquer_backbone is not None:
        props0 = run_conquer(conquer_backbone, frame0, props0)
    seed = pick_proposal(props0, gt0, click_xy=click_xy)
    if seed is None:
        seed = gt0.astype(np.uint8)

    img_sized = cv2.resize(frame0, (args.feat_size, args.feat_size))
    with torch.no_grad():
        feats_prev_raw = dino.extract(img_sized, normalize=False)["feats"]
        feats_prev = feats_prev_raw.cuda().float()
    feats_prev_norm = F.normalize(feats_prev, dim=-1)
    gh, gw = feats_prev_norm.shape[:2]

    cur_mask = seed.astype(np.uint8)
    patch = mask_to_patch(cur_mask, gh=gh, gw=gw)
    prev_centroid = mask_centroid(cur_mask)

    def write_frame(fidx, frame, gt, heat_grid, ot_mask, feats_raw, props, reseeded):
        cv2.imwrite(str(out / "frames" / clip / f"{fidx:05d}.jpg"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        payload = {
            "heat": heat_grid.astype(np.float16),
            "ot_mask": np.packbits(ot_mask.astype(bool)),
            "gt": np.packbits(gt.astype(bool)),
            "shape": np.array([H, W], dtype=np.int32),
            "grid": np.array([gh, gw], dtype=np.int32),
            "reseeded": np.array([int(reseeded)], dtype=np.int8),
        }
        if not args.no_pca:
            payload["pca3"] = _pca3(feats_raw.float().cpu().numpy())
        np.savez_compressed(out / "npz" / clip / f"{fidx:05d}.npz", **payload)
        if props is not None:
            with open(out / "props" / clip / f"{fidx:05d}.pkl", "wb") as fh:
                pickle.dump(_encode_props(props), fh)

    # frame 0: "heat" is just the seed indicator reshaped to the grid
    write_frame(0, frame0, gt0, patch.reshape(gh, gw).cpu().numpy(),
                cur_mask, feats_prev_raw, props0, reseeded=True)

    # ── Chain ───────────────────────────────────────────────────────────────
    for fidx in range(1, n):
        frame = davis.load_frame(clip, fidx)
        gt = davis.load_mask(clip, fidx, instance_id=inst)

        img_sized = cv2.resize(frame, (args.feat_size, args.feat_size))
        with torch.no_grad():
            feats_cur_raw = dino.extract(img_sized, normalize=False)["feats"]
            feats_cur = feats_cur_raw.cuda().float()
        feats_cur_norm = F.normalize(feats_cur, dim=-1)
        gh_c, gw_c = feats_cur_norm.shape[:2]

        heat = propagate_patch(feats_prev_norm, feats_cur_norm, patch, blur=args.blur)

        heat_up = F.interpolate(
            heat.reshape(1, 1, gh_c, gw_c), size=(H, W), mode="bilinear",
            align_corners=False
        )[0, 0].cpu().numpy()
        soft_up = heat_up / (heat_up.max() + 1e-8)
        ot_mask = (soft_up > args.thresh).astype(np.uint8)
        if args.cc_filter:
            ot_mask = largest_cc_near(ot_mask, prev_centroid)

        cur_mask = ot_mask
        patch = heat
        prev_centroid = mask_centroid(cur_mask) or prev_centroid

        # Proposals: always at reseed frames (needed by the chain). Every frame if
        # --props-every, which is what the superpixel-snapping experiment needs.
        is_reseed = args.reseed_interval > 0 and fidx % args.reseed_interval == 0
        props = None
        if is_reseed or args.props_every:
            props = divider.predict(frame)
            if conquer_backbone is not None:
                props = run_conquer(conquer_backbone, frame, props)

        reseeded = False
        if is_reseed and props:
            candidate = pick_proposal(props, ot_mask)
            if candidate is not None and iou(ot_mask, candidate) >= args.reseed_thresh:
                cur_mask = candidate
                patch = mask_to_patch(cur_mask, gh=gh, gw=gw)
                prev_centroid = mask_centroid(cur_mask) or prev_centroid
                reseeded = True

        write_frame(fidx, frame, gt, heat.reshape(gh_c, gw_c).cpu().numpy(),
                    cur_mask, feats_cur_raw, props, reseeded)

        feats_prev_norm = feats_cur_norm

    return {"clip": clip, "n_frames": n, "instance_id": inst, "grid": [gh, gw]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="dumps/davis2016_default")
    ap.add_argument("--davis-root", default="",
                    help="Override the hardcoded DAVIS path in video/loaders/davis.py")
    ap.add_argument("--clips", default="", help="Comma-separated; default = DAVIS 2016 val")
    ap.add_argument("--instance-id", type=int, default=0, help="0 = first instance in frame 0")
    ap.add_argument("--feat-size", type=int, default=1024)
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--reseed-interval", type=int, default=10)
    ap.add_argument("--reseed-thresh", type=float, default=0.3)
    ap.add_argument("--cuvler-score", type=float, default=0.35)
    ap.add_argument("--no-conquer", action="store_true")
    ap.add_argument("--cc-filter", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--props-every", action="store_true", default=True,
                    help="Dump proposals on EVERY frame (needed for superpixel snapping)")
    ap.add_argument("--no-props-every", dest="props_every", action="store_false")
    ap.add_argument("--no-pca", action="store_true",
                    help="Skip the PCA-3 feature image (smaller dump, no feature-CRF offline)")
    args = ap.parse_args()

    if args.davis_root:
        root = Path(args.davis_root)
        davis._DAVIS_ROOT = root
        davis._JPEG = root / "JPEGImages" / "480p"
        davis._ANN = root / "Annotations" / "480p"
    if not davis._JPEG.exists():
        sys.exit(f"DAVIS not found at {davis._JPEG} — pass --davis-root")

    clips = [c.strip() for c in args.clips.split(",") if c.strip()] or DAVIS_2016_VAL
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dino = DenseDINOv3()
    divider = CuVLERDivider(score_thresh=args.cuvler_score)
    conquer_backbone = None if args.no_conquer else load_backbone()

    meta = {"args": vars(args), "clips": []}
    t_all = time.time()
    for i, clip in enumerate(clips, 1):
        t0 = time.time()
        info = dump_clip(clip, dino, divider, conquer_backbone, args, out)
        info["seconds"] = round(time.time() - t0, 1)
        meta["clips"].append(info)
        print(f"[{i}/{len(clips)}] {clip}: {info}", flush=True)
        with open(out / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\nDone in {time.time() - t_all:.0f}s → {out}  ({size_mb:.0f} MB)")
    print(f"Copy it down with:\n  scp -P 58022 -r "
          f"$USER@<workstation>.in.tum.de:{out.resolve()} ./dumps/")


if __name__ == "__main__":
    main()
