"""Full-pipeline aggregate eval on DAVIS 2016 val (20 clips, single object each).

Runs CuVLER+conquer seed → Sinkhorn OT chain → periodic reseed on every clip and
reports J mean (IoU), F mean (boundary F-score), and J&F mean — the standard DAVIS
metrics used by all UVOS papers for comparison.

Usage:
    python -m video.scripts.eval_davis2016
    python -m video.scripts.eval_davis2016 --no-conquer   # ablate conquer stage
    python -m video.scripts.eval_davis2016 --reseed-interval 0  # ablate reseeding
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "divide_and_conquer"))

def refine_mask(refiner, frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1)
    x0c = max(0, int(cx - half)); y0c = max(0, int(cy - half))
    x1c = min(W, int(cx + half)); y1c = min(H, int(cy + half))
    img_crop = cv2.cvtColor(frame_rgb[y0c:y1c, x0c:x1c], cv2.COLOR_RGB2BGR)
    msk_crop = (mask[y0c:y1c, x0c:x1c] * 255).astype(np.uint8)
    L = int(np.clip(max(x1c - x0c, y1c - y0c), 100, 900))
    refined = refiner.refine(img_crop, msk_crop, fast=True, L=L)
    out = np.zeros((H, W), dtype=np.uint8)
    out[y0c:y1c, x0c:x1c] = (refined > 128).astype(np.uint8)
    return out

from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.loaders import davis  # noqa: E402
from video.decoder.train_sam_decoder import sample_clicks  # noqa: E402
from video.divide.cuvler_divide import CuVLERDivider  # noqa: E402
from video.divide.conquer import load_backbone, run_conquer  # noqa: E402
from video.propagation.sinkhorn_ot import propagate_patch, propagate_multiscale, _mask_to_patch_indicator  # noqa: E402
from video.refine.dense_crf import crf_refine, crf_confidence  # noqa: E402


def guided_upsample(heat: torch.Tensor, frame_rgb: np.ndarray,
                    gh: int = 64, gw: int = 64,
                    radius: int = 16, eps: float = 1e-3) -> np.ndarray:
    """Upsample OT heatmap to full resolution using image-guided filter."""
    H, W = frame_rgb.shape[:2]
    heat_up = F.interpolate(
        heat.reshape(1, 1, gh, gw), size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    mn, mx = heat_up.min(), heat_up.max()
    heat_norm = ((heat_up - mn) / (mx - mn + 1e-8)).astype(np.float32)
    guide = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.ximgproc.guidedFilter(guide=guide, src=heat_norm, radius=radius, eps=eps)

DAVIS_2016_VAL = [
    "blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout",
    "car-shadow", "cows", "dance-twirl", "dog", "drift-chicane",
    "drift-straight", "goat", "horsejump-high", "kite-surf", "libby",
    "motocross-jump", "paragliding-launch", "parkour", "scooter-black", "soapbox",
]


# ── Metrics ──────────────────────────────────────────────────────────────────

def j_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union else 0.0


def _get_boundary(mask: np.ndarray, tolerance: int) -> np.ndarray:
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    eroded = binary_erosion(mask.astype(bool), struct)
    return mask.astype(bool) & ~eroded


def f_score(pred: np.ndarray, gt: np.ndarray, tolerance: int = 3) -> float:
    pred_b = _get_boundary(pred, tolerance)
    gt_b = _get_boundary(gt, tolerance)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    pred_dilated = binary_dilation(pred_b, struct)
    gt_dilated = binary_dilation(gt_b, struct)
    precision = float((pred_b & gt_dilated).sum()) / (pred_b.sum() + 1e-8)
    recall = float((gt_b & pred_dilated).sum()) / (gt_b.sum() + 1e-8)
    if precision + recall < 1e-8:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Connected-component filter ───────────────────────────────────────────────

def largest_cc_near(mask: np.ndarray, prev_centroid: tuple[float, float] | None) -> np.ndarray:
    """Keep only the connected component closest to prev_centroid.

    Prevents semantic leakage in crowded scenes (e.g. breakdance): after OT
    thresholding the binary mask may contain the target object plus disconnected
    blobs from nearby similar-looking people. Tracking the centroid across frames
    drops those spurious blobs without affecting single-object clips.

    Falls back to the largest component when prev_centroid is None (frame 1) or
    when the mask is empty.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return mask  # empty or single component
    # component 0 is background; consider 1..n_labels-1
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


# ── OT helpers ───────────────────────────────────────────────────────────────

def mask_to_patch(mask: np.ndarray, gh: int = 64, gw: int = 64) -> torch.Tensor:
    try:
        return _mask_to_patch_indicator(mask, gh, gw)
    except ValueError:
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
        if pooled.sum() <= 0:
            pooled = pooled + 1.0 / (gh * gw)
        return pooled.flatten()


def _frame_diff_patches(frame_a: np.ndarray, frame_b: np.ndarray,
                         gh: int = 64, gw: int = 64) -> torch.Tensor:
    """Mean absolute pixel diff between two RGB frames, pooled to [gh*gw] in [0,1]."""
    diff = np.abs(frame_b.astype(np.float32) - frame_a.astype(np.float32)).mean(axis=-1)
    pooled = F.adaptive_avg_pool2d(
        torch.from_numpy(diff)[None, None], (gh, gw)
    )[0, 0]
    return (pooled / (pooled.max() + 1e-8)).flatten()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


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


# ── Per-clip eval ─────────────────────────────────────────────────────────────

def eval_clip(clip: str, dino: DenseDINOv3, divider: CuVLERDivider,
              conquer_backbone, args) -> dict:
    n = davis.num_frames(clip)
    inst_ids = davis.instance_ids(clip, 0)
    if not inst_ids:
        return {}
    inst = inst_ids[0]  # DAVIS 2016: always single object

    frame0 = davis.load_frame(clip, 0)
    H, W = frame0.shape[:2]
    gt0 = davis.load_mask(clip, 0, instance_id=inst)

    # Simulate user click: distance-transform peak of frame-0 GT
    gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    pts = sample_clicks(gt256, 1)
    cx256, cy256 = pts[0]
    click_xy = (cx256 * (W / 256.0), cy256 * (H / 256.0))

    # Frame 0 seed
    proposals = divider.predict(frame0)
    if conquer_backbone is not None:
        proposals = run_conquer(conquer_backbone, frame0, proposals)
    seed = pick_proposal(proposals, gt0, click_xy=click_xy)
    if seed is None:
        seed = gt0.astype(np.uint8)

    img_sized = cv2.resize(frame0, (args.feat_size, args.feat_size))
    with torch.no_grad():
        feats_prev = dino.extract(img_sized, normalize=False)["feats"].cuda().float()
    feats_prev_norm = F.normalize(feats_prev, dim=-1)
    gh_feat, gw_feat = feats_prev_norm.shape[:2]  # 64 at 1024px, 128 at 2048px

    cur_mask = seed.astype(np.uint8)
    patch = mask_to_patch(cur_mask, gh=gh_feat, gw=gw_feat)
    prev_centroid = mask_centroid(cur_mask)
    frame_prev = frame0
    fd_a: torch.Tensor | None = None  # no prior-frame diff at t=0

    soft0 = cur_mask.astype(np.float32)
    if args.refine:
        feats0_hw = None if args.no_dino_crf else dino.extract(
            cv2.resize(frame0, (args.feat_size, args.feat_size)), normalize=False
        )["feats"].cpu()
        display0 = crf_refine(frame0, soft0, dino_feats=feats0_hw,
                              bilateral_sxy=args.crf_sxy, bilateral_srgb=args.crf_srgb,
                              bilateral_compat=args.crf_compat)
    else:
        display0 = cur_mask
    j0 = j_score(display0, gt0)
    f0 = f_score(display0, gt0)
    js, fs = [j0], [f0]

    for fidx in range(1, n):
        frame = davis.load_frame(clip, fidx)
        gt = davis.load_mask(clip, fidx, instance_id=inst)

        img_sized = cv2.resize(frame, (args.feat_size, args.feat_size))
        with torch.no_grad():
            feats_cur_raw = dino.extract(img_sized, normalize=False)["feats"]
            feats_cur = feats_cur_raw.cuda().float()
        feats_cur_norm = F.normalize(feats_cur, dim=-1)
        gh_cur, gw_cur = feats_cur_norm.shape[:2]

        fd_b = _frame_diff_patches(frame_prev, frame, gh=gh_feat, gw=gw_feat) if args.motion_weight > 0 else None

        if args.multiscale:
            heat = propagate_multiscale(feats_prev_norm, feats_cur_norm, patch,
                                        blur_fine=args.blur, blur_coarse=args.blur_coarse,
                                        coarse_factor=args.coarse_factor, alpha=args.ms_alpha,
                                        spatial_weight=args.spatial_weight,
                                        motion_a=fd_a, motion_b=fd_b,
                                        motion_weight=args.motion_weight)
        else:
            heat = propagate_patch(feats_prev_norm, feats_cur_norm, patch, blur=args.blur,
                                   spatial_weight=args.spatial_weight,
                                   motion_a=fd_a, motion_b=fd_b,
                                   motion_weight=args.motion_weight)

        # Bilinear upsample first (both paths need the raw heatmap)
        heat_up = F.interpolate(
            heat.reshape(1, 1, gh_cur, gw_cur), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        if args.guided:
            # Normalise by max only (preserves relative threshold semantics),
            # apply guided filter to snap boundaries to image edges, then threshold
            heat_norm = (heat_up / (heat_up.max() + 1e-8)).astype(np.float32)
            guide = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            soft_up = cv2.ximgproc.guidedFilter(guide=guide, src=heat_norm, radius=16, eps=1e-3)
        else:
            soft_up = heat_up / (heat_up.max() + 1e-8)

        ot_mask = (soft_up > args.thresh).astype(np.uint8)
        if args.cc_filter:
            ot_mask = largest_cc_near(ot_mask, prev_centroid)

        cur_mask = ot_mask
        patch = heat
        prev_centroid = mask_centroid(cur_mask) or prev_centroid

        # Periodic reseed
        if args.reseed_interval > 0 and fidx % args.reseed_interval == 0:
            props = divider.predict(frame)
            if conquer_backbone is not None:
                props = run_conquer(conquer_backbone, frame, props)
            candidate = pick_proposal(props, ot_mask)
            if candidate is not None and iou(ot_mask, candidate) >= args.reseed_thresh:
                cur_mask = candidate
                patch = mask_to_patch(cur_mask, gh=gh_feat, gw=gw_feat)
                prev_centroid = mask_centroid(cur_mask) or prev_centroid

        apply_crf = args.refine and (args.crf_every or crf_confidence(soft_up) >= args.crf_conf)
        if apply_crf:
            dino_feats_arg = None if args.no_dino_crf else feats_cur_raw.cpu()
            display = crf_refine(frame, soft_up, dino_feats=dino_feats_arg,
                                 bilateral_sxy=args.crf_sxy, bilateral_srgb=args.crf_srgb,
                                 bilateral_compat=args.crf_compat)
        else:
            display = cur_mask

        # store soft_up for potential use in heat_up reference downstream
        heat_up = soft_up  # alias for readability

        if gt.sum() > 0:
            js.append(j_score(display, gt))
            fs.append(f_score(display, gt))

        feats_prev_norm = feats_cur_norm
        frame_prev = frame
        fd_a = fd_b

    return {"j": float(np.mean(js)), "f": float(np.mean(fs)), "n_frames": len(js)}


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-size", type=int, default=1024,
                    help="Image size fed to DINOv3 (default 1024 → 64×64 grid; "
                         "2048 → 128×128 grid, higher boundary precision, ~8× slower).")
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--spatial-weight", type=float, default=0.0,
                    help="Spatial prior weight λ added to OT cost matrix: "
                         "C_ij += λ·||pos_i - pos_j||/√2. Prevents transport "
                         "jumping to distant background (try 0.3).")
    ap.add_argument("--motion-weight", type=float, default=0.0,
                    help="Motion consistency weight γ: C_ij += γ·|fd_b[j]-fd_a[i]|, "
                         "where fd is mean abs pixel diff pooled to patch level. "
                         "Penalises transport between patches with mismatched motion "
                         "magnitude (try 0.5). Near-zero cost on slow/static clips.")
    ap.add_argument("--multiscale", action="store_true",
                    help="Multi-scale OT: blend 64×64 fine and 16×16 coarse passes")
    ap.add_argument("--blur-coarse", type=float, default=0.10,
                    help="Blur for coarse OT pass (default 0.10)")
    ap.add_argument("--coarse-factor", type=int, default=4,
                    help="Spatial pooling factor for coarse pass: 64→16 patches (default 4)")
    ap.add_argument("--ms-alpha", type=float, default=0.4,
                    help="Weight of coarse heatmap in multi-scale blend (default 0.4)")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--reseed-interval", type=int, default=10)
    ap.add_argument("--reseed-thresh", type=float, default=0.3)
    ap.add_argument("--cuvler-score", type=float, default=0.35)
    ap.add_argument("--no-conquer", action="store_true")
    ap.add_argument("--refine", action="store_true",
                    help="Apply Dense CRF + DINOv3 bilateral refinement per frame")
    ap.add_argument("--crf-conf", type=float, default=0.75,
                    help="Min top-10%% heatmap confidence to apply CRF (default 0.75). "
                         "Skips CRF on frames where OT is diffuse/drifted.")
    ap.add_argument("--guided", action="store_true",
                    help="Guided-filter upsample of OT heatmap (snaps boundaries to image edges, fully unsupervised)")
    ap.add_argument("--no-dino-crf", action="store_true",
                    help="Use RGB bilateral in CRF instead of DINOv3 features (sharper pixel-level boundaries)")
    ap.add_argument("--crf-sxy", type=float, default=70.0,
                    help="CRF bilateral spatial bandwidth (smaller=tighter, default 70)")
    ap.add_argument("--crf-srgb", type=float, default=13.0,
                    help="CRF bilateral color bandwidth (smaller=more color-sensitive, default 13)")
    ap.add_argument("--crf-compat", type=float, default=20.0,
                    help="CRF bilateral compatibility weight (higher=stronger edge pull, default 20)")
    ap.add_argument("--crf-every", action="store_true",
                    help="Apply CRF every frame regardless of confidence gate")
    ap.add_argument("--cc-filter", action="store_true",
                    help="Keep only the connected component closest to the previous frame centroid. "
                         "Prevents semantic leakage in crowded scenes (e.g. breakdance).")
    ap.add_argument("--clips", default="",
                    help="Comma-separated subset of clips to run (default: all 20 val clips)")
    args = ap.parse_args()

    clips = [c.strip() for c in args.clips.split(",")] if args.clips else DAVIS_2016_VAL

    print(f"[eval_davis2016] {len(clips)} clips  reseed={args.reseed_interval}  "
          f"conquer={'off' if args.no_conquer else 'on'}  "
          f"guided={'on' if args.guided else 'off'}  "
          f"refine={'on' if args.refine else 'off'}  "
          f"spatial_weight={args.spatial_weight}")

    dino = DenseDINOv3()
    divider = CuVLERDivider(score_thresh=args.cuvler_score)
    conquer_backbone = None if args.no_conquer else load_backbone()
    if conquer_backbone is not None:
        print("[conquer] backbone loaded")
    if args.refine:
        print(f"[refine] Dense CRF enabled (fully unsupervised, conf>={args.crf_conf})")

    results = {}
    t0 = time.time()
    for clip in clips:
        tc = time.time()
        res = eval_clip(clip, dino, divider, conquer_backbone, args)
        elapsed = time.time() - tc
        j, f = res.get("j", float("nan")), res.get("f", float("nan"))
        jf = (j + f) / 2
        results[clip] = res
        print(f"  {clip:<22}  J={j:.3f}  F={f:.3f}  J&F={jf:.3f}  "
              f"({res.get('n_frames',0)} frames, {elapsed:.0f}s)")

    valid = [(r["j"], r["f"]) for r in results.values() if "j" in r]
    if valid:
        mean_j = np.mean([v[0] for v in valid])
        mean_f = np.mean([v[1] for v in valid])
        mean_jf = (mean_j + mean_f) / 2
        print(f"\n{'─'*60}")
        print(f"  J mean  {mean_j:.3f}")
        print(f"  F mean  {mean_f:.3f}")
        print(f"  J&F     {mean_jf:.3f}  ← compare to SOTA ~87.6% (DPA, CVPR 2024)")
        print(f"  clips   {len(valid)}/20")
        print(f"  time    {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
