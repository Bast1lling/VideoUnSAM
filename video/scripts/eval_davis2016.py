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
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
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


def _patch_labels(mask: np.ndarray, gh: int, gw: int) -> torch.Tensor:
    """Binary per-patch label [gh*gw] — patch is positive if >50% covered."""
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
    return (pooled.flatten() > 0.5).float().cuda()


def train_probe(feats_flat: torch.Tensor, y: torch.Tensor,
                steps: int = 100, lr: float = 0.01) -> nn.Module:
    """Test-time-adaptation instance probe: 1-layer linear classifier separating the
    frame-0 seed patches (positive) from the rest (negative) on frozen DINOv3 features.
    Label-free — supervision is the unsupervised seed mask."""
    probe = nn.Linear(feats_flat.shape[1], 1).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n_pos = y.sum().clamp_min(1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=((1 - y).sum().clamp_min(1) / n_pos).detach())
    feats_flat = feats_flat.detach()
    for _ in range(steps):
        opt.zero_grad()
        lossf(probe(feats_flat).squeeze(-1), y).backward()
        opt.step()
    probe.eval()
    return probe


@torch.no_grad()
def probe_score(probe: nn.Module, feats_norm: torch.Tensor) -> torch.Tensor:
    """Per-patch instance probability [N] in [0,1]."""
    flat = feats_norm.reshape(-1, feats_norm.shape[-1])
    return torch.sigmoid(probe(flat).squeeze(-1))


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


def click_centrality(mask: np.ndarray, cx: float, cy: float) -> float:
    """How deep the click sits in the mask, relative to the mask's deepest
    interior point (its distance-transform peak). 1.0 = click is the mask's
    center; near 0 = click is at its edge. The simulated click is the GT
    mask's distance-transform peak, so the true object scores high while a
    merged blob (whose interior peak lies elsewhere) scores low."""
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dmax = float(dt.max())
    return float(dt[int(cy), int(cx)]) / dmax if dmax > 0 else 0.0


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


_GT_PALETTE = None


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Indexed PNG (DAVIS palette, object id 1) for the official toolkit."""
    global _GT_PALETTE
    from PIL import Image
    if _GT_PALETTE is None:
        ann = _REPO / "datasets/davis/DAVIS/Annotations/480p/bear/00000.png"
        _GT_PALETTE = Image.open(ann).getpalette()
    img = Image.fromarray((mask > 0).astype(np.uint8), mode="P")
    img.putpalette(_GT_PALETTE)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ── Per-clip eval ─────────────────────────────────────────────────────────────

_PRED_COLOR = np.array([255, 45, 0])   # stark red-orange
_GT_COLOR = np.array([255, 45, 0])     # same hue so pred/GT panels read as one palette


def _label_bar(out: np.ndarray, label: str) -> None:
    w = max(240, 12 * len(label) + 20)
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


def _render_frame(frame: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                  label: str) -> np.ndarray:
    """Side-by-side: left panel = prediction (stark filled mask + outline),
    right panel = ground truth (same style, separate panel rather than an
    overlaid contour, so each mask is legible on its own)."""
    h, w = frame.shape[:2]

    def _panel(mask: np.ndarray, tag: str) -> np.ndarray:
        panel = frame.copy()
        m = mask.astype(bool)
        panel[m] = (0.35 * panel[m] + 0.65 * _PRED_COLOR).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 2)
        _label_bar(panel, tag)
        return panel

    left = _panel(pred, f"Prediction - {label}")
    right = _panel(gt, "Ground Truth")
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([left, sep, right], axis=1)


def _write_video(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    import shutil, subprocess
    if shutil.which("ffmpeg"):
        h264 = path.with_suffix(".h264.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", str(path), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", str(h264)], check=True, capture_output=True)
        path.unlink()
        h264.rename(path)


def eval_clip(clip: str, dino: DenseDINOv3, divider: CuVLERDivider,
              conquer_backbone, args) -> dict:
    n = davis.num_frames(clip)
    frame0 = davis.load_frame(clip, 0)
    H, W = frame0.shape[:2]
    # DAVIS 2016 protocol: one binary object = union of all 2017 instances
    # (7/20 val clips are split into parts in the 2017 annotations on disk).
    gt0 = (davis.load_mask(clip, 0) > 0).astype(np.uint8)
    if gt0.sum() == 0:
        return {}

    # Simulate user click: distance-transform peak of frame-0 GT
    gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    pts = sample_clicks(gt256, 1)
    cx256, cy256 = pts[0]
    click_xy = (cx256 * (W / 256.0), cy256 * (H / 256.0))
    if args.click_jitter > 0:
        # Robustness test: displace the simulated click by a fixed radius in a
        # per-clip deterministic direction (crc32 so reruns are reproducible).
        rng = np.random.RandomState(zlib.crc32(clip.encode()) & 0x7FFFFFFF)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        click_xy = (float(np.clip(click_xy[0] + args.click_jitter * np.cos(ang), 0, W - 1)),
                    float(np.clip(click_xy[1] + args.click_jitter * np.sin(ang), 0, H - 1)))

    # Frame 0 seed
    proposals = divider.predict(frame0)
    if conquer_backbone is not None:
        proposals = run_conquer(conquer_backbone, frame0, proposals)
    # Label-free seed pick, mirroring demo.py: largest proposal containing the
    # click; if that blob covers >40% of the frame it's a merged region, prefer
    # the smallest valid sub-proposal. GT is used for the click simulation and
    # scoring only — never to select among proposals.
    cx0, cy0 = click_xy
    containing0 = [m for m in proposals if m[int(cy0), int(cx0)] > 0]
    if containing0:
        area_floor = H * W * 0.005
        valid0 = [m for m in containing0 if m.sum() >= area_floor] or containing0
        if args.seed_pick == "smallest":
            seed = min(valid0, key=lambda m: m.sum())
        elif args.seed_pick == "hybrid":
            # Largest, unless it carries the merged-blob signature: too big to
            # be a single object AND the click sits peripherally in it (the
            # click marks the object's interior center, so in a merged blob it
            # lands off the blob's own center). Then arbitrate the remaining
            # candidates by click centrality instead.
            largest0 = max(containing0, key=lambda m: m.sum())
            a_frac = largest0.sum() / (H * W)
            if a_frac > 0.40:
                small0 = [m for m in containing0 if m.sum() >= area_floor]
                seed = min(small0, key=lambda m: m.sum()) if small0 else largest0
            elif a_frac > 0.15 and click_centrality(largest0, cx0, cy0) < 0.6:
                pool0 = [m for m in valid0 if m is not largest0] or valid0
                scored0 = [(click_centrality(m, cx0, cy0), m) for m in pool0]
                best_c = max(s for s, _ in scored0)
                seed = max([m for s, m in scored0 if s >= 0.8 * best_c],
                           key=lambda m: m.sum())
            else:
                seed = largest0
        elif args.seed_pick == "central":
            # Largest candidate in which the click is (near-)central: the
            # click carries positional evidence beyond mere containment.
            scored0 = [(click_centrality(m, cx0, cy0), m) for m in valid0]
            best_c = max(s for s, _ in scored0)
            near0 = [m for s, m in scored0 if s >= 0.8 * best_c]
            seed = max(near0, key=lambda m: m.sum())
        else:  # "largest" — shipped demo.py rule
            largest0 = max(containing0, key=lambda m: m.sum())
            if largest0.sum() > H * W * 0.40:
                small0 = [m for m in containing0 if m.sum() >= area_floor]
                seed = min(small0, key=lambda m: m.sum()) if small0 else largest0
            else:
                seed = largest0
    else:
        seed = max(proposals, key=lambda m: m.sum()) if proposals else None
    if seed is None:
        seed = np.zeros((H, W), dtype=np.uint8)
    if args.oracle_seed:
        # Bypass CuVLER+conquer+click arbitration entirely: seed directly from
        # the frame-0 ground truth, matching the protocol self-supervised
        # correspondence baselines (e.g. DINO nearest-neighbor matching) use --
        # isolates propagation quality alone, with the detector/arbitration
        # confound removed, for a directly comparable J&F.
        seed = gt0.copy()

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

    # Test-time adaptation: train a frame-0 instance probe on the seed mask.
    # Trained when either fusion (--probe) or reseed gating (--probe-reseed)
    # needs it; fused into the OT heat only under --probe.
    probe = None
    probe_bank_x: list[torch.Tensor] = []
    probe_bank_y: list[torch.Tensor] = []
    if (args.probe or args.probe_reseed) and cur_mask.sum() > 0:
        y0 = _patch_labels(cur_mask, gh_feat, gw_feat)
        x0 = feats_prev_norm.reshape(-1, feats_prev_norm.shape[-1])
        probe = train_probe(x0, y0)
        if args.probe_refresh:
            probe_bank_x, probe_bank_y = [x0], [y0]

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

    if args.save_dir:
        _save_mask_png(display0, Path(args.save_dir) / clip / "00000.png")

    render_frames = []
    if args.render_dir:
        render_frames.append(_render_frame(frame0, display0, gt0, f"f0 J={j0:.2f}"))

    for fidx in range(1, n):
        frame = davis.load_frame(clip, fidx)
        gt = (davis.load_mask(clip, fidx) > 0).astype(np.uint8)

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

        # Fuse the appearance probe into the OT heat (patch level). Everything
        # downstream — upsample, threshold, and the patch=heat carry-forward — then
        # uses the fused heat, so probe corrections are fed back into the chain.
        if probe is not None and args.probe:
            heat = heat / (heat.max() + 1e-8)
            heat_ot = heat.reshape(-1)  # pre-fusion OT heat, kept for diagnostics
            score = probe_score(probe, feats_cur_norm)  # [N_b] in [0,1]
            heat = (1.0 - args.probe_weight) * heat + args.probe_weight * score

            # Staleness-signal diagnostic: do label-free per-frame scalars
            # (area-normalized probe peakiness; probe-OT agreement) predict
            # how correct the probe still is? Quality is measured as
            # probe-vs-GT soft IoU — GT used for diagnosis only.
            if args.probe_gate_log and gt.sum() > 0:
                k = max(int(_patch_labels(seed, gh_cur, gw_cur).sum()), 1)
                sflat = score.reshape(-1)
                topk = float(sflat.topk(k).values.mean())
                inter_a = (sflat * heat_ot).sum()
                agree = float(inter_a / (sflat.sum() + heat_ot.sum() - inter_a + 1e-8))
                y_gt = _patch_labels(gt, gh_cur, gw_cur)
                inter = (score * y_gt).sum()
                probe_gt = float(inter / (score.sum() + y_gt.sum() - inter + 1e-8))
                with open(args.probe_gate_log, "a") as fh:
                    fh.write(f"{clip},{fidx},{topk:.4f},{agree:.4f},"
                             f"{probe_gt:.4f}\n")

        # Bilinear upsample first (both paths need the raw heatmap)
        heat_up = F.interpolate(
            heat.reshape(1, 1, gh_cur, gw_cur), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        soft_up = heat_up / (heat_up.max() + 1e-8)

        ot_mask = (soft_up > args.thresh).astype(np.uint8)
        if args.cc_filter:
            ot_mask = largest_cc_near(ot_mask, prev_centroid)

        cur_mask = ot_mask
        patch = heat
        prev_centroid = mask_centroid(cur_mask) or prev_centroid

        # Periodic reseed
        reseed_info = None
        if args.reseed_interval > 0 and fidx % args.reseed_interval == 0:
            props = divider.predict(frame)
            if conquer_backbone is not None:
                props = run_conquer(conquer_backbone, frame, props)
            accept = False
            candidate = None
            if args.probe_reseed and probe is not None:
                # Identity-arbitrated reseed: score every candidate (and the
                # incumbent mask) by soft-IoU against the instance probe — the
                # click's frame-0-anchored appearance evidence. The candidate
                # wins only if it beats the incumbent under that evidence.
                # Unlike the IoU-vs-current-mask gate this cannot be satisfied
                # by coherence with a drifted track, and it can re-acquire the
                # object after full drift (no overlap with the incumbent needed).
                pscore = probe_score(probe, feats_cur_norm)  # [N] in [0,1]

                def probe_soft_iou(m: np.ndarray) -> float:
                    y = _patch_labels(m, gh_cur, gw_cur)
                    inter = (pscore * y).sum()
                    return float(inter / (pscore.sum() + y.sum() - inter + 1e-8))

                incumbent = probe_soft_iou(ot_mask) if ot_mask.sum() > 0 else 0.0
                scored = [(probe_soft_iou(m), m) for m in props if m.sum() > 0]
                if scored:
                    best_s, candidate = max(scored, key=lambda t: t[0])
                    accept = best_s > incumbent
                    cand_gt = iou(gt, candidate) if gt.sum() > 0 else float("nan")
                    reseed_info = (len(props), best_s, cand_gt, accept)
            else:
                candidate = pick_proposal(props, ot_mask)
                if candidate is not None:
                    gate = iou(ot_mask, candidate)
                    cand_gt = iou(gt, candidate) if gt.sum() > 0 else float("nan")
                    accept = gate >= args.reseed_thresh
                    reseed_info = (len(props), gate, cand_gt, accept)
            if candidate is not None and accept:
                cur_mask = candidate
                patch = mask_to_patch(cur_mask, gh=gh_feat, gw=gw_feat)
                prev_centroid = mask_centroid(cur_mask) or prev_centroid
                # Probe refresh: the accepted reseed is the only verified mid-clip
                # evidence of current appearance — retrain the probe on frame 0
                # plus every accepted reseed so far. Frame 0 stays in the bank so
                # one bad reseed cannot rewrite the probe's identity.
                if probe is not None and args.probe_refresh and cur_mask.sum() > 0:
                    probe_bank_x.append(feats_cur_norm.reshape(-1, feats_cur_norm.shape[-1]))
                    probe_bank_y.append(_patch_labels(cur_mask, gh_cur, gw_cur))
                    probe = train_probe(torch.cat(probe_bank_x), torch.cat(probe_bank_y))

        # Guided-filter boundary polish (demo.py semantics): snap the blocky
        # binary mask to image edges for DISPLAY ONLY. Propagation (patch),
        # reseed gating, and CRF confidence all stay on the plain-bilinear
        # path — feeding the edge-snapped mask back compounds erosion.
        if args.guided:
            guide = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            mask_guided = cv2.ximgproc.guidedFilter(
                guide=guide, src=cur_mask.astype(np.float32), radius=8, eps=1e-3)
            display_mask = (mask_guided > 0.5).astype(np.uint8)
        else:
            display_mask = cur_mask

        apply_crf = args.refine and (args.crf_every or crf_confidence(soft_up) >= args.crf_conf)
        if apply_crf:
            dino_feats_arg = None if args.no_dino_crf else feats_cur_raw.cpu()
            display = crf_refine(frame, soft_up, dino_feats=dino_feats_arg,
                                 bilateral_sxy=args.crf_sxy, bilateral_srgb=args.crf_srgb,
                                 bilateral_compat=args.crf_compat)
        else:
            display = display_mask

        if args.save_dir:
            _save_mask_png(display, Path(args.save_dir) / clip / f"{fidx:05d}.png")

        # store soft_up for potential use in heat_up reference downstream
        heat_up = soft_up  # alias for readability

        if gt.sum() > 0:
            js.append(j_score(display, gt))
            fs.append(f_score(display, gt))

        if args.render_dir:
            jj = j_score(display, gt) if gt.sum() > 0 else float("nan")
            render_frames.append(_render_frame(frame, display, gt, f"f{fidx} J={jj:.2f}"))

        # Per-frame diagnostic log: track J, OT-mask J (pre-refine), and every
        # reseed event's gate value vs the candidate's true quality. cand_gt_iou
        # uses GT for diagnosis only — it never influences the pipeline.
        if args.reseed_log:
            j_frame = j_score(display, gt) if gt.sum() > 0 else float("nan")
            j_ot = j_score(cur_mask, gt) if gt.sum() > 0 else float("nan")
            area = cur_mask.sum() / (H * W)
            with open(args.reseed_log, "a") as fh:
                if reseed_info is not None:
                    n_p, gate, cand_gt, acc = reseed_info
                    fh.write(f"{clip},{fidx},{j_frame:.4f},{j_ot:.4f},{area:.4f},"
                             f"1,{n_p},{gate:.4f},{cand_gt:.4f},{int(acc)}\n")
                else:
                    fh.write(f"{clip},{fidx},{j_frame:.4f},{j_ot:.4f},{area:.4f},"
                             f"0,,,,\n")

        feats_prev_norm = feats_cur_norm
        frame_prev = frame
        fd_a = fd_b

    if args.render_dir and render_frames:
        _write_video(render_frames, Path(args.render_dir) / f"{clip}.mp4")

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
    ap.add_argument("--oracle-seed", action="store_true",
                    help="Seed frame 0 directly from ground truth instead of "
                         "CuVLER+conquer+click arbitration. Isolates propagation "
                         "quality alone, for comparison against self-supervised "
                         "correspondence baselines (e.g. DINO nearest-neighbor "
                         "matching) that are also initialized from the GT mask.")
    ap.add_argument("--seed-pick",
                    choices=["largest", "smallest", "central", "hybrid"],
                    default="hybrid",
                    help="Label-free frame-0 arbitration among click-containing "
                         "proposals: hybrid (shipped — largest unless the "
                         "merged-blob signature fires, then click centrality; "
                         "matches demo.py and the paper), largest (old default), "
                         "smallest above a size floor, or central (largest "
                         "candidate whose interior distance-transform peak the "
                         "click sits near)")
    ap.add_argument("--click-jitter", type=float, default=0.0,
                    help="Perturb the simulated frame-0 click by this many "
                         "pixels in a per-clip deterministic random direction "
                         "(robustness test for the oracle-click critique).")
    ap.add_argument("--probe", action="store_true",
                    help="Instance probe (test-time adaptation): train a 1-layer linear "
                         "probe on the frame-0 seed mask and fuse its per-patch score into "
                         "the OT heat each frame. Label-free; helps identity-switch and "
                         "background-confusion clips.")
    ap.add_argument("--probe-weight", type=float, default=0.5,
                    help="Fusion weight: heat = (1-w)*OT + w*probe (default 0.5).")
    ap.add_argument("--probe-reseed", action="store_true",
                    help="Identity-arbitrated reseed: pick and accept reseed candidates "
                         "by soft-IoU with the instance probe's per-patch score (the "
                         "click's appearance evidence) instead of IoU with the current "
                         "OT mask (coherence). Candidate must beat the incumbent mask "
                         "under the probe; no extra threshold. Trains the frame-0 probe "
                         "even without --probe fusion.")
    ap.add_argument("--probe-refresh", action="store_true",
                    help="Retrain the instance probe on every accepted reseed (frame-0 "
                         "patches + all accepted-reseed frames' patches). Attacks the "
                         "frame-0 appearance-staleness regressions (e.g. motocross-jump) "
                         "without any gating heuristic. Requires --probe.")
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
    ap.add_argument("--render-dir", default="",
                    help="If set, write a per-clip video to this directory: side-by-side "
                         "prediction panel and ground-truth panel (per-frame J in the "
                         "prediction panel's label).")
    ap.add_argument("--save-dir", default="",
                    help="Export per-frame indexed PNGs (DAVIS palette, object "
                         "id 1) of the final display mask for the official "
                         "davis2017-evaluation toolkit (semi-supervised task).")
    ap.add_argument("--reseed-log", default="",
                    help="Append per-frame diagnostic CSV rows to this path: "
                         "clip,fidx,j_display,j_ot,area_frac,reseed_attempted,"
                         "n_props,gate_iou,cand_gt_iou,accepted. cand_gt_iou is "
                         "GT-based diagnosis only, never used by the pipeline.")
    ap.add_argument("--probe-gate-log", default="",
                    help="Append per-frame probe-staleness diagnostic CSV rows: "
                         "clip,fidx,topk_mean,probe_ot_agree,probe_gt_softiou "
                         "(topk k = frame-0 seed patch count). probe_gt_softiou "
                         "is GT-based diagnosis only.")
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
