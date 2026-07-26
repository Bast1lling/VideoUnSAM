"""Same feasibility test as eval_cycle_quality_signal.py, but seeded from a REAL
CuVLER+conquer pseudo-mask instead of GT -- the production-realistic case.
Also tests whether CuVLER's own detector confidence (score) fixes the
"self-consistent wrong track" false positive that cycle-agreement alone
falls for (see cycle-agreement-quality-filter-feasibility memory:
drift-chicane, seed_iou=0.00, agree=0.97 -- confidently wrong).

eval_cycle_quality_signal.py answered "does cycle-agreement predict quality
when propagation starts from a perfect mask" (yes, rho=0.858 on 122 samples).
The first (20-clip) run of THIS script answered "does it still work when the
seed itself is noisy" -- rho collapsed to 0.44 and quartiles went
non-monotonic, because a wrong-but-stable seed round-trips just as cleanly
as a right one. Adding CuVLER's own detector score as a second, independent
signal and AND-gating on both recovered most of the lift and specifically
excluded drift-chicane; a held-out 2-fold check (thresholds fit on one half
of clips, applied to the other) confirmed the lift isn't just curve-fitting
on the eval set (+0.187 mean out-of-sample delta vs +0.199 in-sample).

This version scales that same test to the full 64-clip leak-free split (note:
`video/loaders/davis.py` already points at the full DAVIS 2017 480p release --
the "20 clips" used earlier was DAVIS-2016-val-sized subset of that same pool,
not a different, smaller dataset -- so "64 clips" already includes native
DAVIS 2017 sequences, no separate loader needed), and adds:

  1. A keep-rate / quality sweep (not just one threshold pair): rank samples
     by score, sweep the kept-fraction from 5% to 100%, and plot
     kept-fraction vs mean-quality-of-kept for the joint score vs each signal
     alone. This is the actual design curve retraining needs (e.g. "40% at
     0.70" vs "15% at 0.85" imply different training regimes), not a single
     operating point.
  2. Per-clip difficulty tagging (fast / thin / crowded / none), computed
     directly from GT motion/area/instance-count -- no hand-curated labels --
     so keep-rate and quality can be broken down by tag. If the gate
     systematically rejects e.g. all "fast" clips, the retrained model would
     silently never see hard cases, which needs to be known going in, not
     discovered after training.

At 64-clip scale, agreement+score both weakened substantially and the
held-out AND-gate validation went unstable (one fold direction actively
NEGATIVE, see cycle-agreement-quality-filter-feasibility memory). Root cause:
agreement and score are both computed INSIDE the same DINOv3 feature space
the OT propagation itself uses -- new wrong-track cases (sheep, surf,
snowboard) showed HIGH CuVLER score on confidently-wrong seeds, directly
contradicting the "low score flags wrong tracks" hypothesis. A wrong track
that's self-consistent in feature space has no structural reason to also
disagree with itself on either signal.

  3. RAFT optical flow (video/propagation/raft_flow.py) as a THIRD,
     genuinely independent signal -- pixel motion, not DINOv3 features.
     flow_agreement = IoU(flow-warped seed mask, OT-propagated mask): does
     the OT propagation agree with where real pixel motion says the object
     went? RAFT's weights are optical-flow-supervised on synthetic data
     (FlyingChairs/FlyingThings3D), not segmentation-supervised, so this
     doesn't reintroduce segmentation supervision into the label-free
     pipeline. Direct single-jump flow (frame_a -> frame_b), matching the
     project's established finding that direct beats chained for OT
     propagation ([[propagation-ot-vs-alternatives]]).

Both agreement and the round-trip target are measured against the PSEUDO
mask, never GT -- GT is used only to compute true_quality, seed_iou, and
difficulty tags for validating the experiment, exactly like eval_davis2016.py's
convention (clicks simulated from GT to drive the harness, never used as
supervision). Detector score is read straight off CuVLER/conquer's own
output -- no GT involved at all.

    python -m video.scripts.eval_cycle_quality_signal_real_seed \\
        --offsets 5,10,20 --limit-clips 0 --plot-path sweep.png --sweep-csv sweep.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "divide_and_conquer"))
from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis
from video.decoder.train_sam_decoder import sample_clicks
from video.divide.cuvler_divide import CuVLERDivider
from video.divide.conquer import load_backbone, run_conquer_scored
from video.propagation.sinkhorn_ot import propagate
from video.propagation.raft_flow import load_raft, compute_flow, warp_mask_backward

_SPLIT = Path(__file__).resolve().parents[1] / "decoder" / "davis_split.json"

_TAGS = ("fast", "thin", "crowded")
_WRONG_TRACK_CASES = ("drift-chicane", "sheep", "surf", "snowboard")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def pick_proposal_scored(masks: list[np.ndarray], scores: list[float], ref: np.ndarray,
                         click_xy: tuple | None = None) -> tuple[np.ndarray, float] | tuple[None, None]:
    if not masks:
        return None, None
    idx = range(len(masks))
    if click_xy is not None:
        cx, cy = click_xy
        containing = [i for i in idx if masks[i][int(cy), int(cx)] > 0]
        if containing:
            idx = containing
    best = max(idx, key=lambda i: iou(masks[i], ref))
    return masks[best], scores[best]


def clip_motion_area_instances(clip: str, inst: int, max_samples: int = 10) -> dict:
    """GT-only difficulty signal, computed once per clip (no CuVLER/OT involved).

    area_frac: mean fraction of frame covered by the primary instance.
    motion: mean per-sampled-step centroid displacement, normalised by frame
      diagonal, so it's comparable across resolutions.
    n_instances: object count in frame 0 -- DAVIS 2017 annotations are
      multi-instance even though this eval (like eval_davis2016.py) only
      tracks the first/clicked one, so this flags scenes that are crowded
      around the tracked object even if we never touch the other instances.
    """
    n = davis.num_frames(clip)
    step = max(1, n // max_samples)
    idxs = list(range(0, n, step))
    areas: list[float] = []
    centroids: list[tuple[float, float]] = []
    H = W = None
    for fi in idxs:
        m = davis.load_mask(clip, fi, instance_id=inst)
        if H is None:
            H, W = m.shape
        if m.sum() == 0:
            continue
        areas.append(float(m.sum()) / m.size)
        ys, xs = np.nonzero(m)
        centroids.append((float(ys.mean()), float(xs.mean())))
    area_frac = float(np.mean(areas)) if areas else 0.0
    diag = float(np.hypot(H, W)) if H else 1.0
    disp = [np.hypot(centroids[i][0] - centroids[i - 1][0], centroids[i][1] - centroids[i - 1][1])
            for i in range(1, len(centroids))]
    motion = float(np.mean(disp) / diag) if disp else 0.0
    n_instances = len(davis.instance_ids(clip, 0))
    return {"area_frac": area_frac, "motion": motion, "n_instances": n_instances}


def assign_difficulty_tags(clip_stats: dict[str, dict]) -> dict[str, tuple[str, ...]]:
    """Data-driven tercile thresholds over whatever clip set is actually being
    evaluated, rather than fixed magic numbers -- adapts to however many clips
    are in play and keeps class sizes roughly balanced."""
    area_fracs = [d["area_frac"] for d in clip_stats.values()]
    motions = [d["motion"] for d in clip_stats.values()]
    thin_thresh = float(np.percentile(area_fracs, 33))
    fast_thresh = float(np.percentile(motions, 67))
    tags: dict[str, tuple[str, ...]] = {}
    for clip, d in clip_stats.items():
        t = []
        if d["area_frac"] <= thin_thresh:
            t.append("thin")
        if d["motion"] >= fast_thresh:
            t.append("fast")
        if d["n_instances"] >= 2:
            t.append("crowded")
        tags[clip] = tuple(t) if t else ("none",)
    return tags


def sweep_curve(rank_score: np.ndarray, qualities: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """For each kept-fraction f, rank samples by rank_score (desc) and take the
    top f*N; return mean quality of that kept set. Monotonically non-increasing
    if rank_score is any good (best samples get included first)."""
    order = np.argsort(-rank_score)
    sorted_q = qualities[order]
    n = len(qualities)
    return np.array([sorted_q[:max(1, int(round(f * n)))].mean() for f in fractions])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="5,10,20")
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit-clips", type=int, default=0, help="0 = no limit, use the full split")
    ap.add_argument("--split", choices=["clean", "all"], default="clean")
    ap.add_argument("--cuvler-score", type=float, default=0.35)
    ap.add_argument("--no-conquer", action="store_true")
    ap.add_argument("--no-flow", action="store_true", help="skip RAFT flow-consistency signal")
    ap.add_argument("--flow-device", default="cuda")
    ap.add_argument("--sweep-csv", default=None, help="path to write the kept-fraction/quality sweep as CSV")
    ap.add_argument("--plot-path", default=None, help="path to write the sweep plot as PNG")
    args = ap.parse_args()
    offsets = [int(x) for x in args.offsets.split(",")]

    clips = json.load(open(_SPLIT))["clean"] if args.split == "clean" else davis.list_clips()
    if args.limit_clips:
        clips = clips[:args.limit_clips]

    extractor = DenseDINOv3()
    divider = CuVLERDivider(score_thresh=args.cuvler_score)
    conquer_backbone = None if args.no_conquer else load_backbone()
    raft_model = raft_transforms = None
    if not args.no_flow:
        raft_model, raft_transforms = load_raft(device=args.flow_device)
        print("[flow] RAFT loaded")
    print(f"[eval] {len(clips)} clips, offsets={offsets}, blur={args.blur}, "
          f"conquer={'off' if args.no_conquer else 'on'}, flow={'off' if args.no_flow else 'on'}")

    agreements: list[float] = []
    qualities: list[float] = []
    seed_scores: list[float] = []
    flow_agreements: list[float] = []
    seed_ious: list[float] = []
    sample_clips: list[str] = []
    clip_stats: dict[str, dict] = {}
    n_no_seed = 0

    for clip in clips:
        n = davis.num_frames(clip)
        inst_ids = davis.instance_ids(clip, 0)
        if not inst_ids:
            continue
        inst = inst_ids[0]  # DAVIS 2016-style protocol: single tracked object

        frame0 = davis.load_frame(clip, 0)
        H, W = frame0.shape[:2]
        gt0 = davis.load_mask(clip, 0, instance_id=inst)
        if gt0.sum() == 0:
            continue

        clip_stats[clip] = clip_motion_area_instances(clip, inst)

        # Simulate a user click (GT used only to place the click, never as the seed itself)
        gt256 = cv2.resize(gt0, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        pts = sample_clicks(gt256, 1)
        cx256, cy256 = pts[0]
        click_xy = (cx256 * (W / 256.0), cy256 * (H / 256.0))

        proposals, scores = divider.predict_scored(frame0)
        if conquer_backbone is not None:
            proposals, scores = run_conquer_scored(conquer_backbone, frame0, proposals, scores)
        seed, seed_score = pick_proposal_scored(proposals, scores, gt0, click_xy=click_xy)
        if seed is None or seed.sum() == 0:
            n_no_seed += 1
            continue
        mask_a = seed.astype(np.uint8)
        seed_iou = iou(mask_a, gt0)
        seed_ious.append(seed_iou)

        feats0 = extractor.extract(frame0)
        clip_rows = []
        for o in offsets:
            if o >= n:
                continue
            frame_b = davis.load_frame(clip, o)
            feats_b = extractor.extract(frame_b)
            mask_b_gt = davis.load_mask(clip, o, instance_id=inst)
            if mask_b_gt.sum() == 0:
                continue

            fwd = propagate(
                feats0["feats"], feats_b["feats"], mask_a,
                out_size=(frame_b.shape[0], frame_b.shape[1]),
                blur=args.blur, threshold=args.threshold,
            )
            if fwd["mask"].sum() == 0:
                continue
            quality = iou(fwd["mask"], mask_b_gt)  # true quality -- GT used only to score

            back = propagate(
                feats_b["feats"], feats0["feats"], fwd["mask"],
                out_size=(frame0.shape[0], frame0.shape[1]),
                blur=args.blur, threshold=args.threshold,
            )
            agreement = iou(back["mask"], mask_a)  # round-trip vs the PSEUDO seed, not GT

            flow_agreement = float("nan")
            if raft_model is not None:
                flow_b_to_a = compute_flow(raft_model, raft_transforms, frame_b, frame0,
                                           device=args.flow_device)
                flow_warped = warp_mask_backward(mask_a, flow_b_to_a)
                flow_agreement = iou(flow_warped, fwd["mask"])  # OT mask vs independent pixel-motion mask

            agreements.append(agreement)
            qualities.append(quality)
            seed_scores.append(seed_score)
            flow_agreements.append(flow_agreement)
            sample_clips.append(clip)
            clip_rows.append((o, agreement, quality, flow_agreement))
        d = clip_stats[clip]
        print(f"  [{clip}] seed_iou={seed_iou:.2f} seed_score={seed_score:.2f} "
              f"area={d['area_frac']:.3f} motion={d['motion']:.3f} n_inst={d['n_instances']}  " +
              "  ".join(f"o={o}: agree={a:.2f} qual={q:.2f} flow={fl:.2f}" for o, a, q, fl in clip_rows))

    print(f"\n[seed] mean IoU(pseudo, GT) = {np.mean(seed_ious):.3f}  (n={len(seed_ious)}, "
          f"{n_no_seed} clips had no seed)")

    agreements_arr = np.array(agreements)
    qualities_arr = np.array(qualities)
    scores_arr = np.array(seed_scores)
    flow_arr = np.array(flow_agreements)
    has_flow = not args.no_flow and np.isfinite(flow_arr).all() and len(flow_arr) > 0
    n = len(agreements_arr)
    print(f"[n={n}]")
    if n < 3:
        print("Not enough samples for correlation.")
        return

    r, p_r = pearsonr(agreements_arr, qualities_arr)
    rho, p_rho = spearmanr(agreements_arr, qualities_arr)
    print(f"agreement alone: Pearson r={r:.3f} (p={p_r:.2e})  Spearman rho={rho:.3f} (p={p_rho:.2e})")

    r_s, p_rs = pearsonr(scores_arr, qualities_arr)
    rho_s, p_rhos = spearmanr(scores_arr, qualities_arr)
    print(f"score alone:     Pearson r={r_s:.3f} (p={p_rs:.2e})  Spearman rho={rho_s:.3f} (p={p_rhos:.2e})")

    if has_flow:
        r_f, p_rf = pearsonr(flow_arr, qualities_arr)
        rho_f, p_rhof = spearmanr(flow_arr, qualities_arr)
        print(f"flow alone:      Pearson r={r_f:.3f} (p={p_rf:.2e})  Spearman rho={rho_f:.3f} (p={p_rhof:.2e})")

    # Joint linear predictor: quality ~ a*agreement + b*score [+ c*flow] + d
    cols = [agreements_arr, scores_arr] + ([flow_arr] if has_flow else []) + [np.ones(n)]
    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, qualities_arr, rcond=None)
    pred = X @ coef
    ss_res = ((qualities_arr - pred) ** 2).sum()
    ss_tot = ((qualities_arr - qualities_arr.mean()) ** 2).sum()
    r2_joint = 1 - ss_res / ss_tot
    r2_agree_only = r ** 2
    label = "agreement, score, flow" if has_flow else "agreement, score"
    coef_str = f"agree={coef[0]:.3f}, score={coef[1]:.3f}" + (f", flow={coef[2]:.3f}" if has_flow else "")
    print(f"\nJoint linear fit ({label}) -> quality: R^2={r2_joint:.3f}  "
          f"(agreement-alone R^2={r2_agree_only:.3f})  coef=({coef_str}, "
          f"intercept={coef[-1]:.3f})")

    order = np.argsort(agreements_arr)
    quartiles = np.array_split(order, 4)
    print("\nQuartile of agreement -> mean true quality:")
    for i, idx in enumerate(quartiles):
        lo, hi = agreements_arr[idx].min(), agreements_arr[idx].max()
        print(f"  Q{i+1} (agreement {lo:.2f}-{hi:.2f}): mean quality={qualities_arr[idx].mean():.3f}"
              f"  (n={len(idx)})")

    baseline = qualities_arr.mean()
    top_half = qualities_arr[order[n // 2:]]
    print(f"\nUnfiltered mean quality:                  {baseline:.3f}")
    print(f"Top-50%-agreement-only mean quality:       {top_half.mean():.3f}  "
          f"(delta {top_half.mean() - baseline:+.3f})")

    # AND-gates: agreement+score (existing), and agreement+score+flow if available.
    agree_med, score_med = np.median(agreements_arr), np.median(scores_arr)
    and_gate = (agreements_arr >= agree_med) & (scores_arr >= score_med)
    if and_gate.sum() > 0:
        print(f"AND-gate (agree+score) mean quality: "
              f"{qualities_arr[and_gate].mean():.3f}  (delta {qualities_arr[and_gate].mean() - baseline:+.3f}, "
              f"n={and_gate.sum()}/{n} kept)")

    and_gate3 = and_gate
    flow_med = None
    if has_flow:
        flow_med = np.median(flow_arr)
        and_gate3 = and_gate & (flow_arr >= flow_med)
        if and_gate3.sum() > 0:
            print(f"AND-gate (agree+score+flow, flow>={flow_med:.2f}) mean quality: "
                  f"{qualities_arr[and_gate3].mean():.3f}  "
                  f"(delta {qualities_arr[and_gate3].mean() - baseline:+.3f}, "
                  f"n={and_gate3.sum()}/{n} kept)")
        and_gate_af = (agreements_arr >= agree_med) & (flow_arr >= flow_med)
        if and_gate_af.sum() > 0:
            print(f"AND-gate (agree+flow, no score) mean quality: "
                  f"{qualities_arr[and_gate_af].mean():.3f}  "
                  f"(delta {qualities_arr[and_gate_af].mean() - baseline:+.3f}, "
                  f"n={and_gate_af.sum()}/{n} kept)")

    # Does score / flow specifically catch the cases agreement alone is fooled by?
    fooled = (agreements_arr >= agree_med) & (qualities_arr < np.median(qualities_arr))
    if fooled.sum() > 0:
        msg = (f"\nCases where agreement is high but true quality is low (n={fooled.sum()}): "
               f"mean score={scores_arr[fooled].mean():.3f} (overall {scores_arr.mean():.3f})")
        if has_flow:
            msg += f", mean flow={flow_arr[fooled].mean():.3f} (overall {flow_arr.mean():.3f})"
        print(msg)

    # Explicit audit of the known self-consistent-wrong-track cases: does flow catch what
    # agreement+score both missed?
    if has_flow:
        clip_arr_audit = np.array(sample_clips)
        present = [c for c in _WRONG_TRACK_CASES if c in clip_arr_audit]
        if present:
            print(f"\n[wrong-track audit] known confidently-wrong cases (quality~0, but "
                  f"agreement/score high) -- does flow_agreement come in LOW where the others didn't?")
            print(f"  {'clip':16s} {'agree':>6} {'score':>6} {'flow':>6} {'quality':>8}")
            for c in present:
                idx = np.where(clip_arr_audit == c)[0]
                print(f"  {c:16s} {agreements_arr[idx].mean():6.2f} {scores_arr[idx].mean():6.2f} "
                      f"{flow_arr[idx].mean():6.2f} {qualities_arr[idx].mean():8.3f}")

    # ── Held-out 2-fold validation ──────────────────────────────────────────
    clip_arr = np.array(sample_clips)
    unique_clips = list(dict.fromkeys(sample_clips))  # order-preserving dedup
    if len(unique_clips) >= 4:
        half = len(unique_clips) // 2
        fold_a = set(unique_clips[:half])
        fold_b = set(unique_clips[half:])
        is_a = np.array([c in fold_a for c in clip_arr])
        is_b = ~is_a
        print(f"\n[holdout] fold A = {len(fold_a)} clips, fold B = {len(fold_b)} clips")

        def _fit_and_apply(fit_mask: np.ndarray, test_mask: np.ndarray, label: str, use_flow: bool):
            a_med = np.median(agreements_arr[fit_mask])
            s_med = np.median(scores_arr[fit_mask])
            gate_test = (agreements_arr[test_mask] >= a_med) & (scores_arr[test_mask] >= s_med)
            thresh_str = f"agree>={a_med:.2f}, score>={s_med:.2f}"
            if use_flow:
                f_med = np.median(flow_arr[fit_mask])
                gate_test = gate_test & (flow_arr[test_mask] >= f_med)
                thresh_str += f", flow>={f_med:.2f}"
            test_baseline = qualities_arr[test_mask].mean()
            if gate_test.sum() == 0:
                print(f"  {label}: gate kept 0/{test_mask.sum()} test samples, no result")
                return None
            gated_mean = qualities_arr[test_mask][gate_test].mean()
            delta = gated_mean - test_baseline
            print(f"  {label}: fit thresholds ({thresh_str}) -> "
                  f"test-fold baseline={test_baseline:.3f}, gated={gated_mean:.3f} (delta {delta:+.3f}, "
                  f"n={gate_test.sum()}/{test_mask.sum()} kept)")
            return delta

        print("Held-out validation, agree+score gate (thresholds fit on one fold, applied to the other):")
        d_ab = _fit_and_apply(is_a, is_b, "fit=A -> test=B", use_flow=False)
        d_ba = _fit_and_apply(is_b, is_a, "fit=B -> test=A", use_flow=False)
        deltas = [d for d in (d_ab, d_ba) if d is not None]
        if deltas:
            print(f"  mean out-of-sample delta = {np.mean(deltas):+.3f}")

        if has_flow:
            print("Held-out validation, agree+score+flow gate:")
            d_ab3 = _fit_and_apply(is_a, is_b, "fit=A -> test=B", use_flow=True)
            d_ba3 = _fit_and_apply(is_b, is_a, "fit=B -> test=A", use_flow=True)
            deltas3 = [d for d in (d_ab3, d_ba3) if d is not None]
            if deltas3:
                print(f"  mean out-of-sample delta = {np.mean(deltas3):+.3f}")

    # ── Keep-rate / quality sweep ────────────────────────────────────────────
    joint_rank = coef[0] * agreements_arr + coef[1] * scores_arr + (coef[2] * flow_arr if has_flow else 0.0)
    fractions = np.arange(0.05, 1.001, 0.05)
    curve_joint = sweep_curve(joint_rank, qualities_arr, fractions)
    curve_agree = sweep_curve(agreements_arr, qualities_arr, fractions)
    curve_score = sweep_curve(scores_arr, qualities_arr, fractions)
    curve_flow = sweep_curve(flow_arr, qualities_arr, fractions) if has_flow else None

    print("\nKeep-rate / quality-of-kept sweep:")
    header = f"  {'kept%':>6}  {'joint':>6}  {'agree':>6}  {'score':>6}"
    if has_flow:
        header += f"  {'flow':>6}"
    print(header)
    for i, f in enumerate(fractions):
        row = f"  {f*100:5.0f}%  {curve_joint[i]:6.3f}  {curve_agree[i]:6.3f}  {curve_score[i]:6.3f}"
        if has_flow:
            row += f"  {curve_flow[i]:6.3f}"
        print(row)

    if args.sweep_csv:
        with open(args.sweep_csv, "w") as fh:
            cols_hdr = "kept_fraction,quality_joint,quality_agreement_only,quality_score_only"
            if has_flow:
                cols_hdr += ",quality_flow_only"
            fh.write(cols_hdr + "\n")
            for i, f in enumerate(fractions):
                row = f"{f:.3f},{curve_joint[i]:.4f},{curve_agree[i]:.4f},{curve_score[i]:.4f}"
                if has_flow:
                    row += f",{curve_flow[i]:.4f}"
                fh.write(row + "\n")
        print(f"[sweep] wrote {args.sweep_csv}")

    if args.plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
        joint_label = "joint (agreement+score+flow)" if has_flow else "joint (agreement + CuVLER score)"
        ax.plot(fractions * 100, curve_joint, color="#0072B2", linewidth=2, marker="o",
                markersize=4, label=joint_label)
        ax.plot(fractions * 100, curve_agree, color="#E69F00", linewidth=2, marker="o",
                markersize=4, label="agreement only")
        ax.plot(fractions * 100, curve_score, color="#009E73", linewidth=2, marker="o",
                markersize=4, label="CuVLER score only")
        if has_flow:
            ax.plot(fractions * 100, curve_flow, color="#D55E00", linewidth=2, marker="o",
                    markersize=4, label="RAFT flow only")
        ax.axhline(baseline, color="#888888", linewidth=1, linestyle="--", label="unfiltered baseline")
        ax.set_xlabel("kept fraction of pseudo-labels (%)")
        ax.set_ylabel("mean true quality (IoU vs GT) of kept set")
        ax.set_title(f"Stage-5 pseudo-label yield vs quality ({n} clip/offset samples, {len(clips)} clips)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.plot_path)
        print(f"[sweep] wrote {args.plot_path}")

    # ── Difficulty-tag breakdown ─────────────────────────────────────────────
    tags_by_clip = assign_difficulty_tags(clip_stats)
    print("\n[difficulty] per-clip tags (data-driven terciles, area/motion computed from GT):")
    for clip in clips:
        if clip in tags_by_clip:
            print(f"  {clip:20s} {','.join(tags_by_clip[clip])}")

    sample_tags = [tags_by_clip.get(c, ("none",)) for c in sample_clips]
    gate_for_tags = and_gate3 if has_flow else and_gate
    gate_desc = f"agree+score+flow" if has_flow else "agree+score"
    print(f"\n[difficulty] keep-rate and quality by tag at the AND-gate operating point ({gate_desc}):")
    print(f"  {'tag':10s} {'n':>5} {'keep%':>7} {'q_kept':>8} {'q_rejected':>11} {'q_overall':>10}")
    for tag in (*_TAGS, "none"):
        idx = np.array([i for i, ts in enumerate(sample_tags) if tag in ts])
        if len(idx) == 0:
            continue
        kept = gate_for_tags[idx]
        keep_rate = kept.mean() * 100
        q_kept = qualities_arr[idx][kept].mean() if kept.any() else float("nan")
        q_rej = qualities_arr[idx][~kept].mean() if (~kept).any() else float("nan")
        q_all = qualities_arr[idx].mean()
        print(f"  {tag:10s} {len(idx):5d} {keep_rate:6.1f}% {q_kept:8.3f} {q_rej:11.3f} {q_all:10.3f}")
    overall_keep_rate = gate_for_tags.mean() * 100
    print(f"  {'(overall)':10s} {n:5d} {overall_keep_rate:6.1f}%")


if __name__ == "__main__":
    main()
