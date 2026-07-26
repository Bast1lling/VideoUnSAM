"""Offline J&F benchmark over a dumped run. No GPU, no torch, no DINOv3.

Loads the artifacts written by video/scripts/dump_artifacts.py, applies one or more
refiners from refiners.py to every frame, and reports DAVIS J / F / J&F — per clip
and aggregate. Runs on a laptop in seconds, so the edit-measure loop is instant.

    # compare strategies
    python -m video.refinelab.bench --dump dumps/davis2016_default \
        --refiners baseline,guided,snap,guided_snap

    # sweep a parameter
    python -m video.refinelab.bench --dump dumps/davis2016_default \
        --refiners snap --sweep region_thresh=0.2,0.3,0.35,0.4,0.5

    # per-clip detail for one config
    python -m video.refinelab.bench --dump dumps/davis2016_default \
        --refiners guided_snap --params region_thresh=0.35,radius=8 --per-clip
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from video.refinelab.refiners import REFINERS, NEEDS_PCA, NEEDS_FLOW  # noqa: E402

# Parameters that alter the OT chain itself, so an offline result is NOT faithful.
CHAIN_PARAMS = {"thresh"}


# ── DAVIS metrics (identical to video/scripts/eval_davis2016.py) ──────────────

def j_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = (pred | gt).sum()
    return float((pred & gt).sum() / union) if union else 0.0


def _boundary(mask: np.ndarray, tol: int) -> np.ndarray:
    struct = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
    return mask.astype(bool) & ~binary_erosion(mask.astype(bool), struct)


def f_score(pred: np.ndarray, gt: np.ndarray, tol: int = 3) -> float:
    pb, gb = _boundary(pred, tol), _boundary(gt, tol)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    struct = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
    prec = float((pb & binary_dilation(gb, struct)).sum()) / (pb.sum() + 1e-8)
    rec = float((gb & binary_dilation(pb, struct)).sum()) / (gb.sum() + 1e-8)
    return 2 * prec * rec / (prec + rec) if prec + rec > 1e-8 else 0.0


# ── Dump loading ─────────────────────────────────────────────────────────────

def _unpack(bits: np.ndarray, shape) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    return np.unpackbits(bits)[: h * w].reshape(h, w).astype(np.uint8)


def load_frame(dump: Path, clip: str, fidx: int, want_props: bool, want_pca: bool,
              want_flow: bool = False):
    z = np.load(dump / "npz" / clip / f"{fidx:05d}.npz")
    shape = z["shape"]
    h, w = int(shape[0]), int(shape[1])
    gh, gw = (int(x) for x in z["grid"])

    heat = z["heat"].astype(np.float32).reshape(gh, gw)
    heat_up = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    soft_up = heat_up / (heat_up.max() + 1e-8)

    bgr = cv2.imread(str(dump / "frames" / clip / f"{fidx:05d}.jpg"))
    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gt = _unpack(z["gt"], shape)

    props = []
    if want_props:
        p = dump / "props" / clip / f"{fidx:05d}.pkl"
        if p.exists():
            from pycocotools import mask as mask_util
            with open(p, "rb") as fh:
                props = [mask_util.decode(r) for r in pickle.load(fh)]

    pca3 = z["pca3"] if (want_pca and "pca3" in z) else None

    flow_heat = None
    if want_flow:
        fp = dump / "flow" / clip / f"{fidx:05d}.npz"
        if fp.exists():
            fz = np.load(fp)
            flow_heat = fz["flow_heat"].astype(np.float32)

    return soft_up, frame_rgb, gt, props, pca3, flow_heat


def clips_in(dump: Path) -> list[str]:
    return sorted(p.name for p in (dump / "npz").iterdir() if p.is_dir())


def frames_in(dump: Path, clip: str) -> list[int]:
    return sorted(int(p.stem) for p in (dump / "npz" / clip).glob("*.npz"))


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(dump: Path, refiner_name: str, params: dict,
             clips: list[str], per_clip: bool) -> dict:
    fn = REFINERS[refiner_name]
    want_props = "snap" in refiner_name
    want_pca = refiner_name in NEEDS_PCA
    want_flow = refiner_name in NEEDS_FLOW

    rows, all_j, all_f = [], [], []
    for clip in clips:
        js, fs = [], []
        for fidx in frames_in(dump, clip):
            soft_up, frame_rgb, gt, props, pca3, flow_heat = load_frame(
                dump, clip, fidx, want_props, want_pca, want_flow)
            kw = dict(params)
            if want_pca:
                kw["pca3"] = pca3
            if want_flow:
                kw["flow_heat"] = flow_heat
            pred = fn(soft_up, frame_rgb, props, **kw)
            if gt.sum() > 0:
                js.append(j_score(pred, gt))
                fs.append(f_score(pred, gt))
        if not js:
            continue
        j, f = float(np.mean(js)), float(np.mean(fs))
        rows.append({"clip": clip, "j": j, "f": f, "jf": (j + f) / 2, "n": len(js)})
        all_j.extend(js)
        all_f.extend(fs)

    # DAVIS convention: average per-clip, not per-frame
    j = float(np.mean([r["j"] for r in rows])) if rows else 0.0
    f = float(np.mean([r["f"] for r in rows])) if rows else 0.0
    out = {"refiner": refiner_name, "params": params,
           "j": j, "f": f, "jf": (j + f) / 2, "clips": len(rows)}
    if per_clip:
        out["per_clip"] = rows
    return out


def parse_params(s: str) -> dict:
    out = {}
    for kv in (x for x in s.split(",") if x.strip()):
        k, v = kv.split("=", 1)
        try:
            out[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else float(v)
        except ValueError:
            out[k.strip()] = v.strip()
    return out


def parse_sweep(s: str) -> dict:
    """'region_thresh=0.2,0.3,0.4 radius=8,16' -> {name: [values]}"""
    grid = {}
    for token in s.split():
        k, vals = token.split("=", 1)
        parsed = []
        for v in vals.split(","):
            try:
                parsed.append(int(v) if v.lstrip("-").isdigit() else float(v))
            except ValueError:
                parsed.append(v)
        grid[k.strip()] = parsed
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--refiners", default="baseline,guided,snap,guided_snap")
    ap.add_argument("--params", default="", help="k=v,k=v applied to every refiner")
    ap.add_argument("--sweep", default="",
                    help="Space-separated 'key=v1,v2,v3' groups; full cross product")
    ap.add_argument("--clips", default="", help="Comma-separated subset")
    ap.add_argument("--per-clip", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    dump = Path(args.dump)
    if not (dump / "npz").exists():
        sys.exit(f"No dump at {dump} (expected {dump}/npz/)")

    clips = [c.strip() for c in args.clips.split(",") if c.strip()] or clips_in(dump)
    base_params = parse_params(args.params)
    names = [n.strip() for n in args.refiners.split(",") if n.strip()]
    for n in names:
        if n not in REFINERS:
            sys.exit(f"Unknown refiner '{n}'. Available: {', '.join(REFINERS)}")

    if any(p in CHAIN_PARAMS for p in base_params) or \
       any(p in CHAIN_PARAMS for p in parse_sweep(args.sweep)):
        print("!! WARNING: you changed a chain-altering parameter (thresh). The OT\n"
              "!! chain in the dump was run at its own threshold, and the reseed step\n"
              "!! depends on it. Offline numbers are indicative, NOT faithful — confirm\n"
              "!! any win with a real eval_davis2016 run on the cluster.\n")

    configs = []
    grid = parse_sweep(args.sweep)
    if grid:
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            configs.append({**base_params, **dict(zip(keys, combo))})
    else:
        configs = [base_params]

    results = []
    t0 = time.time()
    for name in names:
        for params in configs:
            r = evaluate(dump, name, params, clips, args.per_clip)
            results.append(r)
            tag = " ".join(f"{k}={v}" for k, v in params.items()) or "default"
            print(f"{name:<14} {tag:<40} "
                  f"J={r['j']:.3f}  F={r['f']:.3f}  J&F={r['jf']:.3f}")

    if len(results) > 1:
        best = max(results, key=lambda r: r["jf"])
        base = next((r for r in results if r["refiner"] == "baseline"), None)
        print(f"\nBest: {best['refiner']} {best['params']}  J&F={best['jf']:.3f}")
        if base and best["refiner"] != "baseline":
            print(f"  vs baseline J&F={base['jf']:.3f}  "
                  f"(dJ={best['j'] - base['j']:+.3f}  dF={best['f'] - base['f']:+.3f}  "
                  f"dJ&F={best['jf'] - base['jf']:+.3f})")

    if args.per_clip:
        for r in results:
            print(f"\n--- {r['refiner']} {r['params']}")
            print(f"{'clip':<22}{'J':>8}{'F':>8}{'J&F':>8}")
            for row in sorted(r["per_clip"], key=lambda x: x["jf"]):
                print(f"{row['clip']:<22}{row['j']:>8.3f}{row['f']:>8.3f}{row['jf']:>8.3f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json_out}")
    print(f"\n({len(results)} configs, {len(clips)} clips, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
