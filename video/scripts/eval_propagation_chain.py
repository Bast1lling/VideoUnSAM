"""Aggregate CHAINED OT mask-propagation eval vs the direct single-jump baseline.

Re-applies propagate_chain.py's stride-hop algorithm (propagate in patch space,
sharpen between hops) across many DAVIS clips/instances, alongside a paired direct
(frame 0 -> offset, no hops) baseline computed with the same features — so the
chain-vs-direct delta is read off directly per offset. propagate_chain.py only ever
looked at one clip/instance at a time; eval_propagation.py only covered direct.

Chain state is carried forward across offsets (0->...->10->...->20->...->30), like a
running tracker, rather than restarted from frame 0 each time.

    python -m video.scripts.eval_propagation_chain --offsets 5,10,20,30 --stride 5 --limit-clips 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video.features.dinov3_dense import DenseDINOv3
from video.loaders import davis
from video.propagation.sinkhorn_ot import propagate, propagate_patch, _mask_to_patch_indicator

_SPLIT = Path(__file__).resolve().parents[1] / "decoder" / "davis_split.json"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="5,10,20,30")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--sharpen", type=float, default=4.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit-clips", type=int, default=20)
    ap.add_argument("--split", choices=["clean", "all"], default="clean")
    args = ap.parse_args()
    offsets = sorted(int(x) for x in args.offsets.split(","))

    clips = json.load(open(_SPLIT))["clean"] if args.split == "clean" else davis.list_clips()
    if args.limit_clips:
        clips = clips[:args.limit_clips]

    extractor = DenseDINOv3()
    print(f"[eval] {len(clips)} clips, offsets={offsets}, stride={args.stride}, sharpen={args.sharpen}")

    per_offset_chain: dict[int, list[float]] = {o: [] for o in offsets}
    per_offset_direct: dict[int, list[float]] = {o: [] for o in offsets}

    for clip in clips:
        n = davis.num_frames(clip)
        inst_ids = davis.instance_ids(clip, 0)
        if not inst_ids:
            continue

        feat_cache: dict[int, dict] = {}

        def feats(idx):
            if idx not in feat_cache:
                feat_cache[idx] = extractor.extract(davis.load_frame(clip, idx))
            return feat_cache[idx]

        f0 = feats(0)
        clip_scores = []
        for inst in inst_ids:
            mask_a = davis.load_mask(clip, 0, instance_id=inst)
            if mask_a.sum() == 0:
                continue
            m_patch = _mask_to_patch_indicator(mask_a, f0["grid_h"], f0["grid_w"])
            src_idx = 0
            for o in offsets:
                if o >= n:
                    break
                hops = list(range(src_idx + args.stride, o, args.stride))
                if not hops or hops[-1] != o:
                    hops.append(o)
                for dst_idx in hops:
                    m_patch = propagate_patch(feats(src_idx)["feats"], feats(dst_idx)["feats"],
                                               m_patch, blur=args.blur).cpu()
                    if args.sharpen != 1.0:
                        m_patch = m_patch.clamp_min(0).pow(args.sharpen)
                        m_patch = m_patch / (m_patch.max() + 1e-8)
                    src_idx = dst_idx

                gt = davis.load_mask(clip, o, instance_id=inst)
                if gt.sum() == 0:
                    continue
                H, W = davis.load_frame(clip, o).shape[:2]
                gh_b, gw_b = feats(o)["grid_h"], feats(o)["grid_w"]
                heat_up = F.interpolate(
                    m_patch.reshape(gh_b, gw_b)[None, None], size=(H, W),
                    mode="bilinear", align_corners=False,
                )[0, 0].numpy()
                pred = (heat_up > args.threshold * heat_up.max()).astype(np.uint8)
                chain_score = iou(pred, gt)
                per_offset_chain[o].append(chain_score)

                direct = propagate(f0["feats"], feats(o)["feats"], mask_a,
                                    out_size=(H, W), blur=args.blur, threshold=args.threshold)
                direct_score = iou(direct["mask"], gt)
                per_offset_direct[o].append(direct_score)
                clip_scores.append((o, inst, chain_score, direct_score))

        print(f"  [{clip}] " + "  ".join(f"o={o} i={i}: chain={c:.2f} direct={d:.2f}" for o, i, c, d in clip_scores))

    print()
    for o in offsets:
        c, d = per_offset_chain[o], per_offset_direct[o]
        if c:
            print(f"offset {o:3d}: chain mIoU {np.mean(c):.4f}  direct mIoU {np.mean(d):.4f}  "
                  f"delta {np.mean(c) - np.mean(d):+.4f}  (n={len(c)})")
        else:
            print(f"offset {o:3d}: no valid instances")


if __name__ == "__main__":
    main()
