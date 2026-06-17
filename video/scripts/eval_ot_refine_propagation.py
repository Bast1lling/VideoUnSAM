"""OT-propagation-as-memory: click-prompt frame 0, then for each subsequent frame
OT-propagate the previous frame's predicted mask forward (in the same 64x64 DINOv3
grid used by the decoder) and feed it to the decoder as a `mask_prompts` refinement
-- no new click/box prompts. Pivot away from the learned KV memory bank
([[memory-bank-gate-stuck]]: two training configs both landed at ~0.26-0.30 mIoU,
far below plain OT).

Three columns, all label-free at test time, all starting from the SAME K-click
frame-0 prediction:
  - refine:   OT-propagate prev *refined* prediction -> mask_prompt -> decoder refine.
  - ot-only:  chained OT propagation alone (no decoder refine) -- isolates how much
              refinement adds on top of propagation.
  - carryover: decoder refine using prev frame's own prediction as mask_prompt
              directly (no OT) -- isolates how much OT adds over naive carryover.

Plus the reference OT-from-GT ceiling ([[propagation-ot-vs-alternatives]], native-res
grid, GT-seeded): 0.771/0.727/0.647/0.604/0.529 at offsets 1/5/10/20/30.

    python -m video.scripts.eval_ot_refine_propagation --ckpt checkpoints/unsup_decoder_points_v3.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.loaders import davis  # noqa: E402
from video.decoder.train_sam_decoder import UnsupervisedDecoder, sample_clicks, next_click, _IMG  # noqa: E402
from video.propagation.sinkhorn_ot import propagate_patch, _mask_to_patch_indicator  # noqa: E402

_SPLIT = Path(__file__).resolve().parents[1] / "decoder" / "davis_split.json"
_OT_BASELINE = {1: 0.771, 5: 0.727, 10: 0.647, 20: 0.604, 30: 0.529}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def patch_indicator_or_soft(bin_mask: np.ndarray, soft_mask: np.ndarray, gh: int = 64, gw: int = 64) -> torch.Tensor:
    """_mask_to_patch_indicator, falling back to the soft probability map (or uniform)
    when the binary mask is empty after pooling -- keeps OT propagation alive instead
    of crashing on a degenerate (empty) decoder prediction."""
    try:
        return _mask_to_patch_indicator(bin_mask, gh, gw)
    except ValueError:
        m = torch.from_numpy(soft_mask.astype(np.float32))[None, None]
        pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
        if pooled.sum() <= 0:
            pooled = pooled + 1.0 / (gh * gw)
        return pooled.flatten()


def predict_frame0(model, feats, gts, k):
    """Iterative K-click prediction (no memory/mask prompt). Returns logits[N,1,256,256]."""
    gt256 = [cv2.resize(g, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8) for g in gts]
    N = len(gt256)
    coords, labels = [], []
    for g in gt256:
        pts = sample_clicks(g, 1)
        x, y = pts[0] if pts else (128.0, 128.0)
        coords.append([[x * 4.0, y * 4.0]]); labels.append([1.0])

    prev, logits = None, None
    for it in range(k):
        c = torch.tensor(coords, dtype=torch.float, device="cuda")
        l = torch.tensor(labels, dtype=torch.float, device="cuda")
        with torch.no_grad():
            logits = model(feats, points=(c, l), mask_prompts=prev)
        prev = logits.detach()
        if it < k - 1:
            pred = (prev[:, 0].sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
            for i in range(N):
                nc = next_click(gt256[i], pred[i])
                if nc is None:
                    coords[i].append(coords[i][-1]); labels[i].append(labels[i][-1])
                else:
                    x, y, lab = nc
                    coords[i].append([x, y]); labels[i].append(lab)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unsup_decoder_points_v3.pth")
    ap.add_argument("--clicks", type=int, default=3)
    ap.add_argument("--offsets", default="1,5,10,20,30")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--blur", type=float, default=0.05)
    ap.add_argument("--logit-scale", type=float, default=10.0,
                     help="OT heatmap in [0,1] -> mask_prompt logits via (heat*2-1)*scale.")
    ap.add_argument("--limit-clips", type=int, default=0)
    args = ap.parse_args()
    offsets = sorted(int(x) for x in args.offsets.split(","))
    max_off = offsets[-1]

    clean = json.load(open(_SPLIT))["clean"]
    if args.limit_clips:
        clean = clean[:args.limit_clips]

    dino = DenseDINOv3()
    model = UnsupervisedDecoder().cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])
    print(f"[ckpt] {args.ckpt}")

    per_offset_refine = {o: [] for o in offsets}
    per_offset_ot_only = {o: [] for o in offsets}
    per_offset_carry = {o: [] for o in offsets}

    np.random.seed(0)
    for clip in clean:
        n = davis.num_frames(clip)
        inst_ids = davis.instance_ids(clip, 0)
        gts0 = [davis.load_mask(clip, 0, instance_id=i) for i in inst_ids]
        keep = [(i, g) for i, g in zip(inst_ids, gts0) if g.sum() > 0]
        if not keep:
            continue
        inst_ids, gts0 = zip(*keep)
        gts0 = list(gts0)
        N = len(gts0)

        frame0 = davis.load_frame(clip, 0)
        H, W = frame0.shape[:2]
        img1024 = cv2.resize(frame0, (_IMG, _IMG))
        with torch.no_grad():
            feats_prev = dino.extract(img1024, normalize=False)["feats"].cuda().float()  # [64,64,1024]
        feats_prev_chw = feats_prev.permute(2, 0, 1)[None]
        feats_prev_norm = F.normalize(feats_prev, dim=-1)

        logits0 = predict_frame0(model, feats_prev_chw, gts0, args.clicks)
        soft0_256 = logits0.sigmoid().cpu().numpy()[:, 0]  # [N,256,256]
        bin0_256 = (soft0_256 > 0.5).astype(np.uint8)

        refine_patch = [patch_indicator_or_soft(bin0_256[i], soft0_256[i]) for i in range(N)]
        ot_only_patch = [p.clone() for p in refine_patch]
        carry_prev = [logits0[i:i + 1].detach() for i in range(N)]

        last_frame = min(max_off, n - 1)
        for fidx in range(1, last_frame + 1):
            frame = davis.load_frame(clip, fidx)
            img1024 = cv2.resize(frame, (_IMG, _IMG))
            with torch.no_grad():
                feats_b = dino.extract(img1024, normalize=False)["feats"].cuda().float()
            feats_b_chw = feats_b.permute(2, 0, 1)[None]
            feats_b_norm = F.normalize(feats_b, dim=-1)

            score = fidx in offsets
            gts_f = [davis.load_mask(clip, fidx, instance_id=i) for i in inst_ids] if score else None

            for i in range(N):
                # --- OT + decoder refine (self-click at the OT heatmap's peak, +mask prompt) ---
                heat = propagate_patch(feats_prev_norm, feats_b_norm, refine_patch[i], blur=args.blur)
                heat_256 = F.interpolate(heat.reshape(1, 1, 64, 64), size=(256, 256),
                                          mode="bilinear", align_corners=False)
                mask_prompt = (heat_256 * 2.0 - 1.0) * args.logit_scale
                py, px = np.unravel_index(int(heat.argmax()), (64, 64))
                c = torch.tensor([[[(px + 0.5) * 16.0, (py + 0.5) * 16.0]]], dtype=torch.float, device="cuda")
                l = torch.tensor([[1.0]], dtype=torch.float, device="cuda")
                with torch.no_grad():
                    logits_r = model(feats_b_chw, points=(c, l), mask_prompts=mask_prompt)
                soft_r_256 = logits_r.sigmoid().cpu().numpy()[0, 0]
                bin_r_256 = (soft_r_256 > 0.5).astype(np.uint8)
                refine_patch[i] = patch_indicator_or_soft(bin_r_256, soft_r_256)

                # --- OT only (chained, no refine) ---
                heat_only = propagate_patch(feats_prev_norm, feats_b_norm, ot_only_patch[i], blur=args.blur)
                ot_only_patch[i] = heat_only

                # --- carryover (self-click at prev prediction's peak, +mask prompt, no OT) ---
                prev_soft = carry_prev[i].sigmoid()[0, 0]
                pidx = int(prev_soft.flatten().argmax())
                py_c, px_c = pidx // 256, pidx % 256
                c2 = torch.tensor([[[(px_c + 0.5) * 4.0, (py_c + 0.5) * 4.0]]], dtype=torch.float, device="cuda")
                l2 = torch.tensor([[1.0]], dtype=torch.float, device="cuda")
                with torch.no_grad():
                    logits_c = model(feats_b_chw, points=(c2, l2), mask_prompts=carry_prev[i])
                carry_prev[i] = logits_c.detach()

                if score and gts_f[i].sum() > 0:
                    up_r = F.interpolate(logits_r, (H, W), mode="bilinear", align_corners=False)
                    pred_r = (up_r.sigmoid()[0, 0].cpu().numpy() > args.thresh).astype(np.uint8)
                    per_offset_refine[fidx].append(iou(gts_f[i], pred_r))

                    heat_only_up = F.interpolate(heat_only.reshape(1, 1, 64, 64), size=(H, W),
                                                  mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                    pred_o = (heat_only_up > args.thresh * heat_only_up.max()).astype(np.uint8)
                    per_offset_ot_only[fidx].append(iou(gts_f[i], pred_o))

                    up_c = F.interpolate(logits_c, (H, W), mode="bilinear", align_corners=False)
                    pred_c = (up_c.sigmoid()[0, 0].cpu().numpy() > args.thresh).astype(np.uint8)
                    per_offset_carry[fidx].append(iou(gts_f[i], pred_c))

            feats_prev, feats_prev_norm = feats_b, feats_b_norm

    print(f"{'offset':>8} {'refine':>10} {'ot-only':>10} {'carryover':>10} {'OT (gt-src)':>12}  n")
    for o in offsets:
        r = np.mean(per_offset_refine[o]) if per_offset_refine[o] else float("nan")
        ot = np.mean(per_offset_ot_only[o]) if per_offset_ot_only[o] else float("nan")
        c = np.mean(per_offset_carry[o]) if per_offset_carry[o] else float("nan")
        gt = _OT_BASELINE.get(o, float("nan"))
        print(f"{o:>8} {r:>10.4f} {ot:>10.4f} {c:>10.4f} {gt:>12.4f}  (n={len(per_offset_refine[o])})")


if __name__ == "__main__":
    main()
