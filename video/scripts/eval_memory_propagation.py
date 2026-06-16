"""Compare MemoryDecoder propagation vs the OT baseline and a mask-carryover baseline.

For each instance in the leak-free 64-clip DAVIS split (davis_split.json["clean"]):
  - prompt frame 0 with K=3 iterative correction clicks (eval_clicks.py protocol,
    memory empty), write the anchor memory from the model's OWN K-click
    prediction (not GT -- matches deployment).
  - step through frames 1..max(offsets) with NO new prompts:
      * memory model    -- predicts via MemoryBank (anchor + ring buffer of its
        own past predictions), per-instance (each has its own bank).
      * carryover model -- same checkpoint with memory DISABLED, fed frame
        t-1's own prediction as `mask_prompts` (cheapest possible "memory").
  - report mIoU vs GT at offsets 1/5/10/20/30 for both, alongside the
    [[propagation-ot-vs-alternatives]] OT-from-GT numbers (0.771/0.727/0.647/
    0.604/0.529 -- ceiling reference, propagated from the GT mask not a click
    prompt).

    python -m video.scripts.eval_memory_propagation --ckpt checkpoints/unsup_memory_decoder_v1.pth
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
from video.decoder.train_sam_decoder import sample_clicks, next_click, _IMG  # noqa: E402
from video.decoder.memory_decoder import MemoryDecoder  # noqa: E402
from video.decoder.infer_decoder import iou  # noqa: E402

_SPLIT = Path(__file__).resolve().parents[1] / "decoder" / "davis_split.json"
_OT_BASELINE = {1: 0.771, 5: 0.727, 10: 0.647, 20: 0.604, 30: 0.529}


def predict_frame0(model, feats, gts, k):
    """Iterative K-click prediction with empty memory. Returns (logits[N,1,256,256], img_emb)."""
    gt256 = [cv2.resize(g, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8) for g in gts]
    N = len(gt256)
    coords, labels = [], []
    for g in gt256:
        pts = sample_clicks(g, 1)
        x, y = pts[0] if pts else (128.0, 128.0)
        coords.append([[x * 4.0, y * 4.0]]); labels.append([1.0])

    prev, logits, img_emb = None, None, None
    for it in range(k):
        c = torch.tensor(coords, dtype=torch.float, device="cuda")
        l = torch.tensor(labels, dtype=torch.float, device="cuda")
        with torch.no_grad():
            logits, img_emb = model(feats, memory_bank=None, points=(c, l), mask_prompts=prev)
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
    return logits, img_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/unsup_memory_decoder_v1.pth")
    ap.add_argument("--clicks", type=int, default=3)
    ap.add_argument("--offsets", default="1,5,10,20,30")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--limit-clips", type=int, default=0)
    args = ap.parse_args()
    offsets = sorted(int(x) for x in args.offsets.split(","))
    max_off = offsets[-1]

    clean = json.load(open(_SPLIT))["clean"]
    if args.limit_clips:
        clean = clean[:args.limit_clips]

    dino = DenseDINOv3()
    model = MemoryDecoder().cuda().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])
    print(f"[ckpt] {args.ckpt}  gate={model.memory_reader.gate.item():.4f}")

    per_offset_mem = {o: [] for o in offsets}
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
            feats0 = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()

        logits0, img_emb0 = predict_frame0(model, feats0, gts0, args.clicks)

        mbanks, carry_prev = [], []
        for i in range(N):
            mb = model.new_memory_bank()
            mask64 = F.interpolate(logits0[i:i + 1].sigmoid(), size=(64, 64),
                                    mode="bilinear", align_corners=False)
            with torch.no_grad():
                k_, v_ = model.encode_memory(img_emb0, mask64)
            mb.write_anchor(k_, v_)
            mbanks.append(mb)
            carry_prev.append(logits0[i:i + 1].detach())

        last_frame = min(max_off, n - 1)
        for fidx in range(1, last_frame + 1):
            frame = davis.load_frame(clip, fidx)
            img1024 = cv2.resize(frame, (_IMG, _IMG))
            with torch.no_grad():
                feats = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()

            score = fidx in offsets
            gts_f = [davis.load_mask(clip, fidx, instance_id=i) for i in inst_ids] if score else None

            for i in range(N):
                with torch.no_grad():
                    logits_i, img_emb_i = model(feats, memory_bank=mbanks[i])
                pred_soft = logits_i.sigmoid()
                mask64 = F.interpolate(pred_soft, size=(64, 64), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    k_, v_ = model.encode_memory(img_emb_i, mask64)
                mbanks[i].write_recent(k_, v_)
                if score and gts_f[i].sum() > 0:
                    up = F.interpolate(logits_i, (H, W), mode="bilinear", align_corners=False)
                    pred = (up.sigmoid()[0, 0].cpu().numpy() > args.thresh).astype(np.uint8)
                    per_offset_mem[fidx].append(iou(gts_f[i], pred))

            mp = torch.cat(carry_prev, dim=0)
            with torch.no_grad():
                logits_c, _ = model(feats, memory_bank=None, mask_prompts=mp)
            carry_prev = [logits_c[i:i + 1].detach() for i in range(N)]
            if score:
                up = F.interpolate(logits_c, (H, W), mode="bilinear", align_corners=False).sigmoid()
                for i in range(N):
                    if gts_f[i].sum() == 0:
                        continue
                    pred = (up[i, 0].cpu().numpy() > args.thresh).astype(np.uint8)
                    per_offset_carry[fidx].append(iou(gts_f[i], pred))

    print(f"{'offset':>8} {'memory':>10} {'carryover':>10} {'OT (gt-src)':>12}  n")
    for o in offsets:
        m = np.mean(per_offset_mem[o]) if per_offset_mem[o] else float("nan")
        c = np.mean(per_offset_carry[o]) if per_offset_carry[o] else float("nan")
        ot = _OT_BASELINE.get(o, float("nan"))
        print(f"{o:>8} {m:>10.4f} {c:>10.4f} {ot:>12.4f}  (n={len(per_offset_mem[o])})")


if __name__ == "__main__":
    main()
