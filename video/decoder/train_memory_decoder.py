"""Windowed training for MemoryDecoder (memory_bank.py + memory_decoder.py).

Reuses datasets/davis_video_pseudo (generate_video_pseudo_dataset.py): 26 windows
of T=6 frames (stride 2) from the 26 train-split DAVIS clips, each with OT-tracked
CuVLER+conquer pseudo-masks. For each window, every track that survives all 6
frames is trained as one label-free sequence:

  - frame 0: 1..max-clicks positive clicks (sample_clicks) on the track's
    pseudo-mask, empty memory, BCE+Dice vs the pseudo-mask; write (k0,v0) into the
    anchor slot from the pseudo-mask (the only teacher-forced step).
  - frames 1..5: NO new prompts -- predict from memory alone, BCE+Dice vs that
    frame's track-matched pseudo-mask, plus a soft-IoU temporal-smoothing penalty
    against the previous frame's (detached) prediction; write (k_t,v_t) from the
    model's OWN soft prediction into the ring buffer.

Full backprop-through-time across the 6-frame chain (frozen DINOv3 features, so
the graph is just the ~4.3M-param decoder + memory modules -- cheap). One
optimizer step per window, after accumulating .backward() over all tracks in it.
The base decoder is warm-started from unsup_decoder_points_v3.pth; the memory
modules are random-init with a zero-init residual gate, so training starts at v3
behaviour and gradually learns to use memory.

    python -m video.decoder.train_memory_decoder \
        --coco datasets/davis_video_pseudo/annotations/train.json \
        --image-root datasets/davis_video_pseudo/images \
        --init-from checkpoints/unsup_decoder_points_v3.pth \
        --epochs 30 --cache-feats --out checkpoints/unsup_memory_decoder_v1.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from video.features.dinov3_dense import DenseDINOv3  # noqa: E402
from video.decoder.memory_decoder import MemoryDecoder  # noqa: E402
from video.decoder.train_sam_decoder import sample_clicks, dice_loss, _IMG  # noqa: E402


def load_video_coco(coco_json, image_root, max_tracks=4):
    """Returns a list of windows: {video_id, tracks: [track_id,...], frames: [...]}.

    Only tracks present (track_id != -1) in EVERY frame of a window are kept, so
    each is a clean T=6 sequence. Capped to `max_tracks` per window (smallest
    track_ids -- the earliest-detected, usually larger/more stable objects)."""
    d = json.load(open(coco_json))
    images = {im["id"]: im for im in d["images"]}
    by_image = defaultdict(list)
    for a in d["annotations"]:
        by_image[a["image_id"]].append(a)

    windows = defaultdict(list)
    for im in d["images"]:
        windows[im["video_id"]].append((im["window_pos"], im["id"]))

    out = []
    for vid, frames in windows.items():
        frames.sort()
        track_sets = [{a["track_id"] for a in by_image[img_id] if a["track_id"] != -1}
                       for _, img_id in frames]
        common = sorted(set.intersection(*track_sets)) if track_sets else []
        if not common:
            continue
        common = common[:max_tracks]

        frame_list = []
        for pos, img_id in frames:
            im = images[img_id]
            anns_by_track = {a["track_id"]: a["segmentation"]
                             for a in by_image[img_id] if a["track_id"] in common}
            frame_list.append({
                "file": str(Path(image_root) / im["file_name"]),
                "h": im["height"], "w": im["width"],
                "anns_by_track": anns_by_track,
            })
        out.append({"video_id": vid, "tracks": common, "frames": frame_list})
    return out


def soft_iou(a, b, eps=1.0):
    inter = (a * b).sum()
    union = (a + b - a * b).sum()
    return (inter + eps) / (union + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="datasets/davis_video_pseudo/annotations/train.json")
    ap.add_argument("--image-root", default="datasets/davis_video_pseudo/images")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--memory-lr", type=float, default=1e-3,
                     help="LR for memory_encoder/memory_reader params (incl. gate); "
                          "higher than --lr since these start near-inert.")
    ap.add_argument("--max-tracks", type=int, default=4,
                     help="cap on fully-tracked instances trained per window (memory cost).")
    ap.add_argument("--max-clicks", type=int, default=3,
                     help="frame-0 prompt uses a random 1..max-clicks positive clicks.")
    ap.add_argument("--smooth-weight", type=float, default=0.2,
                     help="weight of the temporal-smoothing soft-IoU penalty (frames 1..5).")
    ap.add_argument("--max-recent", type=int, default=3, help="MemoryBank ring-buffer size.")
    ap.add_argument("--init-from", default="checkpoints/unsup_decoder_points_v3.pth",
                     help="warm-start the base decoder from a prior unsupervised checkpoint.")
    ap.add_argument("--cache-feats", action="store_true",
                     help="precompute frozen-DINOv3 feats once and reuse (encoder is frozen).")
    ap.add_argument("--out", default="checkpoints/unsup_memory_decoder_v1.pth")
    args = ap.parse_args()

    dino = DenseDINOv3()
    for p in dino.backbone.parameters():
        p.requires_grad_(False)

    model = MemoryDecoder(max_recent=args.max_recent).cuda()
    if args.init_from:
        model.load_decoder_weights(args.init_from)
        print(f"[init] base decoder warm-started from {args.init_from}; "
              f"memory modules random-init (gate={model.memory_reader.gate.item():.3f})")
    memory_params = list(model.memory_encoder.parameters()) + list(model.memory_reader.parameters())
    memory_ids = {id(p) for p in memory_params}
    base_params = [p for p in model.parameters() if id(p) not in memory_ids]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": memory_params, "lr": args.memory_lr},
    ])

    data = load_video_coco(args.coco, args.image_root, max_tracks=args.max_tracks)
    n_tracks = sum(len(w["tracks"]) for w in data)
    print(f"[data] {len(data)} windows, {n_tracks} fully-tracked sequences "
          f"({len(data[0]['frames'])} frames each, label-free supervision)")
    print(f"[model] DINOv3 frozen | trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    if args.cache_feats:
        for w in data:
            for f in w["frames"]:
                rgb = cv2.cvtColor(cv2.imread(f["file"]), cv2.COLOR_BGR2RGB)
                img1024 = cv2.resize(rgb, (_IMG, _IMG), interpolation=cv2.INTER_LINEAR)
                with torch.no_grad():
                    feat = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None]
                f["_feats"] = feat.float().cpu()
        print(f"[cache] precomputed DINOv3 feats for "
              f"{sum(len(w['frames']) for w in data)} frames")

    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for w in data:
            opt.zero_grad()
            for track_id in w["tracks"]:
                mb = model.new_memory_bank()
                prev_pred = None
                track_loss = 0.0
                for pos, f in enumerate(w["frames"]):
                    if args.cache_feats:
                        feats = f["_feats"].cuda()
                    else:
                        rgb = cv2.cvtColor(cv2.imread(f["file"]), cv2.COLOR_BGR2RGB)
                        img1024 = cv2.resize(rgb, (_IMG, _IMG), interpolation=cv2.INTER_LINEAR)
                        with torch.no_grad():
                            feats = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()

                    m = mask_util.decode(f["anns_by_track"][track_id])
                    tgt256 = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
                    tgt = torch.tensor(tgt256, dtype=torch.float, device="cuda")[None, None]

                    if pos == 0:
                        n_clicks = np.random.randint(1, args.max_clicks + 1)
                        pts = sample_clicks(tgt256, n_clicks) or [(128.0, 128.0)]
                        coords = [[x * 4.0, y * 4.0] for x, y in pts]
                        c = torch.tensor([coords], dtype=torch.float, device="cuda")
                        l = torch.tensor([[1.0] * len(coords)], dtype=torch.float, device="cuda")
                        logits, img_emb = model(feats, memory_bank=mb, points=(c, l))
                    else:
                        logits, img_emb = model(feats, memory_bank=mb)

                    loss = F.binary_cross_entropy_with_logits(logits, tgt) + dice_loss(logits, tgt)
                    pred_soft = logits.sigmoid()
                    if pos > 0 and prev_pred is not None:
                        loss = loss + args.smooth_weight * (1.0 - soft_iou(pred_soft, prev_pred.detach()))
                    track_loss = track_loss + loss

                    mask_chan = tgt if pos == 0 else pred_soft
                    mask64 = F.interpolate(mask_chan, size=(64, 64), mode="bilinear", align_corners=False)
                    k, v = model.encode_memory(img_emb, mask64)
                    if pos == 0:
                        mb.write_anchor(k, v)
                    else:
                        mb.write_recent(k, v)
                    prev_pred = pred_soft

                track_loss.backward()
                tot += track_loss.item() / len(w["frames"])
                n += 1
            opt.step()
        print(f"[epoch {ep:3d}] loss {tot / max(n, 1):.4f}  gate {model.memory_reader.gate.item():.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "unsupervised": True,
                "encoder": "dinov3-ssl-frozen", "decoder_init": args.init_from,
                "memory_init": "random (zero-init gate)",
                "max_recent": args.max_recent, "max_tracks": args.max_tracks,
                "max_clicks": args.max_clicks, "smooth_weight": args.smooth_weight,
                "epochs": args.epochs, "final_loss": tot / max(n, 1)}, args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
