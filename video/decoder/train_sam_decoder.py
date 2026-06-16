"""Fully-unsupervised mask-decoder trainer.

Architecture only — NO supervised weights anywhere:
  - Encoder:  DINOv3 ViT-L/16 (frozen, self-supervised) — NOT SAM's supervised ViT-H.
  - Decoder:  SAM's PromptEncoder + MaskDecoder *architecture*, RANDOMLY initialised
              (we reuse the design, not the SA-1B-trained weights).
  - Adapter:  1x1 conv mapping DINOv3's 1024-d patch features -> the decoder's
              expected 256-d, 64x64 image embedding (random init, trained).

Supervision is ONLY the CuVLER+conquer pseudo-masks (no human labels); box prompts
are derived from the pseudo-masks themselves. So the whole loop is label-free and
free of supervised pretraining — a genuinely unsupervised decoder.

    python -m video.decoder.train_sam_decoder \
        --coco datasets/davis_pseudo/annotations/train.json \
        --image-root datasets/davis_pseudo/images \
        --epochs 20 --out checkpoints/unsup_decoder.pth
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
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_util

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from video.features.dinov3_dense import DenseDINOv3  # noqa: E402

from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer  # noqa: E402

_IMG = 1024          # DINOv3 at 1024/patch16 -> 64x64 grid == SAM embedding size
_GRID = 64
_EMB = 256


class UnsupervisedDecoder(nn.Module):
    """DINOv3-fed, randomly-initialised SAM-architecture decoder."""

    def __init__(self):
        super().__init__()
        self.adapter = nn.Conv2d(1024, _EMB, kernel_size=1)
        self.prompt_encoder = PromptEncoder(
            embed_dim=_EMB,
            image_embedding_size=(_GRID, _GRID),
            input_image_size=(_IMG, _IMG),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            transformer_dim=_EMB,
            transformer=TwoWayTransformer(depth=2, embedding_dim=_EMB, mlp_dim=2048, num_heads=8),
            num_multimask_outputs=3,
        )
        # NOTE: every module above is randomly initialised — no SAM/SA-1B weights.

    def forward(self, dino_feats_chw, boxes_xyxy=None, mask_prompts=None, points=None):
        # dino_feats_chw: [1, 1024, 64, 64] (frozen);  boxes: [N, 4] in 1024-space
        # points: (coords [N, P, 2], labels [N, P]) click prompts in 1024-space, or None
        # mask_prompts: [N, 1, 256, 256] coarse masks to refine, or None (box-only)
        img_emb = self.adapter(dino_feats_chw)                      # [1, 256, 64, 64]
        sparse, dense = self.prompt_encoder(points=points, boxes=boxes_xyxy, masks=mask_prompts)
        low_res, _ = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return low_res                                              # [N, 1, 256, 256]


def sample_clicks(m, n, rng=np.random):
    """Sample `n` positive clicks from a binary mask, in the mask's native pixel coords.

    Click 0 is the most-interior point (distance-transform peak) — a stable "center"
    click that doesn't land on the noisy boundary; the rest are uniform interior pixels.
    This is the point-prompt analogue of deriving a box from the pseudo-mask: the only
    supervision is still the (label-free) pseudo-mask itself.
    """
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return None
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    fy, fx = np.unravel_index(int(dt.argmax()), dt.shape)
    pts = [(float(fx), float(fy))]
    for _ in range(n - 1):
        i = rng.randint(len(xs))
        pts.append((float(xs[i]), float(ys[i])))
    return pts


def degrade_mask(m):
    """Coarsen a crisp 256x256 mask into a blobby low-res proxy for a pseudo-mask.

    Refinement supervision: the decoder is given this degraded mask (+box) and must
    recover the crisp target. Feeding the *un*-degraded mask would just teach an
    identity copy, so we down-sample to a random coarse resolution and randomly
    erode/dilate the boundary — mimicking the errors of a real CuVLER/conquer mask.
    """
    s = np.random.randint(32, 96)                                   # random coarse res
    small = cv2.resize(m.astype(np.float32), (s, s), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (256, 256), interpolation=cv2.INTER_LINEAR)
    out = (up > 0.5).astype(np.uint8)
    k = np.random.randint(0, 8)                                     # random boundary error
    if k:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, ker) if np.random.rand() < 0.5 else cv2.erode(out, ker)
    return out.astype(np.float32)


def dice_loss(logits, targets, eps=1.0):
    p = logits.sigmoid().flatten(1)
    t = targets.flatten(1)
    num = 2 * (p * t).sum(1) + eps
    den = p.sum(1) + t.sum(1) + eps
    return (1 - num / den).mean()


def next_click(gt, pred):
    """SAM-style correction click: place it in the model's largest error region.

    Returns (x, y, label) in the prompt-encoder's 1024-space (gt/pred are 256-res), where
    label 1 = a positive click on a MISSED foreground region, label 0 = a NEGATIVE click on
    a FALSE-positive region. Negatives are the key signal a box can't give: a click on a
    leaked-in neighbour tells the decoder "not this instance". Picks the deficiency (false-
    neg vs false-pos) with larger area; the click lands on that region's most-interior pixel
    (distance-transform peak) so it isn't a noisy boundary pixel. None if no error remains.
    """
    fn = (gt > 0) & (pred == 0)                 # missed fg -> positive click
    fp = (gt == 0) & (pred > 0)                 # leaked fg -> negative click
    if fn.sum() >= fp.sum():
        region, lab = fn, 1.0
    else:
        region, lab = fp, 0.0
    if region.sum() == 0:
        return None
    pts = sample_clicks(region.astype(np.uint8), 1)
    if pts is None:
        return None
    x, y = pts[0]
    return x * 4.0, y * 4.0, lab                # 256 -> 1024


def iterative_point_step(model, feats, tgt, max_clicks, opt, cold_weight=1.0):
    """One iterative-click training step (SAM recipe). Returns mean per-iteration loss.

    Starts from a single positive click (mask centre), then for a random 1..max_clicks
    iterations re-predicts, samples a correction click (positive or NEGATIVE) from the
    current error region, and feeds the previous mask logits back as the dense prompt.
    Grads accumulate across iterations; opt.step() is called once at the end.

    cold_weight up-weights the iteration-0 (cold, single-click, no mask prompt) loss
    relative to the refinement iterations, to counteract the cold-click conservatism
    that the negative-click correction iterations otherwise induce.
    """
    N = tgt.shape[0]
    gt = (tgt[:, 0] > 0.5).cpu().numpy().astype(np.uint8)        # [N,256,256]
    coords, labels = [], []
    for i in range(N):
        pts = sample_clicks(gt[i], 1)
        x, y = pts[0] if pts else (128.0, 128.0)
        coords.append([[x * 4.0, y * 4.0]]); labels.append([1.0])

    n_iter = np.random.randint(1, max_clicks + 1)
    opt.zero_grad()
    prev, total = None, 0.0
    for it in range(n_iter):
        c = torch.tensor(coords, dtype=torch.float, device="cuda")
        l = torch.tensor(labels, dtype=torch.float, device="cuda")
        logits = model(feats, mask_prompts=prev, points=(c, l))
        loss = F.binary_cross_entropy_with_logits(logits, tgt) + dice_loss(logits, tgt)
        w = cold_weight if it == 0 else 1.0
        (loss * w).backward()
        total += loss.item()
        prev = logits.detach()
        if it < n_iter - 1:
            pred = (prev[:, 0].sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
            for i in range(N):
                nc = next_click(gt[i], pred[i])
                if nc is None:
                    coords[i].append(coords[i][-1]); labels[i].append(labels[i][-1])
                else:
                    x, y, lab = nc
                    coords[i].append([x, y]); labels[i].append(lab)
    opt.step()
    return total / n_iter


def load_coco(coco_json, image_root, refine_targets=False):
    """Each ann -> (input_seg, target_seg). With refine_targets, target_seg is the
    CascadePSP `refined` mask (precomputed); otherwise target == input (echo)."""
    d = json.load(open(coco_json))
    by_img = {im["id"]: {"file": str(Path(image_root) / im["file_name"]),
                         "h": im["height"], "w": im["width"], "anns": []}
              for im in d["images"]}
    for a in d["annotations"]:
        tgt = a["refined"] if (refine_targets and "refined" in a) else a["segmentation"]
        by_img[a["image_id"]]["anns"].append((a["segmentation"], tgt))
    return [v for v in by_img.values() if v["anns"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="datasets/davis_pseudo/annotations/train.json")
    ap.add_argument("--image-root", default="datasets/davis_pseudo/images")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--box-jitter", type=float, default=0.03)
    ap.add_argument("--prompt", choices=["box", "point"], default="box",
                    help="prompt type. 'point' = positive clicks sampled from each pseudo-mask "
                         "(the promptable-clicks model); 'box' = the v1..v3 box lineage.")
    ap.add_argument("--max-clicks", type=int, default=3,
                    help="point mode: each image uses a random 1..max-clicks positive clicks/mask, "
                         "so the model learns both single-click and multi-click prompting.")
    ap.add_argument("--cold-weight", type=float, default=1.0,
                    help="point mode: up-weight the cold (iteration-0, single-click, no mask "
                         "prompt) loss by this factor, to counteract the conservatism that "
                         "negative-click correction iterations otherwise induce on cold 1-click.")
    ap.add_argument("--max-masks", type=int, default=32, help="cap prompts/image for memory")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="train on a deterministic stride-subsample of the images (for data "
                         "learning-curve ablations); 0.25 -> every 4th image, spread across clips.")
    ap.add_argument("--cache-feats", action="store_true",
                    help="precompute frozen-DINOv3 feats once and reuse (~10x faster for many "
                         "epochs; ~16MB/image of CPU RAM). Safe: the encoder is frozen.")
    ap.add_argument("--use-mask-prompt", action="store_true",
                    help="feed a degraded copy of each pseudo-mask as a dense prompt and learn to "
                         "refine it (box+coarse-mask -> mask), instead of box-only. Still label-free.")
    ap.add_argument("--refine-targets", action="store_true",
                    help="implies --use-mask-prompt: feed the RAW pseudo-mask (un-degraded) as the "
                         "prompt and use the precomputed `refined` mask as the target, so the decoder "
                         "learns genuine refinement (raw -> refined) beyond pseudo quality. Requires a "
                         "--coco json with `refined` fields (e.g. make_grabcut_targets.py).")
    ap.add_argument("--target-tag", default="refined",
                    help="label recorded in the checkpoint for the refinement target source "
                         "(e.g. 'grabcut' — keep it honest about supervision: grabcut/densecrf are "
                         "unsupervised, cascadepsp is NOT).")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a prior UNSUPERVISED checkpoint (e.g. unsup_decoder_v2.pth); "
                         "omit to train from random init. Still label-free either way.")
    ap.add_argument("--out", default="checkpoints/unsup_decoder.pth")
    args = ap.parse_args()

    if args.refine_targets:
        args.use_mask_prompt = True            # refinement needs the mask channel

    dino = DenseDINOv3()                       # frozen SSL encoder
    for p in dino.backbone.parameters():
        p.requires_grad_(False)
    model = UnsupervisedDecoder().cuda()
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cuda")["model"])
        print(f"[init] warm-started from {args.init_from} (unsupervised checkpoint)")
    else:
        print("[init] random init (from scratch)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    data = load_coco(args.coco, args.image_root, refine_targets=args.refine_targets)
    if args.frac < 1.0:
        step = max(1, round(1.0 / args.frac))
        data = data[::step]                    # deterministic stride, spread across the set
        print(f"[frac] {args.frac} -> stride {step} -> {len(data)} images")
    print(f"[data] {len(data)} images, "
          f"{sum(len(d['anns']) for d in data)} pseudo-masks (label-free supervision)"
          + (f" | target={args.target_tag}-refined" if args.refine_targets else ""))
    print(f"[model] DINOv3 frozen | decoder params (trainable): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    if args.cache_feats:                       # frozen encoder -> compute feats once, reuse
        for d in data:
            rgb = cv2.cvtColor(cv2.imread(d["file"]), cv2.COLOR_BGR2RGB)
            img1024 = cv2.resize(rgb, (_IMG, _IMG), interpolation=cv2.INTER_LINEAR)
            with torch.no_grad():
                f = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None]
            d["_feats"] = f.float().cpu()       # [1,1024,64,64] on CPU (~16MB each)
        print(f"[cache] precomputed DINOv3 feats for {len(data)} images")

    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for d in data:
            H, W = d["h"], d["w"]
            if args.cache_feats:
                feats = d["_feats"].cuda()
            else:
                rgb = cv2.cvtColor(cv2.imread(d["file"]), cv2.COLOR_BGR2RGB)
                img1024 = cv2.resize(rgb, (_IMG, _IMG), interpolation=cv2.INTER_LINEAR)
                with torch.no_grad():
                    feats = dino.extract(img1024, normalize=False)["feats"].permute(2, 0, 1)[None].cuda().float()

            sx, sy = _IMG / W, _IMG / H
            boxes, inputs, targets = [], [], []
            for in_rle, tgt_rle in d["anns"][:args.max_masks]:
                m = mask_util.decode(in_rle)                           # prompt comes from input
                ys, xs = np.where(m > 0)
                if len(ys) == 0:
                    continue
                if args.prompt == "box":
                    x0, x1, y0, y1 = xs.min() * sx, xs.max() * sx, ys.min() * sy, ys.max() * sy
                    if args.box_jitter > 0:
                        jx, jy = (x1 - x0) * args.box_jitter, (y1 - y0) * args.box_jitter
                        x0 += np.random.uniform(-jx, jx); x1 += np.random.uniform(-jx, jx)
                        y0 += np.random.uniform(-jy, jy); y1 += np.random.uniform(-jy, jy)
                    boxes.append([x0, y0, x1, y1])
                inputs.append(cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST))
                tm = mask_util.decode(tgt_rle) if tgt_rle is not in_rle else m
                targets.append(cv2.resize(tm, (256, 256), interpolation=cv2.INTER_NEAREST))
            if not targets:
                continue
            tgt = torch.tensor(np.stack(targets), dtype=torch.float, device="cuda")[:, None]

            if args.prompt == "point":
                # iterative click correction (positive + negative clicks); steps opt internally
                tot += iterative_point_step(model, feats, tgt, args.max_clicks, opt, args.cold_weight); n += 1
                continue

            boxes_t = torch.tensor(boxes, dtype=torch.float, device="cuda")
            mask_prompts = None
            if args.use_mask_prompt:
                # refine mode: feed the raw input mask (target differs -> learns to refine).
                # echo mode:   degrade the mask so it can't trivially copy the (identical) target.
                src = inputs if args.refine_targets else [degrade_mask(t) for t in inputs]
                mask_prompts = torch.tensor(np.stack(src), dtype=torch.float, device="cuda")[:, None]

            logits = model(feats, boxes_t, mask_prompts)                # [N,1,256,256]
            loss = F.binary_cross_entropy_with_logits(logits, tgt) + dice_loss(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"[epoch {ep:3d}] loss {tot / max(n, 1):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "unsupervised": True,
                "encoder": "dinov3-ssl-frozen", "decoder_init": "random",
                "prompt": ("point" if args.prompt == "point" else
                           ("box+mask" if args.use_mask_prompt else "box")),
                "target": args.target_tag if args.refine_targets else "pseudo-mask",
                "cold_weight": args.cold_weight,
                "epochs": args.epochs, "final_loss": tot / max(n, 1)}, args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
