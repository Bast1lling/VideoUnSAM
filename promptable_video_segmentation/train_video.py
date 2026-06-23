"""
Finetune the VideoIMaskDINOHead on precomputed Divide-and-Conquer masks.

What this does
--------------
* Treats a directory of precomputed .npz masks (e.g. the validation masks from
  imagenet_precompute_parallel.py) as a labelled image pool and splits it into
  *virtual* train / val sets (by file, deterministically with --seed).
* Builds short synthetic video clips on the fly (DiCoVideoDataset): each clip is
  T independently augmented views of one image sharing the same N masks.
* Loads the pretrained UnSAM image model (baseline.pth) into the video model;
  only the new temporal-attention layers are trained by default (the spatial
  layers / pixel decoder / heads stay frozen — see --unfreeze).
* Trains with a SAM-style promptable loss: for each instance we sample one point
  prompt (from the reference frame, shared across frames, exactly as the video
  decoder replicates it).  The decoder emits num_mask_tokens candidate masks per
  prompt per frame; we supervise the best-matching candidate with focal+dice and
  an IoU-prediction MSE — no Hungarian matching / DN bookkeeping required.

Why a self-contained loss
-------------------------
The model is promptable (the prompt *defines* the target), so the prompt→mask
correspondence is known and the standard SAM "minimum over the ambiguity
candidates" loss is the correct, robust training signal here.  This avoids
depending on the image model's tightly-coupled SetCriterion / denoising path.

Run from the repo root, e.g.:
    python promptable_video_segmentation/train_video.py \
        --mask-dir /home/sebastian/data/imagenet1k/masksV1/validation \
        --weights  promptable_video_segmentation/ckpts/baseline.pth \
        --config   promptable_video_segmentation/configs/video_sam_swinT.yaml \
        --frames 2 --epochs 5 --output-dir promptable_video_segmentation/output/video_ft
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

# Use expandable CUDA segments so variable-size batches don't fragment the
# allocator — prevents the "OOM after a few iterations" with mixed image sizes.
# Must be set before torch initialises CUDA. (setdefault: user env still wins.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
# Order matters: promptable_segmentation must precede this package on sys.path so
# the bare ``from model.body…`` imports inside video_mask_dino.py resolve to
# promptable_segmentation/model (not promptable_video_segmentation/model).
# This mirrors gradio_video_seg.py's working setup.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
sys.path.insert(0, str(_THIS))                          # promptable_video_segmentation/
sys.path.insert(0, str(_REPO))                          # UnSAM/  (for the fully-qualified package imports)
sys.path.insert(0, str(_REPO / "promptable_segmentation"))  # ends up first → `model` resolves here

from promptable_video_segmentation.dataset import (
    DiCoVideoDataset,
    collate_video_clips,
    gather_npz,
)
from promptable_video_segmentation.model.video_mask_dino import (
    VideoIMaskDINOHead,
    load_image_weights,
    freeze_non_temporal,
)
from promptable_video_segmentation.model.video_memory_mask_dino import (
    VideoMemoryIMaskDINOHead,
    load_image_weights as load_image_weights_memory,
    freeze_non_memory,
)

# Map --variant → (head class, weight loader, base-freeze fn, new-param prefix).
# Both heads share the same forward I/O contract, so the train loop is identical.
_VARIANTS = {
    "temporal": (VideoIMaskDINOHead,       load_image_weights,        freeze_non_temporal, "temporal_layers"),
    "memory":   (VideoMemoryIMaskDINOHead, load_image_weights_memory, freeze_non_memory,   "memory_layers"),
}

_PRED_SWITCH = {"part": False, "whole": False, "seg": True, "det": True}


# ── model construction ────────────────────────────────────────────────────────

def _load_backbone_weights(backbone, weights_path: str, device: str) -> None:
    """
    Load backbone weights from the baseline checkpoint.

    The checkpoint stores keys as ``model.backbone.*`` (full GeneralizedMaskDINO
    layout); the swin auto-loader expects bare names, so we strip the prefix here
    — mirroring how ``load_image_weights`` strips ``model.sem_seg_head.`` for the
    head.
    """
    ckpt = torch.load(weights_path, map_location=device)
    sd = ckpt.get("model", ckpt)
    prefix = "model.backbone."
    bb_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not bb_sd:   # fall back to a bare-named swin checkpoint
        bb_sd = sd
    missing, _ = backbone.load_state_dict(bb_sd, strict=False)
    real_missing = [k for k in missing if "relative_position_index" not in k
                    and "attn_mask" not in k]   # non-persistent buffers
    if real_missing:
        print(f"  [warn] backbone: {len(real_missing)} keys missing "
              f"(first 5: {real_missing[:5]})")


def build_model(config_path: str, weights_path: str, device: str, variant: str = "temporal"):
    """Build backbone + (no) language encoder + video head, load baseline weights."""
    from promptable_segmentation.utils.arguments import load_opt_from_config_file
    from promptable_segmentation.model.backbone import build_backbone
    from semantic_sam.language import build_language_encoder

    head_cls, load_weights, _, new_prefix = _VARIANTS[variant]

    opt = load_opt_from_config_file(config_path)

    # Skip the swin auto-loader: baseline.pth keys are prefixed (model.backbone.*)
    # so it would silently leave the backbone random. We load it manually below.
    opt["MODEL"]["BACKBONE"]["LOAD_PRETRAINED"] = False

    print("Building backbone …")
    backbone = build_backbone(opt)
    backbone.to(device)

    lang_encoder = build_language_encoder(opt)   # None for TEXT.ARCH: noencoder
    if lang_encoder is not None:
        lang_encoder.to(device).eval()

    print(f"Building {head_cls.__name__} (variant='{variant}') …")
    extra = {"task_switch": {"bbox": True, "mask": True}}
    head = head_cls(opt, backbone.output_shape(), lang_encoder, extra).to(device)

    if os.path.exists(weights_path):
        print(f"Loading baseline weights from {weights_path} …")
        _load_backbone_weights(backbone, weights_path, device)
        missing, _ = load_weights(head, weights_path, map_location=device)
        n_other = sum(1 for k in missing if new_prefix not in k)
        if n_other:
            other = [k for k in missing if new_prefix not in k][:5]
            print(f"  [warn] {n_other} non-{new_prefix} head keys missing (first 5: {other})")
    else:
        print(f"[warn] weights not found at {weights_path}; using random init.")

    num_mask_tokens = head.predictor.num_mask_tokens
    return backbone, head, opt, num_mask_tokens


def set_trainable(head: torch.nn.Module, mode: str, variant: str = "temporal") -> None:
    """Select which parameters to finetune."""
    _VARIANTS[variant][2](head)   # baseline: only the variant's new layers (temporal/memory)
    if mode in ("heads", "decoder"):
        for name, p in head.named_parameters():
            if any(k in name for k in ("mask_embed", "iou_prediction_head",
                                       "bbox_embed", "class_embed")):
                p.requires_grad = True
    if mode == "decoder":
        for name, p in head.named_parameters():
            if name.startswith("predictor."):
                p.requires_grad = True


# ── prompt / target construction ──────────────────────────────────────────────

def _sample_point(mask_hw: torch.Tensor) -> tuple[int, int]:
    """Return (x, y) of a random foreground pixel; falls back to the center."""
    ys, xs = torch.nonzero(mask_hw, as_tuple=True)
    if len(ys) == 0:
        h, w = mask_hw.shape
        return w // 2, h // 2
    k = random.randint(0, len(ys) - 1)
    return int(xs[k]), int(ys[k])


def build_video_target(
    masks_b: torch.Tensor,   # (T, N, Hp, Wp) bool
    num_prompts: int,
    Hp: int,
    Wp: int,
    device: str,
    prompt_frame: "int | None" = None,
):
    """
    Build one head-target dict (shared prompts), the instance index each prompt
    refers to, and which frame the prompt was anchored in.

    The prompt point is sampled from frame *prompt_frame* (random by default) and
    replicated across all frames by the decoder; the temporal layers' prompt-frame
    marker tells the model which frame owns it, so non-prompt frames can
    re-localise the object.  Randomising the prompt frame during training teaches
    a general "propagate from the prompt frame" behaviour and lets you prompt any
    frame at inference.

    Returns (target_dict, inst_idx LongTensor[P], prompt_frame int).
    """
    T, N = masks_b.shape[0], masks_b.shape[1]
    if prompt_frame is None:
        prompt_frame = random.randrange(T)

    if N >= num_prompts:
        inst = random.sample(range(N), num_prompts)
    else:
        inst = [random.randrange(N) for _ in range(num_prompts)]
    inst_idx = torch.tensor(inst, dtype=torch.long)

    ref = masks_b[prompt_frame]   # (N, Hp, Wp)
    pts = []
    for i in inst:
        x, y = _sample_point(ref[i])
        # normalised cxcywh; a 6px click box behaves as a point (matches
        # prepare_targets_interactive in the image model).
        pts.append([x / Wp, y / Hp, 6.0 / Wp, 6.0 / Hp])
    points = torch.tensor(pts, dtype=torch.float, device=device)   # (P, 4)

    target = {
        "points":    points,
        "boxes_dn":  points.clone(),
        "pb":        torch.ones(num_prompts, device=device),
        "box_start": num_prompts,
    }
    return target, inst_idx, prompt_frame


# ── losses ──────────────────────────────────────────────────────────────────────

def _dice_loss(prob: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """prob, tgt: (..., HW). Returns (...,) dice loss."""
    num = 2.0 * (prob * tgt).sum(-1)
    den = prob.sum(-1) + tgt.sum(-1)
    return 1.0 - (num + 1.0) / (den + 1.0)


def compute_sam_loss(
    pred_masks: torch.Tensor,   # (B*T, P*M, h, w)
    pred_ious:  torch.Tensor,   # (B*T, P, M)
    masks_list: list,           # B tensors (T, N_b, Hp, Wp) bool
    inst_list:  list,           # B LongTensors (P,)
    B: int, T: int, M: int,
    weights: dict,
):
    """
    SAM-style promptable loss: per (frame, prompt) supervise the best of the M
    ambiguity candidates with focal+dice; supervise that candidate's IoU pred.
    Returns (total_loss, logging_dict).
    """
    P = inst_list[0].shape[0]
    h, w = pred_masks.shape[-2:]
    pm = pred_masks.view(B, T, P, M, h, w)
    pi = pred_ious.view(B, T, P, M)

    mask_w, dice_w, iou_w = weights["mask"], weights["dice"], weights["iou"]

    tot_mask = pm.new_zeros(())
    tot_iou  = pm.new_zeros(())
    # Metric accumulators kept on-GPU as tensors so there is no per-video CUDA
    # sync; everything is read out with a single .tolist() at the very end.
    present_iou_sum = pm.new_zeros(())   # Σ IoU over present (frame,prompt) pairs
    present_cnt     = pm.new_zeros(())   # # present pairs
    empty_ok_sum    = pm.new_zeros(())   # # absent pairs left (near-)empty
    empty_cnt       = pm.new_zeros(())   # # absent pairs
    n_terms = 0

    for b in range(B):
        inst = inst_list[b]                                    # cpu indices
        gt = masks_list[b][:, inst].to(pm.device).float()      # (T, P, Hp, Wp)
        gt_lr = F.interpolate(
            gt.reshape(T * P, 1, *gt.shape[-2:]), size=(h, w), mode="area"
        ).reshape(T, P, h, w)
        gt_lr = (gt_lr > 0.5).float()                          # (T, P, h, w)

        logits = pm[b]                                         # (T, P, M, h, w)
        tgt = gt_lr.unsqueeze(2).expand(-1, -1, M, -1, -1)     # (T, P, M, h, w)

        bce = F.binary_cross_entropy_with_logits(
            logits, tgt, reduction="none"
        ).flatten(3).mean(-1)                                  # (T, P, M)
        prob = logits.sigmoid().flatten(3)                     # (T, P, M, hw)
        dice = _dice_loss(prob, tgt.flatten(3))                # (T, P, M)

        cand = mask_w * bce + dice_w * dice                    # (T, P, M)
        best_val, best_idx = cand.min(dim=-1)                  # (T, P)
        tot_mask = tot_mask + best_val.mean()

        # IoU supervision on the chosen candidate.
        with torch.no_grad():
            pred_bin = (prob > 0.5).float()                    # (T, P, M, hw)
            inter = (pred_bin * tgt.flatten(3)).sum(-1)
            union = pred_bin.sum(-1) + tgt.flatten(3).sum(-1) - inter
            real_iou = inter / union.clamp(min=1.0)            # (T, P, M)
        chosen_iou = real_iou.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
        pred_iou   = pi[b].gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
        tot_iou = tot_iou + F.mse_loss(pred_iou, chosen_iou)
        n_terms += 1

        # Split metrics, accumulated as tensors (no branch on GPU values).
        with torch.no_grad():
            present = (gt_lr.flatten(2).sum(-1) > 0).float()   # (T, P)
            absent  = 1.0 - present
            present_iou_sum = present_iou_sum + (chosen_iou * present).sum()
            present_cnt     = present_cnt + present.sum()
            chosen_pred = pred_bin.gather(
                2, best_idx[..., None, None].expand(-1, -1, 1, pred_bin.shape[-1])
            ).squeeze(2)                                       # (T, P, hw)
            # "correct" if the chosen mask is essentially empty (<1% of hw)
            correct_empty = (chosen_pred.sum(-1) < 0.01 * (h * w)).float()
            empty_ok_sum = empty_ok_sum + (correct_empty * absent).sum()
            empty_cnt    = empty_cnt + absent.sum()

    tot_mask = tot_mask / max(n_terms, 1)
    tot_iou  = tot_iou / max(n_terms, 1)
    total = tot_mask + iou_w * tot_iou

    # Single GPU→CPU sync for all logged scalars.
    packed = torch.stack([
        total.detach(), tot_mask.detach(), tot_iou.detach(),
        present_iou_sum, present_cnt, empty_ok_sum, empty_cnt,
    ]).tolist()
    loss_v, mask_v, iou_v, p_sum, p_cnt, e_sum, e_cnt = packed
    log = {
        "loss": loss_v, "mask": mask_v, "iou": iou_v,
        "present_iou": p_sum / max(p_cnt, 1.0),
        "empty_acc":   e_sum / max(e_cnt, 1.0),
    }
    return total, log


# ── one forward+loss step ───────────────────────────────────────────────────────

def run_step(backbone, head, batch, T, M, num_prompts, weights, device, amp_enabled):
    images = batch["images"].to(device)                 # (B, T, 3, Hp, Wp)
    B = images.shape[0]
    Hp, Wp = images.shape[-2:]
    frames = images.reshape(B * T, 3, Hp, Wp)           # video-first (b, t)

    # The dataset guarantees each clip's instances are visible in its prompt
    # frame (and possibly absent elsewhere), so anchor the prompt there.
    batch_pf = batch.get("prompt_frames", [0] * B)
    targets, inst_list, pf_list = [], [], []
    for b in range(B):
        tgt, inst, pf = build_video_target(
            batch["masks"][b], num_prompts, Hp, Wp, device, prompt_frame=batch_pf[b]
        )
        targets.append(tgt)
        inst_list.append(inst)
        pf_list.append(pf)
    prompt_frame_idx = torch.tensor(pf_list, dtype=torch.long, device=device)

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        features = backbone(frames)
        out, _ = head(
            features, targets=targets, task="sam",
            extra=_PRED_SWITCH, num_frames=T,
            prompt_frame_idx=prompt_frame_idx,
        )
        loss, log = compute_sam_loss(
            out["pred_masks"], out["pred_ious"],
            batch["masks"], inst_list, B, T, M, weights,
        )
    return loss, log


# ── train / eval epochs ───────────────────────────────────────────────────────

_BAR_KEYS = ("loss", "mask", "iou", "present_iou", "empty_acc")


def train_one_epoch(backbone, head, loader, optimizer, scaler, T, M, num_prompts,
                    weights, device, amp_enabled, grad_clip, epoch,
                    wb=None, global_step=0):
    head.train()
    running = {}
    n = 0
    pbar = tqdm(loader, desc=f"epoch {epoch} [train]", dynamic_ncols=True, leave=True)
    for batch in pbar:
        if not batch:               # entire batch had empty masks
            continue
        loss, log = run_step(backbone, head, batch, T, M, num_prompts,
                             weights, device, amp_enabled)

        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in head.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in head.parameters() if p.requires_grad], grad_clip)
            optimizer.step()

        for k, v in log.items():
            running[k] = running.get(k, 0.0) + v
        n += 1
        global_step += 1
        if wb is not None:
            wb.log({f"train/{k}": v for k, v in log.items()}
                   | {"train/lr": optimizer.param_groups[0]["lr"], "epoch": epoch},
                   step=global_step)
        # Single-line progress bar showing running-average metrics.
        pbar.set_postfix({k: f"{running[k] / n:.3f}"
                          for k in _BAR_KEYS if k in running})
    pbar.close()
    return {k: v / max(n, 1) for k, v in running.items()}, global_step


@torch.no_grad()
def evaluate(backbone, head, loader, T, M, num_prompts, weights, device, amp_enabled,
             epoch=None):
    head.eval()
    running = {}
    n = 0
    desc = f"epoch {epoch} [val]" if epoch is not None else "val"
    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
    for batch in pbar:
        if not batch:
            continue
        _, log = run_step(backbone, head, batch, T, M, num_prompts,
                          weights, device, amp_enabled)
        for k, v in log.items():
            running[k] = running.get(k, 0.0) + v
        n += 1
        pbar.set_postfix({k: f"{running[k] / n:.3f}"
                          for k in _BAR_KEYS if k in running})
    pbar.close()
    return {k: v / max(n, 1) for k, v in running.items()}


# ── checkpoint ──────────────────────────────────────────────────────────────────

def save_checkpoint(head, optimizer, epoch, path, full: bool = False):
    if full:
        state = head.state_dict()
    else:   # only the trainable params (compact; temporal layers + any unfrozen)
        trainable_names = {n for n, p in head.named_parameters() if p.requires_grad}
        state = {k: v for k, v in head.state_dict().items() if k in trainable_names}
    torch.save({"model": state, "epoch": epoch,
                "optimizer": optimizer.state_dict()}, path)
    print(f"  saved checkpoint → {path}")


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-dir", required=True,
        help="Directory of precomputed .npz masks (searched recursively).")
    ap.add_argument("--weights", default=str(_THIS / "ckpts" / "baseline.pth"))
    ap.add_argument("--config",  default=str(_THIS / "configs" / "video_sam_swinT.yaml"))
    ap.add_argument("--output-dir", default=str(_THIS / "output" / "memory_video_ft"))
    ap.add_argument("--ckpt-dir", default=None,
        help="folder to save every epoch's checkpoint into "
             "(default: <output-dir>/checkpoints)")

    ap.add_argument("--frames", type=int, default=2, help="frames per clip (T)")
    ap.add_argument("--batch-size", type=int, default=1, help="videos per batch (B)")
    ap.add_argument("--num-prompts", type=int, default=8,
        help="point prompts (instances) sampled per clip")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=0.01)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--prefetch-factor", type=int, default=4,
        help="batches each worker prefetches (DataLoader); higher hides I/O jitter")
    ap.add_argument("--max-masks", type=int, default=24,
        help="cap the mask pool per clip before augmentation (median ~60/file). "
             "Only a few become prompts, so augmenting all of them is wasted CPU. "
             "Keep >= --num-prompts; 0 disables the cap.")
    ap.add_argument("--image-size", type=int, default=512,
        help="resize every frame to this fixed square size (multiple of 32). "
             "Uniform sizes remove collate-padding waste, stabilise GPU memory, "
             "and speed up cuDNN. Use 0 to keep each image's native resolution.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", choices=["temporal", "memory"], default="temporal",
        help="temporal=per-query TemporalSelfAttention (video_mask_dino); "
             "memory=mask-conditioned memory read (video_memory_mask_dino)")
    ap.add_argument("--unfreeze", choices=["temporal", "heads", "decoder"],
        default="temporal",
        help="temporal=only the variant's new (temporal/memory) layers; "
             "heads=+mask/iou/box heads; decoder=+entire transformer decoder")
    ap.add_argument("--use-divide", action="store_true", default=True)
    ap.add_argument("--no-conquer", dest="use_conquer", action="store_false", default=True)
    ap.add_argument("--crop-min", type=float, default=0.5)
    ap.add_argument("--max-size", type=int, default=720)
    ap.add_argument("--require-all-frames", action="store_true",
        help="legacy: only keep instances visible in EVERY frame. Default (off) "
             "keeps instances visible in the prompt frame and lets them be empty "
             "elsewhere, so the model learns to predict empty masks when the "
             "object leaves the frame.")
    ap.add_argument("--amp", action="store_true", help="enable mixed precision")
    ap.add_argument("--limit", type=int, default=0,
        help="cap number of .npz files used (0 = all); handy for a quick smoke test")
    # ── Weights & Biases ────────────────────────────────────────────────────────
    ap.add_argument("--wandb", action="store_true",
        help="log losses/metrics to Weights & Biases")
    ap.add_argument("--wandb-project", default="unsam-video-ft")
    ap.add_argument("--wandb-entity", default=None,
        help="W&B team/user (defaults to your logged-in default)")
    ap.add_argument("--wandb-run-name", default=None,
        help="name for this run (defaults to a W&B-generated name)")
    ap.add_argument("--wandb-mode", default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode; 'offline' logs locally to sync later")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This model requires CUDA (the decoder allocates .cuda() tensors).")
    device = "cuda"
    # With a fixed --image-size every batch is the same shape, so let cuDNN
    # autotune (and cache) the fastest conv kernels instead of re-picking each
    # step.  Disabled for variable sizes, where it would re-benchmark constantly.
    torch.backends.cudnn.benchmark = args.image_size > 0

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = args.ckpt_dir or os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── data split ──────────────────────────────────────────────────────────────
    files = gather_npz(args.mask_dir, recursive=True)
    if not files:
        raise SystemExit(f"No .npz files found under {args.mask_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.limit:
        files = files[:args.limit]
    n_val = max(1, int(len(files) * args.val_fraction))
    val_files, train_files = files[:n_val], files[n_val:]
    print(f"Dataset: {len(files)} clips → train={len(train_files)}  val={len(val_files)}")

    ds_kwargs = dict(
        num_frames=args.frames, use_divide=args.use_divide,
        use_conquer=args.use_conquer, crop_min=args.crop_min, max_size=args.max_size,
        require_all_frames=args.require_all_frames,
        image_size=(args.image_size if args.image_size > 0 else None),
        max_masks=(args.max_masks if args.max_masks > 0 else None),
    )
    train_ds = DiCoVideoDataset(entries=train_files, **ds_kwargs)
    val_ds   = DiCoVideoDataset(entries=val_files,   **ds_kwargs)

    collate = lambda b: collate_video_clips(b, size_divisibility=32)
    # pin_memory + persistent workers + deeper prefetch keep the (CPU-heavy)
    # augmentation pipeline ahead of the GPU so it doesn't stall between batches.
    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=collate,
                         pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True,
                             prefetch_factor=args.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)

    # ── model ─────────────────────────────────────────────────────────────────────
    backbone, head, opt, M = build_model(args.config, args.weights, device, args.variant)
    backbone.eval()   # backbone is frozen throughout
    for p in backbone.parameters():
        p.requires_grad = False
    set_trainable(head, args.unfreeze, args.variant)

    trainable = [p for p in head.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in head.parameters())
    print(f"Trainable head params: {n_train:,} / {n_total:,} "
          f"({100.*n_train/max(n_total,1):.2f}%) — mode='{args.unfreeze}'")

    dec = opt["MODEL"]["DECODER"]
    weights = {"mask": dec["MASK_WEIGHT"], "dice": dec["DICE_WEIGHT"],
               "iou": dec["IOU_WEIGHT"]}

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ── Weights & Biases ────────────────────────────────────────────────────────
    wb = None
    if args.wandb:
        try:
            import wandb
            cfg = dict(vars(args))
            cfg.update(n_train_params=n_train, n_total_params=n_total,
                       n_train_clips=len(train_files), n_val_clips=len(val_files),
                       loss_weights=weights)
            wb = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.wandb_run_name, mode=args.wandb_mode, config=cfg,
            )
            print(f"W&B logging to project '{args.wandb_project}' "
                  f"(mode={args.wandb_mode}).")
        except Exception as exc:   # noqa: BLE001
            print(f"[warn] W&B init failed ({exc}); continuing without it.")
            wb = None

    # ── training loop ─────────────────────────────────────────────────────────────
    best_val = float("inf")
    best_epoch = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        tr, global_step = train_one_epoch(
            backbone, head, train_loader, optimizer, scaler,
            args.frames, M, args.num_prompts, weights, device,
            args.amp, args.grad_clip, epoch,
            wb=wb, global_step=global_step,
        )
        va = evaluate(backbone, head, val_loader, args.frames, M, args.num_prompts,
                      weights, device, args.amp, epoch=epoch)
        print(f"[epoch {epoch}] "
              f"train: " + " ".join(f"{k}={v:.4f}" for k, v in tr.items())
              + "  |  val: " + " ".join(f"{k}={v:.4f}" for k, v in va.items()))

        if wb is not None:
            wb.log(
                {f"train_epoch/{k}": v for k, v in tr.items()}
                | {f"val/{k}": v for k, v in va.items()}
                | {"epoch": epoch},
                step=global_step,
            )

        # Save every epoch's checkpoint into the dedicated folder.
        save_checkpoint(head, optimizer, epoch,
                        os.path.join(ckpt_dir, f"epoch_{epoch:02d}.pth"))
        if va.get("loss", float("inf")) < best_val:
            best_val = va["loss"]
            best_epoch = epoch
            if wb is not None:
                wb.summary["best_val_loss"] = best_val
                wb.summary["best_epoch"] = epoch

    print(f"Done. Best val loss: {best_val:.4f} (epoch {best_epoch})  "
          f"(checkpoints in {ckpt_dir})")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()


"""
python promptable_video_segmentation/train_video.py \
  --mask-dir /home/sebastian/data/imagenet1k/masksV1/validation \
  --output-dir promptable_video_segmentation/output/video_ft_memory \
  --frames 3 --epochs 5 --image-size 512 --batch-size 8 --amp \
  --num-workers 14 --prefetch-factor 6 --max-masks 8 \
  --variant memory --unfreeze decoder \
  --wandb --wandb-project unsam-video-ft --wandb-run-name v1_memory

  
python promptable_video_segmentation/train_video.py \
  --mask-dir /home/sebastian/data/imagenet1k/masksV1/validation \
  --output-dir promptable_video_segmentation/output/video_ft \
  --frames 3 --epochs 5 \
  --image-size 512 --batch-size 8 --amp --num-workers 8 \
  --wandb --wandb-project unsam-video-ft --wandb-run-name v1_temporal \
  --unfreeze decoder

"""