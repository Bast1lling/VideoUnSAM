# Memory-conditioned promptable video segmentation

This document describes the **memory variant** of the video model
([`model/video_memory_mask_dino.py`](model/video_memory_mask_dino.py)) — a second
way of extending the image-pretrained UnSAM / SemanticSAM model to video that
reuses **all** of UnSAM's weights and adds a SAM‑2‑style mask‑conditioned memory.

It is a sibling of the original temporal variant
([`model/video_mask_dino.py`](model/video_mask_dino.py)); both plug into the same
trainer ([`train_video.py`](train_video.py)) and produce the same output tensors,
so you can switch between them with a single flag (`--variant temporal|memory`).

---

## 1. The task

**Promptable video segmentation.** Given a short clip of `T` frames and a
point/box prompt that identifies an object in *one* frame (the *prompt frame*),
predict that object's mask in **every** frame. This is the video generalisation
of what UnSAM already does on a single image, and the same task SAM 2 solves.

---

## 2. Why the temporal variant under‑performs (recap)

The first variant adds a per‑query `TemporalSelfAttention` after each decoder
layer: each query token attends to *its own state* in the other frames.

Two structural weaknesses:

1. **Only query tokens carry cross‑frame information.** The backbone and pixel
   decoder run on each frame independently, so the *image features* of a
   non‑prompt frame know nothing about the prompt frame. A query has to
   re‑localise a moved/deformed object from a refined reference box plus a tiny
   "is‑this‑the‑prompt‑frame" embedding.
2. **The prompt frame's predicted mask is never fed forward.** The strongest
   available cue — *"here is exactly what the object looks like"* — is discarded.

The memory variant fixes both: it distills the prompt frame's prediction into a
small **memory bank** and lets the queries that generate every other frame's
mask **cross‑attend** to it.

---

## 3. Design overview

The pipeline is unchanged except for one new block per decoder layer:

```
  per frame:  backbone ─► pixel decoder ─► mask_features (B*T, d, h, w)     [frozen, reused]
                                  │
  decoder query prep (shared prompt replicated to all T frames)            [reused]
                                  │
  ┌─────────────────────────── decoder layer i (×9) ───────────────────────────┐
  │  spatial cross-attention        (per frame, identical to image model)  [reused] │
  │            ▼                                                                     │
  │  MASK-CONDITIONED MEMORY READ   (NEW: query frames attend to the prompt          │
  │            ▼                     frame's mask-conditioned memory bank)            │
  │  iterative box refinement       (unchanged)                            [reused] │
  │            ▼                                                                     │
  │  prediction heads               (unchanged; also yields the prompt-frame  [reused]│
  │                                  mask that conditions layer i+1's memory)        │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

Everything outside the **MASK‑CONDITIONED MEMORY READ** box is the original
UnSAM image model running frame‑by‑frame (with `B*T` as the batch size).

---

## 4. The memory bank

The memory module is
[`MaskConditionedMemoryAttention`](model/video_memory_mask_dino.py). For each
**prompt `p`** of each **video `b`**, it builds a tiny memory bank from the
**prompt frame** and lets the `M` mask‑token queries of `p` in every *other*
frame attend to it.

The bank holds `M + 2` tokens:

| token(s) | count | what it is | role |
|---|---|---|---|
| **object pointers** | `M` | the prompt frame's own query states for prompt `p` at this layer | "what was clicked" — identity / instance handle |
| **foreground prototype** | `1` | `mask_features` averaged **inside** the prompt frame's predicted mask for `p` | object appearance (mask‑conditioning) |
| **background prototype** | `1` | `mask_features` averaged **outside** that mask | what to suppress |

The fg/bg prototypes are the **dense‑mask‑prompt** signal: instead of adding a
dense mask to a single‑object image embedding (as SAM does), we pool the
mask‑weighted features into two appearance vectors. This is what makes the
memory *per‑prompt* and lets all `P` prompts share one forward pass (see §7).

Token‑type tags (`nn.Embedding(3, d)`: pointer / fg / bg) are added so attention
can tell the three roles apart. The prototype projections (`fg_proj`, `bg_proj`)
are small linear maps.

---

## 5. Depth‑causal conditioning

The fg/bg prototypes need a predicted mask, but a layer's mask is only known
*after* the layer runs. We resolve this by conditioning **layer `i` on the
prompt frame's mask predicted at layer `i‑1`**:

* at **layer 0** there is no previous mask → the bank is **object pointers only**;
* from **layer 1** on, the previous layer's prompt‑frame mask (mean over the `M`
  ambiguity candidates, `sigmoid`, detached) builds the fg/bg prototypes.

The prediction heads are therefore run on **every** layer in this variant (in
both train and eval) so the conditioning mask is always available — a small
extra cost over the image model's "last‑layer‑only at eval" shortcut.

The prompt frame itself is **never modified** by the memory read (its update is
masked out), so its prediction stays bit‑identical to the image model's. It is
the trusted reference the other frames propagate from.

---

## 6. Identity at initialisation & weight reuse

Each memory layer is **pre‑norm with a zero‑initialised attention output
projection** (`cross.out_proj` weight & bias = 0). At load time:

* the memory residual is exactly **zero**, so the video model reproduces the
  image model **frame‑by‑frame** — verified by the smoke test
  (`Identity-at-init max frame deviation: 0.000e+00`);
* the new behaviour is learned during finetuning; once `out_proj` moves off zero
  the rest of the block receives gradients normally (standard zero‑residual init,
  as in ControlNet/LoRA).

> This is a deliberate improvement over the temporal variant, whose attention is
> *active* at init and so does **not** reproduce the image model on load.

All new parameters live under `predictor.memory_layers.*`. An image checkpoint
loads with `strict=False`; the **only** missing keys are those memory layers
(99 tensors for the 9‑layer SwinT config). Loader + freezer:

* [`load_image_weights`](model/video_memory_mask_dino.py) — strips the
  `model.sem_seg_head.` prefix if present, warns only about genuinely missing
  (non‑`memory_layers`) keys.
* [`freeze_non_memory`](model/video_memory_mask_dino.py) — trains only
  `memory_layers.*` (≈3.56 M params, **12.3 %** of the head).

---

## 7. Multi‑prompt handling (why prototypes, not a dense mask)

UnSAM is **multi‑prompt**: one forward pass handles `P` instances at once, laid
out **prompt‑major** in the query dimension (`pad_size = P*M`, prompt `p` owns
rows `[p*M:(p+1)*M]`), with a block‑diagonal attention mask so prompts don't see
each other.

A literal SAM‑style dense mask prompt is added to the *single shared* image
embedding — but here the embedding is shared across all `P` prompts, so you
can't add `P` different masks to it. The **fg/bg prototype** formulation keeps
the conditioning **per‑prompt**: prompt `p`'s queries attend to a bank built from
*`p`'s own* predicted mask, and all `P` prompts are processed together by folding
`P` into the attention batch dimension. This preserves UnSAM's efficiency and
its ambiguity‑candidate (`M` masks per prompt) behaviour.

---

## 8. Tensor shapes & conventions

Identical conventions to the temporal variant:

* **video‑first batch ordering**: index `b*T + t` is frame `t` of video `b`.
* **prompt‑major queries**: `pad_size = P*M`, `M = num_mask_tokens`
  (`NUM_INTERACTIVE_TOKENS`, 6 in the SwinT config).
* `prompt_frame_idx`: `(B,)` long tensor; per video, which frame owns the prompt
  (defaults to frame 0). Supplied by the dataset
  ([`dataset.py`](dataset.py) `prompt_frame`) and threaded through the head.

Inside [`MaskConditionedMemoryAttention.forward`](model/video_memory_mask_dino.py):

```
output            (P*M, B*T, d)      decoder queries after spatial cross-attn
  └─ reshape ─►   (P, M, B, T, d)
object pointers   O[:, :, b, pf_b, :] → (P, M, B, d)         (gather prompt frame)
fg / bg proto     einsum(mask, mask_features) → (B, P, d)    (mask-pooled appearance)
memory bank       (M+2, P*B, d)
per query frame t (M, P*B, d) ──cross-attn──► residual add (prompt frame masked out)
```

Output contract is unchanged: `pred_masks (B*T, P*M, H, W)`,
`pred_ious (B*T, P, M)`, `pred_boxes (B*T, P*M, 4)`, plus `aux_outputs`.

---

## 9. What changed, file by file

| file | change |
|---|---|
| [`model/video_memory_mask_dino.py`](model/video_memory_mask_dino.py) | **new** — `MaskConditionedMemoryAttention`, `VideoMemoryIMaskDINODecoder`, `VideoMemoryIMaskDINOHead`, `freeze_non_memory`, `load_image_weights` |
| [`model/__init__.py`](model/__init__.py) | export the new classes/utilities |
| [`train_video.py`](train_video.py) | `--variant {temporal,memory}`; `_VARIANTS` table selects head class / loader / freezer; `build_model` and `set_trainable` take `variant` |
| [`configs/video_sam_swinT.yaml`](configs/video_sam_swinT.yaml) | `MEMORY_NHEAD`, `MEMORY_DROPOUT` keys |

Nothing in `promptable_segmentation/` (the image model) is modified — the memory
variant only *imports and subclasses* it.

---

## 10. How to train & run

**Train (memory layers only — recommended first step):**

```bash
python promptable_video_segmentation/train_video.py \
  --mask-dir /home/sebastian/data/imagenet1k/masksV1/validation \
  --variant memory \
  --frames 3 --epochs 5 --num-prompts 8 --batch-size 8 --amp \
  --image-size 512 --num-workers 8 \
  --output-dir promptable_video_segmentation/output/video_ft_memory \
  --wandb --wandb-project unsam-video-ft --wandb-run-name memory_v1
```

* `--variant memory` selects this architecture.
* `--unfreeze {temporal,heads,decoder}` controls *how much else* trains on top of
  the new layers (the base mode is still named `temporal` and means
  "only the variant's new layers" — for `memory` that is `memory_layers.*`).
  Use `heads` to also train the mask/iou/box heads, `decoder` for the whole
  transformer decoder.

**Sanity check (shapes, weight reuse, identity‑at‑init):**

```bash
python promptable_video_segmentation/model/video_memory_mask_dino.py
```

Expected: `memory_layers keys (identity init): 99`, no other missing keys, and
`Identity-at-init max frame deviation: 0.000e+00`.

---

## 11. Limitations & next steps

* **Still trained on synthetic clips.** The clips are `T` augmented views of one
  image ([`dataset.py`](dataset.py)); there is no real motion/occlusion. The
  memory architecture is the right *vehicle* for propagation, but its ceiling is
  capped by this data. Pairing it with real‑motion data (VideoCutLER‑style
  synthetic motion, or real VOS data like DAVIS/YouTube‑VOS) is the highest‑value
  follow‑up.
* **Single memory frame.** The bank is built from the prompt frame only. For
  longer clips, extend to a **running memory bank** over multiple past frames
  (true SAM‑2 streaming) — `pf` would become a set of memory‑frame indices and
  the prototypes/pointers would be concatenated across them.
* **Order‑agnostic.** Like the temporal variant, no frame‑order encoding is used
  (the synthetic clips are order‑free). Add temporal positional encodings if you
  move to ordered real video and want directional propagation.
* **Pooled (not dense) mask conditioning.** The fg/bg prototypes summarise the
  mask into two vectors. A spatially‑dense injection would be stronger but
  requires single‑object processing (see §7) or per‑prompt feature copies.
* **Evaluate on real benchmarks.** Training loss is computed on synthetic clips;
  report **J&F on DAVIS‑2017 val** for an honest measure.
