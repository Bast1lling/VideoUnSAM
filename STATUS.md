# VideoUnSAM — Project Status

> Last updated: 2026-06-16 (evening)

---

## What we're building

An **unsupervised video object segmentation** pipeline that requires no human annotations
and no class labels. A user clicks once on frame 0; the system produces a high-quality
mask for every frame of the clip. No oracle labels are used at any stage except for
evaluation against DAVIS ground truth.

---

## Architecture overview

```
Frame 0 (user click)
        │
        ▼
┌─────────────────────┐
│  1. DIVIDE          │  CuVLER (frozen CascadeMaskRCNN trained on VoteCut pseudo-masks)
│  cuvler_divide.py   │  → class-agnostic region proposals
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. CONQUER         │  DINOv3 spectral merging within each CuVLER bbox (--conquer flag)
│  video/divide/      │  Expands 1 coarse proposal → 30+ tight sub-masks via iterative_merge
│  conquer.py         │  Critical for crowded scenes (breakdance: 0.141 → 0.727 mean IoU)
└────────┬────────────┘
         │ pick sub-mask containing click point, highest IoU
         ▼
┌─────────────────────┐
│  3. SEED MASK       │  Best conquer sub-mask as frame-0 seed
│                     │  dog: 0.84 IoU, breakdance: 0.744 IoU
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. OT PROPAGATION  │  Sinkhorn optimal-transport on DINOv3 64×64 patch features
│  sinkhorn_ot.py     │  chains frame-to-frame, drift ≈ 0.10 IoU / 30 frames
└────────┬────────────┘
         │ every N frames (default: 10)
         ▼
┌─────────────────────┐
│  5. PERIODIC RESEED │  Re-run CuVLER + conquer on current frame; accept best sub-mask
│  propagate_reseed   │  if IoU overlap with OT mask ≥ thresh (default: 0.3)
│                     │  Re-seed shown in orange in output video
└────────┬────────────┘
         │ optional post-processing per frame
         ▼
┌─────────────────────┐
│  6. BOUNDARY REFINE │  CascadePSP applied to a tight crop around each mask
│  (--refine flag)    │  camel: +0.12, breakdance: +0.06 mean IoU
└─────────────────────┘
```

### Key components

| Component | File | Notes |
|---|---|---|
| DINOv3 features | `video/features/dinov3_dense.py` | ViT-L/16, 1024→64×64 patch grid |
| Sinkhorn OT | `video/propagation/sinkhorn_ot.py` | `propagate_patch`, blur=0.05 |
| CuVLER divider | `video/divide/cuvler_divide.py` | CascadeMaskRCNN, score_thresh=0.35 |
| Conquer stage | `video/divide/conquer.py` | DINOv3 spectral merging within CuVLER bboxes |
| CascadePSP refiner | `divide_and_conquer/cascadepsp.py` | crop-based boundary sharpening |
| DAVIS loader | `video/loaders/davis.py` | 480p, 64-clip leak-free eval split |
| Main pipeline | `video/scripts/propagate_reseed.py` | CLI, video output, H.264 encoded |

### Discarded approaches (with reasons)

| Approach | Result | Why discarded |
|---|---|---|
| DINOv3 spectral cut as Divide | AR 0.00 | CuVLER gives AR 0.37; spectral cut can't compete |
| Unsupervised SAM decoder (box/mask prompt) | mIoU 0.52 held-out | Near-saturated — more data didn't help, needed better targets |
| Decoder mask-prompt refinement in the OT loop | Degrades OT signal | Decoder wasn't trained for temporal reasoning; OT alone better |
| KV memory bank (learned gating) | Gate stuck at ~0 | Underperforms plain OT and carryover baselines |
| Chained OT with sharpening (mass^4) | Collapse to 0.057 mIoU | Sharpening destroys the soft mass distribution; unsharpened chain ≈ direct |

---

## Quantitative results

### OT propagation baselines (GT seed, DAVIS 20 clips)

| Offset | mIoU | median |
|---|---|---|
| 1 frame | 0.771 | 0.849 |
| 5 frames | 0.727 | 0.806 |
| 10 frames | 0.647 | 0.753 |
| 20 frames | 0.604 | 0.691 |
| 30 frames | 0.529 | 0.612 |

### CuVLER seed vs decoder seed (dog clip, 60 frames)

| Seed | Frame-0 IoU | Mean IoU / 60 frames |
|---|---|---|
| 3-click unsup decoder | 0.629 | 0.622 |
| CuVLER+conquer | 0.842 | 0.814 |
| CuVLER + reseed every 10f | 0.842 | **0.862** |

### Conquer stage impact (breakdance clip, 84 frames, crowded scene)

| Pipeline | Mean IoU | Median |
|---|---|---|
| CuVLER only (no conquer) | 0.141 | 0.135 |
| + conquer stage | 0.727 | 0.733 |
| + conquer + CascadePSP refine | **0.787** | **0.806** |

Conquer expands CuVLER's single coarse "dancer + crowd" blob into 32 sub-masks; the
spectral cut isolates the dancer (seed IoU 0.744). Without conquer, CuVLER and OT both
agreed on the wrong (crowd-inclusive) region — re-seeding reinforced the error.

### CascadePSP boundary refinement (camel clip, 90 frames)

| | Mean IoU | Frame-0 IoU |
|---|---|---|
| Raw OT + reseed | 0.767 | 0.845 |
| + CascadePSP refine | **0.890** | 0.955 |

### Decoder (click-only, label-free, 64-clip held-out eval)

| Prompt | mIoU |
|---|---|
| 1 click | 0.359 |
| 3 clicks | 0.489 |
| 5 clicks | 0.529 |
| Box | 0.538 |

---

## Current pipeline script

```bash
python -m video.scripts.propagate_reseed \
    --clip breakdance \
    --instance-id 1 \
    --reseed-interval 10 \
    --reseed-thresh 0.3 \
    --conquer \
    --refine \
    --out video/reseed_breakdance_conquer_refined.mp4
```

Key flags:

| Flag | Default | Effect |
|---|---|---|
| `--reseed-interval` | 10 | Re-run CuVLER every N frames (0 = disabled) |
| `--reseed-thresh` | 0.3 | Min IoU to accept a new CuVLER proposal |
| `--conquer` | off | DINOv3 spectral sub-mask generation within CuVLER bboxes |
| `--refine` | off | Apply CascadePSP boundary sharpening per frame |
| `--crop-context` | 0.0 | Guided crop expansion around OT mask bbox at keyframes |
| `--click-crop-radius` | 0 | Fixed-pixel crop radius around frame-0 click (helps tiny objects) |
| `--cuvler-score` | 0.35 | CuVLER proposal score threshold |

Output: 2-panel (raw OT blue | GT green) or 3-panel (raw blue | refined purple | GT green) H.264 mp4.

---

## Known limitations / hard cases

### Fast-moving / small objects
OT's 64×64 patch grid can't track objects that move more than ~1 patch per frame.
Tested on **drone** (71×36px, very fast): frame-0 seed IoU 0.282 with 120×120 crop,
but OT loses it by frame 15. Periodic re-seeding helps if CuVLER re-detects the object.

### Thin/unusual shapes
**Kite-surf**: wide thin kite. CuVLER proposals are blobs; even guided crop gives only 0.156 seed IoU.
OT then propagates a bad seed. Boundary refinement can't fix a fundamentally wrong region.

### Re-seed quality variance
Re-seeding is not always beneficial. On the dog clip, frame 40 reseed (IoU overlap 0.787,
only 2 proposals) slightly degraded the OT mask. Raising `--reseed-thresh` to ~0.5 should
reject weaker re-seeds.

### Self-consistent wrong tracks (largely solved by conquer)
Without conquer, OT locked onto "dancer + crowd" as one region; re-seeding confirmed it because
CuVLER generated the same coarse blob. Adding the conquer stage (DINOv3 spectral sub-masks)
resolved this on **breakdance**: 0.141 → 0.727 mean IoU. Conquer should be on by default for
any clip with multiple people or cluttered background.

---

## Potential future work

### Short-term / incremental

- **Multi-clip aggregate eval** of CuVLER-seeded + reseed OT-chain across all 64 eval clips.
  Currently only single-clip demos. Need `eval_propagation.py`-style sweep with CuVLER seeds.

- **Tune reseed-thresh per difficulty class.** Slow objects: 0.5+ (reject borderline reseeds).
  Fast objects: 0.2 (accept more since OT already drifted). Could be adaptive based on
  how many proposals CuVLER found.

- **Optical flow for fast objects.** RAFT or FlowFormer as an alternative to OT for
  high-motion clips (drone, drift). Could fall back from OT to flow when motion is detected
  as large.

- **Multi-object tracking.** Currently single-instance only (one click → one chain).
  Extending to multi-instance means running N parallel OT chains with conflict resolution.

### Medium-term

- **Temporal self-training (Stage 5).** OT propagation at short range (offset ≤10, mIoU 0.65-0.77)
  is a plausible source of temporal pseudo-labels. Propagate each frame's CuVLER mask a few frames
  forward/backward and use agreement as a consistency loss to re-train the decoder or CuVLER itself.

- **Better boundary targets for the decoder.** The unsupervised decoder is near-saturated
  because its training targets (CuVLER pseudo-masks) are noisy. Using OT-propagated masks or
  CascadePSP-refined masks as training targets might unlock the next quality tier.

- **Make conquer the default.** Currently opt-in via `--conquer`. Given the breakdance result
  (0.141 → 0.727), it should probably always be on. The cost is a second DINOv3 forward pass per
  keyframe — worth profiling to see if it's acceptable for the reseed frames too.

### Longer-term / exploratory

- **Replace OT with learned propagation.** Train a lightweight transformer to propagate patch
  distributions conditioned on appearance features. OT is unsupervised but ignores object-level
  priors; a trained model could learn to stick to foreground.

- **Interactive refinement loop.** The unsupervised decoder supports iterative pos/neg clicks
  (5-click mIoU 0.529 ≈ box 0.538). Combining this with the OT chain could allow a user to
  correct a drifted mask mid-clip and resume propagation from the corrected frame.

- **Scale to longer videos.** Current eval is ≤90 frames. For long videos (500+ frames)
  periodic re-seeding every 10 frames becomes expensive (CuVLER on every keyframe).
  Could thin to every 30-50 frames and only re-seed on keyframes where the OT confidence is low
  (e.g. heat_up.max() < threshold).
