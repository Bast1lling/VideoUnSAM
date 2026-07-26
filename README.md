# Segment Anything without Supervision

Unsupervised SAM (UnSAM) is a "segment anything" model for promptable and automatic whole-image segmentation which does not require human annotations. 

<p align="center"> 
  <img width="1301" alt="teaser_unsam" src="https://github.com/frank-xwang/UnSAM/assets/58996472/0c53071c-bdc8-4424-9e9e-40b8c8c31a18" align="center" >
</p>


> [**Segment Anything without Supervision**](http://arxiv.org/abs/2406.20081)            
> [XuDong Wang](https://frank-xwang.github.io/), [Jingfeng Yang](https://jingfeng0705.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)      
> UC Berkeley            
> NeurIPS 2024            

[[`project page`](https://people.eecs.berkeley.edu/~xdwang/projects/UnSAM/)] [[`arxiv`](http://arxiv.org/abs/2406.20081)] [[`colab (UnSAM)`](https://drive.google.com/file/d/1KyxbFb2JC76RZ1jg7F8Ee4TEmOlpYMe7/view?usp=sharing)] [[`colab (pseudo-label)`](https://drive.google.com/file/d/1aFObIt-xlQmCKk3G7dD8KQxaWhM_RTEd/view?usp=sharing)] [[`bibtex`](#citation)]             


---

## VideoUnSAM Extension

Extending UnSAM into the temporal domain. A user clicks once on frame 0; the
system segments the object across every frame of the clip. **No labels, no
fine-tuning, no supervised components** — fully annotation-free at every stage.

### Pipeline

```
Frame 0 (user click)
        │
        ▼
┌─────────────────────┐
│  1. DIVIDE          │  CuVLER — class-agnostic region proposals
│  video/divide/      │  (trained on MaskCut pseudo-masks, no human labels)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. CONQUER         │  DINOv3 spectral merging within each CuVLER bbox
│  video/divide/      │  expands 1 coarse proposal → 30+ tight sub-masks
│  conquer.py         │  critical for crowded scenes (breakdance: 0.14→0.73)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. OT PROPAGATION  │  Sinkhorn optimal-transport on DINOv3 patch features
│  video/propagation/ │  chains frame-to-frame, drift ~0.10 IoU / 30 frames
└────────┬────────────┘
         │  fused with ↓ (optional, on by default)
         ▼
┌─────────────────────┐
│  3b. INSTANCE PROBE │  1-layer linear probe trained on the frame-0 seed mask
│  video/scripts/     │  (label-free test-time adaptation). Re-acquires the
│  probe_tta.py       │  object after occlusion, rejects look-alike distractors.
└────────┬────────────┘  heat = 0.5·OT + 0.5·probe
         │  every 10 frames
         ▼
┌─────────────────────┐
│  4. PERIODIC RESEED │  Re-run CuVLER+conquer; accept if IoU overlap ≥ 0.3
│  video/scripts/     │  corrects drift, shown in orange in output video
└────────┬────────────┘
         │  optional
         ▼
┌─────────────────────┐
│  5. DENSE CRF       │  Fully-connected CRF with DINOv3 bilateral kernel
│  video/refine/      │  sharpens boundaries, applied only when heatmap is
│  dense_crf.py       │  confident (top-10% mean ≥ 0.65) — pure algorithm
└─────────────────────┘
```

**Instance probe (test-time adaptation).** On the user's click we spend ~1–2 s training a 1-layer linear probe to recognise *that specific object* — positive examples are the seed-mask patches, negatives are the rest of the frame, on frozen DINOv3 features. Fully label-free (supervision is the unsupervised seed). Fused with OT each frame, it attacks the two failure modes no cost-term tweak could: **identity switch** in crowds and **background confusion** (the probe learns to reject look-alike trees/grass). Because it is memoryless it re-acquires the object after an occlusion where plain OT was permanently lost. Over 20 DAVIS clips it is **net +0.031 J&F (0.603 → 0.634)** — but it is a *trade*, not a free win: 13 clips improve (libby +0.29, bmx-trees +0.18, drift-straight +0.11), 6 regress (motocross-jump −0.25, dog −0.09) where the probe overfits frame-0 appearance or its coarse score blurs an already-clean boundary. On by default (the aggregate justifies it); toggle off for clips with large pose/scale change. See [the detailed write-up](#test-time-adaptation-per-clip-instance-probe-the-breakthrough).

### Results — DAVIS 2016 val (20 clips, fully annotation-free)

**Ablation** (adding each component):

| Configuration | J mean | F mean | J&F |
|---|---|---|---|
| No reseed | 0.592 | 0.496 | 0.544 |
| + Periodic reseed | 0.621 | 0.523 | 0.572 |
| + Selective Dense CRF | 0.635 | 0.571 | 0.603 |
| + Instance probe (TTA) | **0.661** | **0.606** | **0.634** |

**Per-clip breakdown** (full pipeline, J&F):

| Clip | J | F | J&F | Notes |
|---|---|---|---|---|
| blackswan | 0.925 | 0.933 | 0.929 | |
| bmx-trees | 0.077 | 0.183 | 0.130 | trees occlude rider, OT latches onto background |
| breakdance | 0.782 | 0.669 | 0.725 | crowd-bleed at seed; conquer sub-masks critical; 0.791 at 2048px |
| camel | 0.807 | 0.713 | 0.760 | |
| car-roundabout | 0.875 | 0.640 | 0.758 | |
| car-shadow | 0.817 | 0.595 | 0.706 | |
| cows | 0.865 | 0.722 | 0.794 | |
| dance-twirl | 0.817 | 0.718 | 0.768 | |
| dog | 0.897 | 0.747 | 0.822 | |
| drift-chicane | 0.000 | 0.003 | 0.001 | car < 1 patch at 64×64 grid; sub-patch object |
| drift-straight | 0.720 | 0.518 | 0.619 | |
| goat | 0.852 | 0.736 | 0.794 | |
| horsejump-high | 0.622 | 0.569 | 0.595 | |
| kite-surf | 0.134 | 0.286 | 0.210 | small kite frequently lost |
| libby | 0.470 | 0.474 | 0.472 | |
| motocross-jump | 0.567 | 0.551 | 0.559 | |
| paragliding-launch | 0.754 | 0.710 | 0.732 | |
| parkour | 0.833 | 0.808 | 0.820 | |
| scooter-black | 0.350 | 0.449 | 0.399 | low contrast against dark background |
| soapbox | 0.529 | 0.405 | 0.467 | |

**Per-clip instance-probe effect** (J&F, sorted by Δ; aggregate **0.603 → 0.634**, +0.031):

| Clip | Base | +Probe | Δ | | Clip | Base | +Probe | Δ |
|---|---|---|---|---|---|---|---|---|
| libby | 0.471 | 0.761 | **+0.290** | | scooter-black | 0.399 | 0.431 | +0.032 |
| bmx-trees | 0.130 | 0.311 | **+0.181** | | breakdance | 0.725 | 0.755 | +0.030 |
| drift-straight | 0.619 | 0.725 | +0.106 | | car-roundabout | 0.758 | 0.767 | +0.009 |
| kite-surf | 0.210 | 0.294 | +0.084 | | drift-chicane | 0.001 | 0.001 | 0.000 |
| soapbox | 0.467 | 0.537 | +0.070 | | dance-twirl | 0.768 | 0.760 | −0.008 |
| car-shadow | 0.706 | 0.754 | +0.048 | | camel | 0.760 | 0.748 | −0.012 |
| parkour | 0.820 | 0.868 | +0.048 | | paragliding-launch | 0.732 | 0.719 | −0.013 |
| goat | 0.794 | 0.839 | +0.045 | | blackswan | 0.929 | 0.896 | −0.033 |
| cows | 0.794 | 0.836 | +0.042 | | dog | 0.822 | 0.730 | −0.092 |
| horsejump-high | 0.595 | 0.626 | +0.031 | | motocross-jump | 0.559 | 0.314 | **−0.245** |

13 clips improve, 6 regress, 1 neutral. Wins concentrate on the hard cases (background confusion, small/lost objects, crowds); losses come from frame-0 appearance overfit under large pose/scale change (motocross-jump) and coarse-probe boundary blur on already-clean clips (dog, blackswan — both lose mainly F).

**Known failure modes:**
- *Sub-patch objects* (drift-chicane): at the default 64×64 grid (16 px/patch), objects smaller than ~2 patches score near zero. The 128×128 grid (`--feat-size 2048`, 8 px/patch) partially addresses this but drift-chicane's car is still sub-patch even at that resolution.
- *Background confusion* (bmx-trees): OT can lock onto background textures (tree branches) that are semantically similar to the foreground object. Periodic reseed only recovers if the drifted mask still overlaps the correct object.
- *Low-contrast objects* (scooter-black, kite-surf): DINOv3 features distinguish these objects from their backgrounds less reliably, leading to diffuse heatmaps and early tracking loss.

---

### Extended ablations & development log

#### 128×128 feature grid (`--feat-size 2048`)

Increasing the DINOv3 input from 1024→2048 px doubles the patch grid from 64×64 to 128×128 (patch size stays 16 px). Confirmed on breakdance:

| Grid | J | F | J&F |
|---|---|---|---|
| 64×64 (1024 px, default) | 0.782 | 0.669 | 0.725 |
| 128×128 (2048 px) | 0.811 | 0.771 | **0.791** |

F score improvement (+0.102) is the main gain. At 64×64, mask boundaries snap to a 16 px grid on 480p frames, producing blocky stepped edges that destroy F score. At 128×128, each patch is 8 px so boundaries follow the object silhouette twice as finely. J also improves (+0.029) because OT matching itself is more precise at higher resolution. Cost: ~8× slower per clip. Available via `--feat-size 2048` and as a toggle in the Gradio demo.

Clips most likely to benefit: those with large J–F gaps (boundary is the bottleneck) — car-roundabout (gap 0.234), car-shadow (0.223), drift-straight (0.203), dog (0.150).

#### Spatial prior on OT cost matrix

Added `--spatial-weight λ` which injects a normalised Euclidean patch-position penalty into the Sinkhorn cost:

```
C_ij += λ · ||pos_i − pos_j||_2 / √2
```

Hypothesis: prevents OT from jumping to semantically similar but spatially distant background patches. Tested at λ=0.3 and λ=1.0 on the two worst-performing clips:

| Clip | λ=0 | λ=0.3 | λ=1.0 |
|---|---|---|---|
| bmx-trees J&F | 0.130 | 0.130 | 0.132 |
| kite-surf J&F | 0.210 | 0.204 | 0.197 |

**Result: no effect.** Root cause: for bmx-trees the confusing tree-bark patches are *adjacent* (1–2 patches away) to the rider, not distant — so spatial distance cannot separate them. For kite-surf the kite moves fast enough that constraining spatial jumps slightly hurts. Spatial prior is kept in the codebase at default 0.0.

#### Motion consistency prior on OT cost matrix

Added `--motion-weight γ` which adds a per-patch frame-difference penalty:

```
C_ij += γ · |fd_b[j] − fd_a[i]|
```

where `fd` is mean absolute pixel change between consecutive frames, pooled to patch level. Hypothesis: object moves differently from static background — penalise transport between patches with mismatched motion magnitudes. On slow clips, both object and background fd are near-zero so the term cancels out and the prior is approximately free.

Full 20-clip eval at γ=0.5:

| | γ=0 | γ=0.5 | Δ |
|---|---|---|---|
| J mean | 0.635 | 0.634 | −0.001 |
| F mean | 0.571 | 0.571 | 0.000 |
| J&F | 0.603 | 0.603 | 0.000 |

**Result: net zero.** Goat improved +0.017 (animal moves relative to static grass), but breakdance was unchanged (the crowd is also dancing — same motion magnitude as the target) and kite-surf regressed −0.020. Clip-level gains and losses cancel. Prior kept at default 0.0.

#### Reseed threshold tuning (breakdance)

Tested `--reseed-thresh` 0.3 / 0.5 / 0.6 on breakdance. All three give identical J&F=0.725. **Conclusion:** the reseed threshold is not the bottleneck for breakdance. Drift accumulates in the frame-to-frame OT step, not at keyframe reseeds. Raising the threshold does not change which reseeds are accepted (they all overlap the OT mask well).

#### Breakdance frame-0 seed quality (CuVLER vs. CuVLER + conquer)

| | Proposals | IoU vs GT | Mask area |
|---|---|---|---|
| CuVLER only | 1 | 0.134 | 60.8% of frame |
| + Conquer | 32 | **0.744** | 10.8% (GT = 10.0%) |

Without conquer, CuVLER generates one large proposal covering 60% of the frame — the entire crowd. Conquer's DINOv3 spectral clustering within each proposal bbox expands this into 32 tight sub-masks; the best one achieves 0.744 IoU against GT and covers only the target dancer. This is the mechanism behind the 0.14→0.725 J&F jump shown in the main ablation table. The mean J over 84 frames (0.782) exceeds the frame-0 seed IoU (0.744) because periodic reseeds find improved proposals at later keyframes.

#### Dancing clip — multi-instance crowd analysis (DAVIS 2017)

The `dancing` clip from DAVIS 2017 contains **3 simultaneous dancers** sharing overlapping screen regions. It is a hard test for single-object OT propagation because the target dancer physically merges with the other two between frames 41–61.

**Seed quality progression** (all runs: user click on dancer 1, frame 0)

| Configuration | Proposals | Seed IoU | Mean IoU | Median IoU |
|---|---|---|---|---|
| Baseline (CuVLER, no conquer) | 1 | 0.049 | 0.035 | — |
| + Conquer | ~38 | 0.518 | 0.387 | 0.499 |
| + Crop-context reseed (1.5×) | ~60 | 0.669 | 0.401 | 0.529 |
| + LAB color fusion (w=0.2) | ~60 | 0.669 | **0.412** | **0.537** |

Color fusion helps frames 1–40 (+0.04–0.05 IoU/frame): the dancers' clothing differs in LAB space, so the additive cost discourages OT from crossing color boundaries. Frames 41–61 are unaffected — when dancers physically overlap, no cost term prevents identity switch.

**Regression check** on benchmark clips with color weight 0.2: blackswan −0.004, camel −0.001. Effectively neutral.

**Frame-0 seed selection fix**

The demo's `max(proposals, key=area)` heuristic was selecting a blob covering 73% of the frame (IoU=0.046) rather than the tight dancer-specific sub-mask (IoU=0.518). CuVLER + conquer generates the correct mask; it just isn't the largest. Fix: if the largest containing-click proposal exceeds **40% of frame area**, assume it is a merged blob and pick the **smallest** proposal instead (minimum 0.5% area floor):

```python
if largest0.sum() > H * W * 0.40:
    valid0 = [m for m in containing0 if m.sum() >= H * W * 0.005]
    seed = min(valid0, key=lambda m: m.sum()) if valid0 else largest0
else:
    seed = largest0
```

**Approaches tested on dancing — rejected**

*Template blending* (α=0.3, 0.5): blend the mean DINOv3 feature of seed patches into OT heat as a persistent appearance prior. Mean IoU crashed from 0.401 to 0.099–0.102. Root cause: all 3 dancers share similar DINOv3 body-part features; the template heat spreads uniformly across all of them and pulls OT mass toward the wrong dancer.

*Click-point tracker as reseed gate*: the Sinkhorn transport plan T already computed for mask propagation can propagate a one-hot click indicator at zero extra cost via `point_b = point_a @ (T/μ)`. The tracked click was used as a hard gate — only accept a reseed proposal if it contains the tracked point. Mean IoU dropped from 0.412 to 0.391. Root cause: the click indicator travels through the same confused OT plan that triggered the identity switch; by frame 50 it had jumped to x=747 (dancer 2), causing all correct reseeds to be rejected. The `point_a` parameter remains in `propagate_patch` for future experimentation, but the hard gate is reverted.

**Fundamental ceiling — frames 41–61**

Dancer 2 rushes from x=772 to x=545 over frames 40–50, physically overlapping dancer 1 (x=539→439). DINOv3 features for the two overlapping bodies become indistinguishable and OT mass switches identity. Per-frame IoU over the collapse: 0.499 → 0.481 → 0.388 → 0.225 → 0.157 → 0.076 → 0.049 → 0.027 → 0.017 → 0.000 over 9 frames. No single-object propagation system can recover without multi-object tracking or re-annotation.

Best achievable with the OT pipeline alone: **mean IoU 0.412 / median 0.537** — an 11.8× improvement over the 0.035 baseline. The instance probe (below) raises this further to **0.439** and, more importantly, converts the permanent post-overlap collapse into a recoverable dip.

#### Competitive multi-object propagation — whole-image tracking (tested)

The bigger structural swing: instead of pushing one mask, partition frame 0 into labels (clicked object + competing scene regions + background) and propagate the **whole label field** through the same OT plan, `argmax`-assigning each target patch to whichever label wins. Because the labels partition the source and the plan's rows sum to 1, every target patch receives equal total mass — a fair competition. This is literally whole-image multi-object video segmentation; the clicked object is one read-out label. Implemented in `video/scripts/propagate_multiobject.py` (`compute_cond` factors the shared plan).

Hypothesis: when a distractor (dancer 2) approaches, its *own* label claims its patches, so the target label's mass cannot leak onto it.

**Matched comparison on dancing** (no reseed, no crop-context, color 0.2 — identical conditions):

| | Mean IoU | Median | Convergence frames 44–61 |
|---|---|---|---|
| Single-object | **0.342** | 0.436 | collapses to **0.000** |
| Multi-object (whole-image) | 0.315 | 0.396 | survives at **~0.04** |

The mechanism **works as designed** — competition prevents the catastrophic total collapse (single-object hits a hard 0.000 when dancer-1's label is fully annihilated; multi-object never dies because dancer-2's label holds its territory). But it **nets slightly negative** because (1) competitor labels nibble the target's boundary patches in the easy frames, exactly where single-object is strongest, and (2) — the decisive reason — at the moment of true physical overlap, dancer-1 and dancer-2's bodies have **genuinely indistinguishable** DINOv3 features (and the same clothing-color discriminator already maxed at w=0.2). Competition only *delays* the switch by a few frames; it cannot prevent it.

At this point we believed the ceiling was the representation — three independent attacks (cost-term priors, cycle consistency, label competition) all failed at what looked like the same wall. **That conclusion was wrong**, and the next section shows why. The multi-object script is kept because it produces a full per-frame multi-object segmentation for free — a capability the single-object pipeline lacks even though it does not improve the single clicked-object metric.

#### Test-time adaptation: per-clip instance probe (the breakthrough)

The "representation ceiling" conclusion above was wrong. The instance-discriminative signal **is** present in frozen DINOv3 features — cosine-similarity OT simply wasn't using it. Cosine weights every feature dimension equally; a trained discriminative direction can up-weight the dims encoding *this* dancer's appearance and down-weight the generic "human body" dims.

On the user's click we spend ~1–2 s training a **1-layer linear probe** on frame 0: positive class = the CuVLER+conquer seed patches, negative class = the rest of the frame, ~100 Adam steps of class-balanced BCE on frozen DINOv3 features. **Fully label-free** — the supervision is the unsupervised seed mask. This is strictly more powerful than the rejected mean-prototype template blend, which could only measure cosine distance to a centroid; the probe with negatives learns a Fisher-style separating direction.

The probe is **memoryless** — it re-recognises the instance from scratch every frame — which gives it two properties OT lacks: it cannot drift (no state to drift), and a transient occlusion cannot cause *permanent* identity loss.

**Probe alone vs. the full OT pipeline on dancing** — the probe, with zero temporal modelling, matches the entire OT chain (0.399 vs 0.405). Crucially its convergence behaviour is opposite: where OT dies to a hard 0.000 from frame 52 onward, the probe dips at the overlap and then **re-acquires** dancer-1 once the dancers separate.

**Fusion** `heat = 0.5·OT + 0.5·probe`, feeding the fused (probe-corrected) mask back into the OT chain, gets the best of both — OT carries the brief overlap, the probe revives identity after and rejects look-alikes:

First, against the **bare OT chain** (no reseed/CRF, in `video/scripts/probe_tta.py`) the fusion `heat = 0.5·OT + 0.5·probe` — feeding the fused mask back into the chain so OT carries the overlap and the probe revives identity after — is a large, uniform win:

| Clip | OT chain only | + Probe fusion | Δ | Failure mode addressed |
|---|---|---|---|---|
| dancing (DAVIS 2017) | 0.342 | **0.439** | +0.097 | identity switch (3 dancers) |
| bmx-trees | 0.076 | **0.229** | +0.153 (3×) | background confusion (trees) |
| dog | 0.726 | **0.832** | +0.106 | drift |
| camel | 0.643 | **0.708** | +0.065 | drift |
| blackswan | 0.788 | 0.789 | +0.001 | (already near-perfect) |

**But against the *full* pipeline (reseed + CRF) the picture is a trade, not a uniform win** — see the per-clip table in [Results](#results--davis-2016-val-20-clips-fully-annotation-free). Reseed already corrects most drift, so on clean clips the probe is redundant *and* its coarse patch-level score blurs the boundary CRF would otherwise sharpen (dog J&F 0.822→0.730, all in F). The 20-clip aggregate is **net +0.031 J&F (0.603 → 0.634)**: the hard-case wins (libby +0.29, bmx-trees +0.18, drift-straight +0.11) outweigh the regressions (motocross-jump −0.25 from frame-0 appearance overfit during the jump; dog/blackswan boundary blur). It is wired into `demo.py` (on by default, toggle "Instance probe") and `video/scripts/eval_davis2016.py` / `probe_tta.py` (`--probe`, `--mode probe|ot|fuse`).

**What it does *not* fix:** the overlap frame itself. On dancing, frames 44–46 (the instant the bodies physically occlude) still drop to ~0.10 — at true occlusion the target is literally hidden, so no appearance method can segment it. The win is that this is now a transient dip that recovers (fusion: 0.10 at f45 → 0.55 at f50 → 0.41 at f61) rather than the permanent death of the OT-only chain (0.000 from f52 on). Frame ~45 remains the genuine hard floor; everything after it is recovered.

---

#### Demo improvements (June 2026)

| Feature | Description |
|---|---|
| Clip preview video | Compiles DAVIS frames to a looping H.264 MP4 and plays it before the user clicks |
| Adaptive seed selection | 40% frame-area threshold distinguishes tight sub-masks from merged blobs at frame 0 |
| Crop-context reseed | At every reseed frame, also runs CuVLER on a 1.5× expanded crop around the OT mask bbox and merges with full-frame proposals |
| LAB color fusion | `OT_COLOR_WEIGHT=0.2` adds per-patch L² distance in LAB color space to the Sinkhorn cost matrix (`cost_addend` parameter) |
| Re-binarized patch carry-forward | Between frames the OT mask is re-binarized via `mask_to_patch` before being passed forward, preventing soft-heat mass accumulation over long clips |

---

#### What did not move the needle

| Approach | Tested | Outcome |
|---|---|---|
| Spatial OT prior | λ=0.3, 1.0 on bmx-trees, kite-surf | No effect — confuser is spatially adjacent |
| Motion OT prior | γ=0.5 on all 20 clips | Net zero — crowd co-moves with target |
| Reseed threshold | 0.3 / 0.5 / 0.6 on breakdance | Identical result — drift is in OT, not reseed |
| Multi-scale OT | coarse+fine blend | No recovery for sub-patch objects |
| Higher blur (ε) | blur=0.10 on fast clips | Smears heatmap, does not recover |
| Template blending | α=0.3, 0.5 on dancing | IoU 0.401→0.099 — DINOv3 body features non-discriminative across dancers |
| Click-point tracker (reseed gate) | dancing, hard gate on point | IoU 0.412→0.391 — click inherits same OT confusion; hard gate reverted |
| Forward-backward cycle consistency | dancing, blackswan; weight 1/2/4 | Erodes mask globally, not just leaks — blackswan 0.856→0.717, dancing 0.405→0.071 |

#### Forward-backward cycle consistency (rejected)

Idea: reuse the Sinkhorn plan `T` to check whether mass pushed into frame B traces back to the source mask. For each B patch `j`, the backward conditional `P(a|j) = T_aj / ν_j` says where its mass came from; `s_j = Σ_a m_a · P(a|j)` is the fraction returning to the seed. Multiply heat by `s_j` to suppress identity-switch and adjacent-background leakage. Near-zero cost (`cycle_weight` parameter in `propagate_patch`).

Two normalisations tested — global-max and object-relative gate. **Both regress everywhere.** The failure is fundamental to entropic OT with a small source: the backward conditional `P(a|·)` is broadly diffuse, so the mass returning to the seed's *few* patches is tiny and noisy for **all** B patches — including genuine object-body patches. The gate therefore erodes the whole mask rather than surgically removing leaks. Erosion scales with how small/diffuse the source is: mild on blackswan (large object, 0.856→0.717), catastrophic on dancing (small object, 0.405→0.071, mask collapses to a core). Kept in the code at default `cycle_weight=0.0` (inert).

### Quickstart

```bash
# Interactive Gradio demo — select a clip, click the object, watch it propagate
python demo.py
python demo.py --share   # public link

# Single clip with video output
python -m video.scripts.propagate_reseed \
    --clip blackswan --instance-id 1 \
    --reseed-interval 10 --reseed-thresh 0.3 \
    --conquer \
    --out video/outputs/blackswan.mp4

# High-quality mode: 128×128 OT grid (sharper boundaries, ~8× slower)
python -m video.scripts.propagate_reseed \
    --clip breakdance --instance-id 1 \
    --reseed-interval 10 --conquer \
    --feat-size 2048 \
    --out video/outputs/breakdance_hq.mp4

# Instance probe (test-time adaptation) — probe-only / OT-only / fused comparison
python -m video.scripts.probe_tta --clip dancing --instance-id 1 \
    --mode fuse --fuse-weight 0.5 \
    --out video/outputs/dancing_probe.mp4

# Full DAVIS 2016 val eval (20 clips, ~15 min) — J&F 0.603
python -m video.scripts.eval_davis2016 --refine --crf-conf 0.65 --crf-compat 20

# …with the instance probe (test-time adaptation) — J&F 0.634 (+0.031)
python -m video.scripts.eval_davis2016 --refine --crf-conf 0.65 --crf-compat 20 --probe

# Full eval at 128×128 grid (~8× slower, higher boundary precision)
python -m video.scripts.eval_davis2016 --refine --crf-conf 0.65 --crf-compat 20 --feat-size 2048
```

### Setup

```bash
# DAVIS 2017 trainval (~800 MB)
mkdir -p datasets/davis && cd datasets/davis
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip -q DAVIS-2017-trainval-480p.zip && cd ../..

# HuggingFace login for DINOv3 weights (first run only)
huggingface-cli login

# pydensecrf for Dense CRF refinement
pip install pydensecrf
```

### Code layout

| Path | What it does |
|---|---|
| `demo.py` | Gradio demo — click-to-track on any DAVIS clip; Standard/High quality + instance-probe toggles |
| `video/features/dinov3_dense.py` | DINOv3 ViT-L/16 dense feature extractor (resolution-agnostic) |
| `video/divide/cuvler_divide.py` | CuVLER class-agnostic proposals |
| `video/divide/conquer.py` | DINOv3 spectral sub-mask generation |
| `video/propagation/sinkhorn_ot.py` | Sinkhorn OT (`propagate_patch`, `propagate_multiscale`, `compute_cond`); optional spatial/motion/color priors, cycle weight |
| `video/scripts/propagate_multiobject.py` | Competitive multi-object / whole-image label propagation (one OT plan, K labels, argmax) |
| `video/scripts/probe_tta.py` | Test-time-adaptation instance probe; `--mode probe\|ot\|fuse` — appearance-based re-acquisition fused with OT |
| `video/refine/dense_crf.py` | Dense CRF boundary refinement |
| `video/scripts/propagate_reseed.py` | Main CLI with video output; `--feat-size`, `--spatial-weight`, `--motion-weight` |
| `video/scripts/eval_davis2016.py` | DAVIS 2016 aggregate eval; `--feat-size`, `--spatial-weight`, `--motion-weight`, `--probe` |
| `video/divide/click_grow.py` | Click-seeded feature-similarity mask extraction |
| `DinoMaskExtraction/` | DINOv3 feature inspector (Basti) |

---

## Updates
- 11/19/2025 UnSAMv2 was released!!!! Check it out at: [GitHub](https://github.com/yujunwei04/UnSAMv2) & [UnSAMv2 project page](https://yujunwei04.github.io/UnSAMv2-Project-Page/)
<img width="2476" height="1276" alt="image" src="https://github.com/user-attachments/assets/81597eec-12c4-4808-814e-61a51e246726" />        

- 10/29/2025 Add Hugging Face support for whole image segmentation [[HF Link](https://huggingface.co/yujunwei04/unsam-whole-image-segmentation)], [[Tutorial Notebook](whole_image_segmentation/hf_demo_whole_image.ipynb)]
  
- 07/01/2024 Initial commit of UnSAM

## Features
- The performance gap between unsupervised segmentation models and SAM can be significantly reduced. UnSAM not only advances the state-of-the-art in unsupervised segmentation by 10% but also achieves comparable performance with the labor-intensive, fully-supervised SAM.
- The supervised SAM can also benefit from our self-supervised labels. By training UnSAM with only 1% of SA-1B images, a lightly semi-supervised UnSAM can often segment entities overlooked by supervised SAM, exceeding SAM’s AR by over 6.7% and AP by 3.9% on SA-1B. 


## Installation
See [installation instructions](INSTALL.md).

## Dataset Preparation
See [Preparing Datasets for UnSAM](datasets/README.md).

## Method Overview

UnSAM has two major stages: 1) generating pseudo-masks with divide-and-conquer and 2) learning unsupervised segmentation models from pseudo-masks of unlabeled data.

### 1. Multi-granular Pseudo-mask Generation with Divide-and-Conquer

Our Divide-and-Conquer approach can be used to provide multi-granular masks without human supervision.

### Divide-and-Conquer Demo

Try out the demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11K2mHhISA7RYY8pKgyeyHO9gnExn-EXl)

If you want to run Divide-and-Conquer locally, we provide `demo_dico.py` that is able to visualize the pseudo-masks.
Please download the CutLER's checkpoint from [here](http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth), and then run it with:
```
cd divide_and_conquer
python demo_dico.py \
    --input /path/to/input/image \
    --output /path/to/save/output \
    --preprocess true \
    --postprocess true \ #postprocess requires gpu 
    --opts MODEL.WEIGHTS /path/to/cutler_checkpoint \
    MODEL.DEVICE gpu
```
We give a few demo images in docs/demos/. Following, we give some visualizations of the pseudo-masks on the demo images.
<p align="center">
  <img src="https://github.com/frank-xwang/UnSAM/assets/58996472/6ea40b0a-7fd3-436b-9b3f-37acbc122fc3" width=100%>
</p>


### 2. Segment Anything without Supervision

### Inference Demo for UnSAM with Pre-trained Models (whole image segmentation)
Try out the UnSAM demo using Colab (no GPU needed): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZHdql8SVHYqQG0BSkpgCPYkfWdiLeor6)

If you want to run UnSAM or UnSAM+ demos locally, we provide `demo_whole_image.py` that is able to demo builtin configs. 
Please download UnSAM/UnSAM+'s checkpoints from the [model zoo](#model-zoo). 
Run it with:
```
cd whole_image_segmentation
python demo_whole_image.py \
    --input /path/to/input/image \
    --output /path/to/save/output \
    --opts \
    MODEL.WEIGHTS /path/to/UnSAM_checkpoint \
    MODEL.DEVICE cpu
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and save the results in the local path.
<!-- For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are: -->
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.

Following, we give some visualizations of the model predictions on the demo images.
<p align="center">
  <img src="https://github.com/frank-xwang/UnSAM/assets/58996472/83f9d9ee-0c2e-4b65-83f7-77852d169d2d" width=100%>
</p>


### Gradio Demo for UnSAM with Pre-trained Models (promptable image segmentation)

The following command will pops up a gradio website link in the terminal, on which users can interact with our model. 
Please download UnSAM/UnSAM+'s checkpoints from the [model zoo](#model-zoo). 
For details of the command line arguments, see `demo_promptable.py -h` or look at its source code
to understand its behavior.
* To run __on cpu__, add `cpu` after `--device`.
```
python demo_promptable.py \
    --ckpt /path/to/UnSAM_checkpoint \
    --conf_files configs/semantic_sam_only_sa-1b_swinT.yaml \
    --device gpu
```

Following, we give some visualizations of the model predictions on the demo images.
<p align="center">
  <img src="https://github.com/frank-xwang/UnSAM/assets/58996472/1b7eb492-2c3d-426f-9f90-bc117ea322eb" width=100%>
</p>


### Model Evaluation
To evaluate a model's performance on 7 different datasets, please refer to [datasets/README.md](datasets/README.md) for 
instructions on preparing the datasets. Next, select a model from the model zoo, specify the "model_weights", "config_file" 
and the path to "DETECTRON2_DATASETS" in `tools/eval.sh`, then run the script.
```
bash tools/{promptable, whole_image}_eval.sh
```

### Model Zoo

#### Whole image segmentation
UnSAM achieves the state-of-the-art results on unsupervised image segmentation, using a backbone of ResNet50 and training 
with only 1% of SA-1B data. We show zero-shot unsupervised image segmentation performance on 7 different datasets, 
including COCO, LVIS, ADE20K, Entity, SA-1B, Part-ImageNet and PACO.   
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Methods</th>
<th valign="bottom">Models</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"># of Train Images</th>
<th valign="bottom">Avg.</th>
<th valign="bottom">COCO</th>
<th valign="bottom">LVIS</th>
<th valign="bottom">ADE20K</th>
<th valign="bottom">Entity</th>
<th valign="bottom">SA-1B</th>
<th valign="bottom">PtIn</th>
<th valign="bottom">PACO</th>
<!-- TABLE BODY -->
</tr>
<tr><td align="center">Prev. Unsup. SOTA</td>
<td valign="center">-</td>
<td valign="center">ViT-Base</th>
<td align="center">0.2M</td>
<td align="center">30.1</td>
<td align="center">30.5</td>
<td align="center">29.1</td>
<td align="center">31.1</td>
<td align="center">33.5</td>
<td align="center">33.3</td>
<td align="center">36.0</td>
<td align="center">17.1</td>
</tr>
<tr><td align="center">UnSAM (ours)</td>
<td valign="center">-</td>
<td valign="center">ResNet50</th>
<td align="center">0.1M</td>
<td align="center">39.2</td>
<td align="center">40.5</td>
<td align="center">37.7</td>
<td align="center">35.7</td>
<td align="center">39.6</td>
<td align="center">41.9</td>
<td align="center">51.6</td>
<td align="center">27.5</td>
</tr>
<tr><td align="center">UnSAM (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/12DvjnXIQsOtBSAAEicd9uhW0TCpnMFyZ/view?usp=sharing">download</a></td>
<td valign="center">ResNet50</th>
<td align="center">0.4M</td>
<td align="center">41.1</td>
<td align="center">42.0</td>
<td align="center">40.5</td>
<td align="center">37.5</td>
<td align="center">41.0</td>
<td align="center">44.5</td>
<td align="center">52.7</td>
<td align="center">29.7</td>
</tr>
</tbody></table>

UnSAM+ can outperform SAM on most experimented benchmarks (including SA-1B), when training UnSAM on 1% of SA-1B with both 
ground truth masks and our unsupervised labels. This demonstrates that the supervised SAM can also benefit from our self-supervised labels.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Methods</th>
<th valign="bottom">Models</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"># of Train Images</th>
<th valign="bottom">Avg.</th>
<th valign="bottom">COCO</th>
<th valign="bottom">LVIS</th>
<th valign="bottom">ADE20K</th>
<th valign="bottom">Entity</th>
<th valign="bottom">SA-1B</th>
<th valign="bottom">PtIn</th>
<th valign="bottom">PACO</th>
<!-- TABLE BODY -->
</tr>
<tr><td align="center">SAM</td>
<td valign="center">-</td>
<td valign="center">ViT-Base</td>
<td align="center">11M</td>
<td align="center">42.1</td>
<td align="center">49.6</td>
<td align="center">46.1</td>
<td align="center">45.8</td>
<td align="center">45.9</td>
<td align="center">60.8</td>
<td align="center">28.3</td>
<td align="center">18.1</td>
</tr>
<tr><td align="center">UnSAM+ (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/1MaCoMLIR6-baaP7p_WriZVhuJoozxTn8/view?usp=sharing">download</a></td>
<td valign="center">ResNet50</td>
<td align="center">0.1M</td>
<td align="center">48.8</td>
<td align="center">52.2</td>
<td align="center">50.8</td>
<td align="center">45.3</td>
<td align="center">49.8</td>
<td align="center">64.8</td>
<td align="center">46.0</td>
<td align="center">32.3</td>
</tr>
</tbody></table>

#### Promptable image segmentation
Despite using a backbone that is 3× smaller and being trained on only 1% of SA-1B, our lightly semi-supervised UnSAM+ surpasses the fully-supervised SAM in promptable segmentation task on COCO.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Methods</th>
<th valign="bottom">Models</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"># of Train Images</th>
<th valign="bottom">Point (Max)</th>
<th valign="bottom">Point (Oracle)</th>
<!-- TABLE BODY -->
</tr>
<tr><td align="center">SAM</td>
<td valign="center">-</td>
<td align="center">ViT-B/8 (85M)</td>
<td align="center">11M</td>
<td align="center">52.1</td>
<td align="center">68.2</td>
</tr>
<tr><td align="center">UnSAM (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/18IilJNw170sKsKBhyIvjfUZ7cZwKyBx7/view?usp=drive_link">download</a></td>
<td align="center">Swin-Tiny (25M)</td>
<td align="center">0.1M</td>
<td align="center">37.6</td>
<td align="center">57.9</td>
</tr>
<tr><td align="center">UnSAM (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/1x5tXWV-HKwQ8dJRjbPPweuHEgsJxN0JF/view?usp=drive_link">download</a></td>
<td align="center">Swin-Tiny (25M)</td>
<td align="center">0.4M</td>
<td align="center">41.3</td>
<td align="center">59.1</td>
</tr>
<tr><td align="center">UnSAM+ (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/1M3lOnSOutQRK4IqBkc3e4vGZ-u2oTkeW/view?usp=sharing">download</a></td>
<td align="center">Swin-Tiny (25M)</td>
<td align="center">0.1M</td>
<td align="center">52.4</td>
<td align="center">69.5</td>
</tr>
</tbody></table>

## License
The majority of UnSAM, CutLER, Detectron2 and DINO are licensed under the [CC-BY-NC license](LICENSE), however portions of the project are available under separate license terms: Mask2Former, Semantic-SAM, CascadePSP, Bilateral Solver and CRF are licensed under the MIT license; If you later add other third party code, please keep this license info updated, and please let us know if that component is licensed under something other than CC-BY-NC, MIT, or CC0.

## Acknowledgement
This codebase is based on CutLER, SAM, Mask2Former, Semantic-SAM, CascadePSP, BFS, CRF, DINO and Detectron2. We appreciate the authors for open-sourcing their codes. 

## Ethical Considerations
UnSAM's wide range of detection capabilities may introduce similar challenges to many other visual recognition methods.
As the image can contain arbitrary instances, it may impact the model output.

## How to get support from us?
If you have any general questions, feel free to email us at [XuDong Wang](mailto:xdwang@eecs.berkeley.edu). If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@article{wang2024segment,
  title={Segment anything without supervision},
  author={Wang, XuDong and Yang, Jingfeng and Darrell, Trevor},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={138731--138755},
  year={2024}
}

@article{yu2025unsamv2,
  title={UnSAMv2: Self-Supervised Learning Enables Segment Anything at Any Granularity},
  author={Yu, Junwei and Darrell, Trevor and Wang, XuDong},
  journal={arXiv preprint arXiv:2511.13714},
  year={2025}
}
```

