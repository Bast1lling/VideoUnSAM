# VideoUnSAM — Project Status

> Last updated: 2026-07-13

A running account of what this project is, what's been tried, what worked,
what didn't, and what the pipeline looks like right now. For usage/CLI
reference see `README.md`; this doc is the "why" and "what's left" companion
to it.

## What we're building

Fully **label-free, unsupervised** promptable video object segmentation: user
clicks an object in frame 0, the pipeline segments and tracks it through the
rest of the clip. No supervised weights anywhere in the chain (no SAM
weights, no COCO/SA-1B pretraining, no human masks) — only frozen
self-supervised DINOv3 features plus classical algorithms (optimal transport,
CRF, guided filter) built on top of them. Evaluated on DAVIS 2016 val (20
clips) with standard J (IoU) / F (boundary) / J&F metrics; clicks are
simulated from GT only to drive the eval harness, never used as supervision.

## Current pipeline

```
Frame 0 (user click)
  └─ CuVLER + conquer  → seed mask (frozen detector, class-agnostic proposals)
       └─ per-frame loop:
            Sinkhorn OT patch propagation (prev frame feats → cur frame feats)
              + instance-probe fusion (test-time adaptation, see below)
              → bilinear upsample → threshold → mask  [propagation state]
            periodic reseed (every 10 frames: re-run CuVLER+conquer,
              accept if IoU vs current mask ≥ 0.3)
            guided-filter boundary polish (display only — never re-enters
              propagation)
            Dense CRF refinement (only on frames above a confidence gate)
```

Concretely:
- **Seed (frame 0):** `video/divide/cuvler_divide.py` (CuVLER) + `video/divide/conquer.py` — hybrid click arbitration (2026-07-13, in demo.py AND `eval_davis2016.py --seed-pick hybrid`, now the default in both): largest click-containing candidate, unless it looks merged (>40% of frame → smallest valid; >15% AND click sits peripherally, centrality <0.6 → re-rank by click centrality). Fixes dance-twirl 0.13→0.75 label-free; DAVIS 2016 full-pipeline headline 0.598→0.624.
- **Propagation:** `video/propagation/sinkhorn_ot.py` — patch-level (64×64 or 128×128 grid) Sinkhorn OT between consecutive frames' DINOv3 features, LAB-color cost blended in.
- **Reseed:** every 10 frames, re-detect and swap in a better mask if it agrees well enough with the current one — the main defense against drift.
- **Instance probe (test-time adaptation):** 1-layer linear probe trained ~100 steps on frame-0 seed patches (positive) vs rest (negative), frozen DINOv3 features. Fused into the OT heat each frame (`heat = 0.5·OT + 0.5·probe`). Memoryless — can't drift, re-acquires the object after occlusion.
- **Guided-filter boundary polish:** `cv2.ximgproc.guidedFilter` snaps the blocky patch-grid mask to real image edges. Display-only, decoupled from propagation (see failure mode below for why that decoupling matters).
- **Dense CRF:** `video/refine/dense_crf.py`, bilateral filter using DINOv3 features or RGB, applied only above a confidence gate (skips diffuse/drifted frames).

Entry points: `demo.py` (interactive Gradio UI — probe and guided filter always on, CRF toggleable), `video/scripts/eval_davis2016.py` (aggregate metric, flags for every stage).

Note: `video/scripts/propagate_reseed.py` and `--conquer`/CascadePSP-refine
described in earlier versions of this doc are the old single-clip demo path
predating CuVLER-as-default and the CRF/probe/guided-filter stack — CuVLER,
conquer, reseed, CRF, and the probe are effectively always-on defaults now,
and CascadePSP was dropped (see "what didn't work").

## Results (20-clip DAVIS 2016 val, J&F)

| Config | J&F (corrected protocol, 2026-07-13) | J&F (old, buggy) |
|---|---|---|
| OT + reseed, no CRF | 0.550 | 0.572 |
| + Selective Dense CRF | 0.580 | 0.603 |
| + Instance probe (TTA) | **0.598** | 0.634 |

SOTA (supervised) is ~0.876 (DPA, CVPR 2024) — not apples-to-apples since
that's not label-free, but it's the honest reference point for the gap.

**Protocol fix (2026-07-13): ladder re-measured (table above); per-clip
numbers elsewhere in this doc are still old-protocol.** Two bugs were found and fixed in
`eval_davis2016.py`: (1) the on-disk annotations are DAVIS 2017 multi-object
and were scored against object 1 only — the 2016 protocol is the union of
all instances (affects 7/20 clips: bmx-trees, horsejump-high, kite-surf,
motocross-jump, paragliding-launch, scooter-black, soapbox); (2) the frame-0
seed pick used GT IoU as tie-breaker (and gt0 fallback) — replaced with
demo.py's label-free logic, so the eval measures what the demo does. Smoke
Corrected full-ladder re-run (logs in `report/rerun_protocol_fix/`): big
per-clip shifts in both directions — bmx-trees 0.31→0.56 and motocross-jump
0.56→0.71 improve under the label-free seed; dance-twirl collapses to 0.25
and breakdance to 0.37 (the GT tie-break had been silently picking their
frame-0 seed — worth investigating whether demo.py's seed pick fails the
same way on those clips). The 2×2 ablation and bmx cascade CSV still need
re-running. Toolkit-compatible union GT built at
`datasets/davis/DAVIS2016eval/`; `--save-dir` exports indexed PNGs for
official-toolkit scoring (probe-config export at
`report/rerun_protocol_fix/export_probe`). The DAVIS 2017 line is a separate
harness and unaffected.

**Guided-filter full 20-clip rerun (2026-07-06): net zero.** With the full
pipeline (`--refine --probe --guided`) the aggregate is J&F 0.634 — identical
to the no-guided baseline (J 0.661→0.664, F 0.606→0.604; per-clip deltas all
within ±0.04, no systematic direction). The earlier 2-clip spot-check's small
win doesn't survive aggregation: with CRF on, guided display only appears on
low-confidence CRF-skipped frames, where edge-snapping a drifted mask doesn't
change IoU. Keep guided in the demo for visual polish; results tables need
no guided row. Note: the first rerun attempt scored 0.320 because
`eval_davis2016.py`'s `--guided` had been wired as the *rejected* design
(filtering the soft heat pre-threshold, contaminating reseed/CRF gates) —
now fixed to match demo.py's validated display-only binary-mask polish.

## What worked

- **CuVLER over CutLER/spectral-cut for the seed detector.** Mask AR 0.46–0.49
  vs CutLER's 0.37 vs a from-scratch DINOv3 spectral cut's 0.00 (no
  objectness prior). [[divide-stage-cutler-vs-spectral]]
- **CuVLER+conquer over the trained decoder as the seed.** Frame-0 IoU 0.842
  vs the from-scratch decoder's 0.629 after 3 clicks — seed quality dominates
  end-to-end quality almost 1:1 since propagation is close to lossless.
- **Pure chained Sinkhorn OT over any learned propagation.** OT-only decays
  only ~0.10 IoU over 30 frames vs a trained decoder-refine (−0.26) or
  carryover (−0.33) or a KV memory bank (worse than both). OT is simple,
  train-free, and just better here. [[ot-chain-beats-decoder-refine]]
- **Periodic reseed.** +0.05 to +0.07 IoU boost at each successful reseed
  (dog: 0.814 → 0.862 mean IoU over 60 frames). [[cuvler-seed-ot-chain-wins]]
- **Instance probe (test-time adaptation).** Net +0.031 J&F on the 20-clip
  aggregate; fixes identity-switch in crowds and background-confusion cases
  that no cost-term tweak could touch, because it's the first approach that
  actually uses appearance *discriminatively* rather than cosine similarity
  (which weights all feature dims equally). [[tta-instance-probe-breakthrough]]
- **Guided-filter display polish, decoupled from propagation.** Small real
  IoU win (dog 0.696→0.700, blackswan 0.775→0.781) once decoupled from the
  propagation chain (see failure below for why decoupling was necessary).

## What didn't work (don't retry these)

- **DINOv3 spectral cut as the seed detector.** AR 0.00 — no learned
  objectness, first cut grabs background texture. [[divide-stage-cutler-vs-spectral]]
- **Decoder-based mask refinement fed OT output back in.** Actively *hurts*
  propagation (0.428→0.173 over 30 frames, worse than doing nothing). Never
  feed OT-propagated masks into the decoder as mask-prompts.
  [[ot-chain-beats-decoder-refine]]
- **KV memory bank (learned temporal memory).** Trained fine (loss
  1.02→0.59) but the residual gate never left ~0 — contributes nothing,
  underperforms even naive carryover. [[memory-bank-gate-stuck]]
- **Mask-prompt refinement decoder (v4/v5), including CascadePSP/GrabCut
  boundary-target training.** No-op on real CuVLER/conquer masks — the
  decoder's target IS the pseudo-mask, so it can only ever reproduce pseudo
  quality, never exceed it. Also: CascadePSP is *supervised*, breaks the
  label-free guarantee if used. [[decoder-mask-prompt-echo]] (This is also
  why CascadePSP no longer appears in the live pipeline, despite being in
  earlier single-clip demos.)
- **Forward-backward OT cycle consistency** (to fix identity-switch/leak
  cases). Erodes the whole mask instead of surgically cutting leaks — net
  negative everywhere (dancing 0.405→0.071). [[cycle-consistency-erodes-mask]]
- **Competitive multi-object OT** (partition scene, argmax-assign). Delays
  identity collapse but doesn't fix it (dancing 0.342→0.315, net negative on
  easy frames). Conclusion at the time: the ceiling on crowded/overlapping
  scenes is the DINOv3 *representation*, not the propagation algorithm.
  [[multiobject-competition-rep-ceiling]] **Partially overturned** by the
  instance probe, which found the discriminative signal *was* linearly
  present after all — cosine similarity just wasn't using it.
- **Sharpened chain propagation** (`propagate_chain.py`'s `sharpen=4.0`
  default). Collapses to near-zero mIoU; unsharpened chaining ≈ direct
  single-jump, sharpening must not be used. [[propagation-ot-vs-alternatives]]
- **Probe refresh at reseed points** (retrain the instance probe on frame-0 +
  every accepted-reseed frame; `--probe-refresh` in `eval_davis2016.py`, kept
  off-default as a documented negative). Net −0.030 J&F on a 6-clip diagnostic
  subset vs a same-day `--probe` control (control reproduced README baselines
  within ±0.006, so deltas are real). The target regression barely moved
  (motocross-jump +0.012, nowhere near the no-probe 0.559), while the probe's
  biggest *wins* were hurt most (libby 0.760→0.698, bmx-trees 0.311→0.250,
  camel 0.749→0.709). Reading: the probe's value is exactly its frame-0 purity
  — refresh injects current-track state into it, turning the memoryless probe
  into a weak memory (the mechanism family that always loses here), and the
  reseed-acceptance gate itself only measures agreement with the current
  track — the same signal class the wrong-track diagnosis showed cannot be
  trusted as a training filter.
- **Guided-filter output fed back into propagation** (the first version of
  the display-only fix, briefly shipped in `demo.py`). Looked crisper but silently
  eroded real coverage — mean IoU vs GT 0.696→0.564 on dog, area/GT ratio
  0.86→0.60 — because the edge-snapped mask compounds shrinkage frame over
  frame when it's also used as next frame's propagation seed. Fixed by
  decoupling: propagation always continues on the plain-bilinear mask;
  guided filter only touches the final display.
- **Spatial penalties in Optimal Transport.** Adding a normalized spatial
  distance penalty to the Sinkhorn cost matrix (to prevent teleportation to
  identical background distractors) yielded strictly zero J&F improvement
  across all tested weights (0.1 to 0.5). This confirms that tracking
  failures are not caused by instantaneous spatial jumps, but rather by
  gradual feature drift that subsequently gets locked in by reseed
  mechanisms. Local frame-to-frame spatial priors cannot prevent this
  gradual drift.

## Whole-image (no-click) line — first video-level result (2026-07-07)

The whole-image pieces (divide+conquer+one-per-object seed, per-object
probe+OT tracking, argmax competition) are now assembled into one evaluated
pipeline: `video/scripts/eval_davis2017_unsupervised.py` — official 30-clip
DAVIS 2017 val, unsupervised multi-object protocol (no click, no GT init;
Hungarian matching of predicted tracks to GT tracks, unmatched GT scores 0).

**Result: J 0.344 / F 0.289 / J&F 0.317 (61 GT objects, in-house protocol)
— beats the SAM2-distilled VVitCutLER baseline (arXiv 2605.17584: J 0.328 /
F 0.159 / J&F 0.244) on both J and F, with zero SAM weights and zero
training, no CRF.**

**TOOLKIT-VERIFIED (2026-07-07): official `davis2017-evaluation` J&F 0.348
(J 0.334 / F 0.362, J-Decay −0.006).** The official number is *higher* than
the in-house 0.317 and is the one to publish. Mechanics: `--save-dir` on
`eval_davis2017_unsupervised.py` exports per-frame indexed PNGs; the toolkit
requires mutually exclusive masks, so overlapping independent tracks are
flattened with whole-track priority by CuVLER seed score (the `_prio`
variant — per-pixel max-soft flattening fragments near-duplicate tracks,
e.g. blackswan toolkit J&F 0.600 vs 0.870 prio; at aggregate the gap is
small, 0.344 vs 0.348, but prio is strictly cleaner). Why verified > in-house:
official `db_eval_boundary` F (0.362) is much more generous than our
reimplementation (0.289), outweighing the small J cost of forced exclusivity
(0.344 → 0.334). Official GT is `Annotations_unsupervised/480p` (downloaded
into `datasets/davis/DAVIS/`), toolkit cloned at
`archive/davis2017-evaluation`, result CSVs in `report/davis2017_toolkit/`.
Rerun: export with `--save-dir <dir>`, then
`python evaluation_method.py --task unsupervised --davis_path
datasets/davis/DAVIS --results_path <dir>_prio`.

**Competition ablation (revises the earlier single-clip conclusion):**
per-patch argmax competition scores 0.301 vs 0.317 independent — net
negative at scale. Where frame-0 seeding over-segments, the junk duplicate
track *steals patches from the real object* under forced exclusivity
(blackswan −0.244, libby −0.197); competition's wins on genuinely
contending tracks are small (dogs-jump/bike-packing ~+0.02). Independent
thresholding is the shipped default (`--compete` opt-in): the Hungarian
matching can simply ignore junk tracks, whereas forced exclusivity lets
them cannibalise real ones.

Failure profile confirms the known open problems, now with numbers:
pool-coverage floors (drift-straight n_pred=0, drift-chicane 0.002,
kite-surf 0.014) and crowd/merge cases (india 0.064, lab-coat 0.054,
breakdance 0.143 — vs 0.73 *with* a click, quantifying exactly what the
click's arbitration is worth). Merge-vs-split at the divide stage remains
the open research question; this number is the honest floor for it.

- **bmx-trees conquer×reseed negative interaction — ROOT-CAUSED (2026-07-06).**
  The 2×2 ablation's one anomaly (conquer+reseed 0.130 vs 0.370/0.320 for
  either alone) is now explained via the `--reseed-log` diagnostic in
  `eval_davis2016.py`: reseed's `pick_proposal` arbitrates by IoU vs the
  *current OT mask* — coherence, with no click available mid-clip. Conquer
  densifies the candidate pool (18–37 vs 1–2), so once OT drifts early
  (onto trees, by ~frame 20) some fine sub-mask always matches the drifted
  mask above the 0.3 gate. Logged cascade: accepted reseeds at f20/f30/f40
  with gate IoU 0.33→0.60→0.76 while true IoU collapses 0.12→0.02→0.008 —
  gate confidence and correctness anti-correlated, the self-consistent
  wrong-track signature *inside the reseed mechanism*. Without conquer the
  f20 coarse candidate gates at 0.048 and is rejected, so the track survives.
  Conclusion: with no click available mid-clip, reseed acceptance can only
  measure agreement with the (possibly drifted) current track, so conquer's
  dense candidate pool — a strength at frame 0 — becomes a liability after
  drift.
  **Predicted fix TESTED (2026-07-06, `--probe-reseed`): surgical success,
  aggregate trade.** Arbitrating reseeds by probe soft-IoU (candidate must
  beat the incumbent; no new hyperparameter) rejects the exact poison
  candidate and rescues bmx-trees 0.130→0.309 with dance-twirl/breakdance
  unharmed — but the 20-clip aggregate is neutral (0.572→0.569 isolated,
  0.634→0.630 full pipeline) because motocross-jump (−0.186), scooter-black
  (−0.178), paragliding-launch (−0.180) lose what bmx gains. Those are
  precisely the frame-0-appearance-staleness clips the probe *fusion* also
  regresses on: both trades are one limitation — **the click's identity
  evidence is time-decaying**, and the only label-free refresh signal
  (accepted reseeds) is coherence-flavored (probe-refresh already measured
  net −0.030). Flag kept, off by default.

- **Sub-patch objects** (drift-chicane, J&F ≈ 0.001): object smaller than
  ~2 patches even at the 128×128 grid (8px/patch). Out of reach for the
  patch-OT paradigm as-is.
- **True physical occlusion** (dancing frames 44–46 dip to ~0.10): the target
  is genuinely hidden, no appearance method can help. The probe recovers
  *after* the occlusion (turns permanent OT death into a transient dip), but
  can't fix the occluded frame itself.
- **Instance probe is a trade, not a free win.** 13/20 clips improve, 6
  regress: motocross-jump −0.245 (frame-0 appearance overfits before a big
  pose/scale change), dog/blackswan −0.03 to −0.09 (coarse patch-level probe
  score blurs an already-clean CRF boundary). Fusion weight is currently a
  fixed 0.5 with no confidence gating.

## Open next steps (not yet done)

1. ~~Confidence-gated / adaptive instance-probe fusion weight~~ **TESTED
   2026-07-07: NEGATIVE — no label-free gate signal exists; don't retry.**
   Diagnostic-first (per-frame `--probe-gate-log` in `eval_davis2016.py`,
   6-clip probe-hurt/probe-win panel with matched no-probe baseline):
   - *Area-normalized probe peakiness* (mean of top-k scores, k = seed patch
     count; fixes the raw-quantile signal, which just measures object size)
     fails in the wrong direction: on motocross-jump (probe's worst clip,
     −0.252) the stale probe stays confidently peaky (0.977→0.944) while its
     true quality collapses (probe-vs-GT soft-IoU 0.25→0.18) — it fires hard
     on patches that still resemble frame 0 (other riders). The probe wins
     (libby, bmx-trees) have *lower* late peakiness (0.80, 0.45). A
     peakiness gate keeps the bad probe and cuts the good ones.
   - *Probe–OT agreement* is perfectly ambiguous: early-window agreement
     0.331 on motocross-jump (probe hurts −0.25) vs 0.332 on libby (probe
     wins +0.29). Identical observable, opposite truth — disagreement says
     *someone* drifted, never who (motocross: OT right/probe wrong;
     libby/bmx: probe right/OT wrong).
   Same pattern as the quality-gate and reseed findings: confidence and
   correctness are uncoupled, so none of the label-free signals tested can
   tell a stale probe from a fresh one. The motocross-jump −0.245 fusion
   regression is irreducible without genuinely independent arbitration
   evidence (i.e., a mid-clip click).
2. ~~Wire the guided-filter fix into `eval_davis2016.py` and rerun~~ **Done
   2026-07-06** — net zero at the aggregate (see Results note above).
3. ~~Probe-gated reseed acceptance~~ **Tested 2026-07-06** — surgical
   success on the diagnosed pathology, aggregate-neutral trade (see failure
   modes above). The real open problem it exposes, unifying it with item 1:
   **keeping identity evidence fresh without track contamination** — every
   probe-consuming mechanism wins where frame-0 appearance holds and loses
   where it goes stale, and all label-free refresh signals measured so far
   are coherence-flavored.
4. Sub-patch objects and true-occlusion frames are known hard floors — not
   worth chasing further with this architecture; would need a different
   mechanism (e.g. native-resolution seeding) or acceptance of the ceiling.
