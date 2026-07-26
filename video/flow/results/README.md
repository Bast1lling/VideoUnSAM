# SMURF fine-tuning on YouTube-VIS — results

**Summary: fine-tuning SMURF on YouTube-VIS improved mask-propagation J&F from
0.805 to 0.868, a +0.063 gain over the pretrained Sintel model. Positive result.**

> **Read this first.** An earlier version of this document concluded the opposite,
> based on visual inspection of flow colour-wheel plots. That conclusion was
> wrong, and the reason it was wrong is itself worth recording — see
> "Result 4" below. Qualitative flow visualisations render the whole frame, so
> background speckle in textureless regions dominates the eye. Mask propagation
> depends only on flow accuracy on and around the object. Always evaluate on a
> task-relevant metric before judging a model.

## Headline: mask-warping J&F on DAVIS 2016 val (20 clips)

**Stride 1** — consecutive frames:

| Method | J | F | J&F | vs copy |
|---|---|---|---|---|
| `copy` — no flow, predict mask(t+1) = mask(t) | 0.7453 | 0.7116 | 0.7285 | — |
| SMURF pretrained (Sintel) | 0.7967 | 0.8126 | 0.8047 | +0.076 |
| **SMURF fine-tuned on YouTube-VIS** | **0.8539** | **0.8820** | **0.8679** | **+0.139** |

**Stride 5** — frames 5 apart, matching the YouTube-VIS training stride:

| Method | J | F | J&F | vs copy |
|---|---|---|---|---|
| `copy` | 0.4913 | 0.3396 | 0.4155 | — |
| SMURF pretrained (Sintel) | 0.5463 | 0.5163 | 0.5313 | +0.116 |
| **SMURF fine-tuned on YouTube-VIS** | **0.6542** | **0.6302** | **0.6422** | **+0.227** |

Three findings:

**1. Fine-tuning helps at both strides.** The fine-tuned model beats pretrained on
every metric in both regimes. Both flow models beat the no-flow control, so the
flow is genuinely contributing rather than the metric being saturated.

**2. The advantage nearly doubles at stride 5.** Fine-tuned minus pretrained goes
from **+0.063** at stride 1 to **+0.111** at stride 5 — a 1.75× larger margin, and
+21% relative. Since YouTube-VIS stores only every 5th frame, this is direct
evidence that the model learned the *large-displacement regime it was trained on*,
not merely "better flow in general". The stride-1 result alone could not
distinguish those two explanations.

The `copy` baseline collapsing from 0.729 to 0.416 confirms stride 5 is a
substantially harder task, so the comparison is discriminative.

**3. Boundary quality is where flow pays off most.** At stride 5, `copy` scores
F=0.340 while the fine-tune reaches F=0.630 — boundary accuracy nearly doubled.
F is the promptable pipeline's specific weakness (J 0.635 vs F 0.571, because OT
masks snap to a 16px patch lattice). Flow is dense and pixel-resolution by
construction, so it supplies exactly the signal the DINOv3 pipeline lacks. This is
the strongest argument for wiring flow into the mask-correspondence stage.

Reproduce with `video/flow/eval_flow_warp_davis.py` (`--stride=1` / `--stride=5`).

All images are `apply_smurf.py` output on DAVIS `blackswan`, first frame pair,
inference at 296×640 for both checkpoints so weights are the only variable.

| File | Checkpoint | Test pairs |
|---|---|---|
| `stride1_pretrained_flow.png` | Sintel pretrained (`ckpt-16`) | consecutive frames |
| `stride1_finetuned_flow.png` | YTVIS fine-tune (`ckpt-15`) | consecutive frames |
| `stride1_finetuned_occlusion.png` | YTVIS fine-tune | consecutive frames |
| `stride5_pretrained_flow.png` | Sintel pretrained | frames 5 apart |
| `stride5_finetuned_flow.png` | YTVIS fine-tune | frames 5 apart |

`smoke_test_sintel_368x496/` holds the original pretrained-only validation run at
368×496 — kept as the reference that the TF 2.15 port works, not comparable to
the above (different resolution).

## Training run

| | |
|---|---|
| Init | `sintel-smurf` pretrained |
| Data | 2,919 YouTube-VIS TFRecords, 88,080 frames, 360×640 |
| Steps | 15,000 (SMURF default 75,000 scaled ×0.2, per the paper's Sintel config) |
| Crop | 296×640, batch size 1 |
| LR | 5e-5, 500 warm-up steps |
| Self-sup | from step 6,250 |
| Hardware | 1× Quadro P6000, ~7h wall (Slurm job 1644232) |

## Important: what this metric is NOT

The 0.868 figure is **not** comparable to the promptable pipeline's 0.624 J&F
headline, and reporting them side by side would be a serious error.

* **0.624 is end-to-end**: one user click on frame 0, propagate ~70 frames, no
  further supervision. Errors compound across the clip.
* **0.868 is single-step and oracle-initialised**: every frame restarts from the
  *ground-truth* mask of the previous frame. No error accumulation whatsoever.

This is an upper bound on per-frame flow-propagation quality. It is meaningful
only against the `copy` control and across checkpoints — which is how the table
above is constructed.

## Result 1 — census loss never decreased

Per-epoch census loss across the whole run:

```
14.92  13.71  11.30  8.37  15.60  14.39  13.03  12.81
10.77  12.34  11.01  15.42  12.94  14.23  13.73
```

Flat. Starts 14.9, ends 13.7, wandering 8.4–15.6 with no trend. Adjacent epochs
differ by 7.2 (8.37 → 15.60), which is the scale of the noise.

**Caveat: this measurement is weak.** With `epoch_length=1000` and batch size 1,
each value is a *single frame pair*, so there are only 15 highly-variable samples.
Any rerun should log a running average every ~50 steps.

For reference, SMURF's census loss on Sintel/KITTI sits well below 1. A value
near 13 means the predicted flow cannot explain the observed motion at all.

## Result 2 — flow quality visibly degraded

Fine-tuned output versus pretrained, at stride 1:

- Swan silhouette blurred; beak and head no longer resolved, neck thickened
- Body boundary lost its sharp edge
- Coloured speckle (green/blue/red) appears across the water and background,
  where pretrained correctly predicts no motion
- Occlusion map became blobby and diffuse instead of thin silhouette lines

PNG file size is a decent proxy: 46KB pretrained vs 75KB fine-tuned for the same
frame. A smooth confident flow field compresses; a noisy one doesn't.

## Result 3 — it is NOT a train/test displacement mismatch

YouTube-VIS stores only every 5th frame, so training pairs are ~5× further apart
in time than the consecutive DAVIS pairs used for testing. The obvious hypothesis
was that the model adapted to large displacement and merely looked bad on small
displacement.

**Tested and rejected.** Evaluating both checkpoints on stride-5 DAVIS pairs —
matching the training regime — the fine-tuned model is *still* worse
(`stride5_*_flow.png`). It loses in both regimes, including the one it trained on.

So this is not domain shift. Training genuinely damaged the model.

## Result 4 — why the visual assessment was wrong (methodological note)

Results 2 and 3 both pointed at "fine-tuning damaged the model". The quantitative
evaluation says the opposite. Both observations are real; the inference from them
was not. Two lessons worth carrying forward:

**Flow visualisations are whole-frame; the task is not.** The colour-wheel plot
weights every pixel equally, so speckle across water and sky dominates the eye.
Mask propagation depends only on flow accuracy on and near the object. The
fine-tuned model got noisier in regions that do not matter and more accurate in
the regions that do. PNG file size — which we used as an entropy proxy — measured
exactly the irrelevant thing.

**Census loss is also whole-frame.** It averages over textureless regions where
photometric matching is ill-posed, so it is dominated by pixels with no bearing
on the downstream task. A flat census curve did not mean nothing was learned; it
meant the training loss could not see what was being learned.

The stride-5 control (Result 3) is still informative, just differently: the
fine-tuned model looked worse there too, which is now explained by the same
whole-frame visual artefact rather than by a real regression. Re-running
`eval_flow_warp_davis.py --stride=5` would confirm this quantitatively and is
listed below.

## Open questions

1. **Does `train_all_frames` exist for YouTube-VIS?** Note this is now a more
   nuanced question than it first appeared. The stride-5 result shows the model
   *benefits* from large-displacement training, so consecutive-frame data would
   not be strictly better — it would be better for stride-1 propagation and
   probably worse for stride-5. Which matters depends on the frame gap the
   correspondence stage actually needs to bridge.
2. **Would better training help further?** The run was batch size 1 with only 15
   logged loss values, and only `ckpt-15` was retained. A rerun with
   `--epoch_length=500` and more frequent logging would show the actual learning
   curve and allow checkpoint bisection. Given fine-tuning already helped, longer
   or better-tuned training is now worth trying rather than speculative.
4. **Wire flow into the mask-correspondence stage.** The F-score gain is the
   result that matters for the wider project — flow supplies pixel-resolution
   boundaries where the DINOv3 OT pipeline snaps to a 16px patch lattice.
