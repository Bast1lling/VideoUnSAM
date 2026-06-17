# Conquer phase – hyperparameter guide

## What the thetas do

`iterative_merge` runs a hierarchical graph-cut on the DINOv3 feature grid.
At each theta it merges neighbouring patches whose **cosine similarity exceeds theta**.

| theta (high) | only very similar patches merge → small, fine-grained segments |
|---|---|
| theta (low)  | even dissimilar patches merge → large, coarse segments |

The default schedule `[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]` is run top-to-bottom, producing one set of segments per level.  All surviving segments from every level are pooled together and then filtered by the coverage threshold and NMS.

---

## Why DINOv3 produces fewer conquer masks than DINO-B/8

### 1. Feature space is more globally uniform

DINO's *key* vectors (extracted via a hook on the last attention layer) are unnormalised and have high within-image variance — nearby patches for different objects differ sharply in cosine similarity.  DINOv3 returns the *last hidden state*, which passes through a final LayerNorm.  LayerNorm compresses the dynamic range: the distribution of pairwise cosine similarities is narrower and shifted closer to 1.

**Consequence:** even at theta = 0.6 most patch pairs already exceed the threshold, so the entire crop merges into one region.  The full-crop segment then fails the coverage check (it covers the divide mask 100 % on the first iteration, gets kept once, and NMS suppresses everything that follows).

### 2. Lower spatial resolution

| Backbone | patch size | local_size=256 → grid |
|---|---|---|
| DINO-B/8  | 8 px  | 32 × 32 = 1024 tokens |
| DINOv3-L/16 | 16 px | 16 × 16 = 256 tokens |

Four times fewer tokens means four times fewer candidate splits for the graph-cut, so the pool of candidate masks is much smaller before coverage/NMS filtering.

---

## How to tune

### Increase spatial resolution (most impactful)

Set **Local crop size → 512**.  With patch_size=16 that gives a 32 × 32 token grid, matching DINO-B/8 at local_size=256.

### Raise the thetas

Because the DINOv3 similarity distribution is compressed towards 1.0, the original schedule `[0.6 … 0.1]` is too coarse – every level merges everything.  Try starting higher:

```
0.95, 0.92, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70
```

This forces the algorithm to split regions that are only very slightly dissimilar.

### Lower the coverage threshold

The default `kept_thresh = 0.9` discards any conquer mask that does not cover ≥ 90 % of the divide-mask crop.  With a uniform feature grid the segments tend to be either the whole crop or tiny fragments.  Lowering to **0.6–0.75** lets more partial segments through.

### Combined starting point for DINOv3

| Parameter | DINO-B/8 default | DINOv3 suggestion |
|---|---|---|
| Local crop size | 256 | **512** |
| Thetas | 0.6 … 0.1 | **0.95, 0.92, 0.90, 0.85, 0.80, 0.70** |
| Coverage threshold | 0.9 | **0.7** |
| NMS IoU | 0.9 | 0.9 (unchanged) |

> DINOv3 runs in full float32 precision, so there is no quantisation-induced similarity collapse on top of the above effects.
