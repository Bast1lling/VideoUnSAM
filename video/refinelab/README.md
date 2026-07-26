# refinelab — offline boundary-refinement bench

Develop mask cleanup/snapping on a laptop. No GPU, no DINOv3, no detectron2.

## Why

The OT heatmap lives on a 64×64 patch grid. At 480p that's 16 px per patch, so bilinear
upsampling + thresholding makes boundaries snap to a coarse lattice. That's the J/F gap
in the current results — **J 0.635 vs F 0.571** — and it's worst exactly where you'd
predict:

| Clip | J | F | gap |
|---|---|---|---|
| car-roundabout | 0.875 | 0.640 | 0.234 |
| car-shadow | 0.817 | 0.595 | 0.223 |
| drift-straight | 0.720 | 0.518 | 0.203 |
| dog | 0.897 | 0.747 | 0.150 |

Refinement is a pure function of `(heatmap, frame, proposals)`. So dump those once on the
cluster, then iterate offline in seconds instead of 30-minute GPU evals.

## Step 1 — dump artifacts (cluster, once)

```bash
salloc --part=PRACT --qos=practical_course --nodes=1 --cpus-per-task=4 \
       --mem=32G --gres=gpu:1,VRAM:24G --time=0-04:00:00

python -m video.scripts.dump_artifacts \
    --davis-root /storage/slurm/$USER/davis/DAVIS \
    --out-dir /storage/slurm/$USER/dumps/davis2016_default
```

~35 min for all 20 clips, ~250 MB. Start with 5–6 clips to sanity-check, using the
high-J/F-gap ones above since those are where refinement should pay off.

Then pull it down:

```bash
scp -P 58022 -r s0041@<workstation>.in.tum.de:/storage/slurm/s0041/dumps/davis2016_default ./dumps/
```

## Step 1b — flow sidecar (TF venv, optional, once)

Only needed for `flow_snap` / `flow_guided_snap`. SMURF (`video/flow/`) needs the
project's separate TF venv (`.venv-tf`, see `video/flow/README.md`) -- it cannot be
imported into the same process as `dump_artifacts.py`, which is PyTorch/DINOv3. So
this is a second pass over the *existing* dump, not an extra flag on step 1:

```bash
source .venv-tf/bin/activate
python -m video.flow.dump_flow_masks \
    --dump dumps/davis2016_default \
    --checkpoint_dir=$HOME/smurf_ckpts/ytvis_finetuned
```

Writes `dump/flow/<clip>/<fidx>.npz` — the previous frame's `ot_mask` warped forward
with SMURF flow, bit-packed like everything else in the dump. `bench.py` loads it with
plain numpy, no TF import needed at bench time. Note this warps the chain's *actual*
`ot_mask` (right or wrong), not ground truth — deliberately different from
`video/flow/eval_flow_warp_davis.py`'s oracle-initialised GT-to-GT numbers, since this
is what a real refiner would have to work with at inference time.

## Step 2 — iterate (laptop)

```bash
pip install numpy opencv-python scipy pycocotools

python -m video.refinelab.bench --dump dumps/davis2016_default \
    --refiners baseline,guided,snap,snap_best,guided_snap
```

`baseline` reproduces current pipeline behaviour, so it's your control. Everything is
measured as a delta against it.

Sweep parameters:

```bash
python -m video.refinelab.bench --dump dumps/davis2016_default \
    --refiners snap --sweep "region_thresh=0.25,0.35,0.45 agg=frac,mean,median"
```

Per-clip detail (sorted worst-first):

```bash
python -m video.refinelab.bench --dump dumps/davis2016_default \
    --refiners guided_snap --params "region_thresh=0.35,radius=8" --per-clip
```

## The refiners

| Name | Idea |
|---|---|
| `baseline` | Threshold the upsampled heatmap. What ships today. |
| `guided` | Guided filter against image structure, then threshold. **`--guided` already exists in `eval_davis2016.py` and has never been evaluated** — no results for it appear anywhere in the README. Cheapest possible win. |
| `snap` | Ask "is this *region* hot?" instead of "is this *pixel* hot?" — score each conquer sub-mask by heat and accept/reject whole regions. Boundaries become object-aligned by construction. |
| `snap_best` | Take only the single best-scoring proposal. Control for whether `snap`'s gains come from boundary quality or from unioning extra regions. |
| `guided_snap` | Guided filter to clean the heat, then region snapping. |
| `crf` / `crf_snap` | The current Dense CRF, for A/B. Needs `pydensecrf`. |
| `flow_snap` | Blend the OT-heat threshold with the SMURF-warped previous-frame mask (needs the flow sidecar, step 1b). `blend` controls the mix; a bad flow warp can drag the fused probability around, so sweep `blend` low-to-high rather than starting at 0.5. |
| `flow_guided_snap` | Same flow-warped mask, but as one more candidate region for `snap`'s region-vote instead of a pixel blend — a bad warp is simply outvoted rather than blended in. Worth A/B'ing against `flow_snap` for exactly that difference. |

`snap` is the idea worth the most attention: conquer already produces ~30 tight,
pixel-resolution sub-masks per frame via DINOv3 spectral clustering. Their boundaries
follow real silhouettes. Reusing them as superpixels costs nothing and sidesteps the
patch lattice entirely.

## Important caveat

The dump's OT chain was run at a fixed threshold, and `ot_mask` feeds back into the chain
at reseed frames via `pick_proposal`. So:

- **Faithful offline:** anything that changes only the *display* mask — all the refiners here.
- **Not faithful offline:** changing `--thresh`, `--guided`, or `--cc-filter` in a way that
  alters the chain. The bench prints a warning if you touch one.

Always confirm a win with a real `eval_davis2016` run on the cluster before putting it in
the README results table.

## Adding your own

Add a function to `refiners.py` with the signature

```python
def my_refiner(soft_up, frame_rgb, props, **params) -> np.ndarray:  # binary uint8 [H, W]
```

and register it in `REFINERS`. It's immediately available to `--refiners` and `--sweep`.
