# SMURF on YouTube-VIS

Unsupervised optical flow (SMURF, Stone et al. CVPR 2021) trained on YouTube-VIS,
to supply the **optical-flow branch** of the mask-correspondence plan: warp masks
between adjacent frames to learn short-term motion.

SMURF ships converters for KITTI, Sintel and Flying Chairs only. This directory
adds YouTube-VIS support.

| File | Role |
|---|---|
| `convert_ytvis_to_tf_records.py` | YouTube-VIS JPEGs → SMURF-format TFRecords |
| `ytvis.py` | SMURF dataset loader; install as `smurf/data/ytvis.py` |

## Environment

TensorFlow, kept strictly separate from the repo's PyTorch venv.

```bash
uv venv --python 3.11 .venv-tf && source .venv-tf/bin/activate
uv pip install "tensorflow[and-cuda]==2.15.1" "tensorflow-addons==0.23.0"
uv pip install absl-py gin-config matplotlib opencv-python scipy six imageio
uv pip install "numpy==1.26.4" --force-reinstall     # MUST be last
```

Version constraints, all load-bearing:

- **TF 2.15.1** — `tensorflow-addons` was archived in 2024 and its final release
  (0.23.0) supports TF ≤ 2.15. TF 2.16 also switched to Keras 3, which breaks a
  2021 codebase.
- **`[and-cuda]`** — pip-installs CUDA 12.2 libs, so no system CUDA toolkit or
  `nvcc` is needed. Essential on a cluster where you aren't root.
- **numpy < 2, installed last** — TF 2.15 is built against numpy 1.x. If
  anything pulls numpy 2 in afterwards you get
  `_ARRAY_API not found` / `numpy.core._multiarray_umath failed to import`.

SMURF's own `requirements.txt` pins `tensorflow==2.4.1`. Ignore it — it has no
wheels for Python ≥3.9 and wants a system CUDA 11.0.

## 1. Get the data

[youtube-vos.org/dataset/vis](https://youtube-vos.org/dataset/vis/), CodaLab
account required. **YouTube-VIS 2022 shares its training set with 2021**, so
either link gives identical frames.

Extract to node-local disk, never onto shared NFS — it's ~90k small files, which
is exactly what the chair's docs warn will degrade the filesystem for everyone.

```bash
mkdir -p /tmp/ytvis && unzip -q train.zip -d /tmp/ytvis
```

Layout: `/tmp/ytvis/train/JPEGImages/<video_id>/00000.jpg, 00005.jpg, ...`

## 2. Convert

```bash
python -m video.flow.convert_ytvis_to_tf_records \
    --data_dir=/tmp/ytvis/train \
    --output_dir=$HOME/datasets/ytvis_tfrecords_360x640 \
    --height=360 --width=640
```

One TFRecord per video (~2,985 files, ~5GB total). These are few, large,
sequential files — safe to keep on shared storage, unlike the raw JPEGs, and
readable from Slurm compute nodes.

Then **always verify before training**:

```bash
python -m video.flow.convert_ytvis_to_tf_records \
    --output_dir=$HOME/datasets/ytvis_tfrecords_360x640 --verify_only
```

This replays SMURF's exact train pipeline (unbatch → sliding window) and prints
the pair shapes and mean absolute frame difference. It catches the failures that
otherwise appear only as mysteriously-bad training: wrong encoding, inconsistent
shapes within a sequence, or records with <2 frames that silently yield no pairs.

`mean|diff|` around 0.02–0.15 is healthy. Near 0.0 means duplicate frames; very
high means a scene cut or mis-sorted frames.

## 3. Wire the loader into SMURF

```bash
cp video/flow/ytvis.py ~/gr-smurf/smurf/data/ytvis.py
```

Then two edits to `smurf/smurf_data.py`. Add to the imports:

```python
from smurf.data import ytvis
```

and in `make_train_dataset`, before the final `else`:

```python
elif 'ytvis' in data_format:
  dataset = ytvis.make_dataset(
      path, mode=mode, seq_len=seq_len,
      shuffle_buffer_size=shuffle_buffer_size,
      height=None if crop_instead_of_resize else height,
      width=None if crop_instead_of_resize else width,
      resize_gt_flow=resize_gt_flow, seed=seed)
```

The dispatcher matches formats by **substring**, so the name must not collide
with `kitti`, `sintel`, `chairs`, `spoof` or `multiframe`. `ytvis` is safe.

## 4. Train

Fine-tune from a pretrained checkpoint rather than starting cold — practical
students get one 24GB GPU for a maximum of 8 hours, and SMURF's default schedule
is 75k steps.

```bash
# pretrained Sintel weights (better for general video than KITTI or chairs)
mkdir -p ~/smurf_ckpts/sintel && cd ~/smurf_ckpts/sintel
for f in checkpoint ckpt-16.index ckpt-16.data-00000-of-00001; do
  curl -sSO "https://storage.googleapis.com/gresearch/smurf/sintel-smurf/$f"
done
```

```bash
python -m smurf.smurf_main \
    --train_on="ytvis:$HOME/datasets/ytvis_tfrecords_360x640" \
    --eval_on="sintel-clean:$HOME/datasets/sintel_clean/train" \
    --checkpoint_dir=$HOME/smurf_ckpts/ytvis_run1 \
    --plot_dir=$HOME/smurf_plots/ytvis_run1 \
    --height=360 --width=640 --global_gpu_batch_size=1
```

Copy the pretrained files into a **fresh** `checkpoint_dir` first so you
fine-tune rather than overwrite the originals. Evaluate on Sintel, not
YouTube-VIS — YTVIS has no flow ground truth, which is why `ytvis.py` raises on
eval mode instead of silently misbehaving.

**Skip the multi-frame self-supervision stage.** SMURF's own README says it
"will take an extremely long time on a single machine" and that they ran it
distributed across many shards.

## Open issue: frame stride

YouTube-VIS stores only every **5th** frame of the source video (`00000, 00005,
00010, ...`). Adjacent files are therefore ~5× further apart in time than the
adjacent frames SMURF was designed around.

This matters because unsupervised flow is trained by photometric consistency:
warp frame 1 to frame 2 and minimise appearance difference. Larger displacement
means more occlusion, more brightness change, and more points leaving the frame —
all of which weaken that signal. Expect degraded results relative to Sintel/KITTI
training, and don't attribute it to a bug.

Two mitigations, in order of preference:

1. **Look for a `train_all_frames` archive** on the YouTube-VOS download page.
   The parent dataset publishes full-frame-rate versions; if one exists for VIS,
   it gives true adjacent pairs and this problem disappears.
2. **Accept it and document it.** Large-displacement flow is arguably the more
   useful regime for mask correspondence anyway, since VideoUnSAM needs to match
   masks across the same stride-5 frames that YouTube-VIS annotates.

Worth resolving before spending GPU hours on a long run.
