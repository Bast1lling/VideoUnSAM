# coding=utf-8
"""SMURF dataset loader for YouTube-VIS TFRecords. Install as smurf/data/ytvis.py.

This is smurf/data/kitti.py's unsupervised-training path, with two changes:

  1. tf.image.decode_png  ->  tf.image.decode_jpeg
     Our records store JPEG frames (see convert_ytvis_to_tf_records.py for why:
     PNG would inflate a 10GB dataset to ~90GB).
  2. Only 'train' and 'video' modes are implemented. YouTube-VIS has no optical
     flow ground truth, so eval/test/supervised modes are meaningless here --
     they raise rather than silently misbehave. Evaluate on Sintel or KITTI.

The windowing logic is copied verbatim from kitti.py so training behaviour is
identical: each record's frame sequence is unbatched, a (size=seq_len, shift=1)
window slides over it, and each window becomes one {'images': [seq_len,H,W,3]}
example.

To wire it into SMURF, add to smurf/smurf_data.py:

    from smurf.data import ytvis                       # with the other imports

and inside make_train_dataset's if/elif chain, BEFORE the final `else`:

    elif 'ytvis' in data_format:
      dataset = ytvis.make_dataset(
          path, mode=mode, seq_len=seq_len,
          shuffle_buffer_size=shuffle_buffer_size,
          height=None if crop_instead_of_resize else height,
          width=None if crop_instead_of_resize else width,
          resize_gt_flow=resize_gt_flow, seed=seed)

Then train with --train_on="ytvis:/path/to/ytvis_tfrecords".

Note the dispatcher matches formats by SUBSTRING, so the name must not collide
with 'kitti', 'sintel', 'chairs', 'spoof' or 'multiframe'. 'ytvis' is safe.
"""

# pylint: skip-file
import os

import numpy as np
import tensorflow as tf

from smurf import smurf_utils


def parse_data(proto, height, width):
  """Parse a SequenceExample of JPEG frames into [seq, H, W, 3] float32."""
  unused_context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
      proto,
      context_features={
          'height': tf.io.FixedLenFeature([], tf.int64),
          'width': tf.io.FixedLenFeature([], tf.int64)
      },
      sequence_features={
          'images': tf.io.FixedLenSequenceFeature([], tf.string)
      })

  def deserialize(image_raw):
    # decode_jpeg, not decode_png -- this is the one substantive difference
    # from kitti.py. channels=3 guards against the occasional greyscale frame.
    image_uint = tf.image.decode_jpeg(image_raw, channels=3)
    return tf.image.convert_image_dtype(image_uint, tf.float32)

  images = tf.map_fn(deserialize, sequence_parsed['images'], dtype=tf.float32)

  if height is not None and width is not None:
    images = smurf_utils.resize(images, height, width, is_flow=False)

  return images


def make_dataset(path,
                 mode,
                 seq_len=2,
                 shuffle_buffer_size=0,
                 height=None,
                 width=None,
                 resize_gt_flow=True,
                 seed=41):
  """Build a YouTube-VIS training dataset.

  Args:
    path: dir of .tfrecord files, or 'dir1,dir2,dir3' to combine several.
    mode: 'train' or 'video'. Anything else raises -- YTVIS has no flow labels.
    seq_len: frames per example; SMURF assumes 2.
    shuffle_buffer_size: 0 disables shuffling.
    height, width: resize target, or None to keep stored resolution.
    resize_gt_flow: unused, kept for signature compatibility with kitti.py.
    seed: controls shard shuffling.

  Returns:
    tf.data.Dataset of {'images': [seq_len, H, W, 3]}, unbatched and unrepeated
    consistently with kitti.make_dataset (it repeats internally).
  """
  if 'train' not in mode and 'video' not in mode:
    raise NotImplementedError(
        f"ytvis loader supports mode 'train' and 'video', got {mode!r}. "
        "YouTube-VIS has no optical-flow ground truth, so there is nothing to "
        "evaluate against -- use --eval_on='sintel-clean:...' instead.")
  if 'sup' in mode:
    raise NotImplementedError(
        "Supervised mode needs ground-truth flow, which YouTube-VIS lacks.")

  if ',' in path:
    l = path.split(',')
    d = '/'.join(l[0].split('/')[:-1])
    l[0] = l[0].split('/')[-1]
    paths = [os.path.join(d, x) for x in l]
  else:
    paths = [path]

  files = [
      os.path.join(d, f)
      for d in paths
      for f in tf.io.gfile.listdir(d)
      if f.endswith('.tfrecord')
  ]
  if not files:
    raise ValueError(f'No .tfrecord files found under {paths}')
  num_files = len(files)

  if 'train' in mode:
    rgen = np.random.RandomState(seed)
    rgen.shuffle(files)

  ds = tf.data.Dataset.from_tensor_slices(files)
  if shuffle_buffer_size:
    ds = ds.shuffle(num_files)

  # Nested dataset: one TFRecordDataset per video file.
  ds = ds.map(tf.data.TFRecordDataset)
  # Parse each video into a frame sequence, then unbatch to individual frames.
  ds = ds.map(lambda x: x.map(
      lambda y: parse_data(y, height, width),
      num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch())
  # Slide a window of seq_len consecutive frames (shift=1 -> every pair).
  window_fn = lambda x: x.window(size=seq_len, shift=1, drop_remainder=True)
  ds = ds.interleave(
      window_fn,
      # 'video' keeps a single clip in temporal order; training interleaves
      # several clips so a batch isn't all from one scene.
      cycle_length=1 if 'video' in mode else min(10, num_files),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if shuffle_buffer_size:
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  ds = ds.repeat()
  ds = ds.flat_map(lambda x: x.batch(seq_len))
  ds = ds.map(lambda x: {'images': x})
  ds = ds.prefetch(10)
  return ds
