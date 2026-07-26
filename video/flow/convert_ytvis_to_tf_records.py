# coding=utf-8
"""Convert YouTube-VIS (2021/2022 train split) to SMURF-compatible TFRecords.

SMURF ships converters for KITTI, Sintel and Flying Chairs only. This adds
YouTube-VIS, whose layout is:

    <data_dir>/JPEGImages/<video_id>/00000.jpg, 00005.jpg, 00010.jpg, ...

One TFRecord is written per video, holding a single tf.train.SequenceExample:

    context      : height (int64), width (int64)
    feature_list : images -> [JPEG bytes, JPEG bytes, ...]

which matches what smurf/data/kitti.py:parse_data expects, EXCEPT that frames
are JPEG-encoded rather than PNG. SMURF's parser hardcodes tf.image.decode_png,
so this must be read with the companion loader video/flow/ytvis.py (installed as
smurf/data/ytvis.py) which uses decode_jpeg.

Why JPEG and not PNG: re-encoding ~90k natural-image frames as lossless PNG
turns a 10GB dataset into roughly 90GB. JPEG keeps it a few GB and decodes
faster. The images are already lossy JPEGs, so re-encoding at quality 95 after a
downscale costs nothing meaningful.

SMURF's train pipeline unbatches each sequence into frames and slides a
(size=seq_len, shift=1) window over them, so storing whole videos yields every
consecutive pair automatically. Do not pre-pair the frames.

    python -m video.flow.convert_ytvis_to_tf_records \
        --data_dir=/tmp/ytvis/train \
        --output_dir=$HOME/datasets/ytvis_tfrecords_360x640 \
        --height=360 --width=640

Then verify before training on it:

    python -m video.flow.convert_ytvis_to_tf_records \
        --output_dir=$HOME/datasets/ytvis_tfrecords_360x640 --verify_only
"""

import os
import sys
import time

from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '',
                    'YouTube-VIS split root, i.e. the dir containing JPEGImages.')
flags.DEFINE_string('output_dir', '', 'Where to write the TFRecords.')
flags.DEFINE_integer('height', 360, 'Target height. YTVIS is mostly 720x1280 '
                     '(16:9); 360x640 preserves that and is divisible by 8, '
                     'which RAFT requires.')
flags.DEFINE_integer('width', 640, 'Target width.')
flags.DEFINE_integer('jpeg_quality', 95, 'Re-encode quality.')
flags.DEFINE_integer('limit_videos', 0, 'Debug: only convert the first N videos.')
flags.DEFINE_integer('min_frames', 2, 'Skip videos with fewer frames than this '
                     '(a pair needs at least 2).')
flags.DEFINE_bool('verify_only', False,
                  'Skip conversion; just read output_dir back through the same '
                  'window pipeline SMURF uses and report what it yields.')
flags.DEFINE_bool('overwrite', False, 'Re-convert videos whose TFRecord exists.')


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _frame_sort_key(filename):
  """YTVIS frames are zero-padded ints ('00000.jpg'), but stride 5 -- so sorting
  must be numeric, not lexicographic, or pairs come out shuffled."""
  return int(os.path.splitext(filename)[0])


def convert_video(video_dir, output_path):
  """Write one video's frames as a single SequenceExample. Returns #frames."""
  frames = [f for f in os.listdir(video_dir) if f.lower().endswith('.jpg')]
  if len(frames) < FLAGS.min_frames:
    return 0
  try:
    frames.sort(key=_frame_sort_key)
  except ValueError:
    # Unexpected naming -- fall back to lexicographic, which is correct for
    # fixed-width zero-padded names anyway.
    frames.sort()

  image_bytes_list = []
  for fname in frames:
    raw = tf.io.read_file(os.path.join(video_dir, fname))
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img[None], [FLAGS.height, FLAGS.width],
                          method=tf.image.ResizeMethod.BILINEAR)[0]
    img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)
    image_bytes_list.append(
        tf.io.encode_jpeg(img, quality=FLAGS.jpeg_quality).numpy())

  example = tf.train.SequenceExample(
      context=tf.train.Features(feature={
          'height': _int64_feature(FLAGS.height),
          'width': _int64_feature(FLAGS.width),
      }),
      feature_lists=tf.train.FeatureLists(feature_list={
          'images': tf.train.FeatureList(
              feature=[_bytes_feature(b) for b in image_bytes_list]),
      }))

  # Write to a temp name then rename, so an interrupted run never leaves a
  # truncated record that would poison training.
  tmp_path = output_path + '.partial'
  with tf.io.TFRecordWriter(tmp_path) as w:
    w.write(example.SerializeToString())
    w.flush()
  os.replace(tmp_path, output_path)
  return len(frames)


def convert_dataset():
  image_root = os.path.join(FLAGS.data_dir, 'JPEGImages')
  if not os.path.isdir(image_root):
    raise ValueError(f'No JPEGImages under {FLAGS.data_dir!r}. Pass the split '
                     f'root, e.g. --data_dir=/tmp/ytvis/train')
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  videos = sorted(d for d in os.listdir(image_root)
                  if os.path.isdir(os.path.join(image_root, d)))
  if FLAGS.limit_videos:
    videos = videos[:FLAGS.limit_videos]

  total_frames = 0
  skipped = 0
  reused = 0
  t0 = time.time()
  for i, vid in enumerate(videos, 1):
    out_path = os.path.join(FLAGS.output_dir, f'{vid}.tfrecord')
    if os.path.exists(out_path) and not FLAGS.overwrite:
      reused += 1
      continue
    n = convert_video(os.path.join(image_root, vid), out_path)
    if n == 0:
      skipped += 1
    total_frames += n
    if i % 50 == 0 or i == len(videos):
      elapsed = time.time() - t0
      rate = i / max(elapsed, 1e-6)
      eta = (len(videos) - i) / max(rate, 1e-6)
      print(f'[{i}/{len(videos)}] frames={total_frames} skipped={skipped} '
            f'reused={reused} {rate:.1f} vid/s eta={eta/60:.1f}min', flush=True)

  print(f'\nWrote {len(videos) - skipped - reused} records to {FLAGS.output_dir}')
  print(f'  frames converted : {total_frames}')
  print(f'  videos skipped   : {skipped} (fewer than {FLAGS.min_frames} frames)')
  print(f'  already existed  : {reused}')
  print(f'  resolution       : {FLAGS.height}x{FLAGS.width}')


def verify():
  """Read the records back exactly the way SMURF's train pipeline does.

  Catches the failure modes that otherwise show up as silently-garbage training:
  wrong encoding, inconsistent shapes within a sequence, or records holding
  fewer than 2 frames (which yield no pairs at all).
  """
  files = [os.path.join(FLAGS.output_dir, f)
           for f in sorted(os.listdir(FLAGS.output_dir))
           if f.endswith('.tfrecord')]
  if not files:
    raise ValueError(f'No .tfrecord files in {FLAGS.output_dir}')
  print(f'{len(files)} records found. Inspecting the first 3.\n')

  for path in files[:3]:
    ds = tf.data.TFRecordDataset(path)
    for proto in ds.take(1):
      ctx, seq = tf.io.parse_single_sequence_example(
          proto,
          context_features={'height': tf.io.FixedLenFeature([], tf.int64),
                            'width': tf.io.FixedLenFeature([], tf.int64)},
          sequence_features={'images': tf.io.FixedLenSequenceFeature([], tf.string)})
      imgs = tf.map_fn(
          lambda s: tf.image.convert_image_dtype(
              tf.image.decode_jpeg(s, channels=3), tf.float32),
          seq['images'], fn_output_signature=tf.float32)
      print(f'  {os.path.basename(path)}: '
            f'{int(ctx["height"])}x{int(ctx["width"])}, '
            f'{imgs.shape[0]} frames, dtype={imgs.dtype.name}, '
            f'range=[{float(tf.reduce_min(imgs)):.3f}, '
            f'{float(tf.reduce_max(imgs)):.3f}]')

  # Now the actual SMURF windowing: unbatch to frames, slide a size-2 window.
  seq_len = 2
  ds = tf.data.Dataset.from_tensor_slices(files[:3])
  ds = ds.map(tf.data.TFRecordDataset)

  def parse(proto):
    _, seq = tf.io.parse_single_sequence_example(
        proto,
        context_features={'height': tf.io.FixedLenFeature([], tf.int64),
                          'width': tf.io.FixedLenFeature([], tf.int64)},
        sequence_features={'images': tf.io.FixedLenSequenceFeature([], tf.string)})
    return tf.map_fn(
        lambda s: tf.image.convert_image_dtype(
            tf.image.decode_jpeg(s, channels=3), tf.float32),
        seq['images'], fn_output_signature=tf.float32)

  ds = ds.map(lambda x: x.map(parse).unbatch())
  ds = ds.interleave(lambda x: x.window(size=seq_len, shift=1, drop_remainder=True),
                     cycle_length=3)
  ds = ds.flat_map(lambda x: x.batch(seq_len))

  print('\nWindowed pairs (what the trainer actually sees):')
  n_pairs = 0
  for pair in ds.take(3):
    print(f'  pair shape {tuple(pair.shape)}  '
          f'mean|diff|={float(tf.reduce_mean(tf.abs(pair[1] - pair[0]))):.4f}')
    n_pairs += 1
  if n_pairs == 0:
    print('  !! NO PAIRS PRODUCED -- records probably hold <2 frames each.')
  else:
    print(f'\nOK: pipeline yields [{seq_len}, {FLAGS.height}, {FLAGS.width}, 3] '
          f'pairs. A mean|diff| of roughly 0.02-0.15 is normal; near 0.0 means '
          f'duplicate frames, very high means a scene cut or mis-ordered frames.')


def main(unused_argv):
  if not FLAGS.output_dir:
    raise ValueError('Must pass --output_dir')
  if FLAGS.verify_only:
    verify()
    return
  if not FLAGS.data_dir:
    raise ValueError('Must pass --data_dir (the split root containing JPEGImages)')
  convert_dataset()
  print('\nNow verify with --verify_only before training.')


if __name__ == '__main__':
  app.run(main)
