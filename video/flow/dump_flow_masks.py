# coding=utf-8
"""Precompute SMURF-flow-warped masks as a refinelab sidecar. TF venv, not the
repo's PyTorch one -- see video/flow/README.md.

Reads an EXISTING dump written by video/scripts/dump_artifacts.py (PyTorch venv,
run separately/earlier), and for every frame t >= 1 warps frame (t-1)'s `ot_mask`
forward into frame t using SMURF backward flow -- the same warp_mask convention
as eval_flow_warp_davis.py. Writes dump/flow/<clip>/<fidx>.npz, bit-packed like
the rest of the dump, so refinelab/bench.py can load it with no TF import at all.

This deliberately warps the CHAIN's own ot_mask (whatever OT actually produced,
right or wrong), not the ground truth -- that's what a real flow_snap refiner
would have available at inference time. It is therefore NOT the same experiment
as eval_flow_warp_davis.py (which is GT-to-GT, oracle-initialised); it measures
flow quality under the pipeline's own, possibly-already-wrong, previous mask.

    # after: python -m video.scripts.dump_artifacts --out-dir dumps/davis2016_default
    python -m video.flow.dump_flow_masks \
        --dump dumps/davis2016_default \
        --checkpoint_dir=$HOME/smurf_ckpts/ytvis_finetuned
"""

import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from smurf import smurf_flags  # pylint:disable=unused-import

from video.flow.eval_flow_warp_davis import warp_mask  # pylint:disable=g-import-not-at-top

FLAGS = flags.FLAGS

flags.DEFINE_string('dump', '', 'Dump directory written by dump_artifacts.py.')
flags.DEFINE_string('checkpoint_dir', '', 'SMURF checkpoint (required).')


def _unpack(bits: np.ndarray, shape) -> np.ndarray:
  h, w = int(shape[0]), int(shape[1])
  return np.unpackbits(bits)[: h * w].reshape(h, w).astype(np.uint8)


def clips_in(dump: str) -> list[str]:
  npz_dir = os.path.join(dump, 'npz')
  return sorted(d for d in os.listdir(npz_dir)
                if os.path.isdir(os.path.join(npz_dir, d)))


def frames_in(dump: str, clip: str) -> list[int]:
  d = os.path.join(dump, 'npz', clip)
  return sorted(int(f[:-4]) for f in os.listdir(d) if f.endswith('.npz'))


def load_frame_rgb(dump: str, clip: str, fidx: int) -> np.ndarray:
  bgr = cv2.imread(os.path.join(dump, 'frames', clip, f'{fidx:05d}.jpg'))
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_ot_mask(dump: str, clip: str, fidx: int) -> np.ndarray:
  z = np.load(os.path.join(dump, 'npz', clip, f'{fidx:05d}.npz'))
  return _unpack(z['ot_mask'], z['shape'])


def main(unused_argv):
  if not FLAGS.dump:
    raise ValueError('--dump is required')
  if not FLAGS.checkpoint_dir:
    raise ValueError('--checkpoint_dir is required')

  import gin
  from smurf import smurf_evaluator
  gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
  smurf = smurf_evaluator.build_network(batch_size=1)
  smurf.update_checkpoint_dir(FLAGS.checkpoint_dir)
  smurf.restore()

  clips = clips_in(FLAGS.dump)
  for i, clip in enumerate(clips, 1):
    frames = frames_in(FLAGS.dump, clip)
    out_dir = os.path.join(FLAGS.dump, 'flow', clip)
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    for t in frames:
      if t == 0 or (t - 1) not in frames:
        continue
      prev_mask = load_ot_mask(FLAGS.dump, clip, t - 1)
      img1 = tf.image.convert_image_dtype(load_frame_rgb(FLAGS.dump, clip, t - 1), tf.float32)
      img2 = tf.image.convert_image_dtype(load_frame_rgb(FLAGS.dump, clip, t), tf.float32)
      _, _, flow_bw = smurf.infer(
          img1, img2, input_height=FLAGS.height, input_width=FLAGS.width,
          infer_occlusion=True, infer_bw=True)
      flow_mask = warp_mask(prev_mask, flow_bw.numpy())

      h, w = flow_mask.shape
      np.savez_compressed(
          os.path.join(out_dir, f'{t:05d}.npz'),
          flow_mask=np.packbits(flow_mask.astype(bool)),
          shape=np.array([h, w], dtype=np.int32),
      )
      n_written += 1
    print(f'[{i}/{len(clips)}] {clip}: wrote {n_written} flow masks', flush=True)

  print(f'\nDone. Sidecar written under {FLAGS.dump}/flow/ -- '
        f'run refinelab/bench.py --refiners flow_snap,flow_guided_snap next.')


if __name__ == '__main__':
  app.run(main)
