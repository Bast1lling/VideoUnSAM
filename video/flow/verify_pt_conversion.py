# coding=utf-8
"""Verify a TF->PyTorch SMURF weight conversion by comparing flow on real frames.

A weight conversion that mismaps layers still produces plausible-looking flow, so
"the output looks like optical flow" proves nothing. This runs BOTH models on the
same DAVIS frame pair and measures the end-point error between them. If the
conversion is correct that error should be a small fraction of a pixel.

It also DETECTS the channel convention rather than assuming it. SMURF's TF code
returns flow as (dy, dx) -- see smurf/data/kitti.py doing `flow[..., ::-1]` before
comparing to (u, v) ground truth -- while TorchVision's RAFT, which the PyTorch
port is built on, returns (dx, dy). Getting this wrong yields a transposed flow
field that looks entirely reasonable in a colour-wheel plot. The script tries both
orders and reports which one matches, so downstream code can be written against a
verified fact instead of a guess.

Needs BOTH frameworks in one environment (the same one used for the conversion):

    cd ~/gr-smurf && source .venv-tf/bin/activate
    export PYTHONPATH=$HOME/VideoUnSAM          # NOT smurf-pytorch, see below
    python -m video.flow.verify_pt_conversion \
        --tf_ckpt_dir=$HOME/smurf_ckpts/ytvis_ft15k \
        --pt_ckpt=$HOME/smurf_ckpts/ytvis_ft15k.pt \
        --pt_repo=$HOME/smurf-pytorch

IMPORTANT -- both repos define a top-level package named `smurf`, and they cannot
coexist on sys.path. Google's TF version (~/gr-smurf/smurf) has no __init__.py so
it is a NAMESPACE package; the PyTorch port has one so it is a REGULAR package.
Python's finder prefers a regular package over a namespace package regardless of
path order, so simply putting smurf-pytorch on PYTHONPATH makes the TF import
fail with `cannot import name 'smurf_flags' from 'smurf'`.

This script therefore uses the TF package first, then evicts it from sys.modules
and inserts the port's path before importing it. Pass the port's location with
--pt_repo instead of adding it to PYTHONPATH.
"""

import os
import sys

from absl import app
from absl import flags

import cv2
import numpy as np
import tensorflow as tf

from smurf import smurf_flags  # pylint:disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_string('tf_ckpt_dir', '', 'Directory of the TF checkpoint.')
flags.DEFINE_string('pt_ckpt', '', 'Path to the converted .pt file.')
flags.DEFINE_string('pt_repo', os.path.expanduser('~/smurf-pytorch'),
                    'Clone of ChristophReich1996/SMURF. Loaded by path rather '
                    'than PYTHONPATH to avoid the `smurf` package collision.')
flags.DEFINE_string('davis_root', '/storage/group/dataset_mirrors/davis', '')
flags.DEFINE_string('clip', 'blackswan', 'DAVIS clip to test on.')
flags.DEFINE_integer('frame', 0, 'First frame of the pair.')
flags.DEFINE_float('tol', 0.5, 'Max acceptable mean EPE between the two models, '
                   'in pixels. The port has minor documented implementation '
                   'differences from official SMURF, so expect small but nonzero '
                   'disagreement -- not bit-identical output.')


def load_rgb(clip, idx):
  p = os.path.join(FLAGS.davis_root, 'JPEGImages', '480p', clip, f'{idx:05d}.jpg')
  bgr = cv2.imread(p)
  if bgr is None:
    raise FileNotFoundError(p)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def epe(a, b):
  """Mean end-point error between two [H, W, 2] flow fields."""
  return float(np.mean(np.sqrt(((a - b) ** 2).sum(axis=-1))))


def main(unused_argv):
  img1 = load_rgb(FLAGS.clip, FLAGS.frame)
  img2 = load_rgb(FLAGS.clip, FLAGS.frame + 1)
  h, w = img1.shape[:2]
  print(f'{FLAGS.clip} frames {FLAGS.frame},{FLAGS.frame + 1}  ({h}x{w})\n')

  # ---- TensorFlow reference -------------------------------------------------
  import gin
  from smurf import smurf_evaluator
  gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
  net = smurf_evaluator.build_network(batch_size=1)
  net.update_checkpoint_dir(FLAGS.tf_ckpt_dir)
  net.restore()
  flow_tf = net.infer(
      tf.image.convert_image_dtype(img1, tf.float32),
      tf.image.convert_image_dtype(img2, tf.float32),
      input_height=FLAGS.height, input_width=FLAGS.width).numpy()
  print(f'TF   flow {flow_tf.shape}  '
        f'ch0 [{flow_tf[...,0].min():+.2f},{flow_tf[...,0].max():+.2f}]  '
        f'ch1 [{flow_tf[...,1].min():+.2f},{flow_tf[...,1].max():+.2f}]')

  # ---- PyTorch port ---------------------------------------------------------
  # The TF model is fully built and its weights restored by this point, and we
  # hold a live reference to it in `net`, so it is safe to evict the TF `smurf`
  # package and let the PyTorch port claim that name instead.
  import torch
  for name in [k for k in sys.modules if k == 'smurf' or k.startswith('smurf.')]:
    del sys.modules[name]
  if not os.path.isdir(os.path.join(FLAGS.pt_repo, 'smurf')):
    raise ValueError(f'No smurf/ package under --pt_repo={FLAGS.pt_repo!r}. '
                     'Clone https://github.com/ChristophReich1996/SMURF there.')
  sys.path.insert(0, FLAGS.pt_repo)
  from smurf import raft_smurf   # now resolves to the PyTorch port

  # Run the port at the SAME internal resolution the TF model used, then map the
  # flow back to native size. Two reasons this matters:
  #   1. RAFT's feature encoder downsamples by 8, so both dimensions must be
  #      divisible by 8. DAVIS is 480x854 and 854 % 8 == 6, which the port
  #      rejects outright. The 296x640 training resolution is divisible by 8.
  #   2. TF's infer() resizes internally to input_height/width and scales the
  #      flow back up. Feeding the port native resolution would compare two
  #      different computations, not two implementations of the same one.
  ih, iw = FLAGS.height, FLAGS.width
  if ih % 8 or iw % 8:
    raise ValueError(f'--height/--width must be divisible by 8, got {ih}x{iw}')

  def to_pt(img):
    small = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_LINEAR)
    # The port documents [-1, 1] normalisation, unlike TF SMURF's [0, 1].
    t = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
    return (2.0 * t - 1.0)[None]

  model = raft_smurf(checkpoint=FLAGS.pt_ckpt)
  model.eval()
  with torch.no_grad():
    out = model(to_pt(img1), to_pt(img2))
  # The model returns the flow at each refinement iteration; the last is final.
  flow_pt = (out[-1] if isinstance(out, (list, tuple)) else out)
  flow_pt = flow_pt[0].permute(1, 2, 0).cpu().numpy()     # [ih, iw, 2]
  print(f'PT   flow {flow_pt.shape} @ {ih}x{iw} before rescale  '
        f'ch0 [{flow_pt[...,0].min():+.2f},{flow_pt[...,0].max():+.2f}]  '
        f'ch1 [{flow_pt[...,1].min():+.2f},{flow_pt[...,1].max():+.2f}]\n')

  # Upsample to native size. Flow VECTORS must be scaled too, not just the grid --
  # a displacement of 5px at 640 wide is 6.67px at 854 wide.
  sy, sx = h / float(ih), w / float(iw)
  up = cv2.resize(flow_pt, (w, h), interpolation=cv2.INTER_LINEAR)

  # Which channel gets sy and which gets sx depends on the channel convention,
  # which is exactly what we are trying to determine -- so build both candidates.
  cand_dydx = up * np.array([sy, sx], dtype=np.float32)   # port is (dy, dx), as TF
  cand_dxdy = (up * np.array([sx, sy], dtype=np.float32))[..., ::-1]  # port is (dx, dy)

  # ---- Compare, testing both channel conventions ----------------------------
  epe_same = epe(flow_tf, cand_dydx)
  epe_swap = epe(flow_tf, cand_dxdy)
  print(f'EPE assuming PT is (dy, dx) = {epe_same:.4f} px')
  print(f'EPE assuming PT is (dx, dy) = {epe_swap:.4f} px')

  best, order = ((epe_same, 'SAME — both are (dy, dx)')
                 if epe_same <= epe_swap else
                 (epe_swap, 'SWAPPED — TF is (dy, dx), PyTorch is (dx, dy)'))
  print(f'\nChannel order: {order}')
  print(f'Best mean EPE: {best:.4f} px')

  if best <= FLAGS.tol:
    print(f'\nPASS — agreement within {FLAGS.tol} px. Conversion looks correct.')
  else:
    print(f'\nFAIL — {best:.4f} px exceeds the {FLAGS.tol} px tolerance.')
    print('A mismapped conversion still yields plausible-looking flow, so do not\n'
          'trust the .pt on visual inspection. Check that --tf_checkpoint pointed\n'
          'at the checkpoint PREFIX (e.g. .../ckpt-15) and not the directory or a\n'
          '.index/.data file.')


if __name__ == '__main__':
  app.run(main)
