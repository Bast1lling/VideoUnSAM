# coding=utf-8
"""Evaluate optical flow by mask-warping J&F on DAVIS.

WHAT THIS MEASURES
------------------
Flow quality expressed in the metric this project actually cares about. For each
consecutive pair (t, t+1): take the GROUND-TRUTH mask at frame t, warp it forward
using the predicted flow, and score the result against the ground-truth mask at
frame t+1 with the standard DAVIS J (region IoU) and F (boundary F-score).

This answers "how well does this flow propagate a mask one frame", which is
precisely what the optical-flow branch of the correspondence plan needs.

WHAT THIS IS *NOT*
------------------
It is NOT comparable to the pipeline's 0.624 J&F headline number, and reporting
it as such would be wrong. Two different tasks:

  * The 0.624 figure is END-TO-END: one user click on frame 0, then propagate
    through the whole clip with no further supervision. Errors accumulate over
    ~70 frames.
  * This is SINGLE-STEP and ORACLE-INITIALISED: every frame starts from the
    ground-truth mask of the previous frame. No error accumulation at all.

So this is an upper bound on per-frame flow-propagation quality. It belongs in a
table of its own, against the baselines below -- not next to the pipeline number.

THE BASELINE THAT MATTERS
-------------------------
`--method copy` skips warping entirely and predicts mask(t+1) = mask(t). On slow
clips that scores surprisingly well. If flow warping does not beat copy, the flow
is contributing nothing, no matter how good its J&F looks in isolation. Always
run both.

    python -m video.flow.eval_flow_warp_davis \
        --checkpoint_dir=$HOME/smurf_ckpts/sintel \
        --davis_root=/storage/group/dataset_mirrors/davis \
        --out=$HOME/flowjf_pretrained.json

    python -m video.flow.eval_flow_warp_davis \
        --method=copy --davis_root=/storage/group/dataset_mirrors/davis \
        --out=$HOME/flowjf_copy.json
"""

import json
import os
import sys
import time

from absl import app
from absl import flags

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import tensorflow as tf

from smurf import smurf_flags  # pylint:disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_string('davis_root', '/storage/group/dataset_mirrors/davis',
                    'DAVIS root containing JPEGImages/480p and Annotations/480p.')
flags.DEFINE_string('clips', '', 'Comma-separated clip names; default is the '
                    'DAVIS 2016 val 20.')
flags.DEFINE_string('method', 'flow', 'flow | copy. "copy" is the no-flow '
                    'control: predict mask(t+1) = mask(t).')
flags.DEFINE_integer('stride', 1, 'Frame gap. Use 5 to match the YouTube-VIS '
                     'training stride.')
flags.DEFINE_integer('max_frames', 0, 'Debug: cap frames per clip.')
flags.DEFINE_string('out', '', 'Optional path to write per-clip JSON results.')

DAVIS_2016_VAL = [
    'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',
    'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane',
    'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby',
    'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox',
]


# ---- DAVIS metrics: identical to video/scripts/eval_davis2016.py -------------

def j_score(pred, gt):
  pred, gt = pred.astype(bool), gt.astype(bool)
  union = (pred | gt).sum()
  return float((pred & gt).sum() / union) if union else 0.0


def _boundary(mask, tol):
  struct = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
  return mask.astype(bool) & ~binary_erosion(mask.astype(bool), struct)


def f_score(pred, gt, tol=3):
  pb, gb = _boundary(pred, tol), _boundary(gt, tol)
  if pb.sum() == 0 and gb.sum() == 0:
    return 1.0
  if pb.sum() == 0 or gb.sum() == 0:
    return 0.0
  struct = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
  prec = float((pb & binary_dilation(gb, struct)).sum()) / (pb.sum() + 1e-8)
  rec = float((gb & binary_dilation(pb, struct)).sum()) / (gb.sum() + 1e-8)
  return 2 * prec * rec / (prec + rec) if prec + rec > 1e-8 else 0.0


# ---- IO ---------------------------------------------------------------------

def frame_path(clip, idx):
  return os.path.join(FLAGS.davis_root, 'JPEGImages', '480p', clip, f'{idx:05d}.jpg')


def mask_path(clip, idx):
  return os.path.join(FLAGS.davis_root, 'Annotations', '480p', clip, f'{idx:05d}.png')


def load_frame_rgb(clip, idx):
  bgr = cv2.imread(frame_path(clip, idx))
  if bgr is None:
    raise FileNotFoundError(frame_path(clip, idx))
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(clip, idx):
  """Binary uint8. The mirror stores DAVIS 2016 style {0,255}, so read as
  greyscale and threshold -- avoids PIL palette handling entirely."""
  m = cv2.imread(mask_path(clip, idx), cv2.IMREAD_GRAYSCALE)
  if m is None:
    raise FileNotFoundError(mask_path(clip, idx))
  return (m > 0).astype(np.uint8)


def num_frames(clip):
  d = os.path.join(FLAGS.davis_root, 'JPEGImages', '480p', clip)
  return len([f for f in os.listdir(d) if f.endswith('.jpg')])


# ---- Warping ----------------------------------------------------------------

def warp_mask(mask_t, flow_bw):
  """Warp mask at frame t into frame t+1 using BACKWARD flow.

  flow_bw is the flow from frame t+1 back to frame t, so for each pixel x in
  frame t+1 it says where that content came from in frame t:

      warped(x) = mask_t(x + flow_bw(x))

  which is a gather, implementable directly with cv2.remap. (Forward flow would
  require a scatter, which leaves holes and needs splatting.)

  NOTE ON CHANNEL ORDER: SMURF returns flow as (dy, dx), not (dx, dy). See
  smurf/data/kitti.py, which does `final_flow = flow[..., ::-1]` before comparing
  against (u, v) ground truth. Getting this backwards produces a plausible-looking
  but transposed warp, so it is easy to miss.
  """
  h, w = mask_t.shape
  dy = flow_bw[..., 0].astype(np.float32)
  dx = flow_bw[..., 1].astype(np.float32)
  xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
  map_x = xs + dx
  map_y = ys + dy
  warped = cv2.remap(mask_t.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
  return (warped > 0.5).astype(np.uint8)


# ---- Eval -------------------------------------------------------------------

def eval_clip(clip, smurf):
  n = num_frames(clip)
  if FLAGS.max_frames:
    n = min(n, FLAGS.max_frames)
  s = FLAGS.stride

  js, fs = [], []
  for t in range(0, n - s, s):
    gt_t = load_mask(clip, t)
    gt_next = load_mask(clip, t + s)
    if gt_next.sum() == 0:
      continue

    if FLAGS.method == 'copy':
      pred = gt_t
    else:
      img1 = tf.image.convert_image_dtype(load_frame_rgb(clip, t), tf.float32)
      img2 = tf.image.convert_image_dtype(load_frame_rgb(clip, t + s), tf.float32)
      _, _, flow_bw = smurf.infer(
          img1, img2, input_height=FLAGS.height, input_width=FLAGS.width,
          infer_occlusion=True, infer_bw=True)
      pred = warp_mask(gt_t, flow_bw.numpy())

    js.append(j_score(pred, gt_next))
    fs.append(f_score(pred, gt_next))

  if not js:
    return None
  j, f = float(np.mean(js)), float(np.mean(fs))
  return {'clip': clip, 'j': j, 'f': f, 'jf': (j + f) / 2, 'pairs': len(js)}


def main(unused_argv):
  clips = [c.strip() for c in FLAGS.clips.split(',') if c.strip()] or DAVIS_2016_VAL

  smurf = None
  if FLAGS.method != 'copy':
    if not FLAGS.checkpoint_dir:
      raise ValueError('--checkpoint_dir is required unless --method=copy')
    import gin
    from smurf import smurf_evaluator
    gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
    smurf = smurf_evaluator.build_network(batch_size=1)
    smurf.update_checkpoint_dir(FLAGS.checkpoint_dir)
    smurf.restore()

  rows = []
  t0 = time.time()
  for i, clip in enumerate(clips, 1):
    r = eval_clip(clip, smurf)
    if r is None:
      continue
    rows.append(r)
    print(f'[{i}/{len(clips)}] {clip:<20} J={r["j"]:.3f} F={r["f"]:.3f} '
          f'J&F={r["jf"]:.3f}  ({r["pairs"]} pairs)', flush=True)

  j = float(np.mean([r['j'] for r in rows]))
  f = float(np.mean([r['f'] for r in rows]))
  print('\n' + '=' * 58)
  print(f'method={FLAGS.method}  stride={FLAGS.stride}  '
        f'ckpt={os.path.basename(FLAGS.checkpoint_dir or "-")}')
  print(f'J mean   {j:.4f}')
  print(f'F mean   {f:.4f}')
  print(f'J&F      {(j + f) / 2:.4f}    over {len(rows)} clips, '
        f'{time.time() - t0:.0f}s')
  print('=' * 58)
  print('Single-step, oracle-initialised: each frame starts from the previous\n'
        'frame\'s GROUND-TRUTH mask. Not comparable to the end-to-end pipeline\n'
        'number -- compare only against --method=copy and other checkpoints.')

  if FLAGS.out:
    with open(FLAGS.out, 'w') as fh:
      json.dump({'method': FLAGS.method, 'stride': FLAGS.stride,
                 'checkpoint_dir': FLAGS.checkpoint_dir,
                 'j': j, 'f': f, 'jf': (j + f) / 2, 'per_clip': rows}, fh, indent=2)
    print(f'\nWrote {FLAGS.out}')


if __name__ == '__main__':
  app.run(main)
