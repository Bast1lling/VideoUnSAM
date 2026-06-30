#!/usr/bin/env bash
# Data learning-curve for the box-only decoder: step-matched (~6600 image-steps) so the
# comparison isolates data diversity, not compute. From scratch each time, eval on the
# leak-free held-out set (64 DAVIS clips, davis_split.json).
set -e
cd /home/nilsc/VideoUnSAM
source .venv/bin/activate

run () {  # frac epochs tag
  echo "########## frac=$1 epochs=$2 (lc_$3) ##########"
  python -u -m video.decoder.train_sam_decoder \
    --coco datasets/davis_train/annotations/train.json \
    --image-root datasets/davis_train/images \
    --frac "$1" --epochs "$2" --cache-feats \
    --out "checkpoints/lc_$3.pth" 2>&1 | grep -E "frac|data|epoch +(0|39|79|159)\b|save"
  echo "----- heldout eval lc_$3 -----"
  python -u -m video.decoder.eval_heldout --ckpt "checkpoints/lc_$3.pth" --frames 3 2>/dev/null \
    | grep -E "heldout|mIoU"
}

run 0.25 160 25
run 0.5  80  50
run 1.0  40  100
echo "########## LEARNING CURVE DONE ##########"
