#!/usr/bin/env bash
set -e
CFG="atss_r18_fpn_2x_street_lr001"
WORKDIR="../TS/checkpoints/transfer_weight/${CFG}"
CONFIG="configs/coco/street/${CFG}.py"
GPUS=2

export CUDA_VISIBLE_DEVICES=0,1
bash tools/dist_train.sh $CONFIG $GPUS --work-dir $WORKDIR  --options DATA_ROOT=$DATA_ROOT #--no-validate

# fps
python tools/benchmark.py configs/coco/street/${CFG}.py
