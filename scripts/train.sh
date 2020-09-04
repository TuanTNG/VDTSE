#!/usr/bin/env bash
set -e
CFG="atss_r50_fpn_1x_street"
WORKDIR="../TS/checkpoints/transfer_weight/${CFG}"
CONFIG="configs/street/${CFG}.py"
GPUS=2
LOAD_FROM="../TS/checkpoints/pretrained/atss_r50_fpn_1x_coco.pth"
export CUDA_VISIBLE_DEVICES=0,1
bash tools/dist_train.sh $CONFIG $GPUS --work-dir $WORKDIR  --options DATA_ROOT=$DATA_ROOT --load_from $LOAD_FROM

# fps
python tools/benchmark.py configs/coco/street/${CFG}.py
