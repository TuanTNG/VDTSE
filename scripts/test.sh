#!/usr/bin/env bash
set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

WORKDIR="/home/cybercore/tank/TS/checkpoints/transfer_weight/atss_r18_fpn_2x_street_lr001"
CONFIG="configs/street/atss_r18_fpn_2x_street_lr001.py"

CHECKPOINT="${WORKDIR}/epoch_12.pth"
RESULT="${WORKDIR}/epoch_12.pkl"

GPUS=2
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --out $RESULT --eval bbox