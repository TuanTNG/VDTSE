 #!/usr/bin/env bash
set -e
CONFIG="configs/coco/street/atss_r50_fpn_1x_street_pretrain.py"

WORKDIR="../checkpoints/transfer_weight/draft"

CHECKPOINT="${WORKDIR}/epoch_12.pth"
DATADIR="/data/Toda/data_for_train/knot_data_set/01_train"
THR=0.5
OUTDIR="cache/street"
python tools/visualize_pred.py $CONFIG --ckpt $CHECKPOINT --data_dir $DATADIR --det_thr $THR --out_dir $OUTDIR