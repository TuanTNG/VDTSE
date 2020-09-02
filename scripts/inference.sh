 #!/usr/bin/env bash
set -e
NAME='atss_r50_fpn_1x_street'
CONFIG="configs/coco/street/${NAME}.py"

WORKDIR="../checkpoints/transfer_weight/${NAME}"

CHECKPOINT="${WORKDIR}/epoch_12.pth"
DATADIR="/home/member/Workspace/tank/TS/data/images/full/full"
THR=0.5
OUTDIR="cache/street"
python tools/visualize_testset.py $CONFIG --ckpt $CHECKPOINT --data_dir $DATADIR --det_thr $THR --out_dir $OUTDIR --num_imgs 200