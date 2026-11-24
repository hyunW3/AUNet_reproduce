#!/usr/bin/env bash
# Capacity curve: identical FEATURES (g_read only -- zero current-patch information), identical
# injection / gate / data / schedule; the ONLY variable is the head's parameter count.
# If dBPB tracks params over a 64x sweep, the Stage-1 gain is capacity, not patch-lag recovery.
# usage: cap_lane.sh <gpu> <hidden>
set -u
cd /mnt/ssd2/hyun2/AUNet
PY=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
AU=main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated
export CUDA_VISIBLE_DEVICES=$1
H=$2
O=reports/patch_lag
TAG=cap_h${H}
ARGS="--kind gread --gate input --inject local --hidden $H"

echo "### [cap h=$H gpu$1] $(date +%H:%M)"
$PY scripts/probes/train_updater.py --ckpt "$AU" --tag $TAG --steps 6000 --bs 4 --lr 1e-3 \
    --eval_every 1000 --outdir $O $ARGS > $O/$TAG.log 2>&1
grep -E "params=|EVAL step 6000|done" $O/$TAG.log | grep -v Warning | tail -3
$PY scripts/probes/train_updater.py --ckpt "$AU" --tag $TAG $ARGS \
    --eval_only $O/${TAG}_updater.pt 2>&1 | grep -E "^\[$TAG\]" | tail -2
echo "### [cap h=$H] done $(date +%H:%M)"
