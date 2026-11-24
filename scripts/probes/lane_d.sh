#!/usr/bin/env bash
# Lane D — snu55 GPU 2.  Co-tenanting one other user's ~3 GB process, authorised 2026-09-01
# for this run only; the standing rule (do not share a card with another user) still holds.
#
# Runs the CONTROL that exactly matches the rg treatment: l3o_rg sees the current patch's bytes
# plus g_read plus patch_offset, so its control must keep g_read and patch_offset and drop only
# the bytes.  (lc1_rg on lane C drops patch_offset too, which makes it the offset ablation.)
set -u
cd /mnt/ssd2/hyun2/AUNet
PY=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
RG=main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated
export CUDA_VISIBLE_DEVICES=2
O=reports/patch_lag
TAG=lc1o_rg
ARGS="--kind gread --use_offset --gate input --inject local"

echo "### [D gpu2] train $TAG $(date +%H:%M)"
$PY scripts/probes/train_updater.py --ckpt "$RG" --tag $TAG --steps 6000 --bs 4 --lr 1e-3 \
    --eval_every 1000 --outdir $O $ARGS > $O/$TAG.log 2>&1
grep -E "params=|EVAL step 6000|done" $O/$TAG.log | grep -v Warning | tail -3
$PY scripts/probes/train_updater.py --ckpt "$RG" --tag $TAG $ARGS \
    --eval_only $O/${TAG}_updater.pt 2>&1 | grep -E "^\[$TAG\]" | tail -2
echo "### [D] done $(date +%H:%M)"
