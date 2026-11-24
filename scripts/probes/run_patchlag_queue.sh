#!/usr/bin/env bash
# Serial patch-lag queue for snu55 GPU 3 (the only GPU free of other tenants).
set -u
cd /mnt/ssd2/hyun2/AUNet
PY=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
CKPT=main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated
export CUDA_VISIBLE_DEVICES=3
O=reports/patch_lag
mkdir -p $O/downstream

ds () {  # ds <tag> [updater.pt]
  local tag=$1; shift
  local extra=""
  [ $# -gt 0 ] && extra="--updater $1"
  echo "### downstream $tag $(date +%H:%M)"
  $PY scripts/probes/eval_downstream_updater.py --ckpt $CKPT --tag "$tag" --bs 16 $extra \
      >> $O/downstream/run.log 2>&1
  grep "^\[$tag\]" $O/downstream/run.log | tail -4
}

tr_ () {  # tr_ <tag> <args...>
  local tag=$1; shift
  echo "### train $tag $(date +%H:%M)"
  $PY scripts/probes/train_updater.py --ckpt $CKPT --tag "$tag" --steps 6000 --bs 4 --lr 1e-3 \
      --eval_every 1000 --outdir $O "$@" > $O/$tag.log 2>&1
  grep -E "EVAL step 6000|done" $O/$tag.log | tail -2
}

ds base_frozen
ds m3_global reports/patch_lag/m3_attn_gread_updater.pt
ds c1_global reports/patch_lag/c1_gread_only_updater.pt

tr_ l3_attn_local  --kind attn --use_gread --gate input --inject local
tr_ lc1_gread_local --kind gread --gate input --inject local

ds l3_local  reports/patch_lag/l3_attn_local_updater.pt
ds lc1_local reports/patch_lag/lc1_gread_local_updater.pt
ds d2_distill reports/patch_lag/d2_combined_updater.pt
echo "### queue done $(date +%H:%M)"
