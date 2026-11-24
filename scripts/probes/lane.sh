#!/usr/bin/env bash
# One serial lane pinned to one GPU.  usage: lane.sh <gpu> <lane-name>
# snu55 has 4 A5000s: 0/1/3 are ours, 2 carries another user's process and is left alone.
set -u
cd /mnt/ssd2/hyun2/AUNet
PY=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
AU=main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated
RG=main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated
export CUDA_VISIBLE_DEVICES=$1
LANE=$2
O=reports/patch_lag

tr_ () {
  local tag=$1 ck=$2; shift 2
  echo "### [$LANE gpu$CUDA_VISIBLE_DEVICES] train $tag $(date +%H:%M)"
  $PY scripts/probes/train_updater.py --ckpt "$ck" --tag "$tag" --steps 6000 --bs 4 --lr 1e-3 \
      --eval_every 1000 --outdir $O "$@" > $O/$tag.log 2>&1
  grep -E "params=|EVAL step 6000|done" $O/$tag.log | grep -v Warning | tail -3
  $PY scripts/probes/train_updater.py --ckpt "$ck" --tag "$tag" "$@" \
      --eval_only $O/${tag}_updater.pt 2>&1 | grep -E "^\[$tag\]" | tail -2
}

case $LANE in
  A)   # aunet offset treatment, then the rg treatment
    tr_ l3o_local "$AU" --kind attn --use_gread --use_offset --gate input --inject local
    tr_ l3o_rg    "$RG" --kind attn --use_gread --use_offset --gate input --inject local
    ;;
  B)   # rg structure check first (fast), then the offset control on aunet
    echo "### [$LANE] rg verify $(date +%H:%M)"
    $PY scripts/probes/patch_lag_verify.py "$RG" > $O/rg_verify.log 2>&1
    grep -E "^T[1-5]|patches|   r=" $O/rg_verify.log | head -12
    echo "### [$LANE] rg oracle $(date +%H:%M)"
    $PY scripts/probes/patch_lag_oracle.py --ckpt "$RG" --tag rg_oracle \
        --modes oracle,prev1,look1 --outdir $O > $O/rg_oracle.log 2>&1
    grep -A12 "by patch_offset" $O/rg_oracle.log | tail -12
    tr_ lc1o_local "$AU" --kind gread --use_offset --gate input --inject local
    ;;
  C)   # waits for lr1 to free the card, then the rg control
    until grep -q "^\[lr1_res_local\] done" $O/lr1_res_local.log; do sleep 60; done
    echo "### [$LANE] lr1 released the GPU $(date +%H:%M)"
    grep -E "EVAL step 6000|done" $O/lr1_res_local.log | tail -2
    tr_ lc1_rg "$RG" --kind gread --gate input --inject local
    ;;
esac
echo "### [$LANE] done $(date +%H:%M)"
