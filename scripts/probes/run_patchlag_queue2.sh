#!/usr/bin/env bash
# Queue 2 — patch_offset feature + BPEByte-rg (root_greedy). snu55 GPU 3, serial.
set -u
cd /mnt/ssd2/hyun2/AUNet
PY=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
AU=main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated
RG=main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated
export CUDA_VISIBLE_DEVICES=3
O=reports/patch_lag

# wait for the currently running job to release the GPU
until grep -q "^\[lr1_res_local\] done" $O/lr1_res_local.log; do sleep 60; done
echo "### lr1 done, queue2 starts $(date +%H:%M)"

tr_ () { local tag=$1 ck=$2; shift 2
  echo "### train $tag $(date +%H:%M)"
  $PY scripts/probes/train_updater.py --ckpt "$ck" --tag "$tag" --steps 6000 --bs 4 --lr 1e-3 \
      --eval_every 1000 --outdir $O "$@" > $O/$tag.log 2>&1
  grep -E "params=|EVAL step 6000|done" $O/$tag.log | grep -v Warning | tail -3
}
ev () { local tag=$1 ck=$2; shift 2
  $PY scripts/probes/train_updater.py --ckpt "$ck" --tag "$tag" "$@" \
      --eval_only $O/${tag}_updater.pt 2>&1 | grep -E "^\[$tag\]" | tail -2
}

# --- A. does patch_offset as an explicit feature help?  (aunet, matched to L3) --------------
tr_ l3o_local "$AU" --kind attn --use_gread --use_offset --gate input --inject local
ev  l3o_local "$AU" --kind attn --use_gread --use_offset --gate input --inject local

# --- B. BPEByte-rg: is the Stage -1 structure the same under BPE-root patching? -------------
echo "### rg verify $(date +%H:%M)"
$PY scripts/probes/patch_lag_verify.py "$RG" > $O/rg_verify.log 2>&1
grep -E "^T[1-5]|patches|   r=" $O/rg_verify.log | head -12

echo "### rg oracle $(date +%H:%M)"
$PY scripts/probes/patch_lag_oracle.py --ckpt "$RG" --tag rg_oracle \
    --modes oracle,prev1,look1 --outdir $O > $O/rg_oracle.log 2>&1
grep -A14 "by patch_offset" $O/rg_oracle.log | tail -14

# --- C. BPEByte-rg: does the treatment/control tie reproduce? -------------------------------
tr_ l3o_rg  "$RG" --kind attn --use_gread --use_offset --gate input --inject local
ev  l3o_rg  "$RG" --kind attn --use_gread --use_offset --gate input --inject local
tr_ lc1_rg  "$RG" --kind gread --gate input --inject local
ev  lc1_rg  "$RG" --kind gread --gate input --inject local
echo "### queue2 done $(date +%H:%M)"
