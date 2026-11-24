#!/usr/bin/env bash
# ece-agpu11 GPUs 5,6,7 -- the two follow-ups the patch_offset decomposition asked for, plus the
# depth control the ece block was missing.  Same recipe/data/steps as sp_rgM_{base,scratchpad1},
# so every number lands in the existing ece block (base = 1.2601 BPB).
#
#   l3o_50m_start     the l3o_50m head (identical params, inputs and dense compute) pointed at
#                     patch_offset == 0 instead of >= 1.  Read against the already-trained
#                     sp_rgM_l3o_50m, so it is self-contained.
#   base_150m         +50 M of trunk depth, no head.  The ece block had no depth reference.
#   scratchpad1_150m  scratchpad + depth together.  offset 0 / offset>=1 split says the two axes
#                     are orthogonal (scratchpad buys -0.0162 at offset>=1 and nothing at 0;
#                     depth buys -0.0165 and -0.0363).  If that holds this beats base_150m.
#
# GPUs 0-3 and 4 carry other tenants' jobs; 5,6,7 were the only idle cards and the run refuses to
# start if that has changed.
set -u
cd ~/AUNet_patchlag/lingua
export TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
PY=~/AUNet/lingua/.venv/bin/python
O=~/AUNet_patchlag/runs/scratchpad
mkdir -p $O

for g in 5 6 7; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -gt 500 ]; then
    echo "### ABORT: GPU $g now holds ${used} MiB -- another tenant moved in, not starting."
    exit 1
  fi
done
echo "### GPUs 5,6,7 idle, starting $(date +%F\ %H:%M)"

for arm in l3o_50m_start base_150m scratchpad1_150m; do
  cfg=apps/aunet/configs/sp_rgM_${arm}.yaml
  echo "### $arm start $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=5,6,7 $PY -m torch.distributed.run --nproc-per-node 3 \
      --master-port 29650 -m apps.aunet.train config=$cfg > $O/sp_rgM_${arm}.log 2>&1
  echo "### $arm done $(date +%H:%M)  last: $(grep -oE 'step: [0-9]+ .*loss: *[0-9.]+' $O/sp_rgM_${arm}.log | tail -1)"
done
echo "### ece2 queue done $(date +%F\ %H:%M)"
