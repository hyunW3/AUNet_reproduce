#!/usr/bin/env bash
# Sequential scratchpad pretraining on ece-agpu11 GPUs 0,1,3 (the three free cards).
# Order: base first (fastest, and it is the reference every other arm is read against).
set -u
cd ~/AUNet_patchlag/lingua
export TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
O=~/AUNet_patchlag/runs/scratchpad
mkdir -p $O
for arm in base scratchpad1 scratchpad1frac; do
  cfg=apps/aunet/configs/sp_rgM_${arm}.yaml
  echo "### $arm start $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=0,1,3 ~/AUNet/lingua/.venv/bin/python -m torch.distributed.run \
      --nproc-per-node 3 --master-port 29620 -m apps.aunet.train config=$cfg \
      > $O/sp_rgM_${arm}.log 2>&1
  echo "### $arm done $(date +%H:%M)  last: $(grep -oE 'step: [0-9]+ .*loss: *[0-9.]+' $O/sp_rgM_${arm}.log | tail -1)"
done
echo "### queue done $(date +%H:%M)"
