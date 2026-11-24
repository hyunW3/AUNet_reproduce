#!/usr/bin/env bash
# snu20 (A5000 x4, all free): the l3o_local arm -- a trained lightweight head instead of a second
# trunk pass -- against its own baseline, on the recipe's own data (dclm_r10lr_2chunk lives here).
# base first: it is the reference the other arm is read against, and it is the cheaper of the two.
set -u
cd /mnt/ssd/hyun/aunet_sp/lingua
export TMPDIR=/mnt/ssd/hyun/tmp OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/mnt/ssd/hyun/aunet_r10lr/venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}
PY=/mnt/ssd/hyun/aunet_r10lr/venv/bin/python
O=/mnt/ssd/hyun/aunet_sp/runs/scratchpad
mkdir -p $O $TMPDIR
for arm in base l3o_local; do
  echo "### $arm start $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --nproc-per-node 4 \
      --master-port 29640 -m apps.aunet.train \
      config=apps/aunet/configs/sp_snu20rgM_${arm}.yaml > $O/sp_snu20rgM_${arm}.log 2>&1
  echo "### $arm done $(date +%H:%M)  $(grep -oE 'step: [0-9]+ .*loss: *[0-9.]+' $O/sp_snu20rgM_${arm}.log | tail -1)"
done
echo "### snu20 queue done $(date +%H:%M)"
