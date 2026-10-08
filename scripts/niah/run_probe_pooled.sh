#!/bin/bash
# Linear probes on patch vectors (S-NIAH-3 + 2 random-UUID copies per item). AU-Net GPU 2, BPEByte GPU 3 (snu55).
S=$(dirname "$(readlink -f "$0")")/probe_pooled.py
cd /mnt/ssd2/hyun2/AUNet/lingua; export PYTHONPATH=$PWD AUNET_ROOT=/mnt/ssd2/hyun2/AUNet; source ../scripts/probes/patch_stats/env.sh
O=/mnt/ssd2/hyun2/AUNet/reports/niah/probe_pooled
M=../main/main/1.3B
run() {  # model ckpt gpu
  CUDA_VISIBLE_DEVICES=$3 .venv/bin/python $S extract --model $1 --ckpt $2 --out $O/$1.pt > $O/$1_extract.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$3 .venv/bin/python $S probe --data $O/$1.pt --out $O/$1.json > $O/$1_probe.log 2>&1
  echo "$1 exit=$?" >> $O/done.log
}
run aunet $M/aunet2_1.3B/checkpoints/0000180000/consolidated 2 &
run bpebyte $M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated 3 &
wait
echo ALL_DONE >> $O/done.log
