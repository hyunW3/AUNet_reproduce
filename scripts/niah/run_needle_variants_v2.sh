#!/bin/bash
# Score the v2 needle variants (tok4, rand4, rand4x4, rand4x5, tok4x11; items_v2.jsonl) with the same scorer as the
# original variants (score_items.py, teacher-forced greedy exact). snu55: AU-Net then Llama on GPU 2, BPEByte on GPU 3.
cd /mnt/ssd2/hyun2/AUNet/lingua; export PYTHONPATH=$PWD AUNET_ROOT=/mnt/ssd2/hyun2/AUNet; source ../scripts/probes/patch_stats/env.sh
O=/mnt/ssd2/hyun2/AUNet/reports/niah/needle_variants
S=../scripts/probes/rep_probes/score_items.py
run() {  # family gpu
  CUDA_VISIBLE_DEVICES=$2 .venv/bin/python $S --family $1 --items $O/items_v2.jsonl --out $O/out/$1_v2.jsonl > $O/out/$1_v2.log 2>&1
  echo "$1 exit=$?" >> $O/out/done_v2.log
}
(run aunet 2; run llama 2) &
run bpebyte 3 &
wait
echo V2_DONE >> $O/out/done_v2.log
