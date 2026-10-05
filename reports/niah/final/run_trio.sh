#!/bin/bash
# Final S-NIAH set (0.5k/1k/2k n=250 + fit-4k) with per-item output, matched trio, snu55 GPU 2.
cd /mnt/ssd2/hyun2/AUNet/lingua; export PYTHONPATH=$PWD AUNET_ROOT=/mnt/ssd2/hyun2/AUNet; source ../scripts/probes/patch_stats/env.sh
P=../reports/niah/sniah123_n250_final_pairs.jsonl; O=../reports/niah/final; M=../main/main/1.3B
for spec in "subword llama $M/llama_1.8B_paper/checkpoints/0000060000/consolidated" \
            "aunet aunet $M/aunet2_1.3B/checkpoints/0000180000/consolidated" \
            "aunet bpebyte $M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=${GPU:-2} .venv/bin/python ../scripts/niah/score_probe_pairs.py --family $1 --tag $2 --ckpt $3 \
    --pairs $P --out $O/trio_cells.jsonl --items_out $O/$2_items.jsonl > $O/$2.log 2>&1
  echo "$2 exit=$?" >> $O/trio_done.log
done
echo ALL_DONE >> $O/trio_done.log
