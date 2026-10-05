#!/bin/bash
# S-NIAH fit-4k cell (BOS + prompt + answer <= 4096 B) for the matched trio, snu55 GPU 2 only.
cd /mnt/ssd2/hyun2/AUNet/lingua; export PYTHONPATH=/mnt/ssd2/hyun2/AUNet/lingua; source ../scripts/probes/patch_stats/env.sh; export AUNET_ROOT=/mnt/ssd2/hyun2/AUNet
P=../reports/niah/sniah123_n250_fit4k_pairs.jsonl; O=../reports/niah/fit4k; M=../main/main/1.3B
for spec in "subword llama $M/llama_1.8B_paper/checkpoints/0000060000/consolidated" \
            "aunet aunet $M/aunet2_1.3B/checkpoints/0000180000/consolidated" \
            "aunet bpebyte $M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python ../scripts/niah/score_probe_pairs.py --family $1 --tag $2 --ckpt $3 \
    --pairs $P --out $O/trio.jsonl > $O/$2.log 2>&1
  echo "$2 exit=$?" >> $O/trio_done.log
done
echo ALL_DONE >> $O/trio_done.log
