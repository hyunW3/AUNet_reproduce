#!/bin/bash
# S-NIAH-3 prompts with the UUID replaced by 7 four-letter groups joined by '-' (pairs_{tok4,rand4}.jsonl):
# tok4 = 4-letter lowercase BPE vocab tokens, rand4 = random letters. Per-byte greedy, AU-Net GPU 2, BPEByte GPU 3.
S=$(dirname "$(readlink -f "$0")")/force_byte_needle.py
cd /mnt/ssd2/hyun2/AUNet/lingua; export PYTHONPATH=$PWD AUNET_ROOT=/mnt/ssd2/hyun2/AUNet; source ../scripts/probes/patch_stats/env.sh
O=/mnt/ssd2/hyun2/AUNet/reports/niah/force_byte
M=../main/main/1.3B
run() {  # model ckpt gpu
  for k in tok4 rand4; do
    CUDA_VISIBLE_DEVICES=$3 .venv/bin/python $S --tag $1 --force none --ckpt $2 --pairs $O/pairs_$k.jsonl --per_byte \
      --out $O/diag_${k}_$1.jsonl > $O/diag_${k}_$1.log 2>&1
    echo "$k $1 exit=$?" >> $O/done.log
  done
}
run aunet $M/aunet2_1.3B/checkpoints/0000180000/consolidated 2 &
run bpebyte $M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated 3 &
wait
echo NEEDLE4_DONE >> $O/done.log
