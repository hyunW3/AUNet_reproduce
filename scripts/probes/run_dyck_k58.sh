#!/usr/bin/env bash
# Dyck-k for k=5,6,7,8, balanced (per-close) + unbalanced (per-closer). 3 models parallel per config.
set -u; cd /mnt/ssd2/hyun2/AUNet
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/scripts/probes:$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack/dyck_k58"; mkdir -p "$O"
LL="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AU="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BP="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }
one(){ CUDA_VISIBLE_DEVICES=$4 env "$2=$3" $PY scripts/probes/dyck.py --models "$1" "${@:5}" 2>&1 | filt; }
cfg(){ local name=$1; shift
  one subword_llama   CKPT_SUBWORD_LLAMA   "$LL" 0 "$@" --out "$O/${name}_llama.jsonl" &
  one aunet_static    CKPT_AUNET_STATIC    "$AU" 1 "$@" --out "$O/${name}_aunet.jsonl" &
  one byte_greedyroot CKPT_BYTE_GREEDYROOT "$BP" 2 "$@" --out "$O/${name}_bpebyte.jsonl" &
  wait; echo "== $name done =="; }
for k in 5 6 7 8; do
  cfg dyck${k}_bal   --k $k --length 40 --maxdepth 6 --n 150
  cfg dyck${k}_unbal --k $k --length 40 --maxdepth 6 --n 150 --unbalanced
done
echo "DYCKK58 DONE"
