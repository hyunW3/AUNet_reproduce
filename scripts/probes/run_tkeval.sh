#!/usr/bin/env bash
# Official TKEval (floatai/TKEval) on the 1.3B paper checkpoints: 10 tasks, 2-shot,
# big tasks subsampled to 500. One model per GPU.
set -u; cd /mnt/ssd2/hyun2/AUNet
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/scripts/probes:$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/tkeval"; mkdir -p "$O"
LL="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AU="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BP="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }
one(){ CUDA_VISIBLE_DEVICES=$4 env "$2=$3" $PY scripts/probes/tkeval_probe.py --models "$1" \
  --out "$O/results_$1.jsonl" 2>&1 | filt > "$O/$1.log"; echo "done $1 exit=$?"; }
one subword_llama   CKPT_SUBWORD_LLAMA   "$LL" 0 &
one aunet_static    CKPT_AUNET_STATIC    "$AU" 1 &
one byte_greedyroot CKPT_BYTE_GREEDYROOT "$BP" 2 &
wait; echo "TKEVAL DONE"
