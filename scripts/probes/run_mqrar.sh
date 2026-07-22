#!/usr/bin/env bash
# Official MQRAR (sundai-research/mqrar-bench) in-context eval, N=4/8/16/32, 3 models parallel.
set -u; cd /mnt/ssd2/hyun2/AUNet
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/scripts/probes:$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack/mqrar"; mkdir -p "$O"
LL="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AU="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BP="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn|Potential|labels = torch"; }
one(){ CUDA_VISIBLE_DEVICES=$4 env "$2=$3" $PY scripts/probes/mqrar_eval.py --models "$1" \
  --n_kv 4 8 16 32 --vocab 256 --seq_len 192 --n_seq 30 --shots 2 --batch_size 16 \
  --out "$O/mqrar_$1.jsonl" 2>&1 | filt; }
one subword_llama   CKPT_SUBWORD_LLAMA   "$LL" 0 > "$O/llama.log"   2>&1 &
one aunet_static    CKPT_AUNET_STATIC    "$AU" 1 > "$O/aunet.log"   2>&1 &
one byte_greedyroot CKPT_BYTE_GREEDYROOT "$BP" 2 > "$O/bpebyte.log" 2>&1 &
wait; echo "MQRAR DONE"
