#!/usr/bin/env bash
# S5 permutation composition at controlled depths 1,2,3 (2-shot), reporting BOTH
# exact-match and 5-way rank choice. One process, 3 models sequentially.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/scripts/probes:$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export CKPT_SUBWORD_LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
export CKPT_AUNET_STATIC="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
export CKPT_BYTE_GREEDYROOT="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
PY="$ROOT/lingua/.venv/bin/python"
CUDA_VISIBLE_DEVICES=3 $PY scripts/probes/s5_depth.py \
  --models subword_llama aunet_static byte_greedyroot --depths 1 2 3 --shots 2 --n 300 \
  --batch_size 16 --out "$ROOT/reports/statetrack/s5_depth_results.jsonl" 2>&1 \
  | grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"
echo "S5DEPTH DONE"
