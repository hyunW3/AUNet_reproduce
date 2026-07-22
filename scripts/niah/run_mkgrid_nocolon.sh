#!/usr/bin/env bash
# Regenerate the MK-NIAH pressure-test grid (K=4, depth x context) with the
# NO-COLON needle ("...is <value>"), firmer cells (n=16).
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export CKPT_SUBWORD_LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
export CKPT_AUNET_STATIC="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
export CKPT_BYTE_GREEDYROOT="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
PY="$ROOT/lingua/.venv/bin/python"
CUDA_VISIBLE_DEVICES=0 $PY scripts/niah/mk_grid.py --K 4 --n_per_cell 16 \
  --contexts 512 1024 2048 4096 6144 --n_depths 10 \
  --out "$ROOT/reports/niah/mk_grid.jsonl" 2>&1 \
  | grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"
echo "MKGRID DONE"
