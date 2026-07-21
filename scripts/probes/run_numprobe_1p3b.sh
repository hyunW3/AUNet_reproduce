#!/usr/bin/env bash
# Number probes on the 1.3B paper checkpoints, one model per GPU.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"
O="$ROOT/reports/statetrack"

CKPT_SUBWORD_LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated" \
  CUDA_VISIBLE_DEVICES=0 $PY scripts/probes/numprobe.py --models subword_llama \
  --n 100 --digits 2 3 4 5 6 --batch_size 16 --out "$O/numprobe_subword_llama.jsonl" \
  > "$O/numprobe_subword_llama.log" 2>&1 &

CKPT_AUNET_STATIC="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated" \
  CUDA_VISIBLE_DEVICES=1 $PY scripts/probes/numprobe.py --models aunet_static \
  --n 100 --digits 2 3 4 5 6 --batch_size 16 --out "$O/numprobe_aunet_static.jsonl" \
  > "$O/numprobe_aunet_static.log" 2>&1 &

CKPT_BYTE_GREEDYROOT="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated" \
  CUDA_VISIBLE_DEVICES=2 $PY scripts/probes/numprobe.py --models byte_greedyroot \
  --n 100 --digits 2 3 4 5 6 --batch_size 16 --out "$O/numprobe_byte_greedyroot.jsonl" \
  > "$O/numprobe_byte_greedyroot.log" 2>&1 &

wait
echo "NUMPROBE DONE"
