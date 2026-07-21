#!/usr/bin/env bash
# cloze arith/copy to 20 digits (RELATIVE CKPT paths -- ckpt_for prepends AUNET_ROOT)
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack"
D="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
CKPT_SUBWORD_LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated" CUDA_VISIBLE_DEVICES=0 \
  $PY scripts/probes/numprobe.py --models subword_llama --subtasks arith_cloze copy_cloze --digits $D --n 100 \
  --out "$O/numprobe_clz20_subword_llama.jsonl" > "$O/clz20_llama.log" 2>&1 &
CKPT_AUNET_STATIC="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated" CUDA_VISIBLE_DEVICES=1 \
  $PY scripts/probes/numprobe.py --models aunet_static --subtasks arith_cloze copy_cloze --digits $D --n 100 \
  --out "$O/numprobe_clz20_aunet_static.jsonl" > "$O/clz20_aunet.log" 2>&1 &
CKPT_BYTE_GREEDYROOT="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated" CUDA_VISIBLE_DEVICES=2 \
  $PY scripts/probes/numprobe.py --models byte_greedyroot --subtasks arith_cloze copy_cloze --digits $D --n 100 \
  --out "$O/numprobe_clz20_byte_greedyroot.jsonl" > "$O/clz20_bpebyte.log" 2>&1 &
wait
echo "CLZ20 DONE"
