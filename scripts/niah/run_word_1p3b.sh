#!/usr/bin/env bash
# S-NIAH word-value retrieval on the 1.3B paper checkpoints: value is an unseen WORD
# (absent from the haystack). Task 4 = essay+word, task 5 = noise+word. 3 models on GPU 0/1/2.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; OUT="$ROOT/reports/niah/word_1p3b"; mkdir -p "$OUT"
LLAMA="$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AUNET="$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BPE="$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
LENGTHS="512 1024 2048 4096 6144"; DEPTHS="0.1 0.3 0.5 0.7 0.9"; NPC=4; BATCH=8
filt() { grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org"; }
word() { local fam=$1 ck=$2 tag=$3 gpu=$4
  for T in 4 5; do
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/niah/niah_probe.py --family "$fam" --ckpt "$ck" --tag "$tag" \
      --task $T --lengths $LENGTHS --depths $DEPTHS --n_per_cell $NPC --batch_size $BATCH \
      --out "$OUT/word_${tag}.jsonl" --per_out "$OUT/word_${tag}_per.jsonl" 2>&1 | filt
  done; }
word subword "$LLAMA" subword_llama   0 > "$OUT/llama.log"   2>&1 &
word aunet   "$AUNET" aunet_static    1 > "$OUT/aunet.log"   2>&1 &
word aunet   "$BPE"   byte_greedyroot 2 > "$OUT/bpebyte.log" 2>&1 &
wait
echo "===== WORD-NIAH DONE -> $OUT ====="
