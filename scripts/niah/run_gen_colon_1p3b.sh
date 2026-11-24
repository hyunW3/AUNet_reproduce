#!/usr/bin/env bash
# Generation-style S-NIAH-1/2/3 on the subword Llama Transformer: free-running
# greedy decode (48 tok) + RULER substring scoring, comparing the needle WITH the
# original RULER colon ("...is: {value}") vs WITHOUT ("...is {value}").
# Same seed -> identical haystack/key/value per pair; only the colon differs.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; OUT="$ROOT/reports/niah/gen_colon"; mkdir -p "$OUT"
LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
LEN="512 1024 2048 4096 6144"; DEP="0.1 0.3 0.5 0.7 0.9"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }

sn(){ local task=$1 gpu=$2;
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/niah/niah_gen_probe.py \
    --family subword --ckpt "$ROOT/$LLAMA" --tag subword_llama \
    --tasks $task --needle_seps "" ":" --lengths $LEN --depths $DEP \
    --n_per_cell 4 --batch_size 8 --max_gen_len 48 \
    --out "$OUT/gen_S${task}.jsonl" --per_out "$OUT/per_S${task}.jsonl" 2>&1 | filt; }

# GPUS: space-separated list of 3 GPU ids for tasks 1/2/3 (default 1 2 3)
read -ra G <<< "${GPUS:-1 2 3}"
sn 1 "${G[0]}" > "$OUT/S1.log" 2>&1 &
sn 2 "${G[1]}" > "$OUT/S2.log" 2>&1 &
sn 3 "${G[2]}" > "$OUT/S3.log" 2>&1 &
wait
echo "GEN_COLON DONE"
