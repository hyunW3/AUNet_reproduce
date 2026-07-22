#!/usr/bin/env bash
# VT (RULER Variable Tracking) sweeping the VALUE length 1..6 digits, at hops 1 and 2.
# Isolates copy fidelity vs value length (R=1 = pure retrieval) and one hop deeper.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/scripts/probes:$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack/vt_digits"; mkdir -p "$O"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }
vt(){ local tag=$1 ckenv=$2 ck=$3 gpu=$4;
  CUDA_VISIBLE_DEVICES=$gpu env "$ckenv=$ck" $PY scripts/probes/vt.py --models "$tag" \
    --hop_counts 1 2 --value_digits 1 2 3 4 5 6 --num_chains 3 --n 50 --batch_size 8 \
    --out "$O/vt_${tag}.jsonl" 2>&1 | filt; }
vt subword_llama   CKPT_SUBWORD_LLAMA   "main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"          0 > "$O/llama.log"   2>&1 &
vt aunet_static    CKPT_AUNET_STATIC    "main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"              1 > "$O/aunet.log"   2>&1 &
vt byte_greedyroot CKPT_BYTE_GREEDYROOT "main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated" 2 > "$O/bpebyte.log" 2>&1 &
wait
echo "VTDIGITS DONE"
