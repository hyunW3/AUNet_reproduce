#!/usr/bin/env bash
# FFLM at 2-shot (matched to the paper's 0-shot config: num_instr=64, n=100,
# n_sparse=1000, max_reads=8) so FFLM is on the same few-shot footing as Dyck/S5.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/fflm/2shot"; mkdir -p "$O"
LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AUNET="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BPE="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }
ff(){ local fam=$1 ck=$2 tag=$3 gpu=$4;
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/fflm/fflm_probe.py --family "$fam" --ckpt "$ROOT/$ck" --tag "$tag" \
    --regimes dense indist sparse --num_instr 64 --n 100 --n_sparse 1000 --max_reads 8 \
    --shots 2 --batch_size 32 --out "$O/ff2_${tag}.jsonl" 2>&1 | filt; }
ff subword "$LLAMA" subword_llama   0 > "$O/llama.log"   2>&1 &
ff aunet   "$AUNET" aunet_static    1 > "$O/aunet.log"   2>&1 &
ff aunet   "$BPE"   byte_greedyroot 2 > "$O/bpebyte.log" 2>&1 &
wait
echo "FFLM2SHOT DONE"
