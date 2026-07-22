#!/usr/bin/env bash
# Extend our Dyck probe: Dyck-4 (balanced) + Dyck-3/Dyck-4 unbalanced-prefix
# (BIG-bench style whole-completion exact-match). 3 models in parallel per config.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/scripts/probes:$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack/dyck_ext"; mkdir -p "$O"
LL="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AU="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BP="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }
one(){ local tag=$1 ckenv=$2 ck=$3 gpu=$4 outn=$5; shift 5;
  CUDA_VISIBLE_DEVICES=$gpu env "$ckenv=$ck" $PY scripts/probes/dyck.py --models "$tag" "$@" \
    --out "$O/${outn}.jsonl" 2>&1 | filt; }
cfg(){ local name=$1; shift;
  one subword_llama   CKPT_SUBWORD_LLAMA   "$LL" 0 "${name}_llama"   "$@" &
  one aunet_static    CKPT_AUNET_STATIC    "$AU" 1 "${name}_aunet"   "$@" &
  one byte_greedyroot CKPT_BYTE_GREEDYROOT "$BP" 2 "${name}_bpebyte" "$@" &
  wait; echo "== $name done =="; }
cfg dyck4_bal   --k 4 --length 40 --maxdepth 6 --n 150
cfg dyck3_unbal --k 3 --length 40 --maxdepth 6 --n 150 --unbalanced
cfg dyck4_unbal --k 4 --length 40 --maxdepth 6 --n 150 --unbalanced
echo "DYCKEXT DONE"
