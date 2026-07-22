#!/usr/bin/env bash
set -u; cd /mnt/ssd2/hyun2/AUNet
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/fflm/hf"; mkdir -p "$O"
LL="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AU="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BP="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn|Repo card|terminate called"; }
ff(){ CUDA_VISIBLE_DEVICES=$4 $PY scripts/fflm/fflm_hf.py --family "$1" --ckpt "$ROOT/$2" --tag "$3" \
  --splits val val_dense val_sparse --n_seq 200 --max_reads 8 --batch_size 32 --out "$O/hf_$3.jsonl" 2>&1 | filt; }
ff subword "$LL" subword_llama   0 > "$O/llama.log"   2>&1 &
ff aunet   "$AU" aunet_static    1 > "$O/aunet.log"   2>&1 &
ff aunet   "$BP" byte_greedyroot 2 > "$O/bpebyte.log" 2>&1 &
wait; echo "FFLMHF DONE"
