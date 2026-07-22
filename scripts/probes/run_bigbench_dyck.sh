#!/usr/bin/env bash
# Official BIG-bench Dyck Languages (multiple_choice = loglik cloze) via the lm_eval
# harness on the paper checkpoints, replacing our home-grown dyck.py.
set -u
cd /mnt/ssd2/hyun2/AUNet/lingua
ROOT="/mnt/ssd2/hyun2/AUNet"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack/bigbench_dyck"; mkdir -p "$O"
T="bigbench_dyck_languages_multiple_choice"
run(){ local app=$1 cfg=$2 ck=$3 tag=$4 gpu=$5;
  CUDA_VISIBLE_DEVICES=$gpu $PY -m "$app" config="$cfg" validation=null \
    harness.tasks="[$T]" harness.limit=1000 harness.include_path="$ROOT/lingua/eval_tasks" \
    ckpt_dir="$ck" dump_dir="$O/$tag" > "$O/$tag.log" 2>&1; echo "done $tag exit=$?"; }
run apps.main.eval  apps/main/configs/eval_llama_1B_b200.yaml     "$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000"              subword_llama   0 &
run apps.aunet.eval apps/aunet/configs/eval_full_5bench_b200.yaml "$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000"                  aunet_static    1 &
run apps.aunet.eval apps/aunet/configs/eval_full_5bench_b200.yaml "$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000"  byte_greedyroot 2 &
wait
echo "BBDYCK DONE"
