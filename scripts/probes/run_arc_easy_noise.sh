#!/usr/bin/env bash
# ARC-Easy clean + noise suite (5 strategies x 3 targets = 15 variants, avg) on the
# three 1.3B checkpoints, via the lm_eval harness. Distinct MASTER_PORT per proc.
set -u
cd /mnt/ssd2/hyun2/AUNet/lingua
ROOT="/mnt/ssd2/hyun2/AUNet"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
export MASTER_ADDR=127.0.0.1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack/arc_easy_noise"; mkdir -p "$O"
T="[arc_easy,arc_easy_noise]"
run(){ local app=$1 cfg=$2 ck=$3 tag=$4 gpu=$5 port=$6;
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$port $PY -m "$app" config="$cfg" validation=null \
    harness.tasks="$T" harness.log_samples=false \
    harness.include_path="$ROOT/lingua/eval_tasks" \
    ckpt_dir="$ck" dump_dir="$O/$tag" > "$O/$tag.log" 2>&1; echo "done $tag exit=$?"; }
run apps.main.eval  apps/main/configs/eval_llama_1B_b200.yaml     "$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000"              subword_llama   0 29531 &
run apps.aunet.eval apps/aunet/configs/eval_full_5bench_b200.yaml "$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000"                  aunet_static    2 29533 &
run apps.aunet.eval apps/aunet/configs/eval_full_5bench_b200.yaml "$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000"  byte_greedyroot 3 29535 &
wait
echo "ARC_EASY_NOISE DONE"
