#!/usr/bin/env bash
# Re-run BIG-bench Dyck for AU-Net + BPEByte WITH per-sample logging, distinct MASTER_PORT per proc.
set -u
cd /mnt/ssd2/hyun2/AUNet/lingua
ROOT="/mnt/ssd2/hyun2/AUNet"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack/bigbench_dyck_samples"; mkdir -p "$O"
T="bigbench_dyck_languages_multiple_choice"
run(){ local ck=$1 tag=$2 gpu=$3 port=$4;
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$port $PY -m apps.aunet.eval \
    config=apps/aunet/configs/eval_full_5bench_b200.yaml validation=null \
    harness.tasks="[$T]" harness.limit=1000 harness.log_samples=true \
    harness.include_path="$ROOT/lingua/eval_tasks" \
    ckpt_dir="$ck" dump_dir="$O/$tag" > "$O/$tag.log" 2>&1; echo "done $tag exit=$?"; }
run "$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000"                  aunet_static    1 29517 &
run "$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000"  byte_greedyroot 2 29519 &
wait
echo "BBDYCK_SAMPLES DONE"
