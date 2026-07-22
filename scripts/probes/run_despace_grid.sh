#!/usr/bin/env bash
# Full despace grid: clean + ctx/ans/all x {10,20,40,70,100}. 5-task MC macro.
set -u; cd /mnt/ssd2/hyun2/AUNet/lingua
ROOT="/mnt/ssd2/hyun2/AUNet"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false MASTER_ADDR=127.0.0.1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
export DESPACE_PROBS=0.1,0.2,0.4,0.7 DESPACE_ANSWER=1 DESPACE_FULL=1
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack/despace_grid"; mkdir -p "$O"; L=500
runmain(){ MASTER_PORT=$3 CUDA_VISIBLE_DEVICES=$2 $PY -m apps.main.eval \
  config=apps/main/configs/eval_llama_1B_b200.yaml validation=null harness.tasks="[despace_mc]" \
  harness.limit=$L harness.include_path="$ROOT/lingua/eval_tasks" ckpt_dir="$1" dump_dir="$O/$4" > "$O/$4.log" 2>&1; echo "done $4 $?"; }
runaunet(){ MASTER_PORT=$3 CUDA_VISIBLE_DEVICES=$2 $PY -m apps.aunet.eval \
  config=apps/aunet/configs/eval_full_5bench_b200.yaml validation=null harness.tasks="[despace_mc]" \
  harness.limit=$L harness.include_path="$ROOT/lingua/eval_tasks" ckpt_dir="$1" dump_dir="$O/$4" > "$O/$4.log" 2>&1; echo "done $4 $?"; }
runmain  "$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000"            0 29911 llama   &
runaunet "$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000"                 1 29912 aunet   &
runaunet "$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000" 2 29913 bpebyte &
wait; echo "DESPACEGRID DONE"
