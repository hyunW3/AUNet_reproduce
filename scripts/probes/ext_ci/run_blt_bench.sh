#!/bin/bash
# BLT-1B standard benchmarks, reproducible: single-seed run (blt_bench.py) + item-bootstrap CI
# (blt_bench_bootstrap.py). snu55 environment (BLT_execution_guide.md §4.2); GPU 2 by default.
#
#   bash run_blt_bench.sh                       # 0-shot, seed 1234, full test sets
#   SHOT=5 SEED=1234 bash run_blt_bench.sh      # few-shot (seed picks the examples)
#   LIMIT=500 GPU=3 bash run_blt_bench.sh       # first 500 items per task
#   TASKS="hellaswag piqa" bash run_blt_bench.sh
set -euo pipefail
A=/mnt/ssd2/hyun2/AUNet; W=$A/runs/ext_ci_snu55; D=$A/scripts/probes/ext_ci
SHOT=${SHOT:-0}; SEED=${SEED:-1234}; GPU=${GPU:-2}; B=${B:-10000}; BSEED=${BSEED:-0}
TASKS=${TASKS:-"hellaswag arc_easy arc_challenge piqa winogrande boolq mmlu_text"}
OUT=${OUT:-$A/reports/blt_bench}
TAG=ds${SHOT}_seed${SEED}${LIMIT:+_limit$LIMIT}
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$W/extra_site:$W/blt_official AUNET_LINGUA=$A/lingua \
EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /mnt/ssd2/hyun2/venvs/vllm011/bin/python $D/blt_bench.py --num_fewshot $SHOT --seed $SEED \
  --tasks $TASKS ${LIMIT:+--limit $LIMIT} --out "$OUT/$TAG.json" 2>&1 | tee "$OUT/$TAG.log"

/mnt/ssd2/hyun2/venvs/vllm011/bin/python $D/blt_bench_bootstrap.py "$OUT/$TAG.json" --B $B --seed $BSEED \
  --out "$OUT/${TAG}_ci.json" | tee "$OUT/${TAG}_ci.md"
