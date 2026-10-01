#!/bin/bash
# Standard benchmarks for any registered model: single-seed run (std_bench.py) + item-bootstrap CI
# (std_bench_bootstrap.py). Picks each model family's interpreter/environment (snu55 paths).
#
#   bash run_std_bench.sh aunet_1.3b                    # 0-shot, seed 1234, full test sets, GPU 2
#   SHOT=5 SEED=1234 GPU=3 bash run_std_bench.sh blt_1b # few-shot (seed picks the examples)
#   LIMIT=500 TASKS="hellaswag piqa" bash run_std_bench.sh llama_1.3b
#   CKPT=<consolidated dir> FAMILY=lingua_aunet bash run_std_bench.sh my_run   # any lingua checkpoint
#   MAX_TOKENS=16384 ...                                # lingua generator max_tokens (default: registry,
#                                                       # or 4096 lingua_main / 16384 lingua_aunet for CKPT)
#   python3 models.py list                              # registered models
set -euo pipefail
MODEL=${1:?usage: run_std_bench.sh MODEL}
D=$(cd "$(dirname "$0")" && pwd)
A=${AUNET_ROOT:-/mnt/ssd2/hyun2/AUNet}
LINGUA=${AUNET_LINGUA:-$A/lingua}
SHOT=${SHOT:-0}; SEED=${SEED:-1234}; GPU=${GPU:-2}; B=${B:-10000}; BSEED=${BSEED:-0}
TASKS=${TASKS:-"hellaswag arc_easy arc_challenge piqa winogrande boolq mmlu_text"}
FAM=${FAMILY:-$(python3 "$D/models.py" family "$MODEL")}
[ -n "$FAM" ] || { echo "unknown model $MODEL (python3 $D/models.py list), or set CKPT + FAMILY"; exit 1; }
OUT=${OUT:-$A/reports/std_bench/$MODEL}
TAG=ds${SHOT}_seed${SEED}${LIMIT:+_limit$LIMIT}
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=$GPU AUNET_ROOT=$A AUNET_LINGUA=$LINGUA EVAL_SUITE=${EVAL_SUITE:-/mnt/ssd2/hyun2/eval_suite}
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
case $FAM in
  blt)
    W=$A/runs/ext_ci_snu55
    PY=/mnt/ssd2/hyun2/venvs/vllm011/bin/python
    export PYTHONPATH=$W/extra_site:$W/blt_official ;;
  hnet)
    PY=${HNET_PYTHON:-/home/hyunwoong/miniconda3/envs/spacebyte/bin/python}
    export HNET_REPO=${HNET_REPO:-/mnt/ssd2/hyun2/hnet} ;;
  lingua_main|lingua_aunet)
    VENV=$LINGUA/.venv; PY=$VENV/bin/python
    NV=$VENV/lib/python3.12/site-packages/nvidia
    export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
    export PYTHONPATH=$LINGUA
    cd "$LINGUA" ;;
  *) echo "unknown family $FAM"; exit 1 ;;
esac

"$PY" "$D/std_bench.py" --model "$MODEL" ${CKPT:+--ckpt "$CKPT" --family "$FAM"} ${MAX_TOKENS:+--max_tokens "$MAX_TOKENS"} \
  --num_fewshot "$SHOT" --seed "$SEED" --tasks $TASKS ${LIMIT:+--limit "$LIMIT"} \
  --out "$OUT/$TAG.json" 2>&1 | tee "$OUT/$TAG.log"

python3 "$D/std_bench_bootstrap.py" "$OUT/$TAG.json" --B "$B" --seed "$BSEED" \
  --out "$OUT/${TAG}_ci.json" | tee "$OUT/${TAG}_ci.md"
