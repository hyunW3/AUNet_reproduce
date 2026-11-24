#!/bin/bash
# Answer surface-form robustness (surface_bench.py) for a registered lingua model.
#   GPU=1 bash run_cfqa.sh bpebyte_1.3b         # 500 items x 9 conditions x {0,3} shots
#   N=20 GPU=1 bash run_cfqa.sh llama_1.3b      # smoke test
set -euo pipefail
MODEL=${1:?usage: run_cfqa.sh MODEL}
D=$(cd "$(dirname "$0")" && pwd)
A=${AUNET_ROOT:-/mnt/ssd2/hyun2/AUNet}
LINGUA=${AUNET_LINGUA:-$A/lingua}
GPU=${GPU:-2}; N=${N:-500}
OUT=${OUT:-$A/reports/surface/$MODEL}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU AUNET_ROOT=$A AUNET_LINGUA=$LINGUA HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export IFEVAL_EAGER_MASK=1
VENV=$LINGUA/.venv; PY=$VENV/bin/python
NV=$VENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$LINGUA
cd "$LINGUA"
"$PY" "$D/surface_bench.py" --model "$MODEL" --n "$N" --out "$OUT/surface_n$N.json" 2>&1 | tee "$OUT/surface_n$N.log"
