#!/usr/bin/env bash
# Train one SVT ablation arm on 2 GPUs, same layout as lb_rg_100M (bs 12 x ga 2 x 2 GPUs = 48).
# Usage: GPUS=5,6 PORT=29771 ./svt_train.sh <arm> [extra overrides...]
set -euo pipefail
ARM="$1"; shift
S=/home/hwbae/AUNet_svt
GPUS="${GPUS:-5,6}"; PORT="${PORT:-29771}"
NPROC=$(( $(echo "$GPUS" | tr -cd , | wc -c) + 1 ))
cd "$S/lingua"
PY=/home/hwbae/AUNet/lingua/.venv/bin/python
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$S/runs/$ARM" "$S/logs"
CUDA_VISIBLE_DEVICES="$GPUS" exec "$PY" -m torch.distributed.run --nproc-per-node "$NPROC" --master-port "$PORT" \
  -m apps.aunet.train "config=$S/configs/$ARM.yaml" "$@"
