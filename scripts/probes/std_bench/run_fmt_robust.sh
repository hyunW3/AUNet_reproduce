#!/bin/bash
# Prompt-format robustness (fmt_robust.py) with crash-resume: finished chunks stay in the jsonl.
#   MODE=grid GPU=0 bash run_fmt_robust.sh llama_1.3b       # MODE: grid | promptsource | demos
#   N=20 MODE=grid GPU=0 bash run_fmt_robust.sh llama_1.3b  # smoke test
set -uo pipefail
MODEL=${1:?usage: run_fmt_robust.sh MODEL}
D=$(cd "$(dirname "$0")" && pwd)
A=${AUNET_ROOT:-/mnt/ssd2/hyun2/AUNet}
LINGUA=${AUNET_LINGUA:-$A/lingua}
GPU=${GPU:-2}; N=${N:-500}; MODE=${MODE:?set MODE}; MAX_RESTARTS=${MAX_RESTARTS:-50}
OUT=${OUT:-$A/reports/fmt_robust/$MODEL}
mkdir -p "$OUT"; J=$OUT/${MODE}_n$N.jsonl; LOG=$OUT/${MODE}_n$N.log
export CUDA_VISIBLE_DEVICES=$GPU AUNET_ROOT=$A AUNET_LINGUA=$LINGUA HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
VENV=$LINGUA/.venv; PY=$VENV/bin/python
NV=$VENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$LINGUA
cd "$LINGUA"
n=0; prev=-1
until "$PY" "$D/fmt_robust.py" --model "$MODEL" --mode "$MODE" --n "$N" --ps_dir "$A/runs/fmt_robust/promptsource" \
    ${MAX_TOKENS:+--max_tokens "$MAX_TOKENS"} --out_jsonl "$J" >> "$LOG" 2>&1; do
  n=$((n+1)); cur=$(cat "$J" 2>/dev/null | wc -l)
  echo "[resume-wrapper] crash $n, $cur chunks done" | tee -a "$LOG"
  [ "$cur" -eq "$prev" ] && { echo "no progress since the last crash"; exit 1; }
  prev=$cur; [ $n -ge $MAX_RESTARTS ] && exit 1
done
echo "[done] $(wc -l < "$J") chunks" | tee -a "$LOG"
