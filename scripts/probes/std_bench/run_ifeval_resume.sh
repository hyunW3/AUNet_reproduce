#!/bin/bash
# IFEval with crash-resume: one prompt per lm-eval call, finished prompts kept in a jsonl, and the
# process restarted after a crash (BPEByte's cached generate path can hit a CUDA illegal memory
# access after a few consecutive calls; a fresh process runs the next prompt fine).
#   SHARD=0/2 GEN=450 CUT=2700 GPU=2 bash run_ifeval_resume.sh bpebyte_1.3b
set -uo pipefail
MODEL=${1:?usage: run_ifeval_resume.sh MODEL}
D=$(cd "$(dirname "$0")" && pwd)
A=${AUNET_ROOT:-/mnt/ssd2/hyun2/AUNet}
LINGUA=${AUNET_LINGUA:-$A/lingua}
GPU=${GPU:-2}; MAX_RESTARTS=${MAX_RESTARTS:-400}
FAM=${FAMILY:-$(python3 "$D/models.py" family "$MODEL")}
OUT=${OUT:-$A/reports/ifeval/$MODEL}
TAG=ifeval${GEN:+_gen$GEN}${CUT:+_cut$CUT}${LIMIT:+_limit$LIMIT}${SHARD:+_shard${SHARD/\//of}}
mkdir -p "$OUT"
X=$A/runs/ifeval
export CUDA_VISIBLE_DEVICES=$GPU AUNET_ROOT=$A AUNET_LINGUA=$LINGUA NLTK_DATA=$X/nltk_data
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
case $FAM in
  lingua_main|lingua_aunet)
    VENV=$LINGUA/.venv; PY=$VENV/bin/python
    NV=$VENV/lib/python3.12/site-packages/nvidia
    export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
    export PYTHONPATH=$LINGUA:$X/extra_site
    cd "$LINGUA" ;;
  *) echo "resume wrapper supports lingua families only (got $FAM)"; exit 1 ;;
esac
n=0; prev=-1
until "$PY" "$D/ifeval_bench.py" --model "$MODEL" ${MAX_TOKENS:+--max_tokens "$MAX_TOKENS"} \
    ${LIMIT:+--limit "$LIMIT"} ${GEN:+--max_gen_toks "$GEN"} ${CUT:+--cut_bytes "$CUT"} ${SHARD:+--shard "$SHARD"} \
    ${ITEMS:+--items "$ITEMS"} --resume_jsonl "$OUT/$TAG.jsonl" --out "$OUT/$TAG.json" >> "$OUT/$TAG.log" 2>&1; do
  n=$((n+1)); cur=$(cat "$OUT/$TAG.jsonl" 2>/dev/null | wc -l)
  echo "[resume-wrapper] crash $n, $cur prompts done" | tee -a "$OUT/$TAG.log"
  [ "$cur" -eq "$prev" ] && { echo "no progress since the last crash: a prompt fails in a fresh process"; exit 1; }
  prev=$cur
  [ $n -ge $MAX_RESTARTS ] && { echo "giving up after $n restarts"; exit 1; }
done
if grep -q "prompt cut to the last" "$OUT/$TAG.log"; then echo "INVALID: prompts were truncated (lower GEN)"; exit 3; fi
tail -6 "$OUT/$TAG.log"
