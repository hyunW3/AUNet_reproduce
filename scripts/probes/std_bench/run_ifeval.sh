#!/bin/bash
# IFEval for a registered model (ifeval_bench.py). Extra deps (langdetect, immutabledict, nltk punkt)
# live in $AUNET_ROOT/runs/ifeval/{extra_site,nltk_data}, not in the family venvs.
#   GPU=1 bash run_ifeval.sh bpebyte_1.3b            # full 541 prompts
#   LIMIT=10 GPU=1 bash run_ifeval.sh llama_1.3b      # smoke test
#   GEN=450 CUT=2700 bash run_ifeval.sh bpebyte_1.3b  # matched budget: byte models 450x6 = 2700 B, every
#   GEN=640 CUT=2700 bash run_ifeval.sh llama_1.3b    #   response scored truncated to 2700 B (BPEByte patch cap)
set -euo pipefail
MODEL=${1:?usage: run_ifeval.sh MODEL}
D=$(cd "$(dirname "$0")" && pwd)
A=${AUNET_ROOT:-/mnt/ssd2/hyun2/AUNet}
LINGUA=${AUNET_LINGUA:-$A/lingua}
GPU=${GPU:-2}
FAM=${FAMILY:-$(python3 "$D/models.py" family "$MODEL")}
OUT=${OUT:-$A/reports/ifeval/$MODEL}
TAG=ifeval${GEN:+_gen$GEN}${CUT:+_cut$CUT}${LIMIT:+_limit$LIMIT}${SHARD:+_shard${SHARD/\//of}}
mkdir -p "$OUT"
X=$A/runs/ifeval
export CUDA_VISIBLE_DEVICES=$GPU AUNET_ROOT=$A AUNET_LINGUA=$LINGUA NLTK_DATA=$X/nltk_data
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
case $FAM in
  blt)
    W=$A/runs/ext_ci_snu55; PY=/mnt/ssd2/hyun2/venvs/vllm011/bin/python
    export PYTHONPATH=$X/extra_site:$W/extra_site:$W/blt_official ;;
  hnet)
    PY=${HNET_PYTHON:-/home/hyunwoong/miniconda3/envs/spacebyte/bin/python}
    export HNET_REPO=${HNET_REPO:-/mnt/ssd2/hyun2/hnet} PYTHONPATH=$X/extra_site ;;
  lingua_main|lingua_aunet)
    VENV=$LINGUA/.venv; PY=$VENV/bin/python
    NV=$VENV/lib/python3.12/site-packages/nvidia
    export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
    export PYTHONPATH=$LINGUA:$X/extra_site
    cd "$LINGUA" ;;
  *) echo "unknown family $FAM"; exit 1 ;;
esac
"$PY" "$D/ifeval_bench.py" --model "$MODEL" ${CKPT:+--ckpt "$CKPT" --family "$FAM"} ${MAX_TOKENS:+--max_tokens "$MAX_TOKENS"} \
  ${LIMIT:+--limit "$LIMIT"} ${GEN:+--max_gen_toks "$GEN"} ${CUT:+--cut_bytes "$CUT"} ${SHARD:+--shard "$SHARD"} ${ITEMS:+--items "$ITEMS"} --out "$OUT/$TAG.json" 2>&1 | tee "$OUT/$TAG.log"
# BPEByte's cached path cuts prompts that do not fit the patch budget: such a run is invalid.
if grep -q "prompt cut to the last" "$OUT/$TAG.log"; then echo "INVALID: prompts were truncated (lower GEN)"; exit 3; fi
