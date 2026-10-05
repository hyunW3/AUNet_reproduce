#!/usr/bin/env bash
# run_node.sh <lc_root> <gpu> <model: llama|aunet|bpebyte|blt> [files...]
# One model over the BLT-weakness probe files with the long-context runner
# (<lc_root>/scripts/longctx/run_longctx.py; resumable). Data/results live in
# <lc_root>/bltweak/{data,results}. env: SHARD=i/n, LIMIT=N (smoke), RESDIR (default results),
# TRIO_PY (python for llama/aunet/bpebyte; default ~/AUNet/lingua/.venv).
set -u
R=$1 GPU=$2 MODEL=$3
shift 3
FILES=${*:-mkniah sniah_nc hashhop dist512 count copy}
E=$R/ext B=$R/bltweak
PY=${TRIO_PY:-$HOME/AUNet/lingua/.venv/bin/python} BS=16 MAXB=
case $MODEL in
  llama)   FAM=subword TAG=subword_llama   CKPT=$R/ckpt/llama ;;
  aunet)   FAM=aunet   TAG=aunet_static    CKPT=$R/ckpt/aunet ;;
  bpebyte) FAM=aunet   TAG=byte_greedyroot CKPT=$R/ckpt/bpebyte ;;
  # official bytelatent + xformers, bs 1, <= 4096 B (byte RoPE table); see ext_models.build_blt
  blt)     FAM=blt     TAG=blt_1b          CKPT=$E/blt_weights PY=${BLT_PY:-$E/venv_blt/bin/python} BS=1 MAXB=4096 ;;
  # cartesia H-Net 1-stage XL via ext_models.build_hnet (its own venv; HNET_REPO = official repo)
  hnet)    FAM=hnet    TAG=hnet_1stage_XL  CKPT=hnet_1stage_XL PY=${HNET_PY:-$E/venv/bin/python} BS=8 ;;
esac
export AUNET_ROOT=$R AUNET_TOK=$R/tokenizer/llama3/tokenizer.model PYTHONPATH=$R/lingua
export EVAL_SUITE=$E/eval_suite BLT_REPO=$E/blt_official HF_HUB_OFFLINE=1
export HNET_REPO=${HNET_REPO:-$HOME/AUNet_fmt/hnet_repo}
# lingua's `apps` would shadow the ext repos' own; EXT_PYTHONPATH adds e.g. snu55's BLT extra_site
case $FAM in blt|hnet) export PYTHONPATH=${EXT_PYTHONPATH:-} ;; esac
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$B/results"
cd "$R/lingua"
for f in $FILES; do
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$R/scripts/longctx/run_longctx.py" --family $FAM \
    --ckpt "$CKPT" --tag $TAG --data "$B/data/$f.jsonl" \
    --out "$B/${RESDIR:-results}/${f}_${TAG}${SHARD:+_s${SHARD/\//of}}.jsonl" --batch_size $BS --max_tokens 32768 \
    ${LIMIT:+--limit $LIMIT} ${SHARD:+--shard $SHARD} ${MAXB:+--max_bytes $MAXB} 2>&1 \
    | grep --line-buffered -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"
done
echo "NODE_DONE $MODEL ${SHARD:-} $(date +%F_%T)"
