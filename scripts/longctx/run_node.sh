#!/usr/bin/env bash
# Run one model over the long-context suite on one GPU, tasks in the planned order
# (1 needle_types -> 2 kv_litm -> 3 ruler_agg -> 4 icl). Resumable (run_longctx.py
# skips finished ids), so re-launching the same command continues a killed job.
#
#   run_node.sh <root> <python> <gpu> <model: llama|aunet|bpebyte|blt|hnet|hnet2> [max_tokens] [batch] [files...]
#   env: SHARD=i/n splits each file by line index (separate result file per shard); LIMIT=N smoke
set -u
R=$1 PY=$2 GPU=$3 MODEL=$4 MAXTOK=${5:-32768} BATCH=${6:-16}
shift 6 2>/dev/null || shift $#
FILES=${*:-needle_types kv_litm ruler_agg icl}
CKPT=$R/ckpt/$MODEL
case $MODEL in
  llama)   FAM=subword TAG=subword_llama ;;
  aunet)   FAM=aunet   TAG=aunet_static ;;
  # NOTE: run_longctx.py now defaults to --bpe_decode inc; the byte_greedyroot results in
  # reports/longctx were produced with the bt loop, so this tag pins bt to stay reproducible
  bpebyte) FAM=aunet   TAG=byte_greedyroot EXTRA="--bpe_decode bt" ;;
  # same paper checkpoint, cached AUNET_INC_PARSE decode (exact causal boundaries) + mask check
  bpebyte_inc) FAM=aunet TAG=byte_greedyroot_inc CKPT=$R/ckpt/bpebyte EXTRA="--bpe_decode inc --check_masks" ;;
  # bt decode again on another host/env: separates decode effect from hardware/numerics noise
  bpebyte_btctl) FAM=aunet TAG=byte_greedyroot_btctl CKPT=$R/ckpt/bpebyte ;;
  # external references (ext_models.py); need BLT_REPO / HNET_REPO / EVAL_SUITE in the env
  blt)     FAM=blt     TAG=blt_1b            CKPT=$R/ext/blt_weights MAX_BYTES=${MAX_BYTES:-4096} ;;
  hnet)    FAM=hnet    TAG=hnet_1stage_XL    CKPT=hnet_1stage_XL ;;
  hnet2)   FAM=hnet    TAG=hnet_2stage_XL    CKPT=hnet_2stage_XL ;;
esac
export AUNET_ROOT=$R AUNET_TOK=$R/tokenizer/llama3/tokenizer.model PYTHONPATH=$R/lingua
# lingua's `apps` package would shadow the BLT repo's own `apps`
case $FAM in blt|hnet) export PYTHONPATH= ;; esac
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
cd "$R/lingua"
for f in $FILES; do
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$R/scripts/longctx/run_longctx.py" --family $FAM \
    --ckpt "$CKPT" --tag $TAG --data "$R/data/longctx/$f.jsonl" \
    --out "$R/results/${f}_${TAG}${SHARD:+_s${SHARD/\//of}}.jsonl" --batch_size "$BATCH" --max_tokens "$MAXTOK" \
    ${LIMIT:+--limit $LIMIT} ${SHARD:+--shard $SHARD} ${MAX_BYTES:+--max_bytes $MAX_BYTES} ${EXTRA:-} 2>&1 \
    | grep --line-buffered -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"
done
echo "NODE_DONE $MODEL $(date +%F_%T)"
