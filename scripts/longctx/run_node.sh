#!/usr/bin/env bash
# Run one model over the long-context suite on one GPU, tasks in the planned order
# (1 needle_types -> 2 kv_litm -> 3 ruler_agg -> 4 icl). Resumable (run_longctx.py
# skips finished ids), so re-launching the same command continues a killed job.
#
#   run_node.sh <root> <python> <gpu> <model: llama|aunet|bpebyte> [max_tokens] [batch] [files...]
#   env: SHARD=i/n splits each file by line index (separate result file per shard); LIMIT=N smoke
set -u
R=$1 PY=$2 GPU=$3 MODEL=$4 MAXTOK=${5:-32768} BATCH=${6:-16}
shift 6 2>/dev/null || shift $#
FILES=${*:-needle_types kv_litm ruler_agg icl}
case $MODEL in
  llama)   FAM=subword TAG=subword_llama ;;
  aunet)   FAM=aunet   TAG=aunet_static ;;
  bpebyte) FAM=aunet   TAG=byte_greedyroot ;;
esac
export AUNET_ROOT=$R AUNET_TOK=$R/tokenizer/llama3/tokenizer.model PYTHONPATH=$R/lingua
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
cd "$R/lingua"
for f in $FILES; do
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$R/scripts/longctx/run_longctx.py" --family $FAM \
    --ckpt "$R/ckpt/$MODEL" --tag $TAG --data "$R/data/longctx/$f.jsonl" \
    --out "$R/results/${f}_${TAG}${SHARD:+_s${SHARD/\//of}}.jsonl" --batch_size "$BATCH" --max_tokens "$MAXTOK" \
    ${LIMIT:+--limit $LIMIT} ${SHARD:+--shard $SHARD} 2>&1 \
    | grep --line-buffered -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"
done
echo "NODE_DONE $MODEL $(date +%F_%T)"
