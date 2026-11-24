#!/bin/bash
# Wave 3: the full scoring-axes A/B, re-run on the SOURCE-BUILT xformers
# (torch 2.6.0.dev20241112+cu121, CUDA 1201, arch 8.6 -- exact match to the runtime).
# Waves 1-2 used the prebuilt cu124 wheel and are kept under blt_rescore/provisional_cu124wheel/.
# Both columns are re-run: BLT's trunk also goes through xformers, so the sdpa column was not
# insulated from the mismatched wheel either.
ROOT=/mnt/ssd2/hyun2/AUNet
BLT=/mnt/ssd2/hyun2/lzspacebyte/blt
OUT=$ROOT/reports_afterAAAIsub/blt_rescore
LOGS=$ROOT/runs/_logs
mkdir -p "$OUT" "$LOGS"
cd $BLT
run() {
  local gpu=$1 attn=$2 port=$3 axis=$4; shift 4
  local tag="${axis}_${attn}"
  CUDA_VISIBLE_DEVICES=$gpu BLT_SUPPRESS_ATTN_ERROR=1 TMPDIR=${TMPDIR:-/tmp} \
    ./.venv/bin/python $ROOT/scripts/probes/flores/blt_rescore_axes.py \
      --axis "$axis" --attn_impl "$attn" --port "$port" --out "$OUT/$tag.json" "$@" \
      > "$LOGS/blt3_$tag.log" 2>&1
  echo "  done $tag rc=$?"
}
T5="hellaswag arc_easy arc_challenge piqa winogrande"
( run 0 xformers 29721 downstream --tasks $T5 --num_fewshot 0 --batch_size 8 ) &
( run 1 sdpa     29731 downstream --tasks $T5 --num_fewshot 0 --batch_size 8 ) &
( run 2 xformers 29741 bpb --batch_size 8
  run 2 xformers 29742 formatted_string --batch_size 8
  run 2 xformers 29743 retrieval --batch_size 8
  run 2 xformers 29744 noise   --limit 500 --batch_size 8
  run 2 xformers 29745 typo    --limit 500 --batch_size 8
  run 2 xformers 29746 despace --limit 500 --batch_size 8 ) &
( run 3 sdpa     29751 bpb --batch_size 8
  run 3 sdpa     29752 formatted_string --batch_size 8
  run 3 sdpa     29753 retrieval --batch_size 8
  run 3 sdpa     29754 noise   --limit 500 --batch_size 8
  run 3 sdpa     29755 typo    --limit 500 --batch_size 8
  run 3 sdpa     29756 despace --limit 500 --batch_size 8 ) &
wait
echo "=== wave 3 complete -> $OUT ==="
