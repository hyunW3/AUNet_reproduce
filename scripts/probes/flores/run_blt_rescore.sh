#!/bin/bash
# BLT-1B scoring axes, xformers (upstream) vs sdpa (the fallback every published BLT number in
# this project was measured with). Same checkpoints, tasks, prompts, batch size -- only the
# entropy model's attention differs, so the delta is attributable to the patching bug.
# Generation axes (CUTE) are excluded: the decode loop still faults under xformers.
ROOT=/mnt/ssd2/hyun2/AUNet
BLT=/mnt/ssd2/hyun2/lzspacebyte/blt
OUT=$ROOT/reports_afterAAAIsub/blt_rescore
LOGS=$ROOT/runs/_logs
mkdir -p "$OUT" "$LOGS"
cd $BLT
run() {  # gpu attn port axis extra... ; tag from axis+attn
  local gpu=$1 attn=$2 port=$3 axis=$4; shift 4
  local tag="${axis}_${attn}"
  CUDA_VISIBLE_DEVICES=$gpu BLT_SUPPRESS_ATTN_ERROR=1 TMPDIR=${TMPDIR:-/tmp} \
    ./.venv/bin/python $ROOT/scripts/probes/flores/blt_rescore_axes.py \
      --axis "$axis" --attn_impl "$attn" --port "$port" --out "$OUT/$tag.json" "$@" \
      > "$LOGS/blt_rescore_$tag.log" 2>&1
  echo "  done $tag rc=$?"
}
T5="hellaswag arc_easy arc_challenge piqa winogrande"
( run 0 xformers 29631 downstream --tasks $T5 --num_fewshot 0 --batch_size 8
  run 0 xformers 29632 bpb --batch_size 8 ) &
( run 1 sdpa     29641 downstream --tasks $T5 --num_fewshot 0 --batch_size 8
  run 1 sdpa     29642 bpb --batch_size 8 ) &
( run 2 xformers 29651 noise   --limit 500 --batch_size 8
  run 2 xformers 29652 typo    --limit 500 --batch_size 8
  run 2 xformers 29653 despace --limit 500 --batch_size 8 ) &
( run 3 sdpa     29661 noise   --limit 500 --batch_size 8
  run 3 sdpa     29662 typo    --limit 500 --batch_size 8
  run 3 sdpa     29663 despace --limit 500 --batch_size 8 ) &
wait
echo "=== BLT rescore complete -> $OUT ==="
