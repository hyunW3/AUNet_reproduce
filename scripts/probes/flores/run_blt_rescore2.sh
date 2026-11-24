#!/bin/bash
# Wave 2 of the BLT scoring-axes A/B: the robustness + retrieval axes that failed in wave 1 on a
# sys.path clash (blt/apps shadowed lingua's apps.aunet). All four GPUs, xformers vs sdpa.
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
      > "$LOGS/blt_rescore_$tag.log" 2>&1
  echo "  done $tag rc=$?"
}
( run 0 xformers 29671 noise            --limit 500 --batch_size 8
  run 0 xformers 29672 formatted_string --batch_size 8 ) &
( run 1 sdpa     29681 noise            --limit 500 --batch_size 8
  run 1 sdpa     29682 formatted_string --batch_size 8 ) &
( run 2 xformers 29691 typo    --limit 500 --batch_size 8
  run 2 xformers 29692 despace --limit 500 --batch_size 8
  run 2 xformers 29693 retrieval --batch_size 8 ) &
( run 3 sdpa     29701 typo    --limit 500 --batch_size 8
  run 3 sdpa     29702 despace --limit 500 --batch_size 8
  run 3 sdpa     29703 retrieval --batch_size 8 ) &
wait
echo "=== wave 2 complete -> $OUT ==="
