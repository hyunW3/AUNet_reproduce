#!/bin/bash
# Final BLT pass on the source-built xformers (torch 2.6.0.dev+cu121, CUDA 1201, arch 8.6).
#   stage 1  scoring axes, xformers vs sdpa      (~1 h)
#   stage 2  native FLORES dev, released vs calibrated theta  (~3 h/theta, sharded)
# One job per GPU: BLT-1B needs ~22 GB of a 24 GB A5000, so two on a card OOMs.
ROOT=/mnt/ssd2/hyun2/AUNet
BLT=/mnt/ssd2/hyun2/lzspacebyte/blt
OUT=$ROOT/reports_afterAAAIsub/blt_rescore
FOUT=$ROOT/reports_afterAAAIsub/flores/raw
LOGS=$ROOT/runs/_logs
mkdir -p "$OUT" "$FOUT" "$LOGS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BLT_SUPPRESS_ATTN_ERROR=1
cd $BLT

ax() {  # gpu attn port axis extra...
  local gpu=$1 attn=$2 port=$3 axis=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu ./.venv/bin/python $ROOT/scripts/probes/flores/blt_rescore_axes.py \
      --axis "$axis" --attn_impl "$attn" --port "$port" --out "$OUT/${axis}_${attn}.json" "$@" \
      > "$LOGS/bltf_${axis}_${attn}.log" 2>&1
  echo "  ax ${axis}_${attn} rc=$?"
}
fl() {  # gpu theta tag langs port
  local gpu=$1 theta=$2 tag=$3 langs=$4 port=$5
  CUDA_VISIBLE_DEVICES=$gpu ./.venv/bin/python $ROOT/scripts/probes/flores/blt_native_flores.py \
      --split dev --tasks_dir $ROOT/lingua/eval_tasks --langs "$langs" --limit 200 \
      --attn_impl xformers --theta "$theta" --port "$port" --out "$FOUT/blt_native_${tag}.json" \
      > "$LOGS/bltf_flores_${tag}.log" 2>&1
  echo "  fl ${tag} rc=$?"
}

echo "=== stage 1: scoring axes $(date -Is)"
T5="hellaswag arc_easy arc_challenge piqa winogrande"
( ax 0 xformers 29771 downstream --tasks $T5 --num_fewshot 0 --batch_size 8 ) &
( ax 1 sdpa     29781 downstream --tasks $T5 --num_fewshot 0 --batch_size 8 ) &
( ax 2 xformers 29791 bpb --batch_size 8
  ax 2 xformers 29792 formatted_string --batch_size 8
  ax 2 xformers 29793 retrieval --batch_size 8
  ax 2 xformers 29794 noise   --limit 500 --batch_size 8
  ax 2 xformers 29795 typo    --limit 500 --batch_size 8
  ax 2 xformers 29796 despace --limit 500 --batch_size 8 ) &
( ax 3 sdpa     29801 bpb --batch_size 8
  ax 3 sdpa     29802 formatted_string --batch_size 8
  ax 3 sdpa     29803 retrieval --batch_size 8
  ax 3 sdpa     29804 noise   --limit 500 --batch_size 8
  ax 3 sdpa     29805 typo    --limit 500 --batch_size 8
  ax 3 sdpa     29806 despace --limit 500 --batch_size 8 ) &
wait
echo "=== stage 1 done $(date -Is)"

echo "=== stage 2: native FLORES, released theta $(date -Is)"
( fl 0 1.335442066192627 rel_a "de,fr"  29811 ) &
( fl 1 1.335442066192627 rel_b "es,it"  29821 ) &
( fl 2 1.335442066192627 rel_c "nl"     29831 ) &
( fl 3 1.335442066192627 rel_d "ko"     29841 ) &
wait
echo "=== stage 2 done $(date -Is)"

echo "=== stage 3: native FLORES, calibrated theta 1.46875 $(date -Is)"
( fl 0 1.46875 cal_a "de,fr" 29851 ) &
( fl 1 1.46875 cal_b "es,it" 29861 ) &
( fl 2 1.46875 cal_c "nl"    29871 ) &
( fl 3 1.46875 cal_d "ko"    29881 ) &
wait
echo "=== ALL DONE $(date -Is)"
