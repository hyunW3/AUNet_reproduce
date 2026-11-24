#!/bin/bash
# BLT-1B FLORES-200 X->en, dev split, n=200, 5-shot greedy, 288-byte budget.
# Runs through the transformers conversion WITH the validated sliding-window shim
# (blt_hf_sliding_window.py), which reproduces native bytelatent patching:
#   released theta 1.335442 -> 4.0716 B/patch   (native 4.0710)
#   calibrated theta 1.46875 -> 4.5101 B/patch  (native 4.5120, target 4.53)
# GPU 3 is left alone -- another job of ours is on it.
ROOT=/mnt/ssd2/hyun2/AUNet
OUT=$ROOT/reports_afterAAAIsub/flores/raw
LOGS=$ROOT/runs/_logs
mkdir -p "$OUT" "$LOGS"
cd $ROOT/lingua
NV=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | paste -sd: -)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # gpu theta tag langs port
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python $ROOT/scripts/probes/flores/blt_flores.py \
      --split dev --tasks_dir $ROOT/lingua/eval_tasks --langs "$4" --limit 200 \
      --theta "$2" --batch_size 1 --out "$OUT/blt_sw_$3.json" \
      > "$LOGS/blt_sw_$3.log" 2>&1
  echo "  done $3 rc=$?"
}
echo "=== released theta $(date -Is)"
( run 0 1.335442066192627 rel_a "de,fr" ) &
( run 1 1.335442066192627 rel_b "es,it" ) &
( run 2 1.335442066192627 rel_c "nl,ko" ) &
wait
echo "=== calibrated theta 1.46875 $(date -Is)"
( run 0 1.46875 cal_a "de,fr" ) &
( run 1 1.46875 cal_b "es,it" ) &
( run 2 1.46875 cal_c "nl,ko" ) &
wait
echo "=== ALL DONE $(date -Is)"
