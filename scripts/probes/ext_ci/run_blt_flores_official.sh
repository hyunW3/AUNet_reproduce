#!/bin/bash
# BLT-1B FLORES-200 X->en, official bytelatent path. snu55 policy: GPUs 2 and 3 only.
#   released threshold 1.335442066192627  -> the paper row
#   1.46875                               -> patch-size-matched variant (4.53 B/patch on DCLM)
ROOT=/mnt/ssd2/hyun2/AUNet
W=$ROOT/runs/ext_ci_snu55
OUT=$ROOT/reports_afterAAAIsub/flores/raw
LOGS=$ROOT/runs/_logs
mkdir -p "$OUT" "$LOGS"
PY=/mnt/ssd2/hyun2/venvs/vllm011/bin/python
export PYTHONPATH=$W/extra_site:$W/blt_official AUNET_LINGUA=$ROOT/lingua EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
cd $ROOT
run() {  # gpu threshold tag langs
  CUDA_VISIBLE_DEVICES=$1 $PY $ROOT/scripts/probes/ext_ci/blt_official_flores.py \
     --blt_weights $W/blt_weights --split dev --langs "$4" --limit 200 --threshold "$2" \
     --out "$OUT/blt_official_$3.json" > "$LOGS/bltoff_flores_$3.log" 2>&1
  echo "  $3 rc=$?"
}
echo "=== released threshold $(date -Is)"
( run 2 1.335442066192627 rel_a "de,fr,es" ) &
( run 3 1.335442066192627 rel_b "it,nl,ko" ) &
wait
echo "=== patch-size-matched threshold 1.46875 $(date -Is)"
( run 2 1.46875 cal_a "de,fr,es" ) &
( run 3 1.46875 cal_b "it,nl,ko" ) &
wait
echo "=== ALL DONE $(date -Is)"
