#!/bin/bash
# Score one probe item file with a list of families on one GPU (waits until the GPU has no compute process).
#   run_queue.sh <items.jsonl> <outdir> <gpu> <family...>
# Host-specific environments (snu55 = this repo; snu234 = ~/aunet_ext).
set -uo pipefail
ITEMS=$1; OUT=$2; GPU=$3; shift 3
mkdir -p $OUT
busy(){ local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}'); nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
for fam in "$@"; do
  [ -f $OUT/$fam.jsonl ] && { echo "skip $fam" >> $OUT/done.log; continue; }
  while busy; do sleep 60; done
  if [ -d "$HOME/aunet_ext" ] && [ ! -d /mnt/ssd2/hyun2/AUNet ]; then             # snu234
    X=$HOME/aunet_ext
    case $fam in
      blt)  CMD="env PYTHONPATH=$X/blt_official EVAL_SUITE=$X/eval_suite RUN_EXT_DIR=$X BLT_SUPPRESS_ATTN_ERROR=1 $X/blt_venv/bin/python" ;;
      hnet) CMD="env EVAL_SUITE=$X/eval_suite HNET_REPO=$X/hnet_repo RUN_EXT_DIR=$X $X/hnet_env/bin/python" ;;
    esac
    SCRIPT=$X/rep_probes/score_items.py
  else                                                                               # snu55
    R=/mnt/ssd2/hyun2/AUNet; NV=$R/lingua/.venv/lib/python3.12/site-packages/nvidia
    case $fam in
      llama|aunet|bpebyte) CMD="env PYTHONPATH=$R/lingua AUNET_ROOT=$R LD_LIBRARY_PATH=$(ls -d $NV/*/lib | paste -sd:) $R/lingua/.venv/bin/python" ;;
      blt)  E=$R/runs/ext_ci_snu55; CMD="env PYTHONPATH=$E/extra_site:$E/blt_official AUNET_LINGUA=$R/lingua EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1 /mnt/ssd2/hyun2/venvs/vllm011/bin/python" ;;
    esac
    SCRIPT=$R/scripts/probes/rep_probes/score_items.py
  fi
  echo "$(date '+%F %T') start $fam GPU$GPU" >> $OUT/done.log
  CUDA_VISIBLE_DEVICES=$GPU $CMD $SCRIPT --family $fam --items $ITEMS --out $OUT/$fam.jsonl ${EXTRA:-} > $OUT/$fam.log 2>&1
  echo "$(date '+%F %T') end $fam exit=$?" >> $OUT/done.log
done
echo ALL_DONE >> $OUT/done.log
