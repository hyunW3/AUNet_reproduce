#!/bin/bash
# 최종 평가 두 건이 끝날 때까지 기다렸다가 결과를 가져와 합친다.
set -u
L=/mnt/ssd2/hyun2/AUNet; RUN=bpebyte_br_greedy_root_1.3B_a100x4; PAD=0000180000
R=/home/hwbae/AUNet/runs/$RUN
until timeout 60 ssh -o BatchMode=yes ece-agpu11 "grep -qa FINAL_EVAL_DONE $R/final_eval.log" 2>/dev/null; do sleep 300; done
for h in a b; do
  rsync -a --exclude='*.log' "ece-agpu11:$R/stage_${PAD}_$h/" "$L/runs/$RUN/stage_${PAD}_$h/" || exit 1
done
timeout 60 ssh -o BatchMode=yes ece-agpu11 "tail -6 $R/final_eval.log"
python3 $L/scripts/probes/merge_final_eval.py
