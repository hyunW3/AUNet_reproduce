#!/bin/bash
# 재학습 BPEByte 의 despace 붕괴가 언제 생겼는지 중간 체크포인트로 되짚는다.
# 최종 180,000 에서 BoolQ despace -2.00 -> -19.85, ARC-E -25.30 -> -35.80 이 됐는데
# noise/typo/pbp 는 논문본과 1pp 안이다. 축 하나만 무너졌으니 그 축만 본다.
set -uo pipefail
L=/mnt/ssd2/hyun2/AUNet
RUN=bpebyte_br_greedy_root_1.3B_a100x4
LOG=$L/runs/robustness_retrain1p3b/despace_trace.log
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
for step in 108000 135000 162000; do
  pad=$(printf "%010d" $step)
  out=$L/runs/robustness_retrain1p3b/despace_$pad
  [ -f "$out/results.json" ] && { say "생략 $step"; continue; }
  [ -d "$L/runs/$RUN/checkpoints/$pad/consolidated" ] || { say "건너뜀 $step (consolidated 없음)"; continue; }
  say "시작 $step"
  DESPACE_TASKS="hellaswag,arc_easy,arc_challenge,piqa,boolq" \
  CFG=apps/aunet/configs/eval_despace_only_bpebyte.yaml OUT_DIR="$out" \
    bash "$L/lingua/scripts/eval/robustness/run_robustness_local.sh" bpebyte "$step" "${GPUS:-2,3}" >> "$LOG" 2>&1
  say "종료 $step results=$([ -f "$out/results.json" ] && echo yes || echo NO)"
done
say "DESPACE_TRACE_DONE"
