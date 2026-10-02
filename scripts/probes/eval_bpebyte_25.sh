#!/bin/bash
# BPEByte 25%(45,000) 평가. 앞서 돌린 stage_0000045000 에는 results.json 이 없다(중단된 흔적).
# 60/75 파이프라인이 GPU 4장을 다 쓰므로 그게 끝난 뒤에 이어서 돌린다.
set -u
L=/mnt/ssd2/hyun2/AUNet
RUN=bpebyte_br_greedy_root_1.3B_a100x4
LOG=$L/reports_afterAAAIsub/milestone_evals.log
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
while pgrep -f "[e]val_bpebyte_60_75.sh" >/dev/null; do sleep 300; done
say "60/75 완료 확인 — 25%(45,000) 시작"
C=$L/runs/$RUN/checkpoints/0000045000
if [ ! -d "$C/consolidated" ]; then
  say "consolidate 시작 45000"
  ( cd "$L/lingua" && .venv/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$C'))" ) >> "$LOG" 2>&1
fi
out=$L/runs/$RUN/stage_0000045000
if [ -f "$out/results.json" ]; then say "평가 생략 45000"; else
  say "평가 시작 45000"
  MASTER_PORT=29583 bash "$L/lingua/scripts/eval/milestone/run_milestone_eval_local.sh" bpebyte 45000 0,1,2,3 \
      > "$L/runs/$RUN/stage_0000045000.log" 2>&1
  say "평가 종료 45000 exit=$? results=$([ -f "$out/results.json" ] && echo yes || echo NO)"
fi
say "BPEBYTE_25_DONE"
