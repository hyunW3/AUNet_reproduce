#!/bin/bash
# 재학습 BPEByte 최종 체크포인트(180,000) 를 가져와 robustness 를 돌린다.
#
# 이 박스에는 162,000 까지만 있다. 180,000 은 ece-agpu11 에만 있는데 지금 SSH 가 안 된다
# (11:20 기준 두 노드 모두 connection timed out). 그래서 붙을 때까지 기다렸다가 전송한다.
#
# llama robustness 가 GPU 2,3 을 쓰는 중이므로, 전송/consolidate 는 GPU 를 안 쓰니 먼저 끝내 두고
# 평가는 llama 큐가 끝난 뒤에 시작한다.
set -uo pipefail
L=/mnt/ssd2/hyun2/AUNet
RUN=bpebyte_br_greedy_root_1.3B_a100x4
PAD=0000180000
C=$L/runs/$RUN/checkpoints/$PAD
LOG=$L/runs/robustness_retrain1p3b/bpebyte_fetch.log
INTERVAL=${INTERVAL:-600}
say(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

reachable(){ timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" 'exit 0' 2>/dev/null; }

say "대기 시작 (${INTERVAL}s 주기)"
while [ ! -d "$C" ] || [ "$(ls "$C" 2>/dev/null | wc -l)" -lt 9 ]; do
  for h in ece-agpu11 ece-agpu18; do
    reachable "$h" || continue
    say "$h 연결됨 — 전송 시작"
    if rsync -a --partial --append-verify \
         "$h:/home/hwbae/AUNet/runs/$RUN/checkpoints/$PAD" "$L/runs/$RUN/checkpoints/" >>"$LOG" 2>&1; then
      say "전송 완료 ($(du -sh "$C" | cut -f1))"; break 2
    fi
    say "전송 실패 @$h"
  done
  sleep "$INTERVAL"
done

if [ ! -d "$C/consolidated" ]; then
  say "consolidate 시작"
  # 이 박스의 torch 는 venv 의 nvidia/*/lib 를 LD_LIBRARY_PATH 로 직접 가리켜야 import 된다
  ( cd "$L/lingua" \
    && NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia \
    && export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}" \
    && .venv/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$C'))" ) >>"$LOG" 2>&1
  [ -d "$C/consolidated" ] || { say "consolidate 실패 — 중단"; exit 1; }
  say "consolidate 완료"
fi

# llama 큐가 GPU 2,3 을 놓을 때까지 기다린다
while pgrep -f "[a]pps.main.eval|[a]pps.aunet.eval" >/dev/null 2>&1; do sleep 300; done
say "GPU 확보 — bpebyte robustness 시작"
GPUS=2,3 bash "$L/scripts/probes/robustness_retrain_queue.sh" bpebyte:180000 >>"$LOG" 2>&1
say "bpebyte robustness 종료 exit=$?"
