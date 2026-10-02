#!/bin/bash
# BPEByte 90%(162,000) 마일스톤 평가.
#   전송(15G) -> consolidate(CPU) -> 평가(GPU0,1)
#
# GPU 선택: 2,3 은 다른 작업(aunet_r10lr, 25시간째)이 쓰고 있으므로 0,1 만 쓴다.
# BPEByte 평가는 CPU 바운드(online greedy BPE 패칭)라 GPU util 0% 로 보이는 게 정상이다.
# 살아있는지 판단할 때는 GPU util 이 아니라 프로세스 CPU 시간을 봐야 한다.
#
# consolidate 에는 LD_LIBRARY_PATH 가 필요하다 — 이 박스의 torch 는 venv 의 nvidia/*/lib 를
# 직접 가리켜야 import 된다(libcusparseLt.so.0). 09-02 에 이걸 빠뜨려 실패했다.
set -u
L=/mnt/ssd2/hyun2/AUNet
RUN=bpebyte_br_greedy_root_1.3B_a100x4
PAD=0000162000
C=$L/runs/$RUN/checkpoints/$PAD
OUT=$L/runs/$RUN/stage_$PAD
LOG=$L/reports_afterAAAIsub/milestone_evals.log
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

[ -f "$OUT/results.json" ] && { say "생략 $PAD (결과 있음)"; exit 0; }

if [ ! -d "$C" ] || [ "$(ls "$C" 2>/dev/null | wc -l)" -lt 9 ]; then
  say "전송 시작 $PAD"
  rsync -a --partial --append-verify \
    "ece-agpu11:/home/hwbae/AUNet/runs/$RUN/checkpoints/$PAD" "$L/runs/$RUN/checkpoints/" \
    || { say "전송 실패 $PAD"; exit 1; }
  say "전송 완료 $PAD ($(du -sh "$C" | cut -f1))"
fi

if [ ! -d "$C/consolidated" ]; then
  say "consolidate 시작 $PAD"
  ( cd "$L/lingua" \
    && NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia \
    && export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}" \
    && .venv/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$C'))" ) >> "$LOG" 2>&1
  [ -d "$C/consolidated" ] || { say "consolidate 실패 $PAD"; exit 1; }
  say "consolidate 완료 $PAD"
fi

say "평가 시작 $PAD (GPU0,1)"
MASTER_PORT=29594 bash "$L/lingua/scripts/eval/milestone/run_milestone_eval_local.sh" bpebyte 162000 0,1 \
    > "$L/runs/$RUN/stage_$PAD.log" 2>&1
say "평가 종료 $PAD exit=$? results=$([ -f "$OUT/results.json" ] && echo yes || echo NO)"
