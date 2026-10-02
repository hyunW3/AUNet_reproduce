#!/bin/bash
# BPEByte 60%(108,000) / 75%(135,000) 마일스톤 평가.
#
# 각 건은 세 단계다: 원격에서 전송(15G) -> consolidate(CPU) -> 평가(GPU).
# 전송과 consolidate 는 GPU 를 안 쓰므로 다음 건의 전송을 앞당겨 겹친다.
#
# 주의: run_milestone_eval_local.sh 는 시작 전에 consolidated/ 존재를 검사하고 없으면 즉시
# exit 1 한다(eval.py 자체는 자동 생성하지만 러너의 사전검사가 먼저 걸린다). 08-28 에 이걸
# 모르고 큐를 돌려 두 건이 42초 만에 실패했다. 그래서 여기서 consolidate 를 명시적으로 먼저 한다.
set -u
L=/mnt/ssd2/hyun2/AUNet
RUN=bpebyte_br_greedy_root_1.3B_a100x4
C=$L/runs/$RUN/checkpoints
LOG=$L/reports_afterAAAIsub/milestone_evals.log
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

fetch(){   # step
  local pad=$1
  if [ -d "$C/$pad" ] && [ "$(ls "$C/$pad" 2>/dev/null | wc -l)" -ge 9 ]; then
    say "전송 생략 $pad (이미 있음)"; return 0
  fi
  say "전송 시작 $pad"
  rsync -a --partial --append-verify \
    "ece-agpu18:/home/hwbae/AUNet/runs/$RUN/checkpoints/$pad" "$C/" \
    && say "전송 완료 $pad ($(du -sh "$C/$pad" | cut -f1))" \
    || { say "전송 실패 $pad"; return 1; }
}

consolidate(){   # step
  local pad=$1
  [ -d "$C/$pad/consolidated" ] && { say "consolidate 생략 $pad"; return 0; }
  say "consolidate 시작 $pad"
  # 이 박스의 torch 는 venv 의 nvidia/*/lib 를 LD_LIBRARY_PATH 로 직접 가리켜야 import 된다
  # (libcusparseLt.so.0). run_milestone_eval_local.sh 에는 있는데 여기서 빠뜨려 09-02 에 실패했다.
  ( cd "$L/lingua" \
    && NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia \
    && export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}" \
    && .venv/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$C/$pad'))" ) >> "$LOG" 2>&1
  [ -d "$C/$pad/consolidated" ] && say "consolidate 완료 $pad" || { say "consolidate 실패 $pad"; return 1; }
}

evaluate(){   # step gpus port
  local pad=$1 step=$((10#$1)) out=$L/runs/$RUN/stage_$1
  [ -f "$out/results.json" ] && { say "평가 생략 $1 (결과 있음)"; return 0; }
  say "평가 시작 $1 (GPU$2)"
  MASTER_PORT=$3 bash "$L/lingua/scripts/eval/milestone/run_milestone_eval_local.sh" bpebyte "$step" "$2" \
      > "$L/runs/$RUN/stage_$1.log" 2>&1
  say "평가 종료 $1 exit=$? results=$([ -f "$out/results.json" ] && echo yes || echo NO)"
}

# 108000 을 먼저 흘려보내고, 그 평가 중에 135000 전송/consolidate 를 겹친다
fetch 0000108000 && consolidate 0000108000 || exit 1
( fetch 0000135000 && consolidate 0000135000 ) &
PREP=$!
evaluate 0000108000 0,1,2,3 29581
wait $PREP
evaluate 0000135000 0,1,2,3 29582
say "BPEBYTE_60_75_DONE"
