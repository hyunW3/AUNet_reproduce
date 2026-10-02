#!/bin/bash
# BPEByte 마일스톤 평가 — 최대 2건 동시, 하나 끝나면 다음 것을 바로 채운다.
#
# 왜 겹쳐도 되는가: BPEByte 평가는 GPU 가 아니라 CPU 바운드다(online greedy BPE 패칭).
# 실측: GPU util 0%, 카드당 9.3GB/24.5GB, 평가 1건이 코어 4개를 100% 로 쓴다.
# 이 박스는 64 코어 / 251GB RAM 이고 load 는 5.5 였으므로, 2건을 겹쳐도 서로를 굶기지 않는다.
# (세션 초반에 GPU util 0% 를 보고 "멈췄다"고 오판해 정상 평가를 두 번 죽인 적이 있다.
#  판단 기준은 GPU util 이 아니라 프로세스 CPU 시간이다.)
#
# 전송/consolidate 는 GPU 를 안 쓰므로 대기 슬롯에서 미리 끝내 둔다.
set -u
L=/mnt/ssd2/hyun2/AUNet
RUN=bpebyte_br_greedy_root_1.3B_a100x4
C=$L/runs/$RUN/checkpoints
LOG=$L/reports_afterAAAIsub/milestone_evals.log
MAXJOBS=${MAXJOBS:-2}
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

nv_env(){   # 이 박스의 torch 는 venv 의 nvidia/*/lib 를 직접 가리켜야 import 된다
  NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia
  export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
}

running(){  # 지금 도는 평가 건수 (launcher 프로세스 기준)
  pgrep -fc "[t]orch.distributed.run.*apps.aunet.eval" 2>/dev/null || echo 0
}

prep(){     # 전송 + consolidate (GPU 불필요)
  local pad=$1
  if [ ! -d "$C/$pad" ] || [ "$(ls "$C/$pad" 2>/dev/null | wc -l)" -lt 9 ]; then
    say "전송 시작 $pad"
    rsync -a --partial --append-verify \
      "ece-agpu18:/home/hwbae/AUNet/runs/$RUN/checkpoints/$pad" "$C/" \
      || { say "전송 실패 $pad"; return 1; }
    say "전송 완료 $pad ($(du -sh "$C/$pad" | cut -f1))"
  fi
  if [ ! -d "$C/$pad/consolidated" ]; then
    say "consolidate 시작 $pad"
    ( cd "$L/lingua" && nv_env && .venv/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$C/$pad'))" ) >> "$LOG" 2>&1
    [ -d "$C/$pad/consolidated" ] || { say "consolidate 실패 $pad"; return 1; }
    say "consolidate 완료 $pad"
  fi
}

launch(){   # pad gpus port  — 백그라운드로 띄우고 즉시 반환
  local pad=$1 step=$((10#$1)) gpus=$2 port=$3
  say "평가 시작 $pad (GPU$gpus)"
  ( MASTER_PORT=$port bash "$L/lingua/scripts/eval/milestone/run_milestone_eval_local.sh" bpebyte "$step" "$gpus" \
        > "$L/runs/$RUN/stage_$pad.log" 2>&1
    say "평가 종료 $pad exit=$? results=$([ -f "$L/runs/$RUN/stage_$pad/results.json" ] && echo yes || echo NO)" ) &
}

i=0
for spec in "0000108000 0,1 29591" "0000135000 2,3 29592"; do
  set -- $spec; pad=$1; gpus=$2; port=$3
  if [ -f "$L/runs/$RUN/stage_$pad/results.json" ]; then say "생략 $pad (결과 있음)"; continue; fi
  prep "$pad" || continue                       # 대기 중에 준비를 끝내 둔다
  while [ "$(running)" -ge "$MAXJOBS" ]; do sleep 120; done
  launch "$pad" "$gpus" "$port"
  sleep 90                                      # 기동이 겹치지 않도록 살짝 띄운다
done
wait
say "ROLLING_DONE"
