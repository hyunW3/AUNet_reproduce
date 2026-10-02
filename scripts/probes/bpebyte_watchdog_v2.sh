#!/bin/bash
# BPEByte 1.3B 학습 감시자 v2 — ece-agpu18.
#
# 이 감시자가 왜 필요한지: 이 학습은 지금까지 두 번, 에러 없이 조용히 죽었다.
#   08-25 02:28  step 82,864 에서 종료 -> 2.5일 방치
#   09-01 11:41  step 139,700 저장 도중 종료 -> 30시간 방치, 체크포인트도 299M 로 잘림
# 둘 다 Traceback / CUDA 오류 없이 "Async dataloader cleaned up" 정상 종료 경로였다.
# 크래시가 아니라 외부 신호로 보이며, 그래서 사람이 눈치채지 못한다.
#
# v1 대비 고친 것:
#   - pgrep 패턴을 브래킷([a]pps)으로 감싼다. v1 은 감시자 자신의 명령줄이 패턴에 걸려
#     "살아있음"으로 오판했고, 그 탓에 죽은 줄 모르고 있었다.
#   - 재시작 전에 잘린 체크포인트를 격리한다. 저장 중 죽으면 distcp 샤드만 남고
#     params.json / train_state 가 없어, 그대로 두면 재개 로직을 헷갈리게 한다.
#   - GPU 를 고정하지 않고 비어 있는 4장을 고른다(0-3 을 남이 쓰고 있을 수 있다).
set -u
A=/home/hwbae/AUNet
RUN=$A/runs/bpebyte_br_greedy_root_1.3B_a100x4
LOG=$RUN/watchdog.log
INTERVAL=300          # 5분마다
STALL_MIN=45          # step 이 45분간 안 늘면 정지로 판단(체크포인트 저장 여유 포함)
MAX_RESTARTS=8
TARGET=180000

say(){ echo "$(date '+%F %T') $*" >> "$LOG"; }
last_step(){ grep -a "step:" "$RUN/train.log" 2>/dev/null | tail -1 | sed -n 's/.*step: *\([0-9]*\).*/\1/p'; }
alive(){ pgrep -fc "[a]pps.aunet.train.*[b]pebyte" 2>/dev/null || echo 0; }

quarantine_partial(){   # 파일 9개 미만인 체크포인트는 저장 중 끊긴 것 -> 치운다
  for d in "$RUN"/checkpoints/[0-9]*; do
    [ -d "$d" ] || continue
    n=$(ls "$d" 2>/dev/null | wc -l)
    if [ "$n" -lt 9 ]; then
      say "잘린 체크포인트 격리: $(basename "$d") (파일 ${n}개)"
      mv "$d" "$RUN/checkpoints/.truncated_$(basename "$d")_$(date +%m%d_%H%M)" 2>/dev/null
    fi
  done
}

free_gpus(){            # 점유 프로세스가 없는 카드 4장
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,utilization.gpu --format=csv,noheader,nounits |
  while IFS=", " read -r idx uuid util; do
    echo "$busy" | grep -q "$uuid" && continue
    [ "$util" -gt 5 ] 2>/dev/null && continue
    echo "$idx"
  done | head -4 | paste -sd, -
}

restarts=0; prev=$(last_step); prev_t=$(date +%s)
say "watchdog v2 시작 (step=${prev:-?}, ${INTERVAL}s 주기, stall ${STALL_MIN}분)"
while :; do
  sleep $INTERVAL
  cur=$(last_step); [ -z "$cur" ] && cur=0; [ -z "$prev" ] && prev=0
  if [ "$cur" -ge "$TARGET" ] 2>/dev/null; then say "목표 $TARGET 도달 (step=$cur) — 종료"; exit 0; fi
  if [ "$cur" -gt "$prev" ] 2>/dev/null; then prev=$cur; prev_t=$(date +%s); continue; fi

  idle=$(( ($(date +%s) - prev_t) / 60 ))
  [ "$idle" -lt "$STALL_MIN" ] && continue
  if [ "$(alive)" -gt 0 ] 2>/dev/null; then
    say "경고: 프로세스 생존하나 step 정체 ${idle}분 (step=$cur) — 재시작 보류"; continue
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then say "재시작 $MAX_RESTARTS 회 초과 — 개입 필요"; exit 1; fi

  restarts=$((restarts+1))
  say "중단 감지 (step=$cur, ${idle}분 정체) — 재시작 #$restarts"
  quarantine_partial
  g=$(free_gpus)
  if [ -z "$g" ] || [ "$(awk -F, '{print NF}' <<< "$g")" -lt 4 ]; then
    say "빈 카드 4장 없음 (현재: ${g:-없음}) — 다음 주기에 재시도"; prev_t=$(date +%s); continue
  fi
  say "GPU $g 에서 재시작"
  ( cd "$A/lingua" && GPUS="$g" setsid nohup bash scripts/train/train_bpebyte_rg_a100x4.sh \
      > "$RUN/restart_wd$(date +%m%d_%H%M).log" 2>&1 < /dev/null & )
  sleep 600
  prev=$(last_step); prev_t=$(date +%s)
  say "재시작 후 step=${prev:-?} (procs=$(alive))"
done
