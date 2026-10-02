#!/bin/bash
# AUNet3 (2.5B, 102857 step / 283.1B bytes) 시작 대기자 — 노드 독점 확보 방식(B안).
#
# 08-29 08:33 시도가 OOM 으로 죽었다. 경합이 아니라 메모리였다:
#   "GPU has 79.15 GiB total, 19.75 MiB free / this process 63.92 GiB / Process ... 15.20 GiB"
# 시작 시점에는 4장이 비어 있었지만 기동하는 2분 사이 다른 사용자가 15.2GB 를 올렸고,
# AUNet3 가 63.9GB 를 쓰면서 79GB 를 넘겼다. 이 모델은 카드당 70.8GB(89%)를 쓰므로 여유가
# 8GB 뿐이고, 남이 조금만 들어와도 무너진다.
#
# 그래서 B안: "4장이 비었다"가 아니라 **노드 전체가 놀고 있을 때만** 시작한다.
#   - 8장 모두 compute 프로세스가 없어야 한다(그 노드를 아무도 안 쓰는 상태)
#   - CONFIRM_S 초 뒤 한 번 더 확인해 그 사이 누가 들어오지 않았는지 본다
#   - 기동 실패 시 로그에서 실제 사유(OOM / 포트 / 경로)를 뽑아 남긴다. 이전 버전은 무조건
#     "경합 추정"이라고 적어서 OOM 을 두 번 놓쳤다.
set -u
L=/mnt/ssd2/hyun2/AUNet
LOG=$L/reports_afterAAAIsub/aunet3_wait.log
INTERVAL=${INTERVAL:-600}
CONFIRM_S=${CONFIRM_S:-120}
NODES="ece-agpu18 ece-agpu11"
say(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

busy_count(){   # compute 프로세스가 붙은 GPU 수 (0 이면 노드 전체 유휴)
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" \
    'nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | grep -c . || true' 2>/dev/null
}
gpu_total(){
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" \
    'nvidia-smi --query-gpu=index --format=csv,noheader | grep -c . || true' 2>/dev/null
}
already_running(){
  for h in $NODES; do
    n=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
         'pgrep -fc "[a]pps.aunet.train.*[a]unet3_1.3B" 2>/dev/null || echo 0' 2>/dev/null)
    if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then echo "$h"; return 0; fi
  done
  return 1
}

say "대기 시작 (B안: 노드 전체 유휴 + ${CONFIRM_S}s 재확인, ${INTERVAL}s 주기, 노드: $NODES)"
while :; do
  if host=$(already_running); then say "이미 $host 에서 실행 중 — 대기자 종료"; exit 0; fi
  for h in $NODES; do
    b=$(busy_count "$h"); t=$(gpu_total "$h")
    [ -z "$b" ] && { say "$h 조회 실패(SSH?) — 건너뜀"; continue; }
    if [ "$b" -ne 0 ] 2>/dev/null; then continue; fi
    say "$h 전체 유휴 감지 (GPU ${t:-?} 장) — ${CONFIRM_S}s 뒤 재확인"
    sleep "$CONFIRM_S"
    b2=$(busy_count "$h")
    if [ "${b2:-1}" -ne 0 ] 2>/dev/null; then say "$h 재확인 실패(그 사이 $b2 장 점유) — 보류"; continue; fi
    say "$h 독점 확인 — AUNet3 시작 (GPU 0,1,2,3)"
    timeout 120 ssh -o BatchMode=yes "$h" \
      "cd /home/hwbae/AUNet/lingua && GPUS=0,1,2,3 MASTER_PORT=29540 setsid nohup bash scripts/train/train_aunet3_a100x4.sh \
       > /home/hwbae/AUNet/runs/aunet3_start_\$(date +%m%d_%H%M).log 2>&1 < /dev/null &" 2>>"$LOG"
    sleep 420
    if host=$(already_running); then say "기동 확인 ($host) — 대기자 종료"; exit 0; fi
    reason=$(timeout 60 ssh -o BatchMode=yes "$h" \
      'f=$(ls -t /home/hwbae/AUNet/runs/aunet3_start_*.log 2>/dev/null | head -1);
       grep -ahoE "OutOfMemoryError|Address already in use|No such file or directory|Permission denied|ChildFailedError" "$f" 2>/dev/null | head -1' 2>/dev/null)
    say "기동 실패 @$h — 사유: ${reason:-불명(로그 확인 필요)}"
  done
  sleep "$INTERVAL"
done
