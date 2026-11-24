#!/bin/bash
# Idle-gate continue-pretrain: wait until all GPUs are free (NEED consecutive polls), then run the
# given lang:arm pairs SEQUENTIALLY via 50_continue_pretrain.sh. Detached-friendly (nohup).
# Generalizes gate_zh_continue.sh to any language. Stop: kill $(cat gate_cont.pid)
#
# Usage:  nohup ./51_gate_continue.sh zh:qwen2 te:bloom code:starcoder2 > gate_cont.log 2>&1 &
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"
PAIRS=("${@:-zh:qwen2}")
POLL=90; NEED=3; FREE_MB=8000; stable=0
echo "PID $$ gate started $(date) pairs=${PAIRS[*]}"; echo $$ > "$here/gate_cont.pid"
while true; do
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  if [ "${procs:-1}" -eq 0 ] && [ "${maxmem:-999999}" -lt "$FREE_MB" ]; then
    stable=$((stable+1)); echo "$(date) free ${stable}/${NEED} (maxmem ${maxmem}MB)"
    if [ "$stable" -ge "$NEED" ]; then
      for p in "${PAIRS[@]}"; do
        lang=${p%%:*}; arm=${p##*:}
        echo "===== $(date) continue-pretrain $lang:$arm ====="
        "$here/50_continue_pretrain.sh" "$lang" "$arm" || echo "$p FAILED"
      done
      echo "$(date) gate done"; exit 0
    fi
  else
    [ "$stable" -ne 0 ] && echo "$(date) reset (procs=$procs maxmem=${maxmem}MB)"; stable=0
  fi
  sleep "$POLL"
done
