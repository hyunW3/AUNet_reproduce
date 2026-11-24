#!/usr/bin/env bash
# Emit one line per BPEByte3 / dropout arm as it reaches a terminal state, then exit.
# Terminal = .DONE written, or metrics reached the final logged step (3340 of 3344), or the log
# shows a crash signature. Failure signatures are watched as explicitly as success: a silent
# monitor must mean "still training", never "died an hour ago".
set -u
RUNS=/mnt/ssd2/hyun2/aunet_r10lr/runs/r10lr
LOGS=$RUNS/_logs
ARMS="r10lr_rgbb7_100M_lr34 r10lr_rgend_100M_lr34 r10lr_rgstats_100M_lr34 r10lr_rgdrop010_100M_lr34 r10lr_rgdropps010_100M_lr34"
declare -A seen
# Announce only NEW transitions. Arms that were already finished before this watcher started are
# history, not events -- re-reporting them on every restart buries the one line that matters.
while read -r l; do seen["runner:$(echo "$l" | md5sum | cut -c1-8)"]=1
done < <(grep -E "\[FAIL\]|\[MISS\]" "$LOGS/runner_bpebyte3.log" 2>/dev/null)

last_step () { [ -f "$RUNS/$1/metrics.jsonl" ] && tail -1 "$RUNS/$1/metrics.jsonl" 2>/dev/null \
  | python3 -c 'import sys,json
try: print(json.loads(sys.stdin.read()).get("global_step",0))
except Exception: print(0)' || echo 0; }

bpb_of () { python3 /mnt/ssd2/hyun2/AUNet/scripts/bpebyte3/collect_results.py 2>/dev/null \
  | awk -v n="$1" 'index($0,n)>0{print $(NF-3)}'; }

# Seed anything already terminal, using the SAME test the loop uses (a run launched outside the
# runner never gets a .DONE, so .DONE alone under-seeds and the arm gets re-announced forever).
for a in $ARMS; do
  if [ -f "$RUNS/$a/.DONE" ] || { s=$(last_step "$a"); [ "${s:-0}" -ge 3334 ] 2>/dev/null; }; then
    seen[$a]=1
  fi
done

while :; do
  alldone=1
  for a in $ARMS; do
    [ -n "${seen[$a]:-}" ] && continue
    log=$LOGS/$a.log
    if [ -f "$RUNS/$a/.DONE" ] || [ "$(last_step "$a")" -ge 3334 ] 2>/dev/null; then
      seen[$a]=1
      echo "[FINISHED] $a  final BPB $(python3 /mnt/ssd2/hyun2/AUNet/scripts/bpebyte3/collect_results.py 2>/dev/null | grep -m1 "${a#r10lr_}" | awk '{print $(NF-3)}')"
      continue
    fi
    # Liveness, not log archaeology. Logs are appended across relaunches, so a stale Traceback from
    # an earlier attempt sits in the file forever and must never be read as a fresh crash (it was,
    # once). A run is dead only if no process currently holds its config AND it has not finished.
    if ! pgrep -f "config=.*/${a}\.yaml" >/dev/null 2>&1; then
      sleep 60                                   # tolerate the gap between relaunch and exec
      if ! pgrep -f "config=.*/${a}\.yaml" >/dev/null 2>&1; then
        seen[$a]=1
        echo "[DEAD] $a at step $(last_step "$a") — no process holds its config; tail: $(grep -E 'Traceback|SignalException|CUDA out of memory' "$log" 2>/dev/null | tail -1 | cut -c1-90)"
        continue
      fi
    fi
    alldone=0
  done
  if [ -f "$LOGS/runner_bpebyte3.log" ]; then
    while read -r l; do
      key="runner:$(echo "$l" | md5sum | cut -c1-8)"
      [ -n "${seen[$key]:-}" ] && continue
      seen[$key]=1; echo "[runner] $l"
    done < <(grep -E "\[FAIL\]|\[MISS\]" "$LOGS/runner_bpebyte3.log" 2>/dev/null)
  fi
  [ "$alldone" = 1 ] && { echo "[ALL ARMS TERMINAL] run collect_results.py for the table"; exit 0; }
  sleep 600
done
