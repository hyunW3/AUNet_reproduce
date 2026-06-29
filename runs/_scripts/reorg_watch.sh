#!/bin/bash
# Idle-gated watcher: re-runs organize_runs.sh until the dirs that are busy now (cmp_g10 = live
# dump_dir; seg_cache/.cache = live-read by the running training) get folded in. organize_runs.sh
# skips any dir an active job touches, so each pass folds in whatever is currently free.
set -u
S=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/_scripts
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs
L=/tmp/reorg_watch.log
AK=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/alert_knock
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
pending(){ for d in cmp_g10 seg_cache .cache; do [ -L "$d" ] || return 0; done; return 1; }
log "watcher up (pid $$): folding cmp_g10 + seg_cache + .cache once the live job frees them"
for i in $(seq 1 160); do            # ~8h cap at 180s
  bash "$S/organize_runs.sh" >>"$L" 2>&1
  if ! pending; then
    log "all pending dirs grouped -> done"
    timeout 90 "$AK" echo "[runs reorg] cmp_g10 + seg_cache + .cache now folded into the organized tree." >/dev/null 2>&1 || true
    exit 0
  fi
  sleep 180
done
log "cap reached; still pending: $(for d in cmp_g10 seg_cache .cache; do [ -L "$d" ] || echo -n "$d "; done)"
