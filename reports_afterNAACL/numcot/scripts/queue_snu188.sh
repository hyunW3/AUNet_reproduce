#!/bin/bash
# Two GPU lanes on snu188 (GPUs 0,1). Each lane waits for the PID it is given (the run already on that GPU), then
# runs its models in order. Trio: every item (items.jsonl); BLT / H-Net: items_ext.jsonl.
# usage: bash queue_snu188.sh <gpu> <pid to wait for or 0> <model> [model ...]
D=/mnt/ssd2/hyunw3/AUNet/reports_afterNAACL/numcot
gpu=$1; wpid=$2; shift 2
log(){ echo "$(date '+%F %T') GPU$gpu $*" >> $D/out/queue.log; }
while [ "$wpid" != 0 ] && kill -0 $wpid 2>/dev/null; do sleep 60; done
for m in "$@"; do
  case $m in blt|hnet) extra="--items $D/items_ext.jsonl --chunk 64";; *) extra="--chunk 128";; esac
  log "start $m"
  bash $D/scripts/run_snu188.sh $m $gpu $extra > $D/out/$m.log 2>&1
  log "end $m exit=$?"
done
log "lane done"
