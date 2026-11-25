#!/usr/bin/env bash
# Offload BLT jobs of run_typoleet_region.sh to a remote host (2026-11-25: info107, A5000 x8, shared).
# Runs on the queue host: pops blt_* lines from <out_dir>/jobs.txt (same flock as the local workers), runs each on a
# remote GPU that has held NO compute process for IDLE_MIN minutes (other tenants' jobs are never joined), copies the
# result json back into <out_dir>, and puts the job back on the queue if it produced no output.
#   run_typoleet_region_remote.sh <out_dir> [host=info107] [gpus="0 1 2 3 4 5 6 7"]
# Remote tree (sync first): ~/hyun2/typoleet_region/{AUNet/{lingua,scripts,reports/format_robustness/items},
#   eval_suite,blt_official,extra_site,blt_weights,venv_blt,out}
set -u
O=$(realpath -m "$1"); H=${2:-info107}; GPUS=${3:-0 1 2 3 4 5 6 7}
IDLE_MIN=${IDLE_MIN:-10}; LIMIT=${LIMIT:-2000}
R='$HOME/hyun2/typoleet_region'
TLB=/mnt/ssd2/hyun2/AUNet/runs/robustness_typoleet_both
Q=$O/jobs.txt
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=60"
say(){ echo "$(date '+%F %T') [$H] $*" >> $O/queue.log; }
thr(){ [ "$1" = 1335 ] && echo 1.335442066192627 || echo 1.6093749403933089; }
popblt(){ ( flock 9; l=$(grep -m1 '^blt_' $Q); [ -n "$l" ] && { grep -vxF "$l" $Q > $Q.tmp; mv $Q.tmp $Q; }; echo "$l" ) 9>$Q.lock; }
pushback(){ ( flock 9; { echo "$1"; cat $Q; } > $Q.tmp; mv $Q.tmp $Q ) 9>$Q.lock; }
outname(){ set -- $1; [ $1 = blt_typo ] && echo blt${2}_typo${4}_$3.json || echo blt${2}_leetregion_$3.json; }

remote_cmd(){ local g=$1; set -- $2
  local env="cd /tmp && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$R/extra_site:$R/blt_official AUNET_LINGUA=$R/AUNet/lingua \
EVAL_SUITE=$R/eval_suite HF_DATASETS_CACHE=$R/hf_datasets HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1"
  if [ $1 = blt_typo ]; then
    echo "$env $R/venv_blt/bin/python $R/AUNet/scripts/probes/ext_ci/run_ext.py --family blt_official --threshold $(thr $2) \
--blt_weights $R/blt_weights --axis typo$4 --tasks $3 --limit $LIMIT --out $R/out/$(outname "$*")"
  else
    echo "$env $R/venv_blt/bin/python -u $R/AUNet/scripts/probes/format_mc/run_format_ext.py --family blt_official \
--threshold $(thr $2) --tasks $3 --variants clean nla_leet nla_leet_opt --blt_weights $R/blt_weights \
--items_dir $R/AUNet/reports/format_robustness/items --limit $LIMIT --cache $R/out/blt${2}_leetregion_$3.cache.jsonl \
--out $R/out/$(outname "$*")"
  fi; }

job(){ local g=$1 l=$2 f; f=$(outname "$l"); set -- $l
  if [ $1 = blt_leet ]; then   # paper cache -> clean scores reused (same as the local runner)
    local c=$O/blt${2}_leetregion_$3.cache.jsonl
    [ -f $c ] || c=$TLB/blt${2}_leetboth_$3.cache.jsonl
    [ -f $c ] && scp -q $c $H:hyun2/typoleet_region/out/blt${2}_leetregion_$3.cache.jsonl
  fi
  say "start [$l] GPU$g"
  $SSH $H "$(remote_cmd $g "$l")" > "$O/logs/${l// /_}.$H.log" 2>&1
  local rc=$?
  scp -q $H:hyun2/typoleet_region/out/$f $O/$f 2>/dev/null
  [ $1 = blt_leet ] && scp -q $H:hyun2/typoleet_region/out/blt${2}_leetregion_$3.cache.jsonl $O/ 2>/dev/null
  if [ -f $O/$f ]; then say "end [$l] GPU$g exit=$rc out=yes"; else say "end [$l] GPU$g exit=$rc out=NO (requeued)"; pushback "$l"; fi
  rm -f $O/.remote_busy_$g; }

declare -A IDLE
mkdir -p $O/logs
say "controller up, gpus=$GPUS idle>=${IDLE_MIN}min"
while true; do
  busy=$($SSH $H 'nvidia-smi --query-gpu=index,uuid --format=csv,noheader; echo ---; nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader' 2>/dev/null) || { sleep 60; continue; }
  now=$(date +%s)
  for g in $GPUS; do
    u=$(awk -F', ' -v g=$g '$1==g{print $2}' <<< "${busy%%---*}")
    if [ -f $O/.remote_busy_$g ]; then IDLE[$g]=$((now - IDLE_MIN * 60)); continue; fi   # mine; reusable right after
    if grep -q "$u" <<< "${busy#*---}"; then unset IDLE[$g]; continue; fi
    [ -z "${IDLE[$g]:-}" ] && IDLE[$g]=$now
    if [ $((now - IDLE[$g])) -ge $((IDLE_MIN * 60)) ]; then
      l=$(popblt); [ -z "$l" ] && continue
      touch $O/.remote_busy_$g; job $g "$l" &
    fi
  done
  if ! grep -q '^blt_' $Q 2>/dev/null && ! ls $O/.remote_busy_* >/dev/null 2>&1; then break; fi
  sleep 60
done
wait; say "controller done"
