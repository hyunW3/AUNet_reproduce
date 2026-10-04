#!/usr/bin/env bash
# BLT PBP-MC with acc + acc_norm at theta 1.34 (released) and 1.61 (calibrated), one job per (theta, task).
# snu55 GPUs 0,1 (user-cleared 2026-10-04 for this robustness follow-up), two workers per GPU.
#   run_snu55_pbp.sh <out_dir> [gpus="0 0 1 1"]
set -u
A=/mnt/ssd2/hyun2/AUNet; S=$(cd "$(dirname "$0")" && pwd); O=$1 GPUS=${2:-0 0 1 1}; mkdir -p $O/logs
W=$A/runs/ext_ci_snu55
Q=$O/jobs.txt
[ -f $Q ] || for th in 1.335442066192627 1.6093749403933089; do for t in hellaswag arc_easy arc_challenge piqa boolq; do echo "$th $t"; done; done > $Q
say(){ echo "$(date '+%F %T') $*" >> $O/queue.log; }
pop(){ ( flock 9; l=$(head -n1 $Q); [ -n "$l" ] && tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q; echo "$l" ) 9>$Q.lock; }
worker(){ local g=$1
  while true; do
    l=$(pop); [ -z "$l" ] && break
    read -r th t <<< "$l"; tag=blt_t${th:0:4}_$t
    say "start [$tag] GPU$g"
    ( cd /tmp && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$W/extra_site:$W/blt_official EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite \
        AUNET_LINGUA=$A/lingua HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1 \
        /mnt/ssd2/hyun2/venvs/vllm011/bin/python -u $S/run_pbp_ext.py --family blt_official --threshold $th \
        --tasks $t --blt_weights $W/blt_weights --out $O/$tag.json ) > $O/logs/$tag.log 2>&1
    say "end [$tag] GPU$g exit=$?"
  done; say "worker GPU$g done"; }
for g in $GPUS; do worker $g & sleep 20; done; wait; say ALL_DONE
