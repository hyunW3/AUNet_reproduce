#!/usr/bin/env bash
# Typo / Leet split by region (2026-11-25): the both-region Typo and Leet axes of the paper table
# (run_typoleet_both.sh, BPEByte from runs/robustness_paper4task_bpebyte) decomposed into
#   ctx : perturb the context only (answer options clean)
#   opt : perturb the answer options only (context clean)
# over HellaSwag / ARC-E / ARC-C / PIQA, for Llama / AU-Net / BPEByte + BLT (theta 1.335, 1.609) + H-Net.
#   Typo : <task>_typoctx / <task>_typoopt (lingua eval_typo_ds, the typoboth seeds -> ctx edit / option edits
#          byte-identical to the matching half of typoboth); within-run clean baseline.
#   Leet : format_mc nla_leet (context only) and nla_leet_opt (options only); the cache of the paper's
#          nla_leet_both run is copied in first, so the clean scores are the paper's.
# Per-model config = the one behind each model's paper both-region number.
#   run_typoleet_region.sh <out_dir> [gpus="0 1 2 3"]
# One worker per GPU pops jobs from <out_dir>/jobs.txt; a job whose output json exists is skipped.
# Code: this checkout's lingua (eval_typo_ds) + scripts; venvs/data/checkpoints from the main AUNet tree.
set -u
O=$(realpath -m "$1"); GPUS=${2:-0 1 2 3}
A=/mnt/ssd2/hyun2/AUNet
S=$(cd "$(dirname "$0")" && pwd)
LW=${LW:-$(cd "$S/../../lingua" && pwd)}        # lingua carrying typoctx / typoopt
VPY=$A/lingua/.venv/bin/python
X=$A/runs/ext_ci_snu55
TLB=$A/runs/robustness_typoleet_both
BPE=$A/runs/robustness_paper4task_bpebyte
ITEMS=$A/reports/format_robustness/items
T4="hellaswag arc_easy arc_challenge piqa"
LIMIT=${LIMIT:-2000}
M=$A/main/main/1.3B
declare -A CK=([llama]=$M/llama_1.8B_paper/checkpoints/0000060000/consolidated
               [aunet]=$M/aunet2_1.3B/checkpoints/0000180000/consolidated
               [bpebyte]=$M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated)
declare -A APP=([llama]=apps.main.eval [aunet]=apps.aunet.eval [bpebyte]=apps.aunet.eval)
# config behind each paper both-region number (typo / leet)
declare -A TCFG=([llama]=$A/runs/robustness_paper1p3b_ext/llama/config.yaml
                 [aunet]=$A/runs/robustness_paper1p3b_ext/aunet/config.yaml
                 [bpebyte]=$BPE/cfg/typo_bpebyte.yaml)
declare -A LCFG=([llama]=$TLB/trio_leet_llama.yaml [aunet]=$TLB/trio_leet_aunet.yaml [bpebyte]=$BPE/cfg/leet_bpebyte.yaml)
declare -A LCACHE=([llama]=$TLB/trio_leet_llama.cache.jsonl [aunet]=$TLB/trio_leet_aunet.cache.jsonl
                   [bpebyte]=$BPE/leet/bpebyte.cache.jsonl)
mkdir -p $O/logs $O/cfg
Q=$O/jobs.txt
if [ ! -f $Q ]; then
  {   # slow BLT typo jobs first so the tail of the queue is short
    for t in hellaswag arc_easy arc_challenge piqa; do for th in 1335 1609; do for r in ctx opt; do echo "blt_typo $th $t $r"; done; done; done
    for r in ctx opt; do echo "hnet_typo $r"; for m in llama aunet bpebyte; do echo "trio_typo $m $r"; done; done
    for th in 1335 1609; do for t in $T4; do echo "blt_leet $th $t"; done; done
    echo "hnet_leet"; for m in llama aunet bpebyte; do echo "trio_leet $m"; done
  } > $Q
fi
NV=$A/lingua/.venv/lib/python3.12/site-packages/nvidia
LDP="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -)"
thr(){ [ "$1" = 1335 ] && echo 1.335442066192627 || echo 1.6093749403933089; }
say(){ echo "$(date '+%F %T') $*" >> $O/queue.log; }
pop(){ ( flock 9; l=$(head -n1 $Q); [ -n "$l" ] && tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q; echo "$l" ) 9>$Q.lock; }
seed_cache(){ [ -f "$2" ] || { [ -f "$1" ] && cp "$1" "$2"; }; }   # paper cache -> clean scores reused

out_of(){ local kind=$1 a=${2:-} b=${3:-} c=${4:-}
  case $kind in
    trio_typo) echo $O/trio_typo${b}_$a/results.json ;;
    trio_leet) echo $O/trio_leet_$a/results.json ;;
    blt_typo)  echo $O/blt${a}_typo${c}_$b.json ;;
    blt_leet)  echo $O/blt${a}_leetregion_$b.json ;;
    hnet_typo) echo $O/hnet_typo$a.json ;;
    hnet_leet) echo $O/hnet_leetregion.json ;;
  esac; }

run(){ local g=$1 kind=$2 a=${3:-} b=${4:-} c=${5:-}
  case $kind in
    trio_typo)   # a=model b=ctx|opt: within-run clean + 8 typo<region> variants per task
      "$VPY" - "${TCFG[$a]}" "$O/cfg/typo${b}_$a.yaml" "$O/trio_typo${b}_$a" "$b" "$LIMIT" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); r = sys.argv[4]
T4 = ["hellaswag", "arc_easy", "arc_challenge", "piqa"]
c["name"] = c["name"].split("_robustness")[0] + f"_robustness_typo{r}"; c["dump_dir"] = sys.argv[3]
c["harness"].update(tasks=T4 + [f"{t}_typo{r}" for t in T4], limit=int(sys.argv[5]))
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
EOF
      ( cd $LW && export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}" TRITON_CACHE_DIR=/tmp/tr_tr_${g}_$BASHPID TORCHINDUCTOR_CACHE_DIR=/tmp/ti_tr_${g}_$BASHPID \
          CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$LW && \
        "$VPY" -m torch.distributed.run --nproc-per-node 1 --master-port $((29900 + g)) -m ${APP[$a]} \
          config=$O/cfg/typo${b}_$a.yaml ckpt_dir=${CK[$a]} dump_dir=$O/trio_typo${b}_$a ) ;;
    trio_leet)
      seed_cache "${LCACHE[$a]}" $O/trio_leet_$a.cache.jsonl
      "$VPY" - "${LCFG[$a]}" "$O/cfg/leet_$a.yaml" "$O/trio_leet_$a" "$LIMIT" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["name"] = c["name"].split("_robustness")[0].replace("_leetboth", "") + "_leetregion"; c["dump_dir"] = sys.argv[3]
c["harness"].update(limit=int(sys.argv[4]))
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
EOF
      ( cd $LW && export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}" && \
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$LW FORMAT_ENTRY=${APP[$a]} FORMAT_ITEMS=$ITEMS \
          FORMAT_CACHE=$O/trio_leet_$a.cache.jsonl FORMAT_VARIANTS=clean,nla_leet,nla_leet_opt FORMAT_TASKS=${T4// /,} \
          MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29950 + g)) RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
          "$VPY" -u $S/format_mc/run_format_lingua.py config=$O/cfg/leet_$a.yaml ckpt_dir=${CK[$a]} dump_dir=$O/trio_leet_$a ) ;;
    blt_typo)    # a=threshold b=task c=ctx|opt
      ( cd /tmp && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$X/extra_site:$X/blt_official AUNET_LINGUA=$LW EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite \
          HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1 \
          /mnt/ssd2/hyun2/venvs/vllm011/bin/python $S/ext_ci/run_ext.py --family blt_official --threshold $(thr $a) \
          --blt_weights $X/blt_weights --axis typo$c --tasks $b --limit $LIMIT --out $(out_of blt_typo $a $b $c) ) ;;
    blt_leet)
      seed_cache $TLB/blt${a}_leetboth_$b.cache.jsonl $O/blt${a}_leetregion_$b.cache.jsonl
      ( cd /tmp && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$X/extra_site:$X/blt_official EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite \
          AUNET_LINGUA=$LW HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1 \
          /mnt/ssd2/hyun2/venvs/vllm011/bin/python -u $S/format_mc/run_format_ext.py --family blt_official --threshold $(thr $a) \
          --tasks $b --variants clean nla_leet nla_leet_opt --blt_weights $X/blt_weights --items_dir $ITEMS --limit $LIMIT \
          --cache $O/blt${a}_leetregion_$b.cache.jsonl --out $(out_of blt_leet $a $b) ) ;;
    hnet_typo)   # a=ctx|opt
      ( cd /tmp && CUDA_VISIBLE_DEVICES=$g AUNET_LINGUA=$LW EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HNET_REPO=/mnt/ssd2/hyun2/hnet \
          HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
          /home/hyunwoong/miniconda3/envs/spacebyte/bin/python $S/ext_ci/run_ext.py --family hnet --axis typo$a \
          --tasks $T4 --limit $LIMIT --out $(out_of hnet_typo $a) ) ;;
    hnet_leet)
      seed_cache $TLB/hnet_leetboth.cache.jsonl $O/hnet_leetregion.cache.jsonl
      ( cd /tmp && CUDA_VISIBLE_DEVICES=$g AUNET_LINGUA=$LW EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HNET_REPO=/mnt/ssd2/hyun2/hnet \
          HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
          /home/hyunwoong/miniconda3/envs/spacebyte/bin/python -u $S/format_mc/run_format_ext.py --family hnet \
          --tasks $T4 --variants clean nla_leet nla_leet_opt --items_dir $ITEMS --limit $LIMIT \
          --cache $O/hnet_leetregion.cache.jsonl --out $(out_of hnet_leet) ) ;;
  esac; }

worker(){ local g=$1
  while true; do
    l=$(pop); [ -z "$l" ] && break
    if [ -f "$(out_of $l)" ]; then say "skip [$l] (output exists)"; continue; fi
    say "start [$l] GPU$g"
    run $g $l > "$O/logs/${l// /_}.log" 2>&1
    say "end [$l] GPU$g exit=$? out=$([ -f "$(out_of $l)" ] && echo yes || echo NO)"
  done; say "worker GPU$g done"; }
for g in $GPUS; do worker $g & sleep 20; done; wait; say ALL_DONE
