#!/usr/bin/env bash
# Typo-both and Leet-both robustness (2026-10-06): the typo / leet perturbation is applied to the context AND to
# every answer option, over HellaSwag / ARC-E / ARC-C / PIQA (BoolQ excluded: yes/no options), for the five
# main-table models (+ BLT at both thresholds). Per-item correctness bits are kept for the paired bootstrap.
#   run_typoleet_both.sh <out_dir> [gpus="0 1 2 3"]
# One worker per GPU pops jobs from <out_dir>/jobs.txt (resumable: finished jobs leave their output json).
# Code: lingua (eval_typo_ds typoboth) + this AUNet checkout (run_ext typoboth, format_mc nla_leet_both).
set -u
O=$1; GPUS=${2:-0 1 2 3}
A=/mnt/ssd2/hyun2/AUNet
LW=$A/lingua
S=$(cd "$(dirname "$0")" && pwd)
X=$A/runs/ext_ci_snu55
ITEMS=$A/reports/format_robustness/items
T4="hellaswag arc_easy arc_challenge piqa"
M=$A/main/main/1.3B
declare -A CK=([llama]=$M/llama_1.8B_paper/checkpoints/0000060000/consolidated
               [aunet]=$M/aunet2_1.3B/checkpoints/0000180000/consolidated
               [bpebyte]=$M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated)
declare -A APP=([llama]=apps.main.eval [aunet]=apps.aunet.eval [bpebyte]=apps.aunet.eval)
mkdir -p $O/logs
Q=$O/jobs.txt
if [ ! -f $Q ]; then
  {
    for th in 1335 1609; do for t in $T4; do echo "blt_typo $th $t"; done; done
    echo "hnet_typo"; for m in llama aunet bpebyte; do echo "trio_typo $m"; done
    for th in 1335 1609; do for t in $T4; do echo "blt_leet $th $t"; done; done
    echo "hnet_leet"; for m in llama aunet bpebyte; do echo "trio_leet $m"; done
  } > $Q
fi
NV=$A/lingua/.venv/lib/python3.12/site-packages/nvidia
LDP="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -)"
thr(){ [ "$1" = 1335 ] && echo 1.335442066192627 || echo 1.6093749403933089; }
say(){ echo "$(date '+%F %T') $*" >> $O/queue.log; }
pop(){ ( flock 9; l=$(head -n1 $Q); [ -n "$l" ] && tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q; echo "$l" ) 9>$Q.lock; }

run(){ local g=$1 kind=$2 a=${3:-} b=${4:-}
  case $kind in
    trio_typo)   # within-run clean baseline + 8 typoboth variants per task, stock eval entry point
      python3 - "$A/runs/robustness_paper1p3b_ext/$a/config.yaml" "$O/trio_typo_$a.yaml" "$O/trio_typo_$a" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); c["name"] = c["name"] + "_typoboth"; c["dump_dir"] = sys.argv[3]
c["harness"]["tasks"] = ["hellaswag", "arc_easy", "arc_challenge", "piqa",
                         "hellaswag_typoboth", "arc_easy_typoboth", "arc_challenge_typoboth", "piqa_typoboth"]
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
EOF
      ( cd $LW && export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}" TRITON_CACHE_DIR=/tmp/tr_tl_$g TORCHINDUCTOR_CACHE_DIR=/tmp/ti_tl_$g \
          CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$LW && \
        $A/lingua/.venv/bin/python -m torch.distributed.run --nproc-per-node 1 --master-port $((29900 + g)) -m ${APP[$a]} \
          config=$O/trio_typo_$a.yaml ckpt_dir=${CK[$a]} dump_dir=$O/trio_typo_$a ) ;;
    trio_leet)
      local mt=16384; [ $a = llama ] && mt=4096
      cat > $O/trio_leet_$a.yaml <<EOF
name: "format_mc_${a}_leetboth"
ckpt_dir: ${CK[$a]}
dump_dir: $O/trio_leet_$a
harness:
  log_samples: false
  limit: 2000
  tasks:
    - despace_mc
validation: null
generator:
  max_tokens: $mt
  dtype: bf16
EOF
      ( cd $LW && export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}" && \
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$LW FORMAT_ENTRY=${APP[$a]} FORMAT_ITEMS=$ITEMS \
          FORMAT_CACHE=$O/trio_leet_$a.cache.jsonl FORMAT_VARIANTS=clean,nla_leet_both FORMAT_TASKS=${T4// /,} \
          MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29950 + g)) RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
          $A/lingua/.venv/bin/python -u $S/format_mc/run_format_lingua.py config=$O/trio_leet_$a.yaml ) ;;
    blt_typo)
      ( cd /tmp && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$X/extra_site:$X/blt_official AUNET_LINGUA=$LW EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite \
          HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1 \
          /mnt/ssd2/hyun2/venvs/vllm011/bin/python $S/ext_ci/run_ext.py --family blt_official --threshold $(thr $a) \
          --blt_weights $X/blt_weights --axis typoboth --tasks $b --limit 2000 --out $O/blt${a}_typoboth_$b.json ) ;;
    blt_leet)
      ( cd /tmp && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$X/extra_site:$X/blt_official EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite \
          AUNET_LINGUA=$LW HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1 \
          /mnt/ssd2/hyun2/venvs/vllm011/bin/python -u $S/format_mc/run_format_ext.py --family blt_official --threshold $(thr $a) \
          --tasks $b --variants clean nla_leet_both --blt_weights $X/blt_weights --items_dir $ITEMS \
          --cache $O/blt${a}_leetboth_$b.cache.jsonl --out $O/blt${a}_leetboth_$b.json ) ;;
    hnet_typo)
      ( cd /tmp && CUDA_VISIBLE_DEVICES=$g AUNET_LINGUA=$LW EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HNET_REPO=/mnt/ssd2/hyun2/hnet \
          HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
          /home/hyunwoong/miniconda3/envs/spacebyte/bin/python $S/ext_ci/run_ext.py --family hnet --axis typoboth \
          --tasks $T4 --limit 2000 --out $O/hnet_typoboth.json ) ;;
    hnet_leet)
      ( cd /tmp && CUDA_VISIBLE_DEVICES=$g AUNET_LINGUA=$LW EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HNET_REPO=/mnt/ssd2/hyun2/hnet \
          HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
          /home/hyunwoong/miniconda3/envs/spacebyte/bin/python -u $S/format_mc/run_format_ext.py --family hnet \
          --tasks $T4 --variants clean nla_leet_both --items_dir $ITEMS \
          --cache $O/hnet_leetboth.cache.jsonl --out $O/hnet_leetboth.json ) ;;
  esac; }

worker(){ local g=$1
  while true; do
    l=$(pop); [ -z "$l" ] && break
    say "start [$l] GPU$g"
    run $g $l > "$O/logs/${l// /_}.log" 2>&1
    say "end [$l] GPU$g exit=$?"
  done; say "worker GPU$g done"; }
for g in $GPUS; do worker $g & sleep 20; done; wait; say ALL_DONE
