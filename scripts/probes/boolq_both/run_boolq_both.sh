#!/usr/bin/env bash
# BoolQ on the both region (context AND the yes/no answer options), 2026-10-06: Noise (5 strategies), Typo (8) and
# Leet, for the five main-table models (+ BLT at both thresholds), so BoolQ can rejoin the four-task robustness
# average under the same protocol. Despace needs no run (despace_mc_boolq_despaceall exists; "yes"/"no" have no
# spaces). Same settings as run_typoleet_both.sh: limit 2000, perturbation seed 1234, within-run clean baseline,
# per-item correctness kept for the paired bootstrap.
#   run_boolq_both.sh <out_dir> [gpus="0 1 2 3"] [job filter regex, e.g. '^(blt|hnet)']
# Paths default to snu55/gpusvr0908; another node overrides LW, X (BLT weights dir), BLTPATH, BLTPY, HNETPY, HNET_REPO,
# EVAL_SUITE, ITEMS, HF_HUB_CACHE (see run_boolq_both_snu20.sh).
# Noise/Typo: boolq_both_tasks.py (overlay sentinels boolq_noiseboth / boolq_typoboth) via launch.py.
# Leet: format_mc nla_leet_both on boolq (no code change; leet seed 0, the context block of the earlier runs).
set -u
O=$1; GPUS=${2:-0 1 2 3}; FILT=${3:-.}
A=/mnt/ssd2/hyun2/AUNet
LW=${LW:-$A/lingua}   # lingua main (paper runs; the typo-leet-both worktree is no longer used)
S=$(cd "$(dirname "$0")/.." && pwd)
L=$S/boolq_both/launch.py
X=${X:-$A/runs/ext_ci_snu55}
ITEMS=${ITEMS:-$A/reports/format_robustness/items}
M=$A/main/main/1.3B
declare -A CK=([llama]=$M/llama_1.8B_paper/checkpoints/0000060000/consolidated
               [aunet]=$M/aunet2_1.3B/checkpoints/0000180000/consolidated
               [bpebyte]=$M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated)
declare -A APP=([llama]=apps.main.eval [aunet]=apps.aunet.eval [bpebyte]=apps.aunet.eval)
mkdir -p $O/logs
Q=$O/jobs.txt
if [ ! -f $Q ]; then
  {
    for th in 1335 1609; do echo "blt_typo $th"; echo "blt_noise $th"; done
    echo "hnet_noise"; echo "hnet_typo"
    for m in llama aunet bpebyte; do echo "trio_nt $m"; done
    for th in 1335 1609; do echo "blt_leet $th"; done
    echo "hnet_leet"; for m in llama aunet bpebyte; do echo "trio_leet $m"; done
    # typo on the context only (passage + question; labels untouched): BoolQ's region in the five-task average
    for th in 1335 1609; do echo "blt_tc $th"; done; echo "hnet_tc"; for m in llama aunet bpebyte; do echo "trio_tc $m"; done
  } | grep -E "$FILT" > $Q
fi
NV=$A/lingua/.venv/lib/python3.12/site-packages/nvidia
LDP="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -)"
thr(){ [ "$1" = 1335 ] && echo 1.335442066192627 || echo 1.6093749403933089; }
say(){ echo "$(date '+%F %T') $*" >> $O/queue.log; }
pop(){ ( flock 9; l=$(head -n1 $Q); [ -n "$l" ] && tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q; echo "$l" ) 9>$Q.lock; }
ES=${EVAL_SUITE:-/mnt/ssd2/hyun2/eval_suite}
HUB=${HF_HUB_CACHE:+HF_HUB_CACHE=$HF_HUB_CACHE}
BLTENV="PYTHONPATH=${BLTPATH:-$X/extra_site:$X/blt_official} AUNET_LINGUA=$LW EVAL_SUITE=$ES HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1"
HNETENV="AUNET_LINGUA=$LW EVAL_SUITE=$ES HNET_REPO=${HNET_REPO:-/mnt/ssd2/hyun2/hnet} HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 $HUB"
BLTPY=${BLTPY:-/mnt/ssd2/hyun2/venvs/vllm011/bin/python}
HNETPY=${HNETPY:-/home/hyunwoong/miniconda3/envs/spacebyte/bin/python}

run(){ local g=$1 kind=$2 a=${3:-}
  case $kind in
    trio_nt|trio_tc)   # within-run clean boolq + (nt) 5 noise-both + 8 typo-both / (tc) 8 typo-context variants
      local tl="boolq_noiseboth,boolq_typoboth"; [ $kind = trio_tc ] && tl="boolq_typoctx"
      python3 - "$A/runs/robustness_paper1p3b_ext/$a/config.yaml" "$O/${kind}_$a.yaml" "$O/${kind}_$a" "$tl" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])); c["name"] = c["name"] + "_boolqboth"; c["dump_dir"] = sys.argv[3]
c["harness"]["tasks"] = ["boolq"] + sys.argv[4].split(",")
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
EOF
      ( cd $LW && export LD_LIBRARY_PATH="$LDP:${LD_LIBRARY_PATH:-}" TRITON_CACHE_DIR=/tmp/tr_bq_$g TORCHINDUCTOR_CACHE_DIR=/tmp/ti_bq_$g \
          CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$LW AUNET_LINGUA=$LW && \
        $A/lingua/.venv/bin/python -m torch.distributed.run --nproc-per-node 1 --master-port $((29800 + g)) $L -m ${APP[$a]} \
          config=$O/${kind}_$a.yaml ckpt_dir=${CK[$a]} dump_dir=$O/${kind}_$a ) ;;
    trio_leet)
      local mt=16384; [ $a = llama ] && mt=4096
      cat > $O/trio_leet_$a.yaml <<EOF
name: "format_mc_${a}_boolq_leetboth"
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
          FORMAT_CACHE=$O/trio_leet_$a.cache.jsonl FORMAT_VARIANTS=clean,nla_leet_both FORMAT_TASKS=boolq \
          MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29850 + g)) RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
          $A/lingua/.venv/bin/python -u $S/format_mc/run_format_lingua.py config=$O/trio_leet_$a.yaml ) ;;
    blt_noise)   # run_ext axis noise; BOOLQ_BOTH_NOISE_SENTINEL maps its boolq_noise sentinel to the both variants
      ( cd /tmp && env CUDA_VISIBLE_DEVICES=$g $BLTENV BOOLQ_BOTH_NOISE_SENTINEL=1 \
          $BLTPY $L $S/ext_ci/run_ext.py --family blt_official --threshold $(thr $a) \
          --blt_weights $X/blt_weights --axis noise --tasks boolq --limit 2000 --out $O/blt${a}_noiseboth_boolq.json ) ;;
    blt_typo)
      ( cd /tmp && env CUDA_VISIBLE_DEVICES=$g $BLTENV \
          $BLTPY $L $S/ext_ci/run_ext.py --family blt_official --threshold $(thr $a) \
          --blt_weights $X/blt_weights --axis typoboth --tasks boolq --limit 2000 --out $O/blt${a}_typoboth_boolq.json ) ;;
    blt_tc)   # BOOLQ_TYPO_CTX maps run_ext's boolq_typoboth sentinel to the context-only variants
      ( cd /tmp && env CUDA_VISIBLE_DEVICES=$g $BLTENV BOOLQ_TYPO_CTX=1 \
          $BLTPY $L $S/ext_ci/run_ext.py --family blt_official --threshold $(thr $a) \
          --blt_weights $X/blt_weights --axis typoboth --tasks boolq --limit 2000 --out $O/blt${a}_typoctx_boolq.json ) ;;
    hnet_tc)
      ( cd /tmp && env CUDA_VISIBLE_DEVICES=$g $HNETENV BOOLQ_TYPO_CTX=1 \
          $HNETPY $L $S/ext_ci/run_ext.py --family hnet --axis typoboth --tasks boolq --limit 2000 \
          --out $O/hnet_typoctx_boolq.json ) ;;
    blt_leet)
      ( cd /tmp && env CUDA_VISIBLE_DEVICES=$g $BLTENV \
          $BLTPY -u $S/format_mc/run_format_ext.py --family blt_official --threshold $(thr $a) \
          --tasks boolq --variants clean nla_leet_both --blt_weights $X/blt_weights --items_dir $ITEMS \
          --cache $O/blt${a}_leetboth_boolq.cache.jsonl --out $O/blt${a}_leetboth_boolq.json ) ;;
    hnet_noise)
      ( cd /tmp && env CUDA_VISIBLE_DEVICES=$g $HNETENV BOOLQ_BOTH_NOISE_SENTINEL=1 \
          $HNETPY $L $S/ext_ci/run_ext.py --family hnet --axis noise --tasks boolq --limit 2000 \
          --out $O/hnet_noiseboth_boolq.json ) ;;
    hnet_typo)
      ( cd /tmp && env CUDA_VISIBLE_DEVICES=$g $HNETENV \
          $HNETPY $L $S/ext_ci/run_ext.py --family hnet --axis typoboth --tasks boolq --limit 2000 \
          --out $O/hnet_typoboth_boolq.json ) ;;
    hnet_leet)
      ( cd /tmp && env CUDA_VISIBLE_DEVICES=$g $HNETENV \
          $HNETPY -u $S/format_mc/run_format_ext.py --family hnet --tasks boolq --variants clean nla_leet_both \
          --items_dir $ITEMS --cache $O/hnet_leetboth_boolq.cache.jsonl --out $O/hnet_leetboth_boolq.json ) ;;
  esac; }

worker(){ local g=$1
  while true; do
    l=$(pop); [ -z "$l" ] && break
    say "start [$l] GPU$g"
    run $g $l > "$O/logs/${l// /_}.log" 2>&1
    say "end [$l] GPU$g exit=$?"
  done; say "worker GPU$g done"; }
for g in $GPUS; do worker $g & sleep 20; done; wait; say ALL_DONE
