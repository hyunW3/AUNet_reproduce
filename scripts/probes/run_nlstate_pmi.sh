#!/usr/bin/env bash
# Natural-language state tracking, SCORING-RULE ABLATION.
#   GPUS="1 2 3" ./run_nlstate_pmi.sh
#
# Same five tasks as run_nlstate.sh, but each task declares three metrics:
#   acc              argmax of the raw SUMMED option log-likelihood  (what the
#                    first run used) -- biased toward SHORT options, and 99.5% of
#                    TSO items have choices of unequal length (mean spread 6.9
#                    chars). This is the suspect: all 9 TSO cells landed BELOW
#                    chance, which is a systematic-bias signature, not noise.
#   acc_norm         argmax of lls / len(choice) -- length-normalised. FREE: same
#                    forward passes, computed from the same lls.
#   acc_mutual_info  argmax of log P(choice|ctx) - log P(choice) -- PMI, removes
#                    each option's unconditional prior (Holtzman et al. 2021).
#                    Costs ~2x runtime (extra unconditional passes).
#
# cf. scripts/probes/listmax_eval.py, which hit exactly this problem: "raw
# argmax lands below chance" -> it reports PMI alongside.
# Writes to reports/statetrack/nlstate_pmi/ so the original run is preserved;
# the `acc` column must reproduce the first run exactly (same passes).
#
# Five tasks, all loglikelihood-scored multiple choice (base 1.3B models cannot
# follow the generate_until/CoT BBH variants, so the MC forms are the only
# meaningful ones -- same reasoning as the earlier BIG-bench Dyck run).
#
#   bb_tso{3,5,7}_mc  tracking_shuffled_objects split by object count.
#                     "pairs of players trade balls" == composition of
#                     transpositions: the NATURAL-LANGUAGE ANALOGUE OF THE S5
#                     PROBE, with 3/5/7 as a difficulty ladder.
#                     chance = 1/3, 1/5, 1/7.  n = 750 / 1250 / 1750.
#   temporal_sequences  track a person's occupied hours -> free slot. chance 1/4, n=1000.
#   navigate            track position through movement steps.  chance 1/2, n=1000.
#
# One job per model (all five tasks in a single harness invocation), dealt into
# one SERIAL lane per GPU so only one 1.3B model is resident per card.
set -u
ROOT=/mnt/ssd2/hyun2/AUNet
cd "$ROOT/lingua"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=127.0.0.1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
PY="$ROOT/lingua/.venv/bin/python"
O="$ROOT/reports/statetrack/nlstate_pmi"; mkdir -p "$O"
INC="$ROOT/scripts/probes/nlstate_tasks"

TASKS="bb_tso3_mc,bb_tso5_mc,bb_tso7_mc,bb_temporal_mc,bb_navigate_mc"

GPUS=${GPUS:-"1 2 3"}
read -ra G <<< "$GPUS"; NG=${#G[@]}

# app|config|ckpt|tag|port
JOBS=(
  "apps.main.eval|apps/main/configs/eval_llama_1B_b200.yaml|$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000|subword_llama|29631"
  "apps.aunet.eval|apps/aunet/configs/eval_full_5bench_b200.yaml|$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000|aunet_static|29633"
  "apps.aunet.eval|apps/aunet/configs/eval_full_5bench_b200.yaml|$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000|byte_greedyroot|29635"
)

run_lane(){
  local lane=$1 gpu=${G[$1]} i
  for ((i=lane; i<${#JOBS[@]}; i+=NG)); do
    IFS='|' read -r app cfg ck tag port <<< "${JOBS[$i]}"
    echo "[gpu$gpu] START $tag" >&2
    CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$port $PY -m "$app" \
      config="$cfg" validation=null \
      harness.tasks="[$TASKS]" harness.include_path="$INC" \
      ckpt_dir="$ck" dump_dir="$O/$tag" > "$O/$tag.log" 2>&1
    echo "[gpu$gpu] DONE  $tag exit=$?" >&2
  done
}
for ((l=0; l<NG; l++)); do run_lane "$l" & done
wait
echo "NLSTATE DONE"
