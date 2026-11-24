#!/usr/bin/env bash
# Natural-language state tracking, SCORING-RULE ABLATION -- snu20 edition.
#   GPUS="0 1 2" ./run_nlstate_pmi_snu20.sh
#
# Same five tasks and three metrics as run_nlstate_pmi.sh, but pathed for snu20:
#   ROOT=/mnt/ssd/hyun/aunet_r10lr, interpreter is $ROOT/venv (NOT lingua/.venv,
#   which does not exist there).
#
#   acc              raw summed option log-likelihood -- what the first run used.
#                    Biased toward SHORT options; 99.5% of TSO items have
#                    unequal-length choices, and all 9 TSO cells landed BELOW
#                    chance, which is a systematic-bias signature.
#   acc_norm         lls / len(choice). FREE -- same forward passes.
#   acc_mutual_info  log P(choice|ctx) - log P(choice). ~2x runtime.
#
# snu20 was verified idle before launch (4x A5000, 3 MiB each, no compute apps)
# per the standing node policy: free VRAM alone is not permission, and a node is
# re-admitted only when every GPU is idle AND nvidia-smi lists no compute apps.
set -u
ROOT=/mnt/ssd/hyun/aunet_r10lr
cd "$ROOT/lingua"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1                 # cache was pre-seeded; no 3-way download race
export MASTER_ADDR=127.0.0.1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
PY="$ROOT/venv/bin/python"
O="$ROOT/reports/statetrack/nlstate_pmi"; mkdir -p "$O"
INC="$ROOT/scripts/probes/nlstate_tasks"

TASKS="bb_tso3_mc,bb_tso5_mc,bb_tso7_mc,bb_temporal_mc,bb_navigate_mc"

GPUS=${GPUS:-"0 1 2"}
read -ra G <<< "$GPUS"; NG=${#G[@]}

JOBS=(
  "apps.main.eval|apps/main/configs/eval_llama_1B_b200.yaml|$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000|subword_llama|29731"
  "apps.aunet.eval|apps/aunet/configs/eval_full_5bench_b200.yaml|$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000|aunet_static|29733"
  "apps.aunet.eval|apps/aunet/configs/eval_full_5bench_b200.yaml|$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000|byte_greedyroot|29735"
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
