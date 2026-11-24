#!/bin/bash
# Common environment for the tokenizer-swap replication bundle.
# SOURCE this (`. 00_env.sh`) — do not execute. Every step script sources it.
#
# The whole study swaps ONLY the eval-time patch-boundary tokenizer on a fixed
# byte-in/byte-out checkpoint (no weight updates) and measures BPB / downstream.
# Central finding: the swap ALWAYS hurts — boundary-DISTRIBUTION binding, not
# coverage/granularity. See ../../PoC/tokenizer_swap_all_results.md.

set -euo pipefail

# --- paths (edit BASE if the tree moves) -------------------------------------
export BASE=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet
export LINGUA=$BASE/lingua
export TOK_DIR=$BASE/tokenizer          # per-tokenizer subdirs hold tokenizer.json
export DATA_DIR=$BASE/data              # BPB corpora (wiki_zh, wiki_te, code_*)
export TASKS=$LINGUA/eval_tasks         # custom lm_eval task defs (cloze / MC)

# --- byte-model checkpoint under study ---------------------------------------
export RUN=$BASE/runs/1.3B/bpebyte_br_greedy_root_1.3B
export CKPT=$RUN/checkpoints/0000180000
# subword reference (CANNOT be swapped — embeddings bound to vocab); apps.main.eval
export LLAMA_RUN=$BASE/runs/1.8B/llama_1.8B_paper
export LLAMA_CKPT=$LLAMA_RUN/checkpoints/0000180000

# --- torch caches ------------------------------------------------------------
export TORCHINDUCTOR_CACHE_DIR=$BASE/runs/.cache/torchinductor
export TRITON_CACHE_DIR=$BASE/runs/.cache/triton
export HF_DATASETS_TRUST_REMOTE_CODE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# --- runtime knobs -----------------------------------------------------------
export NPROC=${NPROC:-4}                # GPUs; 4 on B200, set 2 on shared ece nodes
export MASTER_PORT=${MASTER_PORT:-29515}
export LIMIT=${LIMIT:-0}                # 0 = full set; >0 = per-task doc cap for smoke tests

activate_env() {
  cd "$LINGUA"
  # venv is uv-managed; no system pip
  # shellcheck disable=SC1091
  source "$LINGUA/.venv/bin/activate"
}
