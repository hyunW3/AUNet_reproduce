#!/usr/bin/env bash
# Distractor-bias diagnostic for BIG-bench temporal_sequences -- snu20.
# One model per GPU. See scripts/probes/temporal_mech.py for the hypothesis.
set -u
ROOT=/mnt/ssd/hyun/aunet_r10lr
cd "$ROOT"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/scripts/probes:$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1
PY="$ROOT/venv/bin/python"
O="$ROOT/reports/statetrack/temporal_mech"; mkdir -p "$O"
# RELATIVE to AUNET_ROOT: common.py:ckpt_for() joins them as "$AUNET_ROOT/$rel",
# so an absolute path here silently doubles the prefix.
LL="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AU="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BP="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }
one(){ local tag=$1 ckenv=$2 ck=$3 gpu=$4 rc
  echo "[gpu$gpu] START $tag" >&2
  # NOTE: take PIPESTATUS[0], not $? -- $? is the exit of `filt`, which is 0 even
  # when python died, and that reported a crashed run as "exit=0" once already.
  CUDA_VISIBLE_DEVICES=$gpu env "$ckenv=$ck" $PY scripts/probes/temporal_mech.py \
    --models "$tag" --batch_size 16 --outdir "$O" 2>&1 | filt > "$O/$tag.log" 2>&1
  rc=${PIPESTATUS[0]}
  echo "[gpu$gpu] DONE  $tag exit=$rc" >&2
  [ "$rc" -eq 0 ] || echo "[gpu$gpu] FAILED $tag -- see $O/$tag.log" >&2; }
one subword_llama   CKPT_SUBWORD_LLAMA   "$LL" 0 &
one aunet_static    CKPT_AUNET_STATIC    "$AU" 1 &
one byte_greedyroot CKPT_BYTE_GREEDYROOT "$BP" 2 &
wait
echo "TEMPORAL_MECH DONE"
