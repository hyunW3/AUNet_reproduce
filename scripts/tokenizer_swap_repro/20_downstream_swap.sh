#!/bin/bash
# Downstream / cloze eval under a tokenizer swap (byte model, apps.aunet.eval).
# Same swap mechanism as BPB, but now runs the lm_eval harness on TASKS.
#
# Usage:  ./20_downstream_swap.sh <arm> <tasks-csv> [limit]
#   arm       : llama3 | qwen2 | bloom | starcoder2 | gemma
#   tasks-csv : e.g. hellaswag,arc_easy,arc_challenge,piqa       (§4 English, gemma swap)
#               cmmlu_cloze,ceval-cloze,belebele_cloze           (§1/§5 Chinese, qwen2)
#               xstorycloze_te,belebele_tel                      (§2 Telugu, bloom)
#               mmlu_cloze,humaneval_mc                          (§3 code MC, starcoder2)
#   limit     : per-task doc cap (default $LIMIT; 0 = full)
#
# Custom cloze/MC task defs live under $TASKS (zh_cloze, mmlu_cloze, humaneval_mc,
# belebele_cloze) — see 02_stage_data.sh. Reproduces §4/§5 + the MC rows of §3.
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"; activate_env
ARM=${1:?arm}; TASKS_CSV=${2:?tasks csv}; LIM=${3:-$LIMIT}

OVERRIDE=()
[[ "$ARM" != llama3 ]] && OVERRIDE=( regex_bpe_tokenizer_path="$TOK_DIR/$ARM/tokenizer.json"
                                     force_bpe_online_mode=greedy )

DUMP=$RUN/repro_down_${ARM}
echo "== downstream arm=$ARM tasks=$TASKS_CSV limit=$LIM =="
torchrun --nproc-per-node "$NPROC" --master-port "$MASTER_PORT" -m apps.aunet.eval \
  config=$LINGUA/apps/aunet/configs/eval_gen_bpebyte_chinese.yaml \
  ckpt_dir="$CKPT" dump_dir="$DUMP" \
  harness.include_path="$TASKS" \
  "harness.tasks=[$TASKS_CSV]" harness.limit="$LIM" \
  "${OVERRIDE[@]}"
echo "== done -> $DUMP/results*.json (acc / acc_norm) =="
