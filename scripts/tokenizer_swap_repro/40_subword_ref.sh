#!/bin/bash
# Subword reference arm (llama_1.8B_paper). It is a FLAT subword LMTransformer, so:
#   - it uses apps.main.eval, NOT apps.aunet.eval (hierarchical eval crashes on flat models);
#   - its tokenizer CANNOT be swapped (embeddings are bound to the vocab) — one arm only.
# Provides the iso-byte (NOT iso-param: 1.3B vs 1.8B) baseline: subword compresses BPB best
# (code_multi 0.935 vs byte-l3 0.994) but ties byte on task accuracy.
#
# Usage:  ./40_subword_ref.sh bpb  <domain>          (BPB on staged corpus)
#         ./40_subword_ref.sh down <tasks-csv> [lim] (downstream)
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"; activate_env
MODE=${1:?bpb|down}

if [[ "$MODE" == bpb ]]; then
  DOMAIN=${2:?domain}
  case "$DOMAIN" in
    zh) ROOT=$DATA_DIR/zh_bpb; SRC=wiki_zh ;;
    te) ROOT=$DATA_DIR/te_bpb; SRC=wiki_te ;;
    code_py) ROOT=$DATA_DIR/code_bpb; SRC=python ;;
    code_multi) ROOT=$DATA_DIR/code_multi; SRC='[python,c,cpp,java,javascript,go,rust,php]' ;;
    *) echo bad domain >&2; exit 2 ;;
  esac
  torchrun --nproc-per-node "$NPROC" --master-port "$MASTER_PORT" -m apps.main.eval \
    ckpt_dir="$LLAMA_CKPT" dump_dir="$LLAMA_RUN/repro_bpb_${DOMAIN}" \
    "harness.tasks=[]" validation.root_dir="$ROOT" "validation.sources=[$SRC]"
else
  TASKS_CSV=${2:?tasks csv}; LIM=${3:-$LIMIT}
  # apps.main.eval was patched this session to skip simple_evaluate on empty tasks;
  # here tasks is non-empty so the harness runs normally.
  torchrun --nproc-per-node "$NPROC" --master-port "$MASTER_PORT" -m apps.main.eval \
    ckpt_dir="$LLAMA_CKPT" dump_dir="$LLAMA_RUN/repro_down" \
    harness.include_path="$TASKS" "harness.tasks=[$TASKS_CSV]" harness.limit="$LIM"
fi
