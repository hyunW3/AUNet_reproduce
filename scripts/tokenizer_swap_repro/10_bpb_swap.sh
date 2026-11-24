#!/bin/bash
# BPB (bits/byte = |nll|/bytes/ln2) under a tokenizer swap — the sensitive metric.
# Skips the lm_eval harness (harness.tasks=[]); only eval_on_val scores per-byte NLL
# on the staged corpus, ALWAYS also folding in DCLM *.val.jsonl -> free English ref.
#
# Usage:   ./10_bpb_swap.sh <domain> <arm>
#   domain : zh | te | code_py | code_multi   (staged under $DATA_DIR, see 02_stage_data.sh)
#   arm    : llama3 | qwen2 | bloom | starcoder2
#            llama3 = the checkpoint's OWN scheme (no override, baseline).
#            others = regex_bpe_tokenizer_path=<tok> force_bpe_online_mode=greedy
#                     (force_* REQUIRED — a bare override runs offline/leaky, tanks BPB).
#
# Reproduces §1/§2/§3 of PoC/tokenizer_swap_all_results.md:
#   zh  llama3 2.679 -> qwen2 2.891 (+0.212) ; te llama3 1.235 -> bloom 1.709 (+0.474)
#   code_multi byte-l3 0.994 -> sc2 1.331 (+0.337) ; subword ref 0.935 (see 40_subword_ref.sh)
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"; activate_env
DOMAIN=${1:?domain: zh|te|code_py|code_multi}
ARM=${2:?arm: llama3|qwen2|bloom|starcoder2}

case "$DOMAIN" in
  zh)         ROOT=$DATA_DIR/zh_bpb;   SRC=wiki_zh ;;
  te)         ROOT=$DATA_DIR/te_bpb;   SRC=wiki_te ;;
  code_py)    ROOT=$DATA_DIR/code_bpb; SRC=python ;;
  code_multi) ROOT=$DATA_DIR/code_multi; SRC='[python,c,cpp,java,javascript,go,rust,php]' ;;
  *) echo "bad domain $DOMAIN" >&2; exit 2 ;;
esac

OVERRIDE=()
case "$ARM" in
  llama3) : ;;                          # baseline: checkpoint's own greedy+root+llama3
  qwen2|bloom|starcoder2)
    OVERRIDE=( regex_bpe_tokenizer_path="$TOK_DIR/$ARM/tokenizer.json"
               force_bpe_online_mode=greedy ) ;;
  *) echo "bad arm $ARM" >&2; exit 2 ;;
esac

DUMP=$RUN/repro_bpb_${DOMAIN}_${ARM}
echo "== BPB domain=$DOMAIN arm=$ARM -> $DUMP =="
torchrun --nproc-per-node "$NPROC" --master-port "$MASTER_PORT" -m apps.aunet.eval \
  config=$LINGUA/apps/aunet/configs/eval_bpb_zh.yaml \
  ckpt_dir="$CKPT" dump_dir="$DUMP" \
  "harness.tasks=[]" \
  validation.root_dir="$ROOT" "validation.sources=[$SRC]" \
  "${OVERRIDE[@]}"
echo "== done; BPB in $DUMP/metrics*.json (val/<src>/bpb) =="
