#!/bin/bash
# ADD-not-swap: keep llama3 boundaries AND graft BLOOM's Telugu tokens into the same
# byte-trie (filtered to the Telugu Unicode block U+0C00..U+0C7F). Tests whether adding
# coverage — rather than replacing the boundary distribution — helps.
#
# Result (§2 note): te-BPB 1.717 ≈ full BLOOM swap 1.709 >> llama3 1.235. Adding word-level
# Telugu patches does NOT help — the trunk can't use them; confirms it's boundary-DISTRIBUTION
# binding, not missing vocab. Uses the committed RegexArgs.bpe_augment_* wiring.
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"; activate_env

DUMP=$RUN/repro_bpb_te_augment_bloom
echo "== te-BPB llama3 + BLOOM-Telugu-augment (add, not swap) =="
torchrun --nproc-per-node "$NPROC" --master-port "$MASTER_PORT" -m apps.aunet.eval \
  config=$LINGUA/apps/aunet/configs/eval_bpb_zh.yaml \
  ckpt_dir="$CKPT" dump_dir="$DUMP" \
  "harness.tasks=[]" \
  validation.root_dir="$DATA_DIR/te_bpb" "validation.sources=[wiki_te]" \
  regex_bpe_augment_tokenizer_path="$TOK_DIR/bloom/tokenizer.json" \
  "regex_bpe_augment_unicode_range=[0x0C00,0x0C7F]"
echo "== done -> $DUMP (expect ~1.717, i.e. no gain over full swap) =="
