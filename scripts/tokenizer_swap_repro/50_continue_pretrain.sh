#!/bin/bash
# Continue-pretrain (warm-start) the fixed 1.3B on a NEW language/domain — the follow-up the
# swap study points to: an eval-time swap can't re-target a checkpoint, but CONTINUE-PRETRAINING
# with the new boundaries can (trunk learns the new pooling distribution). Generalizes the two
# committed Chinese configs (bpebyte_1.3B_zh_continue_{llama3,qwen2}.yaml) to any lang×tokenizer.
#
# Recipe (from the zh pilot): init from 180k, 6000 steps, domain-adapt LR 1e-4 + 1k re-warmup,
# 60% target-lang / 40% DCLM English replay (anti-forgetting).
#
# Usage:  ./50_continue_pretrain.sh <lang> <arm>
#   lang : zh | te | code        (train corpus data/<lang>_wiki_train ; DCLM replay auto)
#   arm  : llama3 | qwen2 | bloom | starcoder2
#          the patch-boundary tokenizer the trunk is ADAPTED to. Natural pairing:
#            zh->qwen2   te->bloom   code->starcoder2   (llama3 = keep-original control)
#
# After training, re-run 10_bpb_swap.sh against the NEW checkpoint (edit CKPT= or use the printed
# dump path) to measure whether adaptation closed the swap gap — the open question this answers.
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"; activate_env
LANG_=${1:?lang: zh|te|code}; ARM=${2:?arm: llama3|qwen2|bloom|starcoder2}

case "$LANG_" in
  zh|te|code) TRAIN_SRC=${LANG_}_wiki_train ;;
  *) echo "bad lang $LANG_" >&2; exit 2 ;;
esac
# code corpus isn't "wiki" — allow an override name
[[ "$LANG_" == code ]] && TRAIN_SRC=${CODE_TRAIN_SRC:-code_train}

# boundary tokenizer for the run
if [[ "$ARM" == llama3 ]]; then
  TOK_PATH=$TOK_DIR/llama3/tokenizer.model; TOK_KIND=""
else
  TOK_PATH=$TOK_DIR/$ARM/tokenizer.json;    TOK_KIND="hf"
fi

GEN=$LINGUA/apps/aunet/configs/generated
mkdir -p "$GEN"
CFG=$GEN/bpebyte_1.3B_${LANG_}_continue_${ARM}.yaml
NAME=bpebyte_1.3B_${LANG_}_continue_${ARM}
DUMP=$BASE/runs/1.3B/$NAME

# Build the config by cloning the committed llama3 template and patching the lang/tokenizer keys.
python - "$LINGUA/apps/aunet/configs/bpebyte_1.3B_zh_continue_llama3.yaml" \
         "$CFG" "$NAME" "$DUMP" "$TRAIN_SRC" "$TOK_PATH" "$TOK_KIND" <<'PY'
import sys
from omegaconf import OmegaConf
tmpl, out, name, dump, src, tok, kind = sys.argv[1:8]
c = OmegaConf.load(tmpl)
c.name = name; c.dump_dir = dump
# swap the single training language source (keep the 0.6 target / 0.4 DCLM replay split)
old = [k for k in c.data.sources if k != "dclm_baseline_1.0_2shards_shuffled"][0]
w = c.data.sources.pop(old); c.data.sources[src] = w
c.data.regex.bpe_tokenizer_path = tok
if kind:
    c.data.regex.bpe_tokenizer_kind = kind
elif "bpe_tokenizer_kind" in c.data.regex:
    del c.data.regex.bpe_tokenizer_kind
OmegaConf.save(c, out)
print(f"wrote {out}\n  src={src} tok={tok} kind={kind or '(sp/tiktoken)'}")
PY

echo "== continue-pretrain lang=$LANG_ arm=$ARM (6000 steps, init@180k) -> $DUMP =="
TMPDIR=/var/tmp torchrun --nproc-per-node "$NPROC" --master-port "$MASTER_PORT" \
  -m apps.aunet.train config="$CFG"
echo "== done. new ckpts under $DUMP/checkpoints/ — eval with 10_bpb_swap (point CKPT there) =="
