#!/bin/bash
# Stage the BPB corpora as lingua *.val.jsonl (eval_on_val reads root_dir/<src>/*.val.jsonl,
# one JSON object per line with a "text" field). Best-effort re-staging from public HF sources
# at the sizes used this session (Chinese/Telugu wiki, multi-lang code). Adjust N_* for exact
# doc counts. Custom lm_eval TASK DEFS are separate — see the note at the bottom.
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"; activate_env

N_WIKI=${N_WIKI:-1000}     # wiki docs per language for BPB
N_CODE=${N_CODE:-250}      # code docs per language (§3 used 250 Python; 8 langs for multi)

stage_wiki() {  # <hf_config> <src_name> <root>
  python - "$1" "$2" "$3" "$N_WIKI" <<'PY'
import sys, os, json
from datasets import load_dataset
cfg, src, root, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out = os.path.join(root, src); os.makedirs(out, exist_ok=True)
ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
with open(os.path.join(out, "data.val.jsonl"), "w") as f:
    for i, r in enumerate(ds):
        if i >= n: break
        f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
print(f"  {src}: {min(i+1,n)} docs -> {out}")
PY
}

stage_code() {  # <lang> <root> <src_name>
  python - "$1" "$2" "$3" "$N_CODE" <<'PY'
import sys, os, json
from datasets import load_dataset
lang, root, src, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out = os.path.join(root, src); os.makedirs(out, exist_ok=True)
ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}", split="train")
with open(os.path.join(out, "data.val.jsonl"), "w") as f:
    for i, r in enumerate(ds):
        if i >= n: break
        f.write(json.dumps({"text": r["content"]}) + "\n")
print(f"  code/{lang}: {min(i+1,n)} docs -> {out}")
PY
}

echo "== wiki BPB corpora =="
stage_wiki 20231101.zh wiki_zh "$DATA_DIR/zh_bpb"
stage_wiki 20231101.te wiki_te "$DATA_DIR/te_bpb"

echo "== code BPB corpora =="
stage_code python "$DATA_DIR/code_bpb" python
for lang in python c cpp java javascript go rust php; do
  stage_code "$lang" "$DATA_DIR/code_multi" "$lang"
done

cat <<'NOTE'

== custom lm_eval task defs (NOT staged here) ==
The cloze/MC tasks live under $TASKS ($LINGUA/eval_tasks):
  zh_cloze/      cmmlu_cloze, ceval-cloze         (committed locally)
  mmlu_cloze/    57 subjects, {ctx}{answer-text}  (built on ece; port if missing)
  humaneval_mc/  correct-vs-distractor solution likelihood
  belebele_cloze/ reading-comprehension cloze (zh + tel)
Standard tasks (hellaswag, arc_easy/challenge, piqa, xcopa_zh, xstorycloze_zh/te,
belebele_tel) come from lm_eval's built-in registry — no local def needed.
cmmlu note: datasets>=4 dropped script loaders -> repack to per-subject parquet
(data/cmmlu_local/) and point the task's dataset_path at it.
NOTE
