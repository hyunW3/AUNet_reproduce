#!/usr/bin/env bash
# Extended retrieval tasks at 1.3B (iso-byte, teacher-forced exact-match, no-colon):
# RULER multi-value / multi-query, Sequential-NIAH, NoLiMa-lite (literal vs paraphrase).
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; OUT="$ROOT/reports/niah/ext_1p3b"; mkdir -p "$OUT"
LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AUNET="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BPE="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
LEN="1024 2048 4096"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }
et(){ local fam=$1 ck=$2 tag=$3 gpu=$4;
  for T in mv mq seq; do
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/niah/niah_ext_probe.py --family "$fam" --ckpt "$ROOT/$ck" \
      --tag "$tag" --task $T --K 4 --lengths $LEN --n 50 --batch_size 16 \
      --out "$OUT/ext_${tag}.jsonl" 2>&1 | filt; done
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/niah/niah_ext_probe.py --family "$fam" --ckpt "$ROOT/$ck" \
    --tag "$tag" --task nolima --n_distract 3 --lengths $LEN --n 50 --batch_size 16 \
    --out "$OUT/ext_${tag}.jsonl" 2>&1 | filt; }
et subword "$LLAMA" subword_llama   0 > "$OUT/llama.log"   2>&1 &
et aunet   "$AUNET" aunet_static    1 > "$OUT/aunet.log"   2>&1 &
et aunet   "$BPE"   byte_greedyroot 2 > "$OUT/bpebyte.log" 2>&1 &
wait
echo "EXT DONE"
