#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; OUT="$ROOT/reports/niah/highn_1p3b"; mkdir -p "$OUT"
LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AUNET="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BPE="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
LEN="512 1024 2048 4096 6144"; DEP="0.1 0.3 0.5 0.7 0.9"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }
rest(){ local fam=$1 ck=$2 tag=$3 ckenv=$4 gpu=$5
  echo "[$tag] === S-NIAH-4 ==="
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/niah/niah_probe.py --family "$fam" --ckpt "$ROOT/$ck" --tag "$tag" \
    --task 4 --lengths $LEN --depths $DEP --n_per_cell 10 --batch_size 8 \
    --out "$OUT/sn_${tag}.jsonl" 2>&1 | filt
  echo "[$tag] === MK-NIAH n=200 ==="
  CUDA_VISIBLE_DEVICES=$gpu env "$ckenv=$ROOT/$ck" $PY scripts/probes/mk_niah.py --models "$tag" \
    --needle_counts 1 2 4 8 --target_bytes 2048 --n 200 --batch_size 8 \
    --out "$OUT/mk_${tag}.jsonl" 2>&1 | filt; }
( rest subword "$LLAMA" subword_llama   CKPT_SUBWORD_LLAMA   0 ) > "$OUT/rest_llama.log"   2>&1 &
( rest aunet   "$AUNET" aunet_static     CKPT_AUNET_STATIC    1 ) > "$OUT/rest_aunet.log"   2>&1 &
( rest aunet   "$BPE"   byte_greedyroot  CKPT_BYTE_GREEDYROOT 2 ) > "$OUT/rest_bpe.log"     2>&1 &
wait; echo "NIAH_REST DONE"
