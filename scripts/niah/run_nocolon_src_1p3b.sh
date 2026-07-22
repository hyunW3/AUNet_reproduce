#!/usr/bin/env bash
# Re-run S-NIAH-1..4 + MK-NIAH after the colon was removed from the NEEDLE at the
# source (niah_data.py / mk_niah.py: "...is {value}"), so needle and query "...is"
# are boundary-consistent (no prompt-boundary/partial-token confound). Default mode.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; OUT="$ROOT/reports/niah/nocolon_src_1p3b"; mkdir -p "$OUT"
LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AUNET="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BPE="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
LEN="512 1024 2048 4096 6144"; DEP="0.1 0.3 0.5 0.7 0.9"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }

sn(){ local fam=$1 ck=$2 tag=$3 gpu=$4; for T in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/niah/niah_probe.py --family "$fam" --ckpt "$ROOT/$ck" --tag "$tag" \
    --task $T --lengths $LEN --depths $DEP --n_per_cell 4 --batch_size 8 \
    --out "$OUT/sn_${tag}.jsonl" 2>&1 | filt; done; }

mk(){ local tag=$1 ckenv=$2 ck=$3 gpu=$4;
  CUDA_VISIBLE_DEVICES=$gpu env "$ckenv=$ck" $PY scripts/probes/mk_niah.py --models "$tag" \
    --needle_counts 1 2 4 8 --target_bytes 2048 --n 40 --batch_size 8 \
    --out "$OUT/mk_${tag}.jsonl" 2>&1 | filt; }

( sn subword "$LLAMA" subword_llama   0; mk subword_llama   CKPT_SUBWORD_LLAMA   "$LLAMA" 0 ) > "$OUT/llama.log"   2>&1 &
( sn aunet   "$AUNET" aunet_static     1; mk aunet_static    CKPT_AUNET_STATIC    "$AUNET" 1 ) > "$OUT/aunet.log"   2>&1 &
( sn aunet   "$BPE"   byte_greedyroot  2; mk byte_greedyroot CKPT_BYTE_GREEDYROOT "$BPE"   2 ) > "$OUT/bpebyte.log" 2>&1 &
wait
echo "NOCOLON_SRC DONE"
