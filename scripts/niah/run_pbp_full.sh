#!/usr/bin/env bash
# Full PBP-NIAH grid (as the original run_niah.sh grid), local 100M checkpoints.
# S-NIAH-1/2/3 over lengths x depths + MK-NIAH over K, canonical-vs-space cut per cell.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT"
export AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="$ROOT/lingua/.venv/bin/python"
OUT="$ROOT/reports/niah/pbp_full"; mkdir -p "$OUT"

LLAMA="$ROOT/runs/small/cmp_100M/llama_100M/checkpoints/0000002200/consolidated"
AUNET="$ROOT/runs/small/cmp_100M/aunet_orig_100M/checkpoints/0000001672/consolidated"
BPE="$ROOT/runs/small/cmp_100M/v4_root_greedy_ot/checkpoints/0000006688/consolidated"
LENGTHS="512 1024 2048 4096 6144"; DEPTHS="0.1 0.3 0.5 0.7 0.9"; NPC=4

filt() { grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org"; }

for T in 1 2 3; do
  echo "===== S-NIAH-$T PBP ====="
  $PY scripts/niah/niah_probe.py --family subword --ckpt "$LLAMA" --tag subword_llama   --task $T --pbp --lengths $LENGTHS --depths $DEPTHS --n_per_cell $NPC --batch_size 16 --out "$OUT/pbp_S$T.jsonl" 2>&1 | filt
  $PY scripts/niah/niah_probe.py --family aunet   --ckpt "$AUNET" --tag aunet_static    --task $T --pbp --lengths $LENGTHS --depths $DEPTHS --n_per_cell $NPC --batch_size 16 --out "$OUT/pbp_S$T.jsonl" 2>&1 | filt
  $PY scripts/niah/niah_probe.py --family aunet   --ckpt "$BPE"   --tag byte_greedyroot --task $T --pbp --lengths $LENGTHS --depths $DEPTHS --n_per_cell $NPC --batch_size 16 --out "$OUT/pbp_S$T.jsonl" 2>&1 | filt
done

echo "===== MK-NIAH PBP ====="
CKPT_SUBWORD_LLAMA="runs/small/cmp_100M/llama_100M/checkpoints/0000002200/consolidated" \
CKPT_AUNET_STATIC="runs/small/cmp_100M/aunet_orig_100M/checkpoints/0000001672/consolidated" \
CKPT_BYTE_GREEDYROOT="runs/small/cmp_100M/v4_root_greedy_ot/checkpoints/0000006688/consolidated" \
$PY scripts/probes/mk_niah.py --models subword_llama aunet_static byte_greedyroot \
  --n 40 --target_bytes 2048 --needle_counts 1 2 4 8 --pbp --out "$OUT/mkniah_pbp.jsonl" 2>&1 | filt

echo "===== DONE -> $OUT ====="
