#!/usr/bin/env bash
# Held-out BPB for the three matched 1.3B models across the compression-stability domains.
#
#   bash scripts/probes/run_bpb_domains.sh gate      # English only -- must reproduce tab:bpb_readouts
#   bash scripts/probes/run_bpb_domains.sh all 2     # every domain on GPU 2
#
# BPB is byte-normalised for every family (nll / (ln2 * utf8_bytes)), so subword and byte
# models are directly comparable on identical windows -- same protocol as the paper's
# held-out column. Domains other than English are OUT OF DISTRIBUTION for all three models
# (all English-trained), so cross-domain absolute values are not comparable to each other;
# what is comparable is the model ordering WITHIN a domain, and each model's own English->X
# degradation.
set -uo pipefail

ROOT=/mnt/ssd2/hyun2/AUNet
cd "$ROOT/lingua" || exit 1
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD"
# Checkpoints bake the B200 cluster's tokenizer path into params.json; both loaders honour
# these env overrides (apps/main/generate.py:425, scripts/fflm/fflm_probe.py:39).
export AUNET_ROOT="$ROOT"
export AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
PY=.venv/bin/python

MODE="${1:-gate}"
export CUDA_VISIBLE_DEVICES="${2:-2}"
OUT="$ROOT/reports/compression_stability/bpb_domains.jsonl"

# family:tag:ckpt
MODELS=(
  "subword:Llama:$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
  "aunet:AU-Net:$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
  "aunet:BPEByte:$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
)

declare -A WIN=(
  [en]="$ROOT/reports/bpb_windows.jsonl"
  [zh]="$ROOT/reports/bpb_windows_zh.jsonl"
  [te]="$ROOT/reports/windows/te.jsonl"
  [code]="$ROOT/reports/windows/code.jsonl"
  [despace_p050]="$ROOT/reports/windows/despace_p050.jsonl"
  [despace_p100]="$ROOT/reports/windows/despace_p100.jsonl"
  [random]="$ROOT/reports/windows/random.jsonl"
)

if [ "$MODE" = "gate" ]; then
  SEL=(en)
elif [ "$#" -gt 2 ]; then
  SEL=("${@:3}")                                # explicit domain list (for GPU sharding)
else
  SEL=(en zh te code despace_p050 despace_p100 random)
fi
DOMAINS=()
for s in "${SEL[@]}"; do DOMAINS+=("$s:${WIN[$s]}"); done

for m in "${MODELS[@]}"; do
  IFS=: read -r fam tag ckpt <<<"$m"
  for d in "${DOMAINS[@]}"; do
    IFS=: read -r dom win <<<"$d"
    if [ ! -f "$win" ]; then echo "!! missing windows: $win"; continue; fi
    echo "=== $tag / $dom"
    $PY "$ROOT/scripts/probes/bpb_windows_local.py" \
        --family "$fam" --ckpt "$ckpt" --tag "${tag}|${dom}" \
        --windows "$win" --batch_size 2 --out "$OUT" 2>&1 \
      | grep -v -i "futurewarning\|pynvml\|^  import"
  done
done
echo "done -> $OUT"
