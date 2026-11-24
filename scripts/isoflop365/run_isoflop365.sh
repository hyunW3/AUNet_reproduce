#!/usr/bin/env bash
# Iso-FLOP-365 comparison driver: train the byte arm and the FLOP-matched llama arm to the SAME
# total training FLOPs, then score BPB on the same held-out DCLM bytes. Both arms sit at ~365 M
# FLOP/byte; the only difference is byte-level vs subword representation (and llama's ~2.4x params).
#
# Usage:  bash scripts/isoflop365/run_isoflop365.sh [GPUS] [MASTER_PORT_BASE]
# Run from the repo root (parent of lingua/). Trains sequentially; flip the two train blocks to
# background + wait if you have two nodes. Re-run is safe: training resumes from the last ckpt.
set -euo pipefail
cd "$(dirname "$0")/../../lingua"           # -> lingua/
GPUS="${1:-4}"
PORT="${2:-29570}"
ROOT=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet
export TMPDIR="${TMPDIR:-/var/tmp}"
log(){ echo "[$(date '+%F %T')] $*"; }

# 0) sanity: confirm the two arms are FLOP-matched with the current GPUS
log "iso-FLOP-365 plan (GPUS=$GPUS):"
PYTHONPATH="$PWD" python "$ROOT/scripts/isoflop365/isoflop365_plan.py" --gpus "$GPUS" | sed 's/^/    /'

# 1) BYTE arm — the existing 365 M/byte config, redirected into the isoflop365 run dir
log "training BYTE arm (bpebyte_100M_v4_root_greedy) ..."
torchrun --nproc-per-node "$GPUS" --master-port "$PORT" -m apps.aunet.train \
    config=apps/aunet/configs/bpebyte_100M_v4_root_greedy.yaml \
    dump_dir="$ROOT/runs/isoflop365/bpebyte_v4" name=isoflop365_bpebyte_v4

# 2) LLAMA arm — grown to match FLOP/byte, matched total FLOPs (steps=2218)
log "training LLAMA arm (llama_iso365, dim1024/L18) ..."
torchrun --nproc-per-node "$GPUS" --master-port "$((PORT+1))" -m apps.main.train \
    config=apps/main/configs/llama_iso365.yaml

# 3) BPB eval on the same held-out bytes for both arms
byte_ckpt=$(ls -d "$ROOT"/runs/isoflop365/bpebyte_v4/checkpoints/* 2>/dev/null | sort | tail -1)
llama_ckpt=$(ls -d "$ROOT"/runs/isoflop365/llama_iso365/checkpoints/* 2>/dev/null | sort | tail -1)
log "BPB eval — byte ckpt: $byte_ckpt"
torchrun --nproc-per-node 1 --master-port "$((PORT+2))" -m apps.aunet.eval \
    config=apps/aunet/configs/eval_bpb_bpebyte_v4.yaml ckpt_dir="$byte_ckpt"
log "BPB eval — llama ckpt: $llama_ckpt"
torchrun --nproc-per-node 1 --master-port "$((PORT+3))" -m apps.main.eval \
    config=apps/main/configs/eval_bpb_llama_iso365.yaml ckpt_dir="$llama_ckpt"

# 4) report: BPB = |nll_per_byte| / ln2 from each arm's validation.json
log "iso-FLOP-365 BPB result:"
PYTHONPATH="$PWD" python - "$ROOT" <<'PY'
import json, math, sys, glob, os
root = sys.argv[1]
for arm, d in [("byte  (bpebyte v4)", "runs/isoflop365/bpebyte_v4/eval_bpb"),
               ("llama (dim1024/L18)", "runs/isoflop365/llama_iso365/eval_bpb")]:
    fs = glob.glob(os.path.join(root, d, "validation.json"))
    if not fs:
        print(f"  {arm}: (no validation.json yet)"); continue
    v = json.load(open(fs[0]))
    # average nll_per_byte across sources -> BPB
    npb = [m["nll_per_byte"] for m in v.values() if isinstance(m, dict) and "nll_per_byte" in m]
    bpb = (sum(abs(x) for x in npb)/len(npb))/math.log(2) if npb else float("nan")
    print(f"  {arm}: BPB = {bpb:.4f}  (sources: {list(v.keys())})")
print("  lower BPB wins at equal training FLOPs.")
PY
log "done."
