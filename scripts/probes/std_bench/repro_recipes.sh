#!/bin/bash
# Reproduce one checkpoint's published numbers with std_bench. One block per checkpoint: the original
# scores it should reproduce, the checkpoint (local path + origin), and the exact command.
#
#   bash repro_recipes.sh list                 # checkpoint names
#   bash repro_recipes.sh s760M_rg             # run one checkpoint (GPU 2)
#   GPU=3 bash repro_recipes.sh r10lr_rg_wsd20 # pick the GPU
#   DRY=1 bash repro_recipes.sh s100M_llama    # print the command only
#
# Every command is also runnable by itself (copy it, with AUNET_ROOT=/mnt/ssd2/hyun2/AUNet and
# D=scripts/probes/std_bench). Results: reports/std_bench/<name>/ds0_seed1234[_limitN].{json,log}, *_ci.md.
# Metrics: HS/ARC-E/ARC-C/PIQA acc_norm, BoolQ/WinoGrande acc, 0-shot. max_tokens 16384 = the original evals.
# Copy back a missing checkpoint from its origin:
#   rsync -a <host>:<origin>/consolidated <local dir>/                        # ece, info10x
#   ssh nas "cd <origin> && tar cf - consolidated" | tar xf - -C <local dir>  # NAS has no rsync
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
A=${AUNET_ROOT:-/mnt/ssd2/hyun2/AUNet}
export TOKENIZER_PATH=${TOKENIZER_PATH:-$A/tokenizer/llama3/tokenizer.model}

x() {  # x VAR=value ... <name>: print, then run run_std_bench.sh with those variables
  local name=${!#}; printf '+'; printf ' %q' "${@:1:$#-1}" bash "$D/run_std_bench.sh" "$name"; echo
  [ -n "${DRY:-}" ] || env "${@:1:$#-1}" bash "$D/run_std_bench.sh" "$name"
}

case ${1:-} in

# ===================================================== 100M r10lr LR sweep (reports_afterAAAIsub/r10lr_ledger.json)
# Llama · Transformer, WSD 1.5e-3 (ledger llama/wsd15). Original: HS 38.05  ARC-E 42.55  PIQA 63.71  Avg3 48.10
# ckpt  runs/std_bench_ckpt/llama_wsd15/consolidated  (step 4400)
# origin info107:/home/ansible-test/aunet_r10lr/runs/r10lr/r10lr_llama_100M_wsd15/checkpoints/0000004400
r10lr_llama_wsd15)
  x CKPT=$A/runs/std_bench_ckpt/llama_wsd15/consolidated FAMILY=lingua_main MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy piqa" LIMIT=2000 r10lr_llama_wsd15 ;;

# BPEByte root_greedy, WSD 2.0e-3 (ledger rg/wsd20). Original: HS 34.85  ARC-E 34.45  PIQA 58.38  Avg3 42.56
# ckpt  runs/std_bench_ckpt/rg_wsd20/consolidated  (step 3344)
# origin info103:/mnt/ssd/hyun/aunet_r10lr/runs/r10lr/r10lr_rg_100M_wsd20/checkpoints/0000003344
r10lr_rg_wsd20)
  x CKPT=$A/runs/std_bench_ckpt/rg_wsd20/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy piqa" LIMIT=2000 r10lr_rg_wsd20 ;;

# ===================================================== γ10 scale ladder, 100M (originals: <run>/eval_scaling, evals_fill)
# HS/ARC-E/ARC-C/PIQA on full test sets, BoolQ/WinoGrande on the first 1000 items -> two commands.
# Llama 100M. Original: HS 31.31  ARC-E 42.09  ARC-C 23.72  PIQA 64.25 | BoolQ 61.70  WG 50.60 (L1000)
# ckpt  runs/std_bench_ckpt/g10_llama_100M/consolidated  (step 4400)
# origin ece-agpu11:/home/hwbae/AUNet/runs/cmp_g10/llama_100M/checkpoints/0000004400
s100M_llama)
  x CKPT=$A/runs/std_bench_ckpt/g10_llama_100M/consolidated FAMILY=lingua_main MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa" s100M_llama
  x CKPT=$A/runs/std_bench_ckpt/g10_llama_100M/consolidated FAMILY=lingua_main MAX_TOKENS=16384 \
    TASKS="boolq winogrande" LIMIT=1000 s100M_llama ;;

# AU-Net 100M. Original: HS 28.46  ARC-E 33.67  ARC-C 23.29  PIQA 58.32 | BoolQ 56.70  WG 49.70 (L1000)
# ckpt  runs/std_bench_ckpt/g10_aunet_100M/consolidated  (step 3344)
# origin nas:/var/services/homes/hwbae0326/hyun/runs_backup/100M_adhoc/aunet_100M/checkpoints/0000003344
s100M_aunet)
  x CKPT=$A/runs/std_bench_ckpt/g10_aunet_100M/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa" s100M_aunet
  x CKPT=$A/runs/std_bench_ckpt/g10_aunet_100M/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="boolq winogrande" LIMIT=1000 s100M_aunet ;;

# BPEByte-rg 100M. Original: HS 28.58  ARC-E 32.58  ARC-C 22.61  PIQA 58.11 | BoolQ 55.70  WG 50.00 (L1000)
# ckpt  runs/main/100M_adhoc/rg_100M/checkpoints/0000003344/consolidated
# copy  ece-agpu18:/home/hwbae/AUNet/runs/cmp_g10/rg_100M/checkpoints/0000003344  (copied there 2026-10-01)
s100M_rg)
  x CKPT=$A/runs/main/100M_adhoc/rg_100M/checkpoints/0000003344/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa" s100M_rg
  x CKPT=$A/runs/main/100M_adhoc/rg_100M/checkpoints/0000003344/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="boolq winogrande" LIMIT=1000 s100M_rg ;;

# ===================================================== γ10 scale ladder, 300M (originals: <run>/eval_scaling, evals_fill)
# Llama 300M. Original: HS 42.37  ARC-E 48.82  ARC-C 26.02  PIQA 68.28 | BoolQ 55.80  WG 50.20 (L1000)
# ckpt  runs/std_bench_ckpt/g10_llama_300M/consolidated  (step 13000)
# origin ece-agpu11:/home/hwbae/AUNet/runs/cmp_g10/llama_300M/checkpoints/0000013000
s300M_llama)
  x CKPT=$A/runs/std_bench_ckpt/g10_llama_300M/consolidated FAMILY=lingua_main MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa" s300M_llama
  x CKPT=$A/runs/std_bench_ckpt/g10_llama_300M/consolidated FAMILY=lingua_main MAX_TOKENS=16384 \
    TASKS="boolq winogrande" LIMIT=1000 s300M_llama ;;

# AU-Net 300M. Original: HS 37.16  ARC-E 43.86  ARC-C 25.60  PIQA 64.91 | BoolQ 50.70  WG 50.30 (L1000)
# ckpt  runs/std_bench_ckpt/g10_aunet_300M/consolidated  (step 9900)
# origin nas:/var/services/homes/hwbae0326/hyun/runs_backup/300M/aunet_300M_adhoc/checkpoints/0000009900
s300M_aunet)
  x CKPT=$A/runs/std_bench_ckpt/g10_aunet_300M/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa" s300M_aunet
  x CKPT=$A/runs/std_bench_ckpt/g10_aunet_300M/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="boolq winogrande" LIMIT=1000 s300M_aunet ;;

# BPEByte-rg 300M. Original: HS 37.03  ARC-E 41.75  ARC-C 24.83  PIQA 64.31 | BoolQ 47.30  WG 54.80 (L1000)
# ckpt  runs/main/300M/rg_300M_adhoc/checkpoints/0000009900/consolidated
# origin ece-agpu11:/home/hwbae/AUNet/runs/cmp_g10/rg_300M/checkpoints/0000009900
s300M_rg)
  x CKPT=$A/runs/main/300M/rg_300M_adhoc/checkpoints/0000009900/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa" s300M_rg
  x CKPT=$A/runs/main/300M/rg_300M_adhoc/checkpoints/0000009900/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="boolq winogrande" LIMIT=1000 s300M_rg ;;

# ===================================================== γ10 scale ladder, 760M (originals: <run>/evals_g10, all full test sets)
# Llama 760M. Original: HS 55.73  ARC-E 62.12  ARC-C 31.23  PIQA 73.61  BoolQ 59.82  WG 58.48
# ckpt  main/main/760M/llama_760M/checkpoints/0000026600/consolidated
s760M_llama)
  x CKPT=$A/main/main/760M/llama_760M/checkpoints/0000026600/consolidated FAMILY=lingua_main MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa boolq winogrande" s760M_llama ;;

# AU-Net 760M. Original: HS 52.42  ARC-E 55.68  ARC-C 31.66  PIQA 71.16  BoolQ 58.29  WG 55.33
# ckpt  main/main/760M/aunet_760M/checkpoints/0000060600/consolidated
s760M_aunet)
  x CKPT=$A/main/main/760M/aunet_760M/checkpoints/0000060600/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa boolq winogrande" s760M_aunet ;;

# BPEByte-rg 760M. Original: HS 52.19  ARC-E 55.39  ARC-C 31.48  PIQA 71.44  BoolQ 53.15  WG 54.85
# ckpt  main/main/760M/rg_760M/checkpoints/0000060600/consolidated
s760M_rg)
  x CKPT=$A/main/main/760M/rg_760M/checkpoints/0000060600/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa boolq winogrande" s760M_rg ;;

# ===================================================== 1.3B paper models (originals: <run>/evals_5bench + evals_mmlu_arcc, full)
# Llama (Transformer) 1.3B. Original: HS 62.24  ARC-E 65.45  ARC-C 35.32  PIQA 75.30  BoolQ 63.46  WG 61.56
# ckpt  main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated
s1.3B_llama)
  x CKPT=$A/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated FAMILY=lingua_main MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa boolq winogrande" s1.3B_llama ;;

# AU-Net 1.3B. Original: HS 62.65  ARC-E 65.66  ARC-C 36.52  PIQA 74.21  BoolQ 61.13  WG 61.48
# ckpt  main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated
s1.3B_aunet)
  x CKPT=$A/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa boolq winogrande" s1.3B_aunet ;;

# BPEByte-rg 1.3B. Original: HS 62.47  ARC-E 66.84  ARC-C 37.54  PIQA 74.32  BoolQ 62.05  WG 61.09
# ckpt  main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated
s1.3B_rg)
  x CKPT=$A/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated FAMILY=lingua_aunet MAX_TOKENS=16384 \
    TASKS="hellaswag arc_easy arc_challenge piqa boolq winogrande" s1.3B_rg ;;

list) grep -oE '^[A-Za-z0-9_.]+\)' "$0" | tr -d ')' | grep -v '^list$' ;;
*) sed -n 2,9p "$0"; exit 1 ;;
esac
