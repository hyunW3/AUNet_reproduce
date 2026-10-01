#!/bin/bash
# Exact std_bench commands that reproduce published tables. Each recipe runs run_std_bench.sh with the
# original evaluation's checkpoint, tasks, limit and generator max_tokens (snu55 paths).
#
#   bash repro_recipes.sh r10lr                 # 100M r10lr table, rows llama/wsd15 + rg/wsd20
#   bash repro_recipes.sh scale 760M            # γ10 scale ladder at one scale (100M | 300M | 760M | 1.3B)
#   bash repro_recipes.sh scale all             # all four scales
#   GPU=3 bash repro_recipes.sh scale 100M      # pick the GPU (default 2)
#   DRY=1 bash repro_recipes.sh scale all       # print the commands only
#
# Results: reports/std_bench/<name>/ds0_seed1234[_limitN].{json,log}, *_ci.{json,md}.
#
# Checkpoints (consolidated dirs). Copies under runs/std_bench_ckpt/ were fetched for this; the origin is
# where the original lives if the copy is gone.
#   r10lr_llama_wsd15  runs/std_bench_ckpt/llama_wsd15/consolidated   step 4400
#                      origin info107:/home/ansible-test/aunet_r10lr/runs/r10lr/r10lr_llama_100M_wsd15/checkpoints/0000004400
#   r10lr_rg_wsd20     runs/std_bench_ckpt/rg_wsd20/consolidated      step 3344
#                      origin info103:/mnt/ssd/hyun/aunet_r10lr/runs/r10lr/r10lr_rg_100M_wsd20/checkpoints/0000003344
#   g10 100M llama     runs/std_bench_ckpt/g10_llama_100M/consolidated   step 4400
#                      origin ece-agpu11:/home/hwbae/AUNet/runs/cmp_g10/llama_100M/checkpoints/0000004400
#   g10 100M aunet     runs/std_bench_ckpt/g10_aunet_100M/consolidated   step 3344
#                      origin nas:/var/services/homes/hwbae0326/hyun/runs_backup/100M_adhoc/aunet_100M/checkpoints/0000003344
#   g10 100M rg        runs/main/100M_adhoc/rg_100M/checkpoints/0000003344/consolidated
#   g10 300M llama     runs/std_bench_ckpt/g10_llama_300M/consolidated   step 13000
#                      origin ece-agpu11:/home/hwbae/AUNet/runs/cmp_g10/llama_300M/checkpoints/0000013000
#   g10 300M aunet     runs/std_bench_ckpt/g10_aunet_300M/consolidated   step 9900
#                      origin nas:/var/services/homes/hwbae0326/hyun/runs_backup/300M/aunet_300M_adhoc/checkpoints/0000009900
#   g10 300M rg        runs/main/300M/rg_300M_adhoc/checkpoints/0000009900/consolidated
#   g10 760M           main/main/760M/{llama_760M/checkpoints/0000026600, aunet_760M/checkpoints/0000060600,
#                                      rg_760M/checkpoints/0000060600}/consolidated
#   g10 1.3B (paper)   main/main/1.3B/{llama_1.8B_paper/checkpoints/0000060000, aunet2_1.3B/checkpoints/0000180000,
#                                      bpebyte_br_greedy_root_1.3B/checkpoints/0000180000}/consolidated
# Copy back a missing one with e.g.
#   rsync -a ece-agpu11:<origin>/consolidated runs/std_bench_ckpt/g10_llama_100M/
#   ssh nas "cd <origin> && tar cf - consolidated" | tar xf - -C runs/std_bench_ckpt/g10_aunet_100M   # NAS: no rsync
#
# Original numbers: r10lr -> reports_afterAAAIsub/r10lr_ledger.json (llama/wsd15, rg/wsd20);
# g10 -> <run>/eval_scaling + evals_fill (100M/300M), evals_g10 (760M), evals_5bench + evals_mmlu_arcc (1.3B).
# Every original eval used generator max_tokens 16384 and one lm-eval call per task group as below.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
A=${AUNET_ROOT:-/mnt/ssd2/hyun2/AUNet}
C=$A/runs/std_bench_ckpt
export MAX_TOKENS=16384 TOKENIZER_PATH=${TOKENIZER_PATH:-$A/tokenizer/llama3/tokenizer.model}
T4="hellaswag arc_easy arc_challenge piqa"; T2="boolq winogrande"; T6="$T4 $T2"

run() {  # name ckpt family tasks [limit]
  local cmd="CKPT=$2 FAMILY=$3 MAX_TOKENS=$MAX_TOKENS TASKS=\"$4\" ${5:+LIMIT=$5 }GPU=${GPU:-2} bash $D/run_std_bench.sh $1"
  echo "+ $cmd"
  [ -n "${DRY:-}" ] || CKPT=$2 FAMILY=$3 TASKS="$4" LIMIT=${5:-} bash "$D/run_std_bench.sh" "$1"
}

r10lr() {  # 100M LR-sweep table: HS / ARC-E / PIQA, acc_norm, limit 2000, 0-shot
  run r10lr_llama_wsd15 $C/llama_wsd15/consolidated lingua_main "hellaswag arc_easy piqa" 2000
  run r10lr_rg_wsd20    $C/rg_wsd20/consolidated    lingua_aunet "hellaswag arc_easy piqa" 2000
}

scale() {  # γ10 ladder: 4 tasks full; BoolQ/WinoGrande limit 1000 at 100M/300M, full at 760M/1.3B
  case $1 in
    100M)
      for m in "llama $C/g10_llama_100M/consolidated lingua_main" \
               "aunet $C/g10_aunet_100M/consolidated lingua_aunet" \
               "rg $A/runs/main/100M_adhoc/rg_100M/checkpoints/0000003344/consolidated lingua_aunet"; do
        set -- $m; run s100M_$1 $2 $3 "$T4"; run s100M_$1 $2 $3 "$T2" 1000; done ;;
    300M)
      for m in "llama $C/g10_llama_300M/consolidated lingua_main" \
               "aunet $C/g10_aunet_300M/consolidated lingua_aunet" \
               "rg $A/runs/main/300M/rg_300M_adhoc/checkpoints/0000009900/consolidated lingua_aunet"; do
        set -- $m; run s300M_$1 $2 $3 "$T4"; run s300M_$1 $2 $3 "$T2" 1000; done ;;
    760M)
      run s760M_llama $A/main/main/760M/llama_760M/checkpoints/0000026600/consolidated lingua_main "$T6"
      run s760M_aunet $A/main/main/760M/aunet_760M/checkpoints/0000060600/consolidated lingua_aunet "$T6"
      run s760M_rg    $A/main/main/760M/rg_760M/checkpoints/0000060600/consolidated lingua_aunet "$T6" ;;
    1.3B)
      run s1.3B_llama $A/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated lingua_main "$T6"
      run s1.3B_aunet $A/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated lingua_aunet "$T6"
      run s1.3B_rg    $A/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated lingua_aunet "$T6" ;;
    all) for s in 1.3B 760M 300M 100M; do scale $s; done ;;
    *) echo "scale: 100M | 300M | 760M | 1.3B | all"; exit 1 ;;
  esac
}

case ${1:-} in
  r10lr) r10lr ;;
  scale) scale "${2:?scale: 100M | 300M | 760M | 1.3B | all}" ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
