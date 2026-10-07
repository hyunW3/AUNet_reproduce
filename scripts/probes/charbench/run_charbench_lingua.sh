#!/usr/bin/env bash
# CharBench + Strawberry (gen + cloze) on the matched 1.3B trio through lingua apps.{main,aunet}.eval.
# Items/YAMLs come from build_items.py (default reports/charbench/items). One arm per GPU.
#   bash scripts/probes/charbench/run_charbench_lingua.sh [out] ["llama:0 aunet:1 bpebyte:3"] [limit]
# Each arm writes <out>/<arm>/results.json (+ per-sample jsonl via log_samples).
set -u
L=/mnt/ssd2/hyun2/AUNet
ITEMS=${ITEMS:-$L/reports/charbench/items}
OUT=$(realpath -m "${1:-$L/reports/charbench/runs}")
PLAN=${2:-"llama:0 aunet:1 bpebyte:3"}
LIMIT=${3:-}
GROUPS_=${CB_GROUPS:-"charbench_gen charbench_cloze strawberry_gen strawberry_cloze"}   # task_lists.json keys
M=$L/main/main/1.3B
VENV=$L/lingua/.venv
RUNNER=$L/lingua/scripts/eval/robustness/run_robustness_local.sh
declare -A CK=([llama]=$M/llama_1.8B_paper/checkpoints/0000060000/consolidated
               [aunet]=$M/aunet2_1.3B/checkpoints/0000180000/consolidated
               [bpebyte]=$M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated)
declare -A STEP=([llama]=60000 [aunet]=180000 [bpebyte]=180000)
declare -A BASE=([llama]=apps/main/configs/eval_robustness_llama_paper4.yaml
                 [aunet]=apps/aunet/configs/eval_robustness_aunet2_paper4.yaml
                 [bpebyte]=apps/aunet/configs/eval_robustness_bpebyte_paper4.yaml)
mkdir -p "$OUT/cfg" "$OUT/logs"
say(){ echo "$(date '+%F %T') $*" | tee -a "$OUT/queue.log"; }

mkcfg(){  # arm -> $OUT/cfg/<arm>.yaml (name/generator from the arm's paper config)
  "$VENV/bin/python" - "$L/lingua/${BASE[$1]}" "$OUT/cfg/$1.yaml" "$ITEMS" "$LIMIT" "$GROUPS_" <<'PY'
import json, sys, yaml
c = yaml.safe_load(open(sys.argv[1])); items = sys.argv[3]
names = json.load(open(f"{items}/task_lists.json"))
c["name"] = c["name"].split("_robustness")[0] + "_charbench"
c["harness"] = {"include_path": f"{items}/tasks", "log_samples": True,
                "tasks": [t for k in sys.argv[5].split() for t in names[k]]}
if sys.argv[4]:
    c["harness"]["limit"] = int(sys.argv[4])
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
}

run_arm(){  # arm gpu
  local arm=$1 g=$2 o=$OUT/$1
  if [ -f "$o/results.json" ]; then say "skip $arm (results.json exists)"; return; fi
  mkdir -p "$o"; mkcfg "$arm"
  say "start $arm GPU$g limit=${LIMIT:-all}"
  CFG_OVERRIDE=$OUT/cfg/$arm.yaml CKPT_DIR=${CK[$arm]} OUT_DIR=$o MASTER_PORT=$((29710 + g)) \
    bash "$RUNNER" "$arm" "${STEP[$arm]}" "$g" > "$OUT/logs/$arm.log" 2>&1
  say "end   $arm exit=$? results=$([ -f $o/results.json ] && echo yes || echo NO)"
}

for p in $PLAN; do run_arm "${p%%:*}" "${p##*:}" & done
wait
say "ALL_DONE"
