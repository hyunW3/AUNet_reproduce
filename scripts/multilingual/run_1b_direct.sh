#!/bin/bash
# Direct (zero-shot, no further training) multilingual eval of the English(DCLM)-trained 1B checkpoints —
# the non-Chinese counterpart of reports/zh_cloze_1B.md. Same harness/configs as runs/zh/1B_cloze,
# only the task list changes. Each suite includes an English anchor so transfer = lang / en.
#
# Usage:  scripts/multilingual/run_1b_direct.sh <gpu> <model> [<model> ...]
#   model : llama | rg | aunet_word | aunet_char      (aunet_char runs only the no-space-script subset)
#   ONLY="xcopa arc"  run just these suites (default: all)
#   LIMIT=N  per-task doc cap for every suite (smoke test); unset = full (hellaswag_* capped at 2000)
# Outputs: $OUT/<model>/<suite>/results.json   (skips suites whose results.json already exists)
set -uo pipefail

ROOT=/mnt/ssd2/hyun2/AUNet
LINGUA=$ROOT/lingua
OUT=${OUT:-$ROOT/runs/multilingual/1B_direct}
CK=$ROOT/main/main/1.3B
ZHCFG=$ROOT/runs/zh/1B_cloze            # configs of the Chinese 1B run, reused verbatim
# xcopa / xnli / paws-x copied from lm_eval 0.4.12 with dataset_path fixed to the namespaced HF ids
# (the stock yamls use bare 'xcopa' / 'xnli' / 'paws-x', which current huggingface_hub rejects)
TASKDIR=$(cd "$(dirname "$0")" && pwd)/tasks
GPU=${1:?gpu}; shift
PORT=$((29600 + GPU))

export CUDA_VISIBLE_DEVICES=$GPU
export LD_LIBRARY_PATH=$LINGUA/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HF_DATASETS_TRUST_REMOTE_CODE=1
export AUNET_TOK=$ROOT/tokenizer/llama3/tokenizer.model   # llama params.json points at the NHN tree
export TORCHINDUCTOR_CACHE_DIR=$ROOT/runs/.cache/torchinductor TRITON_CACHE_DIR=$ROOT/runs/.cache/triton

declare -A SUITES=(
  [xnli]="xnli"                                                       # 15 langs, chance 33.3
  [xcopa]="xcopa,copa"                                                # 11 langs + en copa, chance 50
  [xstorycloze]="xstorycloze"                                         # 11 langs (incl. en), chance 50
  [xwinograd]="xwinograd"                                             # 6 langs (incl. en), chance 50
  [pawsx]="pawsx"                                                     # 7 langs (incl. en), chance 50
  [lambada]="lambada_multilingual"                                    # en fr de it es  (Google-Translate, EleutherAI)
  # StableLM-2 re-translation (native-speaker-checked; the GT version was judged too noisy) — run via ONLY=lambada_sl
  [lambada_sl]="lambada_openai_mt_stablelm_en,lambada_openai_mt_stablelm_de,lambada_openai_mt_stablelm_es,lambada_openai_mt_stablelm_fr,lambada_openai_mt_stablelm_it,lambada_openai_mt_stablelm_nl,lambada_openai_mt_stablelm_pt"
  [arc]="arc_challenge,arc_ar,arc_de,arc_es,arc_fr,arc_hi,arc_id,arc_it,arc_ru,arc_vi,arc_zh"   # okapi, chance 25
  [hellaswag]="hellaswag,hellaswag_ar,hellaswag_de,hellaswag_es,hellaswag_fr,hellaswag_hi,hellaswag_id,hellaswag_it,hellaswag_ru,hellaswag_vi"
  # no-space scripts only: the AU-Net word-vs-char boundary swap matters here
  [nospace]="xnli_th,xnli_zh,xcopa_th,xcopa_zh,xstorycloze_zh,xstorycloze_my,xwinograd_jp,xwinograd_zh,paws_ja,paws_ko,paws_zh,arc_zh"
)
ORDER=(xstorycloze xcopa xwinograd pawsx lambada arc xnli hellaswag)

cd "$LINGUA" || exit 1
for model in "$@"; do
  case $model in
    llama)      mod=apps.main.eval;  cfg=$ZHCFG/llama/config.yaml;      ck=$CK/llama_1.8B_paper/checkpoints/0000060000; extra=() ;;
    rg)         mod=apps.aunet.eval; cfg=$ZHCFG/rg/config.yaml;         ck=$CK/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000; extra=() ;;
    aunet_word) mod=apps.aunet.eval; cfg=$ZHCFG/aunet_word/config.yaml; ck=$CK/aunet2_1.3B/checkpoints/0000180000; extra=() ;;
    aunet_char) mod=apps.aunet.eval; cfg=$ZHCFG/aunet_char/config.yaml; ck=$CK/aunet2_1.3B/checkpoints/0000180000; extra=(regex_strategy_override=char1) ;;
    *) echo "unknown model $model"; exit 1 ;;
  esac
  if [[ $model == aunet_char ]]; then suites=(nospace); else suites=(${ONLY:-${ORDER[@]}}); fi
  for s in "${suites[@]}"; do
    dump=$OUT/$model/$s
    [[ -f $dump/results.json ]] && { echo "[skip] $model/$s"; continue; }
    lim=${LIMIT:-}; [[ -z $lim && $s == hellaswag ]] && lim=2000
    mkdir -p "$dump"
    echo "[$(date '+%F %T')] start $model/$s gpu=$GPU limit=${lim:-full}"
    .venv/bin/python -m torch.distributed.run --nproc-per-node 1 --master-port $PORT -m $mod \
      config=$cfg ckpt_dir=$ck dump_dir=$dump \
      harness.include_path=$TASKDIR \
      "harness.tasks=[${SUITES[$s]}]" ${lim:+harness.limit=$lim} \
      "${extra[@]}" > "$dump/eval.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] end   $model/$s rc=$rc"
  done
done
