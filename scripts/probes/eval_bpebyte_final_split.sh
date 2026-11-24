#!/bin/bash
# BPEByte 1.3B 최종(180,000) 마일스톤 평가 — ece-agpu11 A100.
#
# 평가를 둘로 쪼개 GPU 1장씩 동시에 돌린다. 이 평가는 GPU 가 아니라 CPU 바운드라
# (online greedy BPE 패칭) 카드를 더 준다고 빨라지지 않는다. 90%(162,000) 평가는
# 2 GPU 로 20시간 걸렸다. 태스크를 반으로 나눠 병렬로 돌리면 벽시계가 절반이 된다.
#
# 분할 기준은 continuation 수(= 실제 일량):
#   A  hellaswag 40k + arc_easy 9.5k + arc_c 4.7k + piqa 3.7k + winogrande 2.5k + boolq 6.5k ~= 67k
#   B  mmlu_text 56k + hellaswag_gen + arc_easy_gen                                        ~= 56k + 생성
#
# run_milestone_eval.sh 는 master-port 가 29540 으로 고정이라 두 건을 겹칠 수 없다.
# 그래서 여기서 torchrun 을 직접 부른다.
set -uo pipefail
R=/home/hwbae/AUNet
RUN=bpebyte_br_greedy_root_1.3B_a100x4
PAD=0000180000
CKPT=$R/runs/$RUN/checkpoints/$PAD
LOG=$R/runs/$RUN/final_eval.log
VENV=$R/lingua/.venv
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

cd $R/lingua
export TMPDIR=/tmp
export TRITON_CACHE_DIR=/tmp/triton_${USER}_$(hostname -s)
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_$(hostname -s)
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

if [ ! -d "$CKPT/consolidated" ]; then
  say "consolidate 시작 $PAD"
  $VENV/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$CKPT'))" >> "$LOG" 2>&1
  [ -d "$CKPT/consolidated" ] || { say "consolidate 실패"; exit 1; }
  say "consolidate 완료 ($(du -sh $CKPT/consolidated|cut -f1))"
else
  say "consolidate 생략 (이미 있음)"
fi

launch(){  # half gpu port
  local half=$1 gpu=$2 port=$3
  local out=$R/runs/$RUN/stage_${PAD}_$half
  if [ -f "$out/results.json" ]; then say "생략 $half (결과 있음)"; return 0; fi
  mkdir -p "$out"
  say "평가 시작 $half (GPU$gpu, port $port)"
  ( CUDA_VISIBLE_DEVICES=$gpu $VENV/bin/python -m torch.distributed.run \
        --nproc-per-node 1 --master-port $port -m apps.aunet.eval \
        config=apps/aunet/configs/eval_final_bpebyte_$half.yaml \
        ckpt_dir="$CKPT" dump_dir="$out" > "$R/runs/$RUN/stage_${PAD}_$half.log" 2>&1
    say "평가 종료 $half exit=$? results=$([ -f $out/results.json ] && echo yes || echo NO)" ) &
}

launch a 3 29561
sleep 60                     # 기동이 겹치지 않도록
launch b 4 29562
wait
say "FINAL_EVAL_DONE"
