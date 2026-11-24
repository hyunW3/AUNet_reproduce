#!/bin/bash
# 179,300 체크포인트 평가 — 180,000 의 ARC-E/BoolQ 하락이 체크포인트 단위 흔들림인지
# 학습 후반의 체계적 하락인지 가르는 검사. 700 step 차이면 loss 는 사실상 같다.
set -u
R=/home/hwbae/AUNet; RUN=bpebyte_br_greedy_root_1.3B_a100x4; PAD=0000179300
C=$R/runs/$RUN/checkpoints/$PAD; LOG=$R/runs/$RUN/final_eval.log; VENV=$R/lingua/.venv
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
cd $R/lingua; export TMPDIR=/tmp TRITON_CACHE_DIR=/tmp/triton_${USER}_$(hostname -s) TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_$(hostname -s)
if [ ! -d "$C/consolidated" ]; then
  say "consolidate 시작 $PAD"
  $VENV/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$C'))" >> "$LOG" 2>&1
  [ -d "$C/consolidated" ] || { say "consolidate 실패 $PAD"; exit 1; }
  say "consolidate 완료 $PAD"
fi
for t in a mmlu; do
  out=$R/runs/$RUN/stage_${PAD}_$t; mkdir -p $out
  say "179300 평가 시작 $t (GPU0)"
  CUDA_VISIBLE_DEVICES=0 $VENV/bin/python -m torch.distributed.run --nproc-per-node 1 \
    --master-port 2958$([ $t = a ] && echo 1 || echo 2) -m apps.aunet.eval \
    config=apps/aunet/configs/eval_179300_$t.yaml ckpt_dir=$C dump_dir=$out \
    > $R/runs/$RUN/stage_${PAD}_$t.log 2>&1
  say "179300 평가 종료 $t exit=$? results=$([ -f $out/results.json ] && echo yes || echo NO)"
done
say "EVAL_179300_DONE"
