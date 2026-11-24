#!/bin/bash
# 최종 평가의 남은 세 태스크를 각각 별도 잡으로. mmlu_text 는 Avg7 에 필요하므로 최우선.
# hellaswag_gen(10,042문서)이 arc_easy_gen(2,376문서)보다 4배 크므로 GPU 를 2장 준다.
set -u
R=/home/hwbae/AUNet; RUN=bpebyte_br_greedy_root_1.3B_a100x4; PAD=0000180000
CKPT=$R/runs/$RUN/checkpoints/$PAD; LOG=$R/runs/$RUN/final_eval.log; VENV=$R/lingua/.venv
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
cd $R/lingua
export TMPDIR=/tmp TRITON_CACHE_DIR=/tmp/triton_${USER}_$(hostname -s) TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_$(hostname -s)
go(){ # tag gpus port
  local tag=$1 gpus=$2 port=$3 out=$R/runs/$RUN/stage_${PAD}_$1 ng
  ng=$(awk -F, '{print NF}' <<< "$gpus")
  [ -f "$out/results.json" ] && { say "생략 $tag"; return; }
  mkdir -p "$out"; say "평가 시작 $tag (GPU$gpus, $ng장)"
  ( CUDA_VISIBLE_DEVICES=$gpus $VENV/bin/python -m torch.distributed.run \
      --nproc-per-node $ng --master-port $port -m apps.aunet.eval \
      config=apps/aunet/configs/eval_final_bpebyte_$tag.yaml ckpt_dir="$CKPT" dump_dir="$out" \
      > "$R/runs/$RUN/stage_${PAD}_$tag.log" 2>&1
    say "평가 종료 $tag exit=$? results=$([ -f $out/results.json ] && echo yes || echo NO)" ) &
}
go mmlu  0   29571
sleep 45
go hsgen 1,2 29572
sleep 45
go aegen 3   29573
wait
say "SINGLES_DONE"
