#!/bin/bash
# BLT-1B robustness, 두 threshold arm.
#   A: 공개 config 값        theta = 1.335442066192627  (DCLM 4.071 B/patch)
#   B: bytes/patch 4.53 보정  theta = 1.46875            (DCLM 4.51-4.54 B/patch)
# batch_size 1 고정 — BLT 는 배치 구성에 정확도가 의존한다(모듈 docstring 참조).
#
#   bash run_blt_rob_arms.sh <A|B|sdpa> <gpu> [limit]
set -u
ARM=${1:?A|B|sdpa}
GPU=${2:?gpu index}
LIMIT=${3:-500}
B=/mnt/ssd2/hyun2/lzspacebyte/blt
OUT=/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/blt_rob
mkdir -p "$OUT"

# bytelatent 의 setup_torch_distributed 는 MASTER_PORT 를 SLURM_JOB_ID 로 시드한 난수로 정한다
# (distributed.py:258, 기본 job_id=-1). arm 을 동시에 띄우면 전부 같은 포트(28805)를 잡고
# 두 번째부터 EADDRINUSE 로 죽는다.
# SLURM_JOB_ID 를 주는 건 답이 아니다 — get_is_slurm_job() 이 켜지면서 SLURM_NTASKS 까지 요구한다.
# LOCAL_RANK 를 세팅해 torchrun 모드로 인식시키면(distributed.py:210) MASTER_PORT 를 그대로 쓴다.
case "$ARM" in
  A)    TH=1.335442066192627; IMPL=xformers; TAG=A_released; PORT=28811 ;;
  B)    TH=1.46875;           IMPL=xformers; TAG=B_bpp4.53;  PORT=28812 ;;
  # 진단용: 저장소의 run_blt_eval.py / run_blt_robustness.py 가 실제로 쓰는 (틀린) 경로.
  # 512바이트 위에서 조용히 다른 패칭을 하므로 그 차이의 크기를 재기 위한 arm.
  sdpa) TH=1.335442066192627; IMPL=sdpa;     TAG=diag_sdpa;  PORT=28813 ;;
  *) echo "unknown arm: $ARM"; exit 1 ;;
esac

echo "arm=$TAG theta=$TH attn=$IMPL gpu=$GPU limit=$LIMIT"
cd "$B"
BLT_SUPPRESS_ATTN_ERROR=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
  .venv/bin/python /mnt/ssd2/hyun2/AUNet/scripts/probes/blt_robustness_native.py \
    --threshold "$TH" --attn_impl "$IMPL" --batch_size 1 --limit "$LIMIT" \
    --out "$OUT/${TAG}_n${LIMIT}.json" > "$OUT/${TAG}_n${LIMIT}.log" 2>&1
echo "exit=$? -> $OUT/${TAG}_n${LIMIT}.json"
