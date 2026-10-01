# std_bench — 표준 벤치마크 통합 측정

모든 모델을 **같은 lm-eval 호출**로 채점한다. 과제 YAML, seed, 과제별 지표(`run_ext.METRIC`)가 같고, 프로세스 1개·GPU 1장을 쓴다.

| 파일 | 역할 |
|---|---|
| `models.py` | 모델 등록부와 로더. family마다 harness가 다르다 |
| `std_bench.py` | 단일 seed 측정. 과제별 점수, 평균, 문항별 정답 여부, manifest를 JSON으로 저장한다 |
| `std_bench_bootstrap.py` | 문항 bootstrap 95% CI. 여러 모델을 한 번에 넣을 수 있고, `--ref`를 주면 대응 차이 CI와 p값을 낸다 |
| `run_std_bench.sh` | family에 맞는 인터프리터·환경을 골라 위 두 단계를 실행한다 (snu55 경로) |

## 등록 모델 (`python3 models.py list`)

| 이름 | family | harness | 설정 |
|---|---|---|---|
| `blt_1b` | blt | eval_suite `BLTHarness` | 공식 bytelatent, xformers, threshold 1.3354, **배치 1** |
| `hnet_1stage_XL`, `hnet_2stage_XL` | hnet | eval_suite `HNetHarness` | 배치 8 (`run_ext.py`와 동일) |
| `llama_1.3b` | lingua_main | `apps.main.eval.EvalHarnessLM` | 논문 Transformer, step 60000, max_tokens 4096 |
| `aunet_1.3b` | lingua_aunet | `apps.aunet.eval.EvalHarnessLM` | step 180000, max_tokens 16384 |
| `bpebyte_1.3b` | lingua_aunet | `apps.aunet.eval.EvalHarnessLM` | br_greedy_root, step 180000, max_tokens 16384 |

lingua 모델 설정은 논문 matched 평가(`runs/robustness_paper1p3b/<arm>/config.yaml`)와 같다. bf16, temperature 0이고 나머지 EvalArgs 옵션은 기본값이다.
등록되지 않은 lingua 체크포인트는 `CKPT=<consolidated dir> FAMILY=lingua_main|lingua_aunet`으로 돌린다. generator `max_tokens`는 기본이 4096(lingua_main) / 16384(lingua_aunet)이고, 원 평가와 다르면 `MAX_TOKENS=`로 맞춘다(packing이 바뀌어 문항 결과가 달라질 수 있다).
새 모델을 고정해 두려면 `models.py`의 `MODELS`에 한 줄 추가한다.

## 실행

```bash
D=scripts/probes/std_bench
bash $D/run_std_bench.sh aunet_1.3b                       # 0-shot, seed 1234, 전체 test set, GPU 2
GPU=3 bash $D/run_std_bench.sh blt_1b
SHOT=5 SEED=1234 bash $D/run_std_bench.sh llama_1.3b      # few-shot (seed가 예시를 고른다)
LIMIT=500 TASKS="hellaswag piqa" bash $D/run_std_bench.sh bpebyte_1.3b
CKPT=/path/to/consolidated FAMILY=lingua_aunet bash $D/run_std_bench.sh my_run
# 결과: reports/std_bench/<model>/ds<shot>_seed<seed>[_limit<N>].{json,log}, *_ci.{json,md}

# 여러 모델 비교 (같은 과제·limit·shot의 실행끼리, 문항 인덱스를 공유하는 대응 bootstrap)
python3 $D/std_bench_bootstrap.py reports/std_bench/*/ds0_seed1234.json \
  --ref reports/std_bench/llama_1.3b/ds0_seed1234.json --out reports/std_bench/cmp_ds0.json
```

기본 과제: HellaSwag, ARC-E, ARC-C, PIQA (acc_norm) · WinoGrande, BoolQ (acc) · MMLU-text (acc, 57과목을 문항 수 가중으로 합침).

## 재현성 (2026-10-01 확인)

- **seed:** 0-shot에서는 점수를 바꾸지 않는다. few-shot에서는 seed가 예시를 고른다(MMLU-text는 항상 dev 앞 n개).
- **같은 설정 반복:** 같은 모델·과제 목록·limit·GPU 1장이면 문항 단위로 똑같이 나온다(llama, H-Net에서 확인). bootstrap도 `--seed`가 같으면 결과가 같다.
- **BLT:** 요청을 하나씩 처리해서 과제 목록이 달라도 문항 결과가 같다. `reports/blt_batch_invariance`의 배치 1 결과와 문항 단위로 일치한다.
- **lingua 모델 주의:** harness가 여러 요청을 한 시퀀스로 packing하므로, 함께 묶이는 요청이 바뀌면 bf16 수치가 바뀌어 일부 문항이 뒤집힌다. 과제 목록, limit, GPU 수를 바꾸면 문항 단위 결과가 조금 달라질 수 있다. 논문 robustness 실행(3-GPU, noise/typo 과제와 함께 실행)과 비교하면 같은 2000문항 중 0.5–0.8%가 다르고 점수 차는 0.2pt 이하다(llama/AU-Net, HellaSwag·ARC-E).
- **manifest:** 결과 JSON에 코드 커밋과 dirty 여부(diff 해시), 패키지 버전, GPU, 가중치 SHA256, 명령줄이 남는다. lingua 작업 사본이 dirty이면 `repos.lingua.dirty=true`다.

## 재현 레시피 (`repro_recipes.sh`)

체크포인트 하나에 블록 하나씩 들어 있다. 블록마다 원래 점수, 체크포인트의 로컬 경로와 원본 위치(ece, NAS, info10x), 그 체크포인트만 돌리는 명령이 있다. 원 평가와 같은 과제, limit, `max_tokens 16384`를 쓴다.

```bash
bash scripts/probes/std_bench/repro_recipes.sh list               # 체크포인트 이름 14개
bash scripts/probes/std_bench/repro_recipes.sh s760M_rg           # 760M BPEByte-rg 하나만 (GPU 2)
GPU=3 bash scripts/probes/std_bench/repro_recipes.sh r10lr_rg_wsd20
DRY=1 bash scripts/probes/std_bench/repro_recipes.sh s100M_llama  # 명령만 출력
```

블록 안의 명령은 그대로 복사해서 따로 실행해도 된다. 예: 760M BPEByte-rg (원래 점수 HS 52.19, ARC-E 55.39, ARC-C 31.48, PIQA 71.44, BoolQ 53.15, WG 54.85)

```bash
CKPT=/mnt/ssd2/hyun2/AUNet/main/main/760M/rg_760M/checkpoints/0000060600/consolidated \
FAMILY=lingua_aunet MAX_TOKENS=16384 TASKS="hellaswag arc_easy arc_challenge piqa boolq winogrande" \
  bash scripts/probes/std_bench/run_std_bench.sh s760M_rg
```

- r10lr 두 행: 2026-10-01 재현에서 원본과 점수·문항이 모두 같았다(원 평가도 GPU 1장).
- γ10 스케일: 원 평가는 ece 클러스터 GPU 4장이라 문항 단위로는 조금 다를 수 있다. 점수가 오차 범위 안인지로 판단한다.
