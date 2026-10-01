# BLT-1B 실행 가이드 (평가 · patch 측정)

외부 참조 모델 BLT(`facebook/blt-1b`)를 논문 표에 넣을 때 쓰는 **유일한 올바른 설정**을 정리한다.
2026-09-30 재평가의 근거와 검증 결과를 함께 적었다.
원인 분석: `paper_results`, `reports/next_byte_entropy/blt_rerun.md`, 메모리 `blt-entropy-window-bug`.

---

## 1. 한 줄 요약

**공식 `facebookresearch/blt` 코드 + 공개 가중치 + `attn_impl=xformers`(모든 sliding window 적용) + threshold `1.335442066192627` + 배치 크기 1.**

---

## 2. 필수 설정

### 2.1 모델 · 코드

| 항목 | 값 |
|---|---|
| 본 모델 가중치 | `facebook/blt-1b` (`config.json` + `model.safetensors`, gated) |
| 엔트로피 모델 가중치 | `facebook/blt-entropy` |
| 코드 | 공식 `facebookresearch/blt` (`bytelatent` 패키지) |
| 로딩 | `ByteLatentTransformer.from_pretrained(<blt_1b dir>)`, `LMTransformer.from_pretrained(<entropy dir>)` |
| 정밀도 | bf16 |
| attention 구현 | `xformers` (공개 config 기본값) |

### 2.2 Attention 창 — 가장 중요

공개 config에 정의된 창을 **모두** 적용해야 한다.
`attn_impl=xformers`이면 `bytelatent`가 config를 읽어 자동으로 적용한다.

| 구성 요소 | 설정 |
|---|---|
| 엔트로피 모델 | `attn_bias_type=local_block_causal`, `sliding_window=512` |
| Local encoder / decoder | `local_attention_window_len=512` |
| Global transformer | `block_causal` (창 없음) |

### 2.3 Patching

| 항목 | 값 |
|---|---|
| 모드 | entropy |
| threshold | **`1.335442066192627`** (공개 값) |
| 규칙 | 위치 p의 다음 바이트 엔트로피 > θ → p+1에서 새 patch 시작. BOS(토큰 0)와 첫 바이트(토큰 1)는 항상 시작점 |
| monotonicity | 없음 |
| 최대 patch 길이 | 제한 없음 (`max_patch_length=None`) |

### 2.4 입력

| 항목 | 값 |
|---|---|
| tokenizer | `BltTokenizer(vocab_size_unit_1=256, bpe_delim=False, add_bos=True, add_eos=False)` — 바이트 b → id b+4, BOS=1 |
| 최대 문맥 | 4,096바이트. 넘으면 앞쪽을 잘라낸다 (원래 평가와 동일) |
| few-shot 제외 | 3-shot MMLU, 5-shot BoolQ/MMLU는 4,096바이트를 넘으므로 BLT에서 제외 (`--`) |

### 2.5 배치 — 배치 크기 1

`eval_suite/blt_eval/harness.py::BLTHarness`는 배치를 128바이트 / 64 patch의 배수로 padding한다.
그래서 배치 크기가 1보다 크면 **같은 배치의 다른 문항에 따라 점수가 달라진다.**

2026-10-01 측정 (`reports/blt_batch_invariance`): 0-shot, 6개 과제(HellaSwag/ARC-E/ARC-C/PIQA acc_norm, WinoGrande/BoolQ acc) × 500문항.
순변화는 3,000문항 전체에서 배치 1 대비 정답 수 증감, p는 대응 부호 검정.

| harness | 배치 1 | 배치 4 | 배치 8 | 배치 16 | 배치 1 대비 순변화 |
|---|---|---|---|---|---|
| 현재 (행별 엔트로피, 8/29 수정) | 63.03 | 62.67 | 62.77 | 62.63 | −0.27~−0.40pt (p=0.10–0.27, 유의하지 않음). 과제당 문항 1–3%가 정답↔오답으로 뒤집힘 |
| 수정 전 (`code_appendix` 판) | 62.93 | 60.97 | 61.47 | 61.67 | −1.27~−1.97pt (배치 4 p=0.005, 배치 8 p=0.032). 과제당 문항 6–25%가 뒤집힘 |

원인은 두 가지다.

- **수정 전 harness:** BLT의 `calculate_entropies`가 배치를 한 줄로 이어 붙여 8,192바이트 단위로 다시 자른다. 앞 문항이 다음 문항의 엔트로피 문맥(512 창 안)에 섞여 요청당 logprob이 평균 약 1 nat 바뀐다. 현재 harness는 `_row_entropies`로 행마다 따로 계산한다.
- **현재 harness에도 남는 것:** bf16 엔트로피 모델이 배치 shape에 따라 조금씩 다른 값을 내서 threshold 근처에서 patch 경계가 뒤집힌다 (요청 128개 중 24–33개). 엔트로피 모델을 fp32로 돌리면 경계 변경은 0건이 되고 순변화도 −0.13~−0.17pt로 줄지만, 본 모델의 bf16 수치 차이(평균 0.06 nat)는 남는다. 배치 1은 반복해도 결과가 비트 단위로 같다.

배치 크기 > 1에서는 torch.compile된 flex-attention `create_block_mask`(동적 shape)가 가끔 `CUDA illegal memory access`로 죽는다.
`torch._dynamo.config.automatic_dynamic_shapes=False`로 피할 수 있지만 2–3배 느려진다.

이전 판의 예시 중 ARC-Easy acc 배치 16 0.687 / 배치 8 0.650 (`reports/verify_blt_1335`)은 8/5에 수정 전 harness로 잰 값이다.
HellaSwag 첫 300문항(배치 16 63.3 / 배치 1 62.3)과 S-NIAH(배치 8 0.91 / 배치 1 0.96)는 측정 harness를 확인할 수 없어 근거로 쓰지 않는다.

---

## 3. 금지 설정

| 설정 | 문제 |
|---|---|
| `attn_impl=sdpa` + `BLT_SUPPRESS_ATTN_ERROR=1` | `create_causal_mask`가 모든 창 마스크를 **조용히 일반 causal로 바꾼다.** 논문 당시 원인이다. 512바이트 이후 patch가 약 1바이트로 무너진다 |
| HF transformers 구현(`itazap/blt-1b-hf`)을 **수정 없이** 사용 | local encoder/decoder와 엔트로피 모델에 창이 없다. 512바이트 이하 입력에서는 공식과 같지만, 그보다 길면 결과가 다르다 (S-NIAH: 창 없음 0.47, 엔트로피 창만 적용 0.12, 공식 0.96). 쓰려면 세 부분 모두에 창 마스크를 넣어야 한다 (`reports/next_byte_entropy/blt_rerun.md`, §7) |
| **엔트로피 창만 복원** | 아무것도 안 한 것보다 나쁘다. S-NIAH 0.12 < 창 없음 0.47. 부분 수정은 금지 |
| threshold 2.85 | 공개 값이 아니다 (DCLM 4.5 B/patch에 맞춘 값). 짧은 프롬프트에서 patch가 13–22바이트로 과도하게 길어져 정확도가 9–25pt 떨어진다 (`reports/blt_threshold_ab`) |
| 배치 크기 > 1 (현재 harness) | 배치 구성에 따라 문항이 1–3% 뒤집히고 평균이 약 0.3pt 흔들린다. 가끔 CUDA 크래시도 난다 (2.5절) |
| 배치 크기 > 1 (수정 전 harness) | 평균이 1.3–2.0pt 떨어진다 (2.5절) |
| `BLT_ALLOW_MISSING_FLEX_ATTENTION=1` | cross-attention의 flex attention까지 꺼져서 forward가 실패한다 |

---

## 4. 실행 방법

### 4.1 공통 러너

`scripts/probes/ext_ci/run_ext.py --family blt_official`

- 공식 모델을 불러온 뒤 `BLTHarness`(배치 1)로 lm-eval 과제, 섭동(Noise/Typo/Despace), S-NIAH를 채점한다.
- 문항별 결과(`samples`, despace `bits`, S-NIAH `per_item`)를 JSON으로 저장한다.
- 섭동 seed는 **1234**다. matched 모델의 `numpy_random_seed`와 같아서, 모든 모델이 바이트 단위로 같은 섭동 입력을 받는다.

```bash
# 예: 0-shot 다운스트림
python run_ext.py --family blt_official --blt_weights <weights dir> \
  --axis downstream --num_fewshot 0 \
  --tasks hellaswag arc_easy arc_challenge piqa winogrande boolq mmlu_text \
  --out out/blt_official/ds0.json

# axis 종류: downstream | noise | typo | despace | sniah | bpb

# bpb: 보유 윈도우 jsonl({text, n_bytes})에 대한 byte-normalised BPB.
#      eval_suite/common/axes.py::bpb, scripts/probes/bpb_windows_local.py와 같은 프로토콜이라
#      matched 모델 값과 직접 비교된다. H-Net은 --hnet_model 로 1stage/2stage 선택.
python run_ext.py --family blt_official --blt_weights <weights dir> \
  --axis bpb --windows data/flores_bpb/de.jsonl --out out/blt_official/bpb_de.json
# 섭동은 --limit 2000, 과제별로 나눠 돌리면 병렬화가 쉽다 (--tasks piqa 등)
```

**표준 벤치마크만 재현할 때:** 모델 공통 통합 스크립트 `scripts/probes/std_bench/`를 쓴다 (사용법과 재현성 조건은 그 폴더의 `README.md`).
BLT는 `blt_1b`로 등록돼 있고 배치 1로 고정된다.

```bash
bash scripts/probes/std_bench/run_std_bench.sh blt_1b             # 0-shot, seed 1234, 전체 test set
SHOT=5 SEED=1234 bash scripts/probes/std_bench/run_std_bench.sh blt_1b
LIMIT=500 GPU=3 bash scripts/probes/std_bench/run_std_bench.sh blt_1b
# 결과: reports/std_bench/blt_1b/ds<shot>_seed<seed>[_limit<N>].json, *_ci.md
```

2026-10-01에 6개 과제 × 500문항으로 돌려 `reports/blt_batch_invariance`의 배치 1 결과와 문항 단위로 같음을 확인했다.

### 4.2 환경별 준비

| 서버 | 환경 | 실행 |
|---|---|---|
| snu234 | `~/aunet_ext/blt_venv` (MRGUI env의 torch 2.8 + xformers 0.0.32, `--system-site-packages`, lm-eval 0.4.11) | `PYTHONPATH=~/aunet_ext/blt_official EVAL_SUITE=~/aunet_ext/eval_suite AUNET_LINGUA=~/aunet_ext/lingua blt_venv/bin/python run_ext.py ...` |
| snu55 (= 로컬 `gpusvr0908`) | `vllm011` python (torch 2.8 + xformers 0.0.32, pip 없음) + `runs/ext_ci_snu55/extra_site` (`--target` 설치). 가중치는 `runs/ext_ci_snu55/blt_weights` | `PYTHONPATH=runs/ext_ci_snu55/extra_site:runs/ext_ci_snu55/blt_official /mnt/ssd2/hyun2/venvs/vllm011/bin/python run_ext.py ...` (`runs/ext_ci_snu55/worker.sh` 참고). GPU 2,3만 사용 |
| info103 | `/mnt/ssd/ansible-test/AUNet/venv` — 로컬 lingua venv와 패키지 동일(177개, torch 2.7.0+cu128 · xformers 0.0.30 · lm_eval 0.4.12). matched 모델 체크포인트 3종 보유. **BLT/H-Net용 설정은 아직 없음** (필요하면 blt_official·flash-attn 추가 필요) | A5000 4장, 파일시스템 비공유 |

**H-Net은 인터프리터가 다르다.** `flash_attn` + `mamba_ssm`이 필요해서 vllm011로는 안 되고
conda `spacebyte` 환경을 쓴다. `extra_site`는 vllm011 전용 보충 경로라 conda 환경에 붙이면
`pandas._libs` 충돌이 나므로 **BLT에만** 넣는다.

```bash
# H-Net
AUNET_LINGUA=<lingua> EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HNET_REPO=/mnt/ssd2/hyun2/hnet \
  /home/hyunwoong/miniconda3/envs/spacebyte/bin/python run_ext.py --family hnet ...
```

### 4.3 주의 사항 (실제로 겪은 문제)

1. **포트 충돌.** `bytelatent`는 torchrun 밖에서 `MASTER_PORT`를 `random.Random(SLURM_JOB_ID or -1)`로 정한다. 그래서 항상 28805가 나오고, 여러 작업을 동시에 띄우면 `EADDRINUSE`로 실패한다. `run_ext.py`는 빈 포트로 process group을 직접 초기화해서 이 문제를 피한다.
2. **`apps` 패키지 충돌.** BLT 저장소에 `apps/__init__.py`가 있어서 AU-Net의 `apps.aunet`을 가린다. lingua 쪽 `apps/`와 `apps/aunet/`에도 `__init__.py`가 있어야 한다.
3. **xformers 0.0.30 이상.** `bytelatent/distributed.py`가 사라진 op(`efficient_attention_forward_cutlass`)를 import 시점에 참조한다. 로컬 복사본에서 `hasattr` 검사로 우회했다. activation checkpointing 목록일 뿐이라 추론 결과에는 영향이 없다.
4. **스크립트 구조.** `setup_torch_distributed`가 자식 프로세스를 띄우므로 스크립트 본문을 `if __name__ == "__main__":` 안에 둬야 한다.
5. **lm-eval 버전.** 0.4.8은 옛 데이터셋 스크립트(`piqa.py`)를 써서 `datasets` 3.x 이상에서 실패한다. 0.4.11 이상을 쓴다. 섭동 코드가 쓰는 `TaskManager.load`가 없는 버전에서는 `run_ext.py`의 shim이 대신한다.

---

## 5. 검증 (실행 전후 반드시 확인)

`scripts/probes/ext_ci/check_blt_window.py`는 평가 harness와 **같은 호출**(`_row_entropies` → `patcher.patch`)로 DCLM 320개 창을 측정한다.

| 확인 항목 | 정상 (창 적용) | 창이 빠진 경우 |
|---|---|---|
| DCLM bytes/patch | **≈ 4.08** (측정 4.075) | 1.151 |
| 위치 512 이후 평균 엔트로피 | 0.78–0.81 (512 이전 0.90과 비슷) | 1.8 → 2.56으로 급등 |
| 512 이후 threshold 초과 비율 | ≈ 24% | 95–99% |
| 0-shot HellaSwag / WinoGrande (full) | 70.3 / 65.0 | — |
| S-NIAH 평균 (n=250/셀, ≤4k) | 0.963 | 0.45–0.47 |

앞의 세 항목 중 하나라도 "창이 빠진 경우"와 같으면 결과를 쓰지 않는다.

---

## 6. 현재 기준값 (2026-09-30, `reports/ext_ci/blt_official`)

| 항목 | 값 (95% CI 절반 폭) |
|---|---|
| 다운스트림 Avg5, 0 / 3 / 5-shot | 63.3±1.0 / 66.2±0.9 / 66.6±1.0 |
| S-NIAH-1 / 2 / 3, 평균 | 0.930 / 0.983 / 0.975, 평균 0.963 |
| Noise / Typo / Despace \|ΔAcc\| | 13.0±0.7 / 4.2±0.5 / 5.9±0.9 |
| PBP | 0.00 (5개 과제, 창 적용) |
| clean 다운스트림 B/patch (Avg5 과제) | 3.66 |

이전 논문 값(0-shot 59.1, 3-shot 59.4, 5-shot 57.2, S-NIAH 0.45)은 창이 빠졌고 배치 버그도 있던 상태의 결과라 쓰지 않는다.

---

## 7. HF transformers 포트를 쓰려면 (upstream 수정 진행 중)

포트를 쓸 수밖에 없는 경우(공식 코드가 없는 환경 등)를 위해 정리한다. **세 곳 전부** 창을 넣어야 하며,
하나라도 빠지면 쓰지 않는다.

| 구성요소 | 포트에서 필요한 조치 |
|---|---|
| 엔트로피 모델 | `BltPatcherConfig.sliding_window=512` + `BltPatcher.forward`가 `create_sliding_window_causal_mask` 사용 |
| local encoder / decoder | `BltConfig.local_attention_window_len=512` + `BltModel.forward`의 byte-level 마스크가 windowed |
| global transformer | 그대로 causal (창 없음) |

- 즉시 쓸 수 있는 구현: `reports/next_byte_entropy/code/blt_window.py` (`install(512)`). `position_ids is None`
  으로 global 호출을 구분한다. 검증됨 — S-NIAH 0.96(공식 0.963).
- **upstream 반영**: transformers 이슈 [#49185](https://github.com/huggingface/transformers/issues/49185),
  PR [#49188](https://github.com/huggingface/transformers/pull/49188). config 기반 분기로 같은 수정을 넣었고,
  위 shim과 logit이 비트 단위로 동일함을 확인했다(400/1200/3000바이트). 변환 스크립트가 두 필드를
  버리던 것도 함께 고쳤다. 머지되면 포트를 수정 없이 쓸 수 있다.
- 원인: `facebook/blt-1b`의 `local_attention_window_len: 512`와 `facebook/blt-entropy`의
  `sliding_window: 512`가 변환 과정에서 유실됐다. 포트 config에는 아무 창 필드도 없다
  (`patcher_config.attn_bias_type: local_block_causal`만 남아 있고 이걸 읽는 코드가 없다).

---

## 8. 아직 남은 일

- **Forward latency 재측정.** 현재 값 1315.3 ms는 창 없이, 또 기록에 따르면 threshold 2.85에서 측정됐다. 위 설정(창 적용, threshold 1.335)으로 다른 모델과 같은 **A100-80GB**에서 다시 재야 한다.
- 재측정 전까지 main table과 `throughput_1B.tex`의 BLT latency는 `°`로 표시한다.
