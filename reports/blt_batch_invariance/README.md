# BLT-1B: 평가 배치 크기에 따른 점수 변화 (2026-10-01)

공식 bytelatent BLT-1B (xformers, threshold 1.3354, native 512 창), `run_ext.build_blt_official` + `eval_suite/blt_eval/harness.py`.
0-shot, 6개 과제 × 500문항 (HellaSwag/ARC-E/ARC-C/PIQA acc_norm, WinoGrande/BoolQ acc), 요청 8,995개. snu55 GPU 2,3.

| 변형 | 설명 | 평균 bs1 | bs4 | bs8 | bs16 | 순변화 (sign-test p) |
|---|---|---|---|---|---|---|
| fixed | 현재 harness (행별 엔트로피, 8/29 수정) | 63.03 | 62.67 | 62.77 | 62.63 | −0.37 / −0.27 / −0.40pt (p=0.14 / 0.27 / 0.10) |
| fixed_entfp32 | fixed + 엔트로피 모델 fp32 | 62.80 | 62.63 | 62.67 | 62.63 | −0.17 / −0.13 / −0.17pt (p≥0.42) |
| unfixed | 수정 전 harness (= code_appendix 판, `calculate_entropies`가 배치를 한 줄로 이음) | 62.93 | 60.97 | 61.47 | 61.67 | −1.97 / −1.47 / −1.27pt (p=0.005 / 0.032 / 0.064) |

과제별 표·뒤집힌 문항 수·요청별 |Δlogprob|: `summary.md`. 문항별 데이터: `out/*.json`.

## 원인 (diag.py, diag*.json — 요청 128개)
1. **unfixed**: 배치를 flatten → 8192 단위로 재분할하므로 문항 i의 앞부분이 문항 i−1 뒤에 이어진 채 엔트로피가 계산된다(512 창 안). 요청당 평균 |Δlogp| ≈ 1.0–1.3 nat, 최대 155 nat. 과제당 6–25% 문항 정답 뒤집힘.
2. **fixed에도 남는 것**: bf16 엔트로피 모델이 배치 shape에 따라 미세하게 다른 값을 내고, threshold 근처에서 patch 경계가 뒤집힌다. 같은 문항 8개 복제 배치에서 24/128, 긴 문항과 함께 배치에서 33/128 요청의 경계 변경. 엔트로피 fp32 → 경계 변경 0건.
3. 남는 미세 차이(평균 0.06 nat)는 본 모델 bf16 커널 수치 차이. bs1 반복은 비트 단위로 동일(결정적).

## 크래시
bs>1에서 비결정적으로 `CUDA illegal memory access` (torch.compile된 flex-attention `create_block_mask` 커널, 동적 shape).
`verify_blt_1335`의 bs4 실패도 같은 것. `STATIC_SHAPES=1` (automatic_dynamic_shapes=False, recompile 한도 상향)이면 통과하지만 2–3배 느리다.

## 결론
- 현재 harness: 배치 크기에 따라 점수가 **조금** 흔들린다(과제당 1–3% 문항 뒤집힘, 평균 −0.3pt 경향, 통계적으로 유의하지 않음). 논문 숫자는 bs1 유지.
- 수정 전 harness: 평균 1.3–2.0pt 하락, 유의. `BLT_execution_guide.md` §2.5의 예시(ARC-E 0.687 vs 0.650 등)는 8/5 측정으로 이 버그가 섞인 값이다.

재현: `STATIC_SHAPES=1 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=runs/ext_ci_snu55/extra_site:runs/ext_ci_snu55/blt_official EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HF_DATASETS_OFFLINE=1 /mnt/ssd2/hyun2/venvs/vllm011/bin/python reports/blt_batch_invariance/bs_sweep.py --variant fixed --bs 1 4 8 16`
