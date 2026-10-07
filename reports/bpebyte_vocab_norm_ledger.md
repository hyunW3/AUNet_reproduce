# BPEByte vocab_norm 기록 (2026-10-07 기준)

**정책**: 논문의 BPEByte 숫자는 모두 AU-Net 공식 generator(`vocab(x)`, vocab_norm 없음)로 채점합니다. AU-Net과 같은 방식입니다.
lingua 기본값도 이 방식입니다(`34c5791`). `BPEBYTE_VOCAB_NORM=1`을 줄 때만 vocab_norm을 적용합니다. 자세한 내용은 `lingua/apps/aunet/SCORING_POLICY.md`에 있습니다.
**vocab_norm 기간**: lingua `7e58c5c`(2026-10-05 11:52) ~ `297fd75`(2026-10-07 13:09). 이 기간에 `make_generator`를 거친 BPEByte 평가는 vocab_norm이 적용됐습니다.
같은 수정이 들어간 브랜치 `superword-level2`(`8c74e40`, `871564c`)는 opt-out이 없어 지금도 vocab_norm입니다.
`std_bench/models.py`와 fast-decode latency 경로는 영향을 받지 않습니다.
**이후 기록**: lingua `fa54a2d`부터 `apps.aunet.eval`이 `eval_meta.json`의 `scoring` 필드(`scoring_generator`, `vocab_norm`)와 로그 `[scoring]`에 채점 방식을 남깁니다.

## 논문에 쓰이는 BPEByte 결과
| 논문 위치 | 결과 경로 | 생성 | vocab_norm | 근거 |
|---|---|---|---|---|
| Downstream 0/3/5-shot (메인 표, 부록) | `runs/downstream_ci/bpebyte_{0,3,5}shot` | 09-29 | **없음** | 기간 이전 |
| LAMBADA | `reports/std_bench/bpebyte_1.3b` | 10-04~06 | **없음** | std_bench가 공식 generator를 직접 생성 |
| S-NIAH | `reports/niah/final/bpebyte_items.jsonl` | 10-02 | **없음** | 기간 이전 |
| PBP, Noise | `runs/robustness_paper1p3b{,_ext}/bpebyte` | 08-23, 08-29 | **없음** | 기간 이전 |
| Despace | `runs/robustness_despace_bits/bpebyte` | 09-28 | **없음** | 기간 이전 |
| Typo, Leet (메인/상세/범주 표, sig_holm, Fig.1) | `runs/robustness_paper4task_nonorm/{typo,leet}/bpebyte` | 10-07 | **없음** | `BPEBYTE_VOCAB_NORM=0`, lingua `aebc6a4`. 옛 경로 재현: Despace 2/640비트(vocab_norm이면 27/640), PIQA clean 72.63(옛 72.58) |
| 100M parser ablation | `reports_NAACL/parser_ablation_100M` | 10-05~06 | **없음** | ece 스냅샷 `3528f19`(기간 이전 코드) |
| BoolQ context-only Typo (`tab:boolq_robust`, −5.79 ± 1.4) | `runs/robustness_boolq_nonorm/trio_tc_bpebyte` | 10-07 13:52 | **없음** | `eval_meta.scoring.vocab_norm=False`. Overleaf `0f9b27b`에 반영(이전 vocab_norm 실행 `runs/robustness_boolq_ctx/trio_tc_bpebyte`는 −6.39) |
| BoolQ Leet both (부록 "n0 97–100%, 37–39%") | `runs/robustness_boolq_nonorm/trio_leet_bpebyte` | 10-07 13:46 | **없음** | `eval_meta.scoring.vocab_norm=False`. 정확도 37.3%(이전 37.2%)라 문장은 그대로 성립 |
| BPB 0.908 | 논문 커밋 `7fafb65` | 07-29 | 해당 없음 | `model.forward`(학습과 같은 경로)로 계산 |
| Latency | fast-decode 경로 | — | 해당 없음 | 채점이 아니라 속도 측정. 이 경로는 원래 vocab_norm을 적용 |

## 논문에 쓰이지 않는 vocab_norm 결과 (참고)
- `runs/robustness_boolq_ctx/trio_tc_bpebyte`, `runs/robustness_boolq_both/trio_leet_bpebyte`: 위의 nonorm 재실행으로 대체됐습니다.
- `runs/robustness_typoleet_both/trio_{typo,leet}_bpebyte` (10-06): 위의 nonorm 재실행으로 대체됐습니다.
- `runs/robustness_boolq_both/trio_nt_bpebyte` (10-06): `five_task_summary.json`에만 쓰입니다.
- `runs/robustness_paper4task/{noise_pbp,despace}/bpebyte` (10-07): vocab_norm 효과를 보려고 일부러 돌린 비교 실행입니다.
- `reports/fewshot_probe/smoke_merged/root_greedy`, `reports_NAACL/fewshot_vocab_norm *_vnfix`, `reports_NAACL/inference_speed/verify_vocab_norm_2026-10-05`: 비교·검증용입니다.
- `runs/bpebyte_br_greedy_root_1.3B/eval_floresA_fix`, `eval_floresAX_s*` (10-05, `AUNET_FIX_VOCAB_NORM=1`): FLORES는 논문에서 `\iffalse`로 빠져 있습니다. 다시 넣으려면 재실행해야 합니다.
