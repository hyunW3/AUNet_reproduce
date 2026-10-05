# 3차 실험 계획 — echo 취약성의 원인과 범위 (2026-10-03)

배경 (`FINDINGS.md` 2차): 질문 앞에 선택지를 나열하면(echo_all) BLT만 추가로 −0.106 떨어지고(BPEByte 대비 DiD), 나열된
첫 선택지를 88% 고른다. 오답 hint는 −0.077. 반면 무관 distractor needle(S-NIAH, −0.04)이나 질문 반복에는 BLT가 가장 강했다.
질문은 두 가지다. **(Q1) 원인이 patch 병합(entropy model이 앞의 문자열을 보고 선택지를 굵게 묶음)인가, BLT의 강한
in-context 복사 성향인가?** **(Q2) 어떤 종류의 context 정보에 취약한가?**

공통: ARC-E / ARC-C / PIQA / HellaSwag 각 500문항(`data/mc_items.jsonl`), 선택지 문자열과 채점(raw-loglik argmax)은
불변, context만 바꾼다. clean 기준은 2차 `echo/clean`과 같은 prompt다. 5개 모델(Llama / AU-Net / BPEByte / BLT-1B official / H-Net 1-stage XL).
H-Net은 2차 결과가 없으므로 이번에 `echo.jsonl`(clean 포함)도 함께 채점한다.
지표는 paired Δ vs clean, BLT − 각 모델 DiD(item bootstrap 95% CI), 그리고 patch probe의 정답 선택지 patch 수다.

## E1. 거리 분리 — 메커니즘 판별 (모델 수정 없음)

BLT의 entropy model과 local encoder/decoder는 512 B sliding window이고, global model은 전체 문맥을 본다. echo와 질문 사이에
~700 B filler를 넣으면 entropy model은 echo를 못 보지만 global model은 여전히 볼 수 있다.

| cond | prompt 구성 |
|---|---|
| `clean_fill` | filler + 질문 (filler 자체의 효과 통제) |
| `echo_all_near` | filler + Options(...) + 질문 |
| `echo_all_far` | Options(...) + filler + 질문 |
| `echo_wrong_near` / `echo_wrong_far` | 같은 배치의 `Hint: <오답>` |

near와 far는 같은 bytes이고 순서만 다르다. 예측은 다음과 같다.
- **patch 병합이 원인이면**: far에서 BLT의 정답 선택지 patch 수가 clean 수준(~13.6)으로 돌아오고, BLT의 추가 하락도 사라진다.
- **복사 성향이 원인이면**: patch는 돌아오지만 추가 하락이 남는다.

trio의 near→far 변화가 "거리에 따른 복사 약화"의 대조군이다(DiD-of-DiD).

## E2. segmentation 이식 — 직접 인과 개입 (BLT만)

BLT forward에 patch 경계를 외부에서 넣는다(`model(toks, patch_lengths=…)`). bytes는 그대로 두고 **선택지 구간의 경계만** 바꾼다.

| cond | context | 선택지 구간 경계 |
|---|---|---|
| T0 `echo_nat` | echo_all | 자연(entropy) — 2차 재현 |
| T1 `echo_cleanseg` | echo_all | clean prompt에서 같은 선택지가 받은 경계 이식 |
| T2 `clean_nat` | clean | 자연 — 2차 재현 |
| T3 `clean_echoseg` | clean | echo_all에서 받은 (병합된) 경계 이식 |
| T4 `echo_byte` / T5 `clean_byte` | echo / clean | 선택지 구간 1-byte patch (최대 세분) |

판정은 이렇다. T1이 T0보다 회복되고 첫 선택지 편향이 줄면 병합이 원인이다. T3가 T2보다 떨어지면 병합만으로도 충분히 해롭다.
T0과 T2가 2차 harness 점수를 재현하는지 먼저 확인한다(경계 규칙 검증).

## Q2 perturbation 그룹 (5개 모델)

| 그룹 | cond | 구성 | 질문 |
|---|---|---|---|
| A 무관 distractor | `dist1` / `dist2` / `dist4` | 무관한 일반 상식 문장 1/2/4개를 질문 앞에 | 일반적 주의 분산 |
| | `indomain` | 같은 task의 다른 문항 질문 1개를 앞에 | 같은 분포의 방해 |
| B 오답의 정보 형태 | `soft_wrong` | "Some people think the answer is: ⟨오답⟩." | hint 표지 없이도 끌리는가 |
| | `student_wrong` | "A student answered: ⟨오답⟩." | |
| | `neg_wrong` | "It is NOT true that the answer is: ⟨오답⟩." | 부정을 무시하고 복사하는가 |
| | `neg_gold` | "It is NOT true that the answer is: ⟨정답⟩." | 〃 (복사면 오히려 정답↑) |
| C 문자열 일치 정도 | `pre25` / `pre50` / `pre75` | Hint: 오답 앞 25/50/75%(단어 경계) | 글자 그대로의 prefix 일치량 dose |
| | `suf50` | Hint: 오답 뒤 50% | 시작부 일치가 필요한가(병합은 prefix를 보고 일어남) |
| | `bow` | Hint: 오답 단어들을 섞어서 | 의미/어휘는 같고 문자열 순서는 다름 |
| D 위치 / 순서 | `list_gold_first` / `list_gold_last` | Options 목록에서 정답을 처음 / 끝에 | 첫 선택지 편향이 위치 복사인가 |
| | `list_two` | 정답 + 오답 1개만 무작위 순서로 나열 | 나열 개수 |

해석 틀: **BLT의 추가 하락이 C에서 글자 그대로의 prefix(pre*, echo 100%)에만 생기고 suf50 / bow에서 사라지며, E1에서
far일 때도 사라진다면** "BLT는 문맥에 등장한 후보 문자열에 segmentation 수준에서 취약하다"는 주장이 된다. A에서도
BLT가 약하다면 일반적인 distraction 취약성으로 범위를 넓힌다.

## 산출물

- 데이터 `data/probe3.jsonl` (`build_tasks3.py`), 결과 `results/**/probe3_*.jsonl`, E2 `results/**/transplant_blt_1b*.jsonl`
  (`blt_transplant.py`), patch `patches/p3_*.jsonl`
- 표 `RESULTS3.md` (`report3.py`), 결론은 `FINDINGS.md` 4차 섹션
