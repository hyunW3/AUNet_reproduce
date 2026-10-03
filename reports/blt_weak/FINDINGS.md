# BLT 약점 probe — 결론 (2026-10-02)

후보와 근거는 `CANDIDATES.md`, 전체 표는 `RESULTS.md` (`scripts/probes/blt_weak/report.py`로 생성), 그림은
`dist512.png`. 4개 모델(Llama-1.8B / AU-Net / BPEByte-rg 1.3B / BLT-1B official)이 byte-identical prompt
10,200개를 같은 runner(`run_longctx.py`, greedy + substr / rank)로 채점. BLT는 bs 1, ≤4096 B.

## 한 줄 요약

**가설(후보 1·2)은 대부분 반증됐다.** BLT patching의 메커니즘 차이(반복 구간 병합, 512 B window에 따른
segmentation 변화)는 patch probe에서 그대로 확인됐지만, **정확도로는 BLT가 거의 모든 probe에서 BPEByte와
같거나 앞선다.** BPEByte가 앞선 셀은 near-collision MK-NIAH `rand` K=2(0.96 vs 0.85, 95% CI가 겨우 분리)와
dist512 hash d=2048(0.74 vs 0.67, CI 겹침) 정도이고, 같은 probe의 다른 K / d에서는 BLT가 앞서므로 일관된 우위가 아니다.

## 주요 수치

| probe | BPEByte | BLT-1B | 비고 |
|---|---:|---:|---|
| HashHop 3584 B | 0.25 | **0.92** | BPEByte는 길이에 따라 0.77→0.25로 감소, BLT는 ≤2 KB에서 1.00 |
| MK-NIAH K=8 (word key) | 0.61 | **0.98** | |
| near-collision MK-NIAH, Δ vs rand (평균 nc_*) | −0.34 | −0.33 | 하락 폭 동일, 절대값은 BLT가 높음 (nc_last K=8: 0.30 vs 0.57) |
| S-NIAH-1/2/3 `distract` Δ | −0.20 | **−0.04** | BLT는 무관 distractor에 거의 영향 없음 |
| S-NIAH-1/2/3 `key_nc` Δ | −0.61 | **−0.41** | |
| S-NIAH-1/2/3 `keyval_nc` Δ | −0.65 | −0.53 | |
| count (few-shot, 20 options) | 0.04–0.08 | 0.08–0.10 | **전 모델 chance(0.05) 근처** → 판별력 없음 |
| copy rand L=128 | 0.44 | **1.00** | |
| copy periodic_mut L=32 / 64 | 0.36 / 0.04 | 0.42 / 0.14 | 전 모델 붕괴; Llama 0.54 / 0.11 |
| dist512 hash, d ≤ 512 → d > 512 | 0.90 → 0.74 | 0.99 → 0.86 | BLT는 d ≥ 2048에서만 0.67–0.72 (BPEByte 0.74 / 0.60과 동급) |

## 메커니즘은 확인됨 (patch probe, §8)

- **반복 병합**: count `word`(평균 48 B) 구간이 BLT는 2.9 patch, BPEByte는 7.4 patch이다. copy에서 BLT가 생성하는
  답(128 B random)은 **2.1 patch**에 담긴다. 앞선 copy 결과가 1.00이므로 긴 patch가 정확한 복사를 막지 못한다.
  local decoder가 patch 내부의 byte를 충분히 복원한다.
- **512 B window**: dist512 hash에서 BLT query key(12 B)의 patch 수는 d가 커지면서 1.1(d ≤ 448) → 4.5(512–768)
  → 11.5(≥ 2048)로 바뀌고, 정답 span도 1.7 → 7 → 11.9로 바뀐다. 같은 문자열의 segmentation이 거리에 따라 달라진다는
  예측은 정확히 맞다. BPEByte는 모든 d에서 6.7 patch로 일정하다. 다만 BLT 정확도 하락은 d ≥ 2048에서만 나타나고
  (1.00 → 0.67–0.72), 512 경계 자체에서 생기는 하락은 작다(0.94).
- **near-collision**: `nc_last`에서 BLT query key는 1.4 patch로 병합된다(`rand`는 5.2). 그런데도 정확도 하락 폭은
  다른 모델과 같다. 병합은 일어나지만 손해로 이어지지 않는다.

## 부수 발견 (논문에 쓸 만한 것)

1. **BLT 시스템 취약성**: official BLT는 copy의 반복성 입력(`periodic`, `words_mut`)에서만 94/600 row(16%)가
   `CUDA illegal memory access`로 crash했다. `rand`와 `periodic_mut`(entropy가 높은 입력)에서는 0건이다. 매우 긴
   patch가 생기는 입력에서 kernel이 깨지는 것으로 보인다. crash row는 표에서 제외했으므로 해당 셀의 BLT n은
   76–90이다. 정확도 약점이 아니라 robustness / serving 약점이다(후보 6과 같은 축).
2. **우리 byte 모델의 거리 약점**: hash needle에서 **AU-Net은 d ≥ 448부터 0.00**이고(d=384에서 0.64),
   BPEByte도 d=448에서 0.96 → 0.72로 떨어진다. num needle에서는 모두 1.00이다. 즉 단어 경계가 없는 12-byte
   random 문자열이 needle에서 ~450 B 이상 떨어지면 trio의 byte 단계가 복사하지 못한다. local(byte) attention
   window 쪽 한계로 보인다. 같은 hierarchical 구조이고 경계가 448–512 사이라는 점이 근거다. BLT의 약점이 아니라
   **우리 모델의 약점**이다.
3. HashHop / MK-NIAH의 BLT arm: 기존 trio 결과(`reports/niah/hashhop_1p3b`)와 같은 생성기로 만들었다. BLT가 크게
   앞선다(1.3B matched 비교에서 BLT는 외부 reference이므로 공정성 각주 필요: 학습 데이터·토큰 수가 다름).

## 다음에 해볼 만한 것

- 남은 후보 3(echo / priming)과 4(OOD 입력에서 patch 수 폭주 → **compute / latency** 비교)는 정확도 대신
  **효율 축**이다. BLT가 정확도로는 강하므로 BPEByte의 우위는 "같은 정확도에서 예측 가능한 patch 수와 결정성"
  쪽에서 주장하는 편이 현실적이다.
- count probe는 1B base 모델에서 floor다. 쓰려면 0–5 범위의 쉬운 셋으로 다시 만들거나 rank 대신 비교형
  ("A가 B보다 많나?")으로 바꿔야 한다.
- 위 부수 발견 2(trio hash-needle 거리 붕괴)는 local window 설정을 확인한 뒤 따로 파고들 가치가 있다.

---

# 2차: 후보 3–6 (2026-10-02)

같은 4개 모델, 같은 runner. MC는 ARC-E / ARC-C / PIQA / HellaSwag 각 500문항(lm-eval 표준 prompt, raw-loglik acc)이다.
Δ는 같은 문항 쌍으로 계산한 paired 값이고, DiD는 BLT의 Δ에서 상대 모델의 Δ를 뺀 값이다. 표는 `RESULTS.md` §9–12,
DiD bootstrap 표는 `delta_ci.md`(`delta_ci.py`, 2000 iters)에 있다.

## 한 줄 요약

**후보 3(echo / priming)에서 처음으로 유의하고 BLT에 특유한 약점이 나왔다.** 선택지를 질문 앞에 나열하면 BLT는
−0.21, 나머지 세 모델은 −0.11 떨어진다(DiD −0.106 [−0.133, −0.079]). BLT는 나열된 **첫 번째 선택지를 88%** 고른다
(나머지 60–64%, clean에서는 모두 38%). 후보 4(OOD)는 **compute 쪽 불안정성**으로 확인됐고, 후보 5·6은 BPEByte의
우위가 작거나 없었다.

## #3 echo / priming

| variant | Llama Δ | AU-Net Δ | BPEByte Δ | BLT Δ | DiD BLT−BPEByte [95% CI] |
|---|---:|---:|---:|---:|---:|
| echo_all (선택지 나열) | −0.111 | −0.112 | −0.105 | **−0.212** | **−0.106 [−0.133, −0.079]** |
| echo_wrong (오답 hint) | −0.483 | −0.490 | −0.467 | **−0.543** | **−0.077 [−0.095, −0.059]** |
| echo_gold (정답 hint) | +0.443 | +0.422 | +0.432 | +0.420 | −0.011 [−0.028, +0.005] |
| repeat_q (질문 2회) | −0.008 | −0.008 | −0.010 | −0.002 | +0.009 (n.s.) |

- echo_all의 DiD는 ARC-E −0.22, ARC-C −0.10, PIQA −0.15이고 HellaSwag만 +0.04다. echo_wrong은 4개 task 모두 음수다.
- **메커니즘(patch probe)**: 정답 선택지(평균 53.6 B)는 BLT에서 clean 13.6 patch → echo_all 3.8, echo_gold 2.8로
  병합된다. BPEByte는 모든 variant에서 11.0으로 일정하다. 앞에 나온 문자열이 BLT에서는 다른 segmentation으로 바뀐다.
- **해석 주의**: 위치 편향(첫 선택지 88%)은 "나열된 문자열을 그대로 복사하려는" prior다. 네 모델 모두 이 편향을
  보이며, BLT에서 가장 강하다. patch 병합 때문인지 BLT의 강한 in-context 복사 능력 때문인지(1차 NIAH에서 BLT가
  가장 강했다)는 이 실험으로 분리되지 않는다. 논문에서는 "priming / 선택지 echo에 대한 BLT의 취약성,
  segmentation 변화와 동반됨" 정도로 쓰는 것이 안전하다.

## #5 byte 삽입 (질문 텍스트만)

전 모델에서 |Δ| ≤ 0.03으로 작다. 유의한 DiD는 **homoglyph −0.019 [−0.034, −0.004]** 하나뿐이다(BLT −0.029 vs BPEByte −0.010,
4개 task 모두 같은 방향). zwsp / shy / nbsp / emoji는 n.s.다. prompt bytes/patch는 BLT 3.75 → 2.69(homoglyph)로 가장 크게
잘게 쪼개지지만 정확도 손실은 작다. → 후보 5는 "BLT가 약간 더 민감" 수준이고 headline감은 아니다.

## #4 OOD 입력: compute 불안정성 (확인됨)

| | BLT | BPEByte | Llama (tok) |
|---|---:|---:|---:|
| B/patch en | 3.94 | 4.60 | 4.57 |
| B/patch random / hex / base64 | **1.00 / 1.00 / 1.00** | 1.39 / 1.77 / 1.45 | 1.33 / 1.76 / 1.40 |
| B/patch boilerplate / repeat_line | **30.0 / 69.9** | 4.58 / 3.13 | 4.52 / 3.12 |
| 최대 \|log2(B/patch ÷ en)\| | **4.15** | 1.72 | 1.78 |
| latency en → random (ms, bs 1, 3 KB) | 60 → 126 (**×2.1**) | 114 → 169 (×1.5) | 44 → 87 (×2.0) |

- BLT patch 수는 domain에 따라 70배 범위로 흔들린다. 고-entropy 입력(random / hex / base64 / UUID)에서는 1 B/patch로 떨어져
  BPEByte보다 global position이 40–77% 많고, 반복 입력에서는 한 patch가 70 B가 된다. compute와 메모리를 입력만 보고
  예측할 수 없다는 점은 serving 관점의 약점이다.
- 다만 **BPB는 BLT가 모든 domain에서 가장 낮고**(code 0.49 vs BPEByte 0.88, zh 1.21 vs 2.74), 절대 latency도 BLT가
  BPEByte보다 빠르다. BPEByte latency에는 CPU parser overhead가 포함된다. 따라서 "BPEByte가 더 효율적"이 아니라
  "BPEByte의 비용이 더 예측 가능하다"(drift 1.72 vs 4.15)고 써야 한다.

## #6 결정성

| BLT 경계 변화 | windows with ≥1 flip | flips / window |
|---|---:|---:|
| 재계산 / prefix(streaming) / +1024 pad / batch 8 | 0% | 0 |
| **bf16 → fp32 entropy model** | **64%** | 2.96 (zh 10.95) |

- official harness의 per-row entropy 경로는 batch / padding에 대해 경계 수준에서 불변이었다. `blt_batch_invariance`에서 본
  bs>1 점수 변화는 경계가 아니라 본 모델 forward의 batch 의존성에서 온 것으로 보인다(추정).
- 경계는 **수치 정밀도에 의존**한다. fp32 entropy model로 바꾸면 64% window에서 경계가 바뀐다. BPEByte / AU-Net은
  모든 조건에서 0이다(입력 bytes의 순수 함수).

## 종합 — BPEByte의 우위로 주장할 수 있는 것

1. **선택지 echo / priming 강건성**: 유의한 정확도 차이다(−0.106, −0.077). 단 메커니즘 귀속에는 주의가 필요하다.
2. **비용 예측 가능성**: domain 간 patch 수 drift가 1.72 vs 4.15이고, BLT 고-entropy 입력에서 latency가 ×2.1이 된다.
3. **segmentation 결정성**: 경계가 정밀도와 무관하다(BLT는 bf16↔fp32에서 64% window flip). 반복 입력에서 BLT kernel
   crash가 16% 발생했다(1차).
그 외 정확도 축(NIAH 계열, copy, insertion)에서는 BLT가 같거나 강하다.

---

# 3차: NoLiMa-lite (2026-10-02) — `NOLIMA.md`

`niah_ext_data.make_nolima`로 셀당 200개(literal과 paraphrase는 같은 item에 query 명사만 다름, 3 distractor needle,
1024/2048/3200 B)를 만들어 4개 모델을 같은 runner로 채점했다. 기존 BLT 수치(`reports/niah/n250/blt_probe_pairs_hfwin.json`)는
HF port 결과라 512 B 이후 local window가 빠져 신뢰할 수 없어서 대체한다.

| | Llama | AU-Net | BPEByte | BLT-1B |
|---|---:|---:|---:|---:|
| literal (전 길이) | 0.814 | 0.691 | 0.761 | **0.980** |
| paraphrase (전 길이) | 0.518 | 0.421 | 0.383 | **0.704** |
| NoLiMa 효과 (literal − paraphrase, paired) | +0.296 | +0.269 | **+0.378** | +0.276 |

- BLT paraphrase − BPEByte = **+0.321 [+0.278, +0.365]**, − Llama = +0.186 [+0.139, +0.231]. 모든 길이에서 BLT가 가장 높다.
- literal과 paraphrase의 차이(잠재 연상 비용)도 BPEByte가 가장 크다. → NoLiMa에서도 BPEByte의 우위는 없다.
- caveat: 이 runner(greedy 생성 + substr)의 BPEByte literal 1024는 0.88로, 기존 teacher-forced scorer(`probe_compare`)의
  0.98보다 낮다. BPEByte의 online-BT 생성 경로와 teacher forcing의 차이이며, 모델 간 비교는 같은 runner 안에서만 한다.

---

# 4차: echo 취약성의 원인과 범위 (2026-10-03) — `EXPERIMENTS_3.md`, `RESULTS3.md`

5개 모델(Llama / AU-Net / BPEByte / **H-Net 1-stage XL** / BLT-1B), MC 2,000문항 × 21개 조건(+2차 echo 조건 H-Net 추가),
그리고 BLT segmentation 이식(E2, 2,000문항 × 6조건)을 돌렸다. DiD = BLT Δ − 상대 모델 Δ(paired, item bootstrap 95% CI)이다.

## 결론

1. **원인은 patch 병합이 아니다.** BLT는 context에 등장한 후보 문자열 / 어휘에 더 강하게 끌린다. 즉 in-context
   lexical priming이 더 강하다.
   - **E2 (직접 개입)**: echo context에서 선택지 경계를 clean 때 경계로 되돌려도 Δ는 **+0.007 [−0.004, +0.018]**로 회복되지
     않는다. 첫 선택지 선택률도 0.88 → 0.91로 그대로다. 1-byte patch로 강제해도 +0.007이다. echo_all의 −0.212는
     segmentation을 바꿔도 남는다.
   - **C bag-of-words**: 오답 단어를 섞어 hint로 주면 BLT 정답 선택지 patch 수가 13.45(clean 13.62)로 병합이 전혀 없다.
     그래도 DiD vs BPEByte는 **−0.082 [−0.103, −0.061]**로 C 그룹에서 가장 크다. `suf50`(patch 12.17)도 −0.045다.
   - **E1 거리 분리**: echo를 512 B 밖으로 옮기면 병합은 일부 풀리지만(정답 patch 4.28 → 7.79) BLT의 추가 하락은 남는다
     (echo_all DiD −0.078 → −0.066, far−near DiD² +0.011 n.s.; echo_wrong은 far에서 −0.071로 오히려 커짐).
   - 참고로 clean context에 병합 경계를 이식하면 −0.082다. 병합 patch는 context가 뒷받침할 때만 자연스럽다는 뜻이며,
     echo 손실의 원인은 아니다.
2. **취약 범위 = "답 후보가 정보로 제시될 때"이고, 일반적인 distraction은 아니다.**
   - A 무관 / 같은 분포의 distractor 문장 1–4개: 모든 모델 |Δ| ≤ 0.015, DiD n.s. → BLT는 일반적 방해에 약하지 않다.
     1차 S-NIAH distract에서 BLT가 가장 강했던 것과도 일치한다.
   - B 오답을 정보로 제시: soft("Some people think…") **−0.089**, student **−0.083**, 부정문("It is NOT true…") **−0.096**
     (모두 DiD vs BPEByte, **). **부정은 모든 모델이 무시한다**: 정답을 부정문으로 제시하면(neg_gold) 오히려 모든 모델이
     +0.42–0.44 오르고 DiD는 0이다. 1B급 base 모델 공통의 "문자열 복사" 성향이다.
   - C 오답 prefix 일치량에 따른 dose-response: 25% −0.017(n.s.) → 50% −0.055 → 75% −0.071 → 100%(echo_wrong) −0.077.
     오답 끌림 비율(같은 오답을 고른 비율의 증가)도 BLT가 모든 조건에서 가장 높다(neg_wrong +0.81 vs BPEByte +0.70).
   - D 위치 편향: 정답을 목록 처음에 두면 BLT +0.243(BPEByte +0.122), 끝에 두면 **−0.406**(BPEByte −0.191,
     DiD −0.216). 두 개만 나열하면 다른 모델은 오르는데(+0.05–0.07) BLT만 −0.032다.
3. **H-Net은 그 중간이다.** echo_all Δ −0.146, list_gold_last −0.307로 trio보다 취약하지만 BLT보다는 덜하다.
   B 전부, C의 pre75 / suf50 / bow, D 전부에서 BLT − H-Net DiD가 유의하다(BLT가 더 강하게 끌림, |DiD| 0.02–0.10;
   list_gold_first는 같은 편향 때문에 BLT가 +0.046 더 이득). pre25 / pre50은 n.s.다.

## 논문용 정리

- 주장할 수 있는 것: "BLT는 context에 나열 / 언급된 답 후보에 과도하게 끌린다(priming / positional copy bias). matched
  BPEByte 대비 −0.08 ~ −0.22, 무관 distractor에는 차이 없음."
- 주장하면 안 되는 것: "entropy patching(병합) 때문에 생긴다." E2와 bag-of-words가 이를 반증한다. 원인은 BLT 모델
  자체(학습 데이터 / 규모 / 강한 in-context 복사 능력)일 가능성이 높고, 이 실험으로는 tokenizer-free 설계와
  분리되지 않는다. BLT는 다른 데이터·규모로 학습된 외부 reference라는 점을 각주로 남겨야 한다.

---

# 5차: 정답 위치 균형 + 채점 방식 분리 (2026-10-03) — `POSITION_BALANCED.md`

각 문항의 선택지 순서를 섞어 정답 위치를 A/B/C/D에 균등하게 배치했다(4지선다 373/372/375/375). 선택지 목록 위치(질문 앞 /
질문 뒤 표준 MCQ)와 채점 방식(선택지 텍스트 / 알파벳 / "(A) 텍스트")을 각각 따로 채점했다(OR 합산 없음).
5개 모델과 기준 prompt 모두 snu55에서 채점했다.

- **불균형 착시가 아니다.** 균형을 맞춘 "앞에 나열 + 텍스트 채점"에서도 BLT 0.362 vs trio 0.445–0.451, H-Net 0.422로
  원래 순서(0.366)와 같다. 정답 위치별로 보면 BLT는 A 0.761 / B 0.183 / C 0.115 / D 0.179(편차 0.647)이고,
  trio는 편차 0.27–0.32, H-Net은 0.48이다.
- **알파벳 채점은 다섯 모델 모두 chance다**(4지선다 0.25–0.26). 1B급 base 모델은 "A/B/C/D로 답하기"를 못 하고,
  각 모델의 알파벳 prior만 측정된다. 예: 질문 뒤 표준 MCQ에서 Llama는 B/D, AU-Net은 D, H-Net은 A를 주로 고르고,
  BLT는 상대적으로 고르게 분포한다(편차 0.139). 이 형식으로는 모델 간 비교가 무의미하다.
- **"(A) 텍스트" 채점**: 앞에 나열하면 BLT와 H-Net이 A로 가장 쏠린다(A 예측 0.87 / 0.83).
- **질문 뒤 표준 MCQ + 텍스트 채점**은 모든 모델에서 chance 근처다(0.25–0.29). 선택지를 질문 뒤에 두면 lm-eval
  cloze 형식보다 모두 크게 떨어진다.
- 결론: BLT의 위치 / 첫 선택지 편향은 정답 위치 균형과 무관하게 재현된다. 알파벳 채점은 이 규모에서 판별력이 없으므로
  텍스트 이어쓰기(cloze) 채점이 맞는 지표다.
