# Plan — Direction A: BPE vocabulary as a *swappable boundary prior*

**작성일** 2026-08-11 · **상태** 설계 (학습 미실행) · **대상** BPEByte 후속 논문
**선행 문서** `PoC/tokenizer_swap_all_results.md`, `plans/chinese_continue_pretrain_plan.md`,
`plans/plan_compression_stability.md`, `plans/plan_stacked_bpebyte.md`, `plans/plan_entropy_patching.md`

---

## 0. TL;DR

1. **naive eval-time swap은 이미 4개 도메인에서 전부 실패했고**(zh +0.212, code +0.337, te +0.474 BPB,
   EN downstream −2.3 acc_norm; `PoC/tokenizer_swap_all_results.md`), 실패 원인은 coverage도
   granularity도 아니라 **boundary-position 분포에 대한 binding**이다. 따라서 "swap이 된다/안 깨진다"
   류의 논문은 물론이고, **학습 측 개입 없이 positive gain을 주장하는 어떤 설계도 이미 반증되어 있다.**
2. 그러므로 이 논문의 필수 재료는 **boundary-invariant training** (M1) 하나다. 나머지 후보
   (test-time selection M3, efficiency knob M4)는 전부 "swap이 non-destructive하다"를 전제로 하는데
   그 전제가 거짓이므로, M1 없이는 단독으로 성립하지 않는다.
3. **추천**: M1(= per-document trie 샘플링 + stochastic boundary dropout) 을 *baseline을 대체하는*
   pretraining 레시피로 삼고, 그 위에 M2(local enc/dec만 푸는 초저가 adaptation)를 "swap + N GPU-분"
   비용 주장으로 얹는다. M3/M4는 M1이 통과한 뒤 **무료로 딸려오는 측정**이다.
4. **Vehicle: Chinese를 기각하고 code(StarCoder2 / 자체 fit 코드 vocab)를 1순위**, **English ×
   SuperBPE를 2순위**로 한다. 이유는 §6.
5. **Go/no-go**: 100M law 레시피(53,504 step, 21 GB, ≈17 GPU-h/arm)에서 arm 3개.
   게이트는 "swap penalty 60% 이상 축소 + in-distribution BPB tax ≤ +0.010".
6. 최대 리스크: **invariance가 in-distribution 성능을 깎는 것(regularization tax)**, 그리고
   *invariance는 성공했는데 어떤 trie도 llama3보다 낫지 않아 gain이 0으로 수렴하는 것*(§11).

---

## 1. 문제의식

현재 논문(`paper_overleaf/sections/4-method.tex` §OnlineBPE)의 서술은 이미
"BPE vocabulary는 input/prediction unit이 아니라 **computation boundary에 대한 statistical prior**"다.
그러나 이 주장은 논문 안에서 **한 번도 실험적으로 사용되지 않는다.** llama3 vocab 하나만 쓰고,
그 vocab을 바꿨을 때 무슨 일이 일어나는지 보고하지 않는다. `6-conclusion.tex:16`의 주석 처리된
future-work 문단이 정확히 이 구멍을 자인한다:

> "OnlineBPE inherits the vocabulary of a single pretrained tokenizer (Llama 3, 128K entries) as its
> boundary prior. A systematic ablation over the vocabulary source, size, and training domain ...
> we leave this study to future work."

Prior가 진짜로 prior라면 세 가지가 따라나와야 한다.

- (a) **선택 가능성** — 어떤 vocab이 좋은 boundary prior인가? (크기, merge-rank 절단, 도메인 정합,
  whitespace-crossing unit, entropy patcher를 trie로 distill)
- (b) **교체 가능성** — I/O가 byte이므로 embedding/head 수술 없이 학습 후 trie만 갈아끼울 수 있다.
- (c) **학습 가능성** — prior 자체를 데이터로부터 fitting 할 수 있다.

이 문서는 (a)+(b)를 다루되, **user가 명시적으로 거부한 결론**("swap이 가능하다 / 안 깨진다")을
피하고 **positive downstream gain**을 낼 수 있는 조건을 설계한다.

---

## 2. 핵심 주장 (what we must be able to write)

> **주장 H.** BPE vocabulary는 모델 가중치와 독립적인 *교체 가능한 hyperparameter*다.
> 단, 그렇게 만들려면 **학습 때 boundary 분포에 대한 invariance를 주입해야** 하며, 그렇게 학습된
> 모델은 (i) 원래 trie로도 손해가 없고, (ii) 추론 시 도메인 정합 trie로 바꾸면 **동일 가중치로
> 목표 도메인 성능이 실제로 향상**되며, (iii) trie 선택은 latency/BPB Pareto 상의 자유 축이 된다.

성공 판정은 **(ii)** 하나로 한다. (i)은 비용 조건, (iii)은 부가 측정이다.
(ii)가 실패하면 논문의 헤드라인을 바꾼다 (§11 fallback).

---

## 3. 왜 naive swap이 실패했는가 — 근거

### 3.1 이미 측정된 4개의 실패 (전부 동일 checkpoint, weight update 0)

`bpebyte_br_greedy_root_1.3B @180k`, eval-time override
`regex_bpe_tokenizer_path=<tok> force_bpe_online_mode=greedy`
(재현 번들: `scripts/tokenizer_swap_repro/`, 결과: `PoC/tokenizer_swap_all_results.md`)

| swap | domain | metric | llama3 | swapped | Δ |
|---|---|---|---:|---:|---:|
| →Qwen2 | wiki-zh | BPB | **2.679** | 2.891 | **+0.212** |
| →Qwen2 | DCLM-EN (ref) | BPB | **0.828** | 0.876 | **+0.047** |
| →StarCoder2 | code multi (8 lang, paired 95% CI) | BPB | **0.994** [.96,1.03] | 1.331 [1.29,1.38] | **+0.337** [.32,.36] |
| →StarCoder2 | HumanEval-MC | acc_norm | **56.7** | 48.8 | **−7.9** |
| →BLOOM | wiki-te | BPB | **1.235** | 1.709 | **+0.474** |
| →Gemma | HS/ARC-E/ARC-C/PIQA 평균 | acc_norm | **60.2** | 57.9 | **−2.3** (HellaSwag −4.6 ≈ 9 stderr) |

독립 확인 하나 더 — **swap이 아니라 augment(추가)여도 똑같이 망가진다.**
`reports/compression_stability/bpb_augment_te.jsonl` : llama3 trie에 BLOOM의 U+0C00–0C7F만 이식
(`bpe_augment_tokenizer_path` / `bpe_augment_unicode_range`, `regex_cutting.py:103-104, 353-404`) →
Telugu CR 1.57 → 7.40 B/patch로 "압축은 개선"되지만 te BPB **1.2594 → 1.7248/1.7312 (+0.47)**,
en/zh는 불변(0.90797 → 0.90793/0.91500). 즉 **개선 방향으로 vocab을 고쳐도 손해**다.

### 3.2 원인 진단 — coverage도 granularity도 아니고 *boundary position 분포*다

결정적 증거는 **Gemma × English**다. llama3와 Gemma는 영어에서 **평균 granularity가 사실상 동일**
(원 보고 5.12 B/patch 동률; 이 계획에서 재측정 시 llama3 4.56 / gemma 4.40, §12 표1)한데도
downstream이 2–5 pt 떨어진다. 평균 patch 길이 가설·vocab coverage 가설 모두 이 한 셀에서 죽는다.

이 계획에서 그 축을 **정량화**했다 (신규 측정, §12 표3). 두 trie의 boundary 집합에 대한 Jaccard로
"경계가 얼마나 다른 곳에 찍히는가"를 잰다:

| 도메인 | vs llama3 | boundary disagreement (1−Jaccard) | 대응하는 측정된 손해 |
|---|---|---:|---|
| EN | qwen2 | **0.022** | EN BPB **+0.047** |
| EN | gemma | **0.101** | downstream **−2.3 acc_norm** |
| code | starcoder2 | **0.297** | code BPB **+0.337** |
| zh | qwen2 | **0.564** | zh BPB **+0.212** |
| EN | superbpe | **0.668** | (미측정) |

읽는 법: **경계의 2.2%만 옮겨도 홈 도메인 BPB가 0.047 오른다.** 이 민감도가 이 프로젝트의 실제
현상이며, 동시에 논문의 새로운 측정 축이다. (단조는 아니다 — zh는 disagreement가 높은데 손해는
code보다 작다. 도메인 난이도·기저 CR이 섞여 있어 단일 스칼라 법칙으로 쓰면 안 되고,
"동일 도메인 안에서 disagreement ↑ → 손해 ↑" 정도로만 쓴다.)

**부차 원인 — CR shift.** te augment는 CR을 4.7× (2.2 bit) 거칠게 만들면서 +0.47을 냈다. 즉 경계
위치 이동과 별개로 **patch가 굵어지는 방향은 추가 손해**다 (local decoder가 한 patch state에서
복원해야 할 byte 수가 학습 시 분포를 벗어남; `PoC/PoC_down_pooling.md`의 content-starvation과 같은 축).
반대로 despace처럼 **가늘어지는 방향**은 BPEByte에서 상대적으로 안전하다
(`reports/compression_stability/summary.md`: bpebyte en 4.58 → despace_p100 3.60, I=0.35).

### 3.3 반대편 증거 — **trained-in 이면 vocab 선택은 실제로 효과가 있다**

이게 이 방향을 살려두는 근거다.

- `reports/superbpe_level2_100M.html` (100M, law 21 GB): SuperBPE(200k, 72,713 superword,
  level-2가 level-1 안에 100% nesting)를 **level-2 prior로 학습에 넣으면** causal arm 기준
  BPB **1.085 → 1.032**, downstream all-6 **40.60 → 42.87**. 즉 *더 나은 vocabulary prior는
  학습에 들어가 있을 때 BPB와 downstream을 동시에 올린다.*
- 반대로 vocab **정체성**의 효과는 약하다: `reports/zh_cloze_100M_21G.md` — 중국어를 처음부터
  학습하면 rg-llama3 27.3 vs rg-qwen2 27.2 (**0.1 pt**). 같은 파일에서 **granularity** 축은
  1.5 pt 움직인다 (AU-Net word ~15 B/patch 26.2 vs AU-Net char ~3 B/patch 27.7).
  → "어떤 vocab이냐"보다 **"얼마나 굵게 자르냐"**가 지배적. sub-claim (a)는 vocab 이름 sweep이
  아니라 **CR을 통제한 sweep**으로 설계해야 한다 (§7 V-fit / rank truncation).

### 3.4 그래서 무엇이 참이어야 하는가

positive gain이 나오려면 다음 두 항의 부호 합이 양이어야 한다.

```
Δ(swap) =  −G(prior quality)   +   P(distribution shift)
            도메인 정합 trie가         경계 위치/굵기가 학습 분포를
            주는 진짜 이득            벗어나서 내는 벌점
```

현재 측정은 전부 `P ≫ G`다. 그리고 §3.3은 `G > 0`이 실재함을 보인다(SuperBPE, trained-in).
**따라서 논문의 기술적 기여는 `P → 0`을 만드는 것이고, 그것이 곧 boundary-invariant training이다.**
`P`를 0으로 못 만들면 어떤 vehicle을 골라도 gain은 나오지 않는다.

---

## 4. 후보 메커니즘 비교 (정직한 평가)

| # | 메커니즘 | 작동 원리 | null result가 적용되는가 | 최소 반증 실험 | 비용 | 판정 |
|---|---|---|---|---|---|---|
| **M1a** | **stochastic boundary dropout** — 학습 시 level_mask에서 경계를 확률 p로 삭제/삽입 | 실패 원인이 *경계 위치 binding*이므로 위치 자체를 흔든다. BPE-dropout/subword regularization의 직접 유사물 | **적용 안 됨** (학습 분포를 바꾼다) | 100M, p∈{0,0.1,0.2}; gemma-swap penalty와 EN BPB | 코드 ~20줄, 17 GPU-h/arm | **채택 (주력)** |
| **M1b** | **multi-trie sampling** — 문서마다 trie를 pool에서 추출 | 여러 vocab의 경계 분포를 동시에 학습 → segmentation-agnostic 표현 | 적용 안 됨 | 100M, pool={llama3,qwen2,gemma,starcoder2} | 코드 ~60줄, 17 GPU-h/arm | **채택 (보조/ablation)** |
| **M2** | **cheap boundary adaptation** — trunk freeze, byte enc/dec(=`layers[3,·]`의 앞 3층)만 새 trie로 파인튜닝 | binding은 주로 pooling 인터페이스에 있음. trunk의 지식은 보존 | 적용 안 됨. 단 "swap만으로 된다"는 주장은 포기 | 1.3B ckpt + code/SC2, 500~2000 step | **6,000 step = 33분 wall (4×B200) = 2.2 GPU-h**; adapter-only는 ~5분 | **채택 (비용 주장 담당)** |
| **M3** | **test-time trie 선택** — 입력마다 무학습 기준으로 trie 선택 | 선택지가 전부 손해면 선택할 것이 없다 | **그대로 적용됨 — 단독으로는 죽어 있다** | M1 통과 후에만 의미 | M1에 종속, 추가 ~0 | 조건부 (M1의 부산물) |
| **M4** | **efficiency knob** — 거친 trie로 trunk step 수를 줄이고 BPB와 교환 | 굵어지는 방향이 가장 위험한 방향(§3.2, te +0.47)이라 baseline에선 Pareto가 열등 | **그대로 적용됨** | M1 통과 후 rank-truncation sweep | ~0 (eval만) | 조건부 (M1의 부산물) |
| **M5** | **full continue-pretrain** (기존 `50_continue_pretrain.sh`) | 새 경계로 재학습하면 당연히 회복 | 적용 안 됨 | 이미 config 존재 | 2.2 GPU-h/6k step | **upper bound 참조용** (그 자체로는 논문 X — "재학습하면 된다"는 자명) |
| **M6** | **trie를 데이터로부터 fit** (V-fit: 목표 도메인에서 128k BPE 직접 학습) | off-the-shelf tokenizer는 vocab 크기·학습 도메인·CR이 전부 confound. 직접 fit하면 통제 가능 | 해당 없음(직접 개입) | CPU 1시간 | ~1 CPU-h | **채택 (sub-claim (a)의 올바른 형태)** |

### 각 후보에 대한 정직한 코멘트

- **M1a (boundary dropout)** — 가장 싸고 실패 축을 정확히 겨냥한다. 위험은 두 가지. (i) *tax*:
  in-distribution BPB가 나빠질 수 있다 (BPE-dropout도 downstream을 조금 깎는 경우가 있다).
  (ii) *과잉 일반화*: 경계를 무작위로 흔들면 모델이 아예 경계를 무시하고 byte-level LM처럼 되어
  hierarchy의 이득(=속도)만 잃을 수 있다. 그래서 게이트에 "EN BPB tax ≤ +0.010"을 반드시 넣는다.
  구현상 주의: eval 경로(`_online_levels_mask_bytes`, `regex_cutting.py:1067`)에는 dropout을 넣지
  않는다(학습 전용 정규화). `level_mask[0]=1`은 반드시 보존(`regex_cutting.py:702`).
- **M1b (multi-trie)** — M1a보다 "현실적인 대안 segmentation"을 학습시키므로 transfer가 더 잘 될
  가능성이 있으나, 비용은 같고 구현은 더 무겁다(§10). 그리고 pool에 없는 trie로의 일반화는
  보장되지 않는다 — 반면 M1a는 pool-free다. **M1a 우선, M1b는 ablation.**
- **M2** — "swap + N GPU-분 ≪ 283 GB 재학습"은 정직하고 실용적인 주장이고, 비용 수치가 실제로 강하다
  (전체 사전학습 180,000 step = **16.6 h wall / 66 B200-GPU-h**; adaptation 500 step = **~3분**).
  다만 이것만 있으면 논문이 "adapter 논문"이 되고 thesis(=vocab은 자유 hyperparameter)는 약해진다.
  M1의 보완재로만 쓴다.
- **M3** — 매력적이지만 §3.1이 사망 선고를 내린 상태다. M1이 통과해야 부활한다. 부활 시 무학습 기준
  후보: (i) 학습 CR과의 거리 |log2(CR_trie/CR_train)| 최소화 (완전 무료), (ii) 후보 K개에 대해
  모델 자신의 BPB 최소화 (K회 forward — 무학습이지만 무료는 아님).
- **M4** — baseline에서는 이미 진 싸움이다(굵게 = 가장 위험). M1 이후엔 rank-truncation으로
  **한 vocab 안에서 CR을 연속적으로 조절**할 수 있으므로(§12 표2: llama3 top-K 절단으로 EN CR
  2.32→4.56) confound 없는 Pareto 그림이 공짜로 나온다.
- **M5** — 이미 config가 커밋되어 있다(`bpebyte_1.3B_zh_continue_{llama3,qwen2}.yaml`,
  `scripts/tokenizer_swap_repro/50_continue_pretrain.sh`: init@180k, 6,000 step, lr 1e-4 + 1k
  re-warmup, 60/40 replay). 논문에는 "adaptation 상한선"으로만 등장시킨다.

**새로 제안하는 후보 — M7 (기각)**: *canonicalization* — 새 trie의 경계를 학습 trie의 경계 집합에
투영해서 "가장 가까운 llama3-호환 segmentation"을 만든다. 매력적으로 들리지만, 그러면 정의상
새 vocab의 정보를 버리는 것이므로 `G→0`이 된다. 기각.

---

## 5. 추천안

> **BPEByte-Inv** = (기존 아키텍처) + (학습 시 stochastic boundary dropout, M1a) 
> ⊕ (선택적) per-document multi-trie sampling (M1b) 
> ⊕ 추론 시 trie 교체, 필요하면 **local enc/dec만 수 분 adaptation** (M2)

핵심은 **M1a를 별도 arm이 아니라 baseline 레시피 자체로 삼는 것**이다. 그래야
"BPEByte-Inv가 BPEByte를 대체한다"는 서술이 가능하고, 표가 arm 폭발 없이 정리된다.

논문의 서사는 3단이다.

1. **현상 보고**: byte-in/byte-out인데도 모델은 학습 trie의 *경계 위치*에 묶여 있다.
   경계의 2.2%만 옮겨도 홈 도메인 BPB +0.047. (새 측정 축: boundary disagreement)
2. **처방**: boundary dropout으로 이 binding을 제거한다. tax는 ≤ X.
3. **수확**: 이제 vocabulary가 추론 시 자유 변수다 →
   (a) 코드 도메인에서 도메인 정합 trie가 llama3를 **이긴다**(headline),
   (b) SuperBPE로 trunk step 24% 절감이 품질 손해 없이 가능(Pareto),
   (c) trie 선택 규칙이 어떤 고정 trie보다 낫다.

---

## 6. Vehicle 결정 — Chinese를 왜 버리는가

| vehicle | 모델 역량 | 경계 차이 | 측정된 baseline penalty | downstream 지표 | 판정 |
|---|---|---|---|---|---|
| **code** (StarCoder2 / V-fit) | **있음** — code BPB 0.808(py)/0.994(multi) ≈ 영어 수준; HumanEval-MC 56.7 (chance 25) | disagree **0.297** | **+0.337 BPB [.32,.36]**, HumanEval-MC **−7.9** | HumanEval-MC (chance 대비 유의) | **1순위** |
| **EN × SuperBPE** | 최상 (홈 도메인, 전 벤치마크 가동) | disagree **0.668**, CR 4.56→5.67 (+24%) | 미측정 (반드시 먼저 측정) | HS/ARC/PIQA/Wino 전부 | **2순위** — trained-in 100M 근거 존재(§3.3) |
| **despaced / noised EN** | 최상 (despace grid 5 tasks × 3 region × 5 p 이미 존재) | llama3 vocab의 **45.2%가 leading-space 엔트리**(§12 표1) → despace 시 절반이 죽음 | 미측정 | despace macro-acc (rg all100 0.428) | 3순위 (robustness 축 강화용) |
| **Chinese** | **없음** — ceval/cmmlu/xcopa/xstorycloze 전부 chance, cloze도 구제 실패 | disagree 0.564 | +0.212 BPB | **없음(전부 chance)** | **기각** |
| **Telugu** | 없음 (llama3 완전 문자 0개) | 최대 | +0.474 BPB | 없음 | 기각 (극단점 참조용) |

**Chinese 기각 사유 (3개, 각각 독립적으로 치명적)**

1. **downstream이 존재하지 않는다.** `PoC/tokenizer_swap_all_results.md` §1,§5 —
   letter/cloze/xcopa/xstorycloze 전부 chance. user가 요구한 "Chinese downstream에서 Qwen2 trie가
   llama3 trie를 이긴다"는 **현재 checkpoint로는 측정 자체가 불가능**하다. continued pretraining이
   전제되는데, 그러면 논문의 주장이 "swap"에서 "재학습"으로 이동해 thesis가 무너진다.
2. **재학습해도 vocab 축이 안 움직인다.** `reports/zh_cloze_100M_21G.md` — 처음부터 중국어로
   full budget 학습하면 llama3 27.3 vs qwen2 27.2 (**0.1 pt**). 즉 중국어에서 "Chinese-aware trie가
   낫다"는 가설은 이미 **trained-in 조건에서조차 약하게 반증**되어 있다. 여기에 논문을 걸면 안 된다.
3. **효과 크기와 CI**: 중국어는 단일 BPB 점(+0.212). code는 8개 언어 paired CI(+0.337 [.32,.36])와
   accuracy 지표를 동시에 가진다. 반증 가능성이 훨씬 선명하다.

**code를 1순위로 놓는 이유**: 역량 있음 + penalty가 크고 CI가 좁음 + chance를 넘는 accuracy 지표
존재 + 새 pretraining 코퍼스 불필요(모델은 DCLM 영어로만 학습되었고 code는 transfer 도메인이므로
"재학습 없이 도메인 이식"이라는 서사와 정확히 일치) + "domain-matched vocabulary"가 코드에서
가장 자연스럽고 논쟁이 없는 프레이밍.

**주의**: StarCoder2를 그대로 쓰면 안 된다. SC2는 49k vocab이라 code에서 CR이 **낮다**
(3.58 vs llama3 4.19, §12 표1) → 품질이 같아도 연산이 손해다. **V-fit**(§7)으로 llama3와 동일한
128k 규모의 코드 BPE를 직접 학습해서 vocab 크기 confound를 제거하고 CR ≥ llama3를 확보한다.

---

## 7. 실험 매트릭스 (scale × arm × 비용)

측정된 단가: byte 모델 **B200 4GPU 0.332 s/step, 1.57 M bytes/step**
(`plans/chinese_continue_pretrain_plan.md:72`), A100 0.508 s/step (`reports/b4_pilot_estimate.md`).
100M law 레시피 실측 페이스 ~1.78 step/s (2 GPU) → 53,504 step ≈ **8.3 h ≈ 17 GPU-h**
(`reports/leaderboard_100M_training.md:57-78`).

### Stage 0 — 무학습 준비 (CPU 중심, ~1일)

| 작업 | 산출 | 비용 |
|---|---|---|
| S0.1 boundary disagreement 행렬 (trie쌍 × 도메인) | 새 측정 축, 논문 Fig.1 | CPU 분 (§12 표3 이미 시작) |
| S0.2 rank-truncation CR 곡선 (llama3 top-K) | CR 연속 knob (§12 표2) | CPU 분 |
| S0.3 **V-fit**: code / despaced-EN 코퍼스에서 128k BPE 직접 학습 (HF `tokenizers`) | confound 없는 domain-matched vocab | CPU ~1 h |
| S0.4 baseline swap penalty 보강 측정: EN×SuperBPE, EN×V-fit, code×V-fit (1.3B, eval only) | headline 표의 baseline 행 | ~1 GPU-h/셀 × 6 |
| S0.5 distilled trie arm 확인 (`tokenizer/boundary_v56/`, `data/boundary_distill.py`) | sub-claim (a)의 "entropy→trie" 다리 | ~0 |

### Stage 1 — 100M go/no-go (게이트, ~51 GPU-h)

레시피 고정: `lingua/portable_aunetlaw/configs/bpebyte_rg_100M.yaml` (dims [512,768], layers [3,10],
53,504 step, global batch 48, seq 8192, 21 GB, lr 3.4e-3/warmup 3000).

| arm | 변경점 | 비용 |
|---|---|---|
| **A0** | 기존 rg baseline (이미 학습됨, BPB 1.196) | 0 (재사용) |
| **A1** | + boundary dropout p_del=0.15, p_ins=0.05 | 17 GPU-h |
| **A2** | + boundary dropout p_del=0.30, p_ins=0.10 | 17 GPU-h |
| **A3** | + multi-trie sampling {llama3, qwen2, gemma, starcoder2} (per-doc md5 draw) | 17 GPU-h |

각 arm에 대해 **swap penalty 프로파일**을 잰다 (EN×gemma, EN×superbpe, code×SC2, code×V-fit,
zh×qwen2, te×bloom) — arm당 ~6 GPU-h. 총 Stage 1 ≈ **51 + 24 ≈ 75 GPU-h**.

### Stage 2 — 1.3B headline (게이트 통과 시, ~90 GPU-h)

| arm | 내용 | 비용 |
|---|---|---|
| **B0** | 기존 `bpebyte_br_greedy_root_1.3B@180k` | 0 (재사용) |
| **B1** | BPEByte-Inv 1.3B, 180,000 step, 283 GB, Stage 1 최적 p | **16.6 h wall / 66 GPU-h** |
| **B2** | B1 + local-only adaptation (trunk freeze, 500–2,000 step, 목표 도메인 + 새 trie) | **3–11분 wall / 0.2–0.7 GPU-h** per 도메인 |
| **B3** | (참조 상한) B0 + full continue-pretrain 6,000 step, 새 trie | 2.2 GPU-h per 도메인 |
| eval | headline 표 전 셀 + downstream 5-bench + HumanEval-MC + despace grid | ~20 GPU-h |

**총 프로그램 ≈ 190 GPU-h (B200 4GPU 노드 기준 약 2일 점유).** 300M 티어는 건너뛴다
(law 레시피 120,752 step ≈ 2.5–3일/arm으로 비용 대비 정보가 낮고, 100M→1.3B ladder는 이미
`reports/downstream_scaling.md`로 검증되어 있음).

---

## 8. Headline 표 mock

주장이 성립하려면 **굵은 칸의 부호**가 아래와 같아야 한다.

### 표 A (헤드라인) — "같은 가중치, trie만 교체"

| model | trie | EN BPB ↓ | EN avg4 acc ↑ | **code BPB ↓** | **HumanEval-MC ↑** | trunk steps/KB ↓ |
|---|---|---:|---:|---:|---:|---:|
| BPEByte (baseline) | llama3 (train) | 0.908 | 60.2 | 0.994 | 56.7 | 219 |
| BPEByte (baseline) | V-fit-code | *(=A)* | *(=B)* | **1.33** (측정됨, SC2) | **48.8** (측정됨) | — |
| **BPEByte-Inv (ours)** | llama3 | ≤ 0.918 (tax≤.010) | ≥ 59.2 | ≈ 0.99 | ≈ 56 | 219 |
| **BPEByte-Inv (ours)** | **V-fit-code** | — | — | **≤ 0.97** ✅ | **≥ 59** ✅ | ≤ 219 |
| BPEByte-Inv + 3분 adapt | V-fit-code | — | — | **≤ 0.94** | **≥ 61** | ≤ 219 |
| (참조) full continue-pretrain | V-fit-code | — | — | 상한 | 상한 | — |

**claim이 성립하는 조건**: 4행의 code BPB가 1행보다 **낮고**(≥0.02 차이, CI 비중첩),
HumanEval-MC가 **+2 pt 이상 높고**, 3행의 EN BPB tax가 +0.010 이하.
(1행 0.908/60.2, 2행 1.331/48.8은 **이미 측정된 실수**다. 나머지가 채워야 할 칸.)

### 표 B (메커니즘) — swap penalty가 사라지는가

| arm | EN×gemma ΔBPB | EN×superbpe ΔBPB | code×SC2 ΔBPB | zh×qwen2 ΔBPB | te×bloom ΔBPB |
|---|---:|---:|---:|---:|---:|
| baseline (측정치) | (미측정, acc −2.3) | 미측정 | **+0.337** | **+0.212** | **+0.474** |
| BPEByte-Inv | ≤ +0.05 | ≤ +0.05 | **≤ 0.00** | ≤ +0.08 | ≤ +0.19 |

### 그림 1 — swap penalty vs boundary disagreement

x축 = 1−Jaccard(§12 표3), y축 = ΔBPB. baseline은 가파른 상승 직선, Inv는 평평. 도메인별 마커.
이 그림 하나가 "현상 → 처방 → 효과"를 전부 담는다.

### 그림 2 — Pareto (M4)

x축 = bytes/patch (llama3 rank-truncation top-K로 연속 스윕: EN 2.32→4.56, §12 표2 + superbpe 5.67),
y축 = EN BPB. baseline 곡선은 학습점 밖에서 급격히 열화, Inv 곡선은 완만 → "trie는 추론 시
latency/quality 다이얼".

---

## 9. Go / No-go 게이트

### G0 (Stage 0, 무학습) — 진입 조건
- baseline의 **EN×SuperBPE swap penalty가 실제로 존재**해야 한다(ΔBPB > +0.02 또는 Δacc < −1.0).
  만약 여기서 penalty가 이미 0이면 §3의 진단이 부분적으로 틀린 것이므로 계획을 재작성한다.
- V-fit-code trie가 code에서 **CR ≥ llama3(4.19 B/patch)** 를 만족해야 한다. 아니면 vocab을 키운다.

### G1 (Stage 1, 100M) — 1.3B 진입 결정. **셋 다 통과해야 진행.**

| gate | 조건 | 근거/노이즈 기준 |
|---|---|---|
| **G1-a (invariance)** | 최소 한 arm에서 `code×SC2`와 `EN×gemma` 두 penalty가 **A0 대비 ≥60% 축소** | penalty 자체가 0.3 규모라 60%면 노이즈(±0.01)를 크게 상회 |
| **G1-b (no tax)** | 해당 arm의 100M in-distribution EN BPB ≤ A0 + **0.010** | 100M BPB 1.196; seed noise floor는 1.3B에서 ±0.005 (`reports/bpb_ci_1.3B.md`) |
| **G1-c (positive)** | 해당 arm에서 `code×V-fit`의 ΔBPB ≤ **−0.010** (즉 swap이 **이득**으로 부호 전환) | headline 주장의 100M 예고편 |

- G1-a·b 통과 / G1-c 실패 → **조건부 진행**. 1.3B는 돌리되 헤드라인을 §11의 F1으로 교체.
- G1-a 실패 → M1a 포기, **M2(adaptation)** 로 논문 축을 전환 (F2).
- G1-b만 실패(tax > 0.010) → p를 낮춰 재시도 1회, 그래도 실패면 M1b(multi-trie)로 교체.
  multi-trie는 "실제 존재하는 segmentation만" 학습시키므로 tax가 작을 가능성이 있다.

### G2 (Stage 2, 1.3B) — 논문 제출 결정
표 A의 4행이 1행을 **code BPB에서 −0.02 이상, HumanEval-MC에서 +2 pt 이상** 이기고,
3행의 EN 5-bench avg가 B0 대비 −1.0 pt 이내.

---

## 10. 구현 touch-points (코드)

전부 `lingua/apps/aunet/` 기준. **학습 경로와 eval 경로가 별도 구현이므로 항상 양쪽을 고쳐야 한다.**

| 항목 | 파일:라인 | 해야 할 일 |
|---|---|---|
| 학습 mask | `data/regex_cutting.py:654-703` (`online_levels_mask`) | dropout 적용 지점 (`bounds` 조립 직후 678-697). `level_mask[0]=1` 보존(702) |
| eval mask | `data/regex_cutting.py:1058-1148` (`_online_levels_mask_bytes`) | **dropout 미적용** (학습 전용). 새 trie 경로만 |
| per-doc 무작위 추출 템플릿 | `data/regex_cutting.py:705-723` (`_hybrid_sample_boundary`) | `md5(text) ^ seed` 기반 결정적 draw — multi-trie/dropout 모두 이 패턴을 복사 (resume 정확성 유지; `PrefetchState`에 RNG가 없음) |
| 단일 BPE stage 잠금 | `regex_cutting.py:661-663`, `739-741` | multi-trie는 stage 확장이 아니라 **동일 stage의 trie 교체**이므로 이 assert를 건드릴 필요 없음 (stacked 계획과 독립) |
| trie 캐시 | `regex_cutting.py:323-351` (`_byte_trie` 단일 프로퍼티) | trie **레지스트리**로 확장 (name→trie), 워커당 1회 빌드 |
| **seg cache 서명** | `regex_cutting.py:227-239` (`_seg_sig`), `1048` (호출 키) | dropout rate·trie pool·seed를 반드시 서명에 포함. 안 하면 `REGEX_SEG_CACHE`가 **옛 segmentation을 조용히 반환**한다 |
| vocab rank truncation | `data/byte_trie.py:330-364` (`build_byte_trie_from_tokenizer`) | **현재 truncation 옵션 없음.** `bpe_vocab_max_rank` 추가 (llama3 tiktoken은 파일의 rank 컬럼, HF는 merges 순서) |
| eval-time swap | `eval.py:124-138`, `641-663` | 이미 지원 (`regex_bpe_tokenizer_path` + **`force_bpe_online_mode=greedy` 필수** — 없으면 offline/leaky) |
| swap 재현 번들 | `scripts/tokenizer_swap_repro/{10_bpb_swap.sh,20_downstream_swap.sh,30_augment_not_swap.sh}` | 그대로 재사용 |
| adaptation(M2) | `train.py` | trunk freeze 플래그가 **없음** — `requires_grad=False` 를 level≥1 모듈에 거는 옵션 추가 필요 |
| continue-pretrain(M5) | `scripts/tokenizer_swap_repro/50_continue_pretrain.sh`, `configs/bpebyte_1.3B_zh_continue_{llama3,qwen2}.yaml` | 그대로 재사용 (init@180k, 6k step, lr 1e-4) |
| distilled trie | `data/boundary_distill.py:215-217,274-312`, `tokenizer/boundary_v56/prefix_free_vocab.txt` | sub-claim (a) "entropy patcher → trie" arm |

**함정 목록**
- `force_bpe_online_mode` 없이 `regex_bpe_tokenizer_path`만 주면 offline/leaky 경로로 떨어져 BPB가
  무너진다 (`PoC/tokenizer_swap_all_results.md` 말미).
- Gemma는 SentencePiece metaspace(`▁`, vocab의 48.9%)라 byte 복원 경로가 다르다. 이미
  `lingua/tokenizer.py:248-283`이 3가지 family(byte-level BPE / tiktoken / metaspace)를 자동 처리한다.
- online BPE trie 파싱은 **CPU-side**라 zh/code에서 data-bound가 될 수 있다
  (`reports/b4_pilot_estimate.md`). multi-trie는 워커 CPU 부하를 늘리므로 `prefetch_size`/워커 수 확인.

---

## 11. 리스크 / fallback

| 리스크 | 확률 | 완화 | 발생 시 |
|---|---|---|---|
| **R1 tax** — invariance가 in-distribution 성능을 깎는다 | **중** | p sweep, multi-trie로 교체 | p를 낮춰 "부분 invariance + 3분 adaptation" 조합으로 전환 |
| **R2 gain 없음** — invariance는 성공(P→0)했는데 어떤 trie도 llama3를 못 이긴다(G=0) | **높음** — `zh_cloze_100M_21G.md`가 이미 vocab 정체성 효과 0.1 pt를 보고 | V-fit으로 진짜 도메인 정합 vocab을 만들고, CR을 통제해 granularity 이득(1.5 pt 축)을 노린다 | → **F1** |
| **R3 disagreement 축이 단조가 아님** | 중 (이미 zh/code에서 역전) | 도메인 내 비교로 한정, 도메인 간에는 회귀 대신 순위만 주장 | 그림 1을 도메인별 패널로 분리 |
| **R4 hierarchy 붕괴** — dropout이 세면 모델이 경계를 무시 | 중 | bytes/patch 및 trunk step 수를 학습 중 로깅 | p 축소 |
| **R5 100M→1.3B 비전이** | 중 | 100M에서 penalty 축소율이 60% 이상일 때만 진행 | Stage 2 중단, 100M/300M 결과로 workshop |
| **R6 CPU 병목** | 낮 | trie 레지스트리 사전 빌드 + seg-cache | 워커 증설 |

### Fallback 논문 (우선순위 순)

- **F1 — "Boundary-distribution binding in hierarchical byte models"** (R2 발생 시).
  gain이 0이어도 **현상 자체가 새롭고 반직관적**이다: byte-in/byte-out 모델인데 경계의 2.2%만
  옮겨도 홈 도메인 BPB가 +0.047 오른다. 기여는 (i) 4개 도메인 × 5개 vocab의 swap-penalty 지도,
  (ii) boundary-disagreement라는 예측 지표, (iii) dropout으로 binding을 제거하면 penalty가 사라진다는
  처방, (iv) "그럼에도 어떤 vocab도 이기지 못한다"는 negative result. 이건 여전히 논문이다.
  **표 A의 3행(tax)과 표 B만 있어도 성립한다.**
- **F2 — "Retargeting a byte LM in minutes"** (R2 + G1-a 실패 시).
  M2 중심. "전체 사전학습 66 GPU-h vs local-only adaptation 3분"이라는 비용 대비가 실측으로 강하다.
  단 thesis는 "vocab은 자유 hyperparameter"에서 "vocab은 싸게 갈아끼울 수 있는 hyperparameter"로 약화.
- **F3 — compression-stability 확장** (`plans/plan_compression_stability.md`와 병합).
  학습 0. parser별 CR 안정성 + vocab augment 복구 arm. 이미 데이터가 대부분 존재
  (`reports/compression_stability/`). AAAI 본문 motivation 실험으로 흡수 가능.

---

## 12. 부록 — 이 계획 작성 중 수행한 신규 측정

전부 CPU, `ByteTrie.greedy_tokenize_boundaries` (= 논문의 OnlineBPE greedy/root) 직접 호출.
윈도우는 기존 프로토콜 재사용 (`reports/bpb_windows.jsonl` 등, 비중첩 ≤3,840 B).
**아직 스크립트로 커밋되지 않았다 — `scripts/probes/boundary_disagreement.py` 로 커밋할 것.**

### 표1 — vocab 조성 (전 엔트리) 및 도메인별 CR (bytes/patch, 윈도우 120개)

| vocab | 크기 | leading-space % | inner-space % | EN | zh | code | te | despace_p100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3 | 128,000 | **45.2** | 0.3 | **4.56** | 2.86 | 4.20 | 1.57 | 3.60 |
| llama3_superbpe | 200,000 | 59.3 | **32.2** | **5.67** | 2.84 | 4.22 | 1.98 | 3.69 |
| qwen2 | 151,643 | 35.0 | 0.2 | 4.48 | 3.03 | 4.10 | 1.84 | 3.53 |
| starcoder2 | 49,152 | 32.8 | 0.9 | 3.95 | 2.53 | **3.58** | 2.09 | 3.21 |
| bloom | 250,680 | 55.6 | 0.3 | 4.52 | 3.42 | 3.87 | **5.46** | 3.38 |
| gemma (metaspace) | 80,847 | 48.9 (`▁`) | — | 4.40 | 3.08 | 3.41 | 3.77 | 3.59 |
| polyglot-ko | 30,000 | 64.4 | 0.0 | 2.02 | 1.70 | 1.53 | 1.01 | 1.82 |
| llama3 + no-space 변형(합집합) | 185,878 | — | — | 4.58 | 2.86 | 4.23 | 1.57 | **3.90** |

읽을 점 3개.
(1) llama3 엔트리의 **45.2%가 leading space**로 시작 → despace 시 vocab의 절반이 무효화되고 CR이
4.56→3.60으로 떨어진다. despace 취약성의 parser-level 원인이다.
(2) leading-space를 벗긴 변형을 **추가**만 해도 EN은 불변(4.56→4.58)이고 despace만 3.60→3.90으로
회복된다 — 도메인 국소적 augment가 가능함을 보여준다(단 §3.1의 te 사례처럼 BPB는 별도 확인 필요).
(3) StarCoder2는 code에서 오히려 CR이 **낮다** → 그대로 쓰면 안 되는 이유(§6).
llama3 EN 4.56은 `reports/compression_stability/summary.md`의 4.58과 일치(게이트 통과).

### 표2 — llama3 merge-rank top-K 절단에 따른 CR (윈도우 100개)

| \|V\| | EN | zh | code | despace_p100 |
|---:|---:|---:|---:|---:|
| 1,000 | 2.32 | 1.02 | 2.16 | 1.90 |
| 4,000 | 3.14 | 1.10 | 2.84 | 2.37 |
| 16,000 | 3.94 | 1.38 | 3.57 | 2.86 |
| 32,000 | 4.22 | 1.71 | 3.84 | 3.12 |
| 64,000 | 4.45 | 2.06 | 4.07 | 3.43 |
| 128,000 | 4.56 | 2.86 | 4.19 | 3.61 |

→ **한 vocab 안에서 CR을 연속으로 조절하는 confound-free knob**이 존재한다.
merge rank는 그대로 두고 크기만 바꾸므로 "vocab 크기" 축과 "vocab 도메인" 축을 분리할 수 있다.
(`byte_trie.py`에 truncation 옵션이 없으므로 `bpe_vocab_max_rank` 추가 필요, §10.)

### 표3 — boundary disagreement (1 − Jaccard(경계집합), vs llama3, 윈도우 80개)

| vs llama3 | EN | code | zh |
|---|---:|---:|---:|
| qwen2 | **0.022** | 0.028 | 0.564 |
| gemma | 0.101 | 0.268 | 0.578 |
| starcoder2 | 0.177 | **0.297** | 0.408 |
| bloom | 0.229 | 0.420 | 0.674 |
| superbpe | 0.668 | 0.335 | 0.065 |

이 표와 §3.1의 손해를 짝지으면 §3.2의 진단이 된다. 특히 **qwen2/EN = 0.022 ↔ EN BPB +0.047**.

### 재현 방법 (스케치)

```python
import sys; sys.path.insert(0, "lingua")
from apps.aunet.data.byte_trie import ByteTrie
# 1) vocab → raw bytes: llama3=base64(tokenizer.model), HF=gpt2 byte-decoder 역변환,
#    gemma=metaspace('▁'→' ');  lingua/lingua/tokenizer.py:248-283 과 동일 규약
# 2) trie.insert(each) ; 3) trie.greedy_tokenize_boundaries(window_bytes)
# 4) CR = total_bytes / total_patches ("shared boundary byte counted once")
# 5) disagreement = 1 - |B_a ∩ B_b| / |B_a ∪ B_b|
```

---

## 13. 다음 행동 (순서)

1. `scripts/probes/boundary_disagreement.py` 커밋 (§12 재현).
2. **S0.4 실행** — 1.3B baseline에 EN×SuperBPE / EN×V-fit swap penalty 측정 (~6 GPU-h).
   G0 게이트. 여기서 penalty가 없으면 계획 전체를 재작성한다.
3. **V-fit** — code 코퍼스(`data/code_bpb`, `data/code_multi`)에서 128k BPE 학습 (CPU 1h).
4. `regex_cutting.py`에 boundary dropout 구현 + `_seg_sig` 갱신 + 단위 테스트
   (train/eval mask 분기, `level_mask[0]` 보존, seg-cache 무효화).
5. **Stage 1 (100M) 3 arm 발사** → G1 판정.
6. G1 통과 시 Stage 2 (1.3B, 16.6 h wall) 1 arm.
