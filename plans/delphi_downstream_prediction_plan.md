# Delphi 방식 downstream 예측 재현 — BPEByte-rg

원문: [openathena.ai/blog/delphi](https://openathena.ai/blog/delphi/) §"Predictions are
accurate for downstream evals as well"

## 1. 배경과 목표

Delphi 는 downstream 점수를 **2단계**로 예측한다. 객관식 정확도는 "정답이 모든 오답보다
확률이 높아지는" 순간에만 움직이는 이산량이라, 모델이 계속 좋아져도 점수는 chance 근처에
평평하게 눌려 있다.

1. **soft metric** — compute 에 대해 매끄럽게 변하는 연속 대리지표 (객관식: 오답 대비 정답의
   로그확률 / 생성: reference completion 의 BPB) 에 scaling law 를 fit 하고 외삽
2. **sigmoid** — 관측 모델 풀에서 soft → hard(정확도) 매핑을 fit 하고 둘을 합성

### 헤드라인 주장 (재설정)

처음에는 "기존 hard-metric fit 의 MAE 3.3 pp 를 이긴다"를 목표로 잡았으나, 데이터를 보고
**더 방어 가능하고 더 흥미로운 주장**으로 바꾼다:

> **soft metric 은 6개 태스크 전부에서 compute 에 대해 단조롭게 개선된다 — 정확도가 단조롭지
> 않은 태스크(BoolQ, ARC-C, WinoGrande)에서도 그렇다. 그리고 합성 예측은 그 태스크들에 대해
> 애초에 예측값을 낼 수 있는 유일한 방법이다.**

이유: BPEByte-rg 의 BoolQ 정확도는 사다리를 따라 **비단조**이고 (아래), 99M ARC-C 는
chance(25.02%) **미만**, WinoGrande 는 296M→680M 사이에서 binomial SE(1.40 pp)의 1/14 만
움직인다. 단조 hard-metric scaling law 로는 BoolQ 를 애초에 fit 할 수 없다.
Schaeffer 류의 "이산화가 연속적 진전을 가린다"의 교과서적 사례다.

MAE 비교도 물론 한다 (§8). 단 **macro-average binomial 노이즈 바닥이 ≈0.44 pp** 이므로
**0.6 pp 미만의 개선은 측정 불가**임을 미리 명시한다.

### 기존 측정치의 출처 불일치 (반드시 재측정해야 하는 이유)

BPEByte-rg 0-shot 정확도가 저장소 안에서 **서로 다르게 기록돼 있다**:

| 출처 | 100M | 300M |
|---|---|---|
| `reports/downstream_data.json` | 30.7 / 34.6 / 23.5 / 59.4 / 39.1 / 50.2 | 31.5 / 34.7 / 23.9 / 61.7 / 38.7 / 51.6 |
| `reports/scaling_downstream_table.md` | 30.7 / 34.6 / 23.5 / 59.4 / 39.1 / 50.2 | **37.8 / 38.5 / 25.2 / 64.8 / 49.1 / 51.8** |

(순서: hellaswag / arc_easy / arc_challenge / piqa / boolq / winogrande)

300M 행이 크게 다르다. 저장소에 300M rg 런이 둘(`rg_300M` 44630스텝 23.4 GB,
`rg_300M_adhoc` 9900스텝 62.3 GB) 있고, 과거 평가들이 수개월에 걸쳐 서로 다른 경계 플래그
아래에서 돌았기 때문이다 (`reports/performance_summary_per_scale.md` 의 "three measurement
regimes" 경고). **따라서 하나의 고정 eval config 로 전 체크포인트를 재측정한다 (§7.3).**
위 숫자들은 동기 부여용으로만 쓰고, 논문 숫자는 전부 이번 스윕에서 나온다.

### 사용자 확정 범위

BPEByte-rg 단일 패밀리 · 6-bench 0-shot (hellaswag, arc_easy, arc_challenge, piqa,
winogrande, boolq) · sigmoid 는 **우리 체크포인트(중간 스텝 포함)** 로 fit · **1.3B 런 전체 held-out**.

---

## 2. 체크포인트 인벤토리 (전수 검증 완료)

`D` = 학습 소비 바이트(`optim/total_tokens`; byte 모델이라 base token 1 = 1 byte).
`lr%` = `lr / lr_max`. **굵게** = γ=10 iso-ratio rung (~210 bytes/param).

모든 런이 `scheduler: cosine`, `lr_min_ratio: 0.01`, `cycle_length: 1.0` 이므로
**각 런의 마지막 체크포인트만 anneal 이 끝났다.** 중간 체크포인트는 cosine 중턱(LR 이 최대의
42~88 %)이라 같은 D 로 예산이 잡힌 런보다 체계적으로 나쁘다 — Chinchilla 의 고전적
"Approach 1 vs 중간 체크포인트" 편향이다.

### fit 집합 = **anneal 끝난 ≤760M 7점**

| N | D (GB) | D/N | C (FLOPs) | 위치 |
|---|---|---|---|---|
| 98.59M | 10.519 | 106.7 | 3.86e18 | `runs/small/ablation_100M/v4_root_greedy/checkpoints/0000001672` |
| 98.59M | **21.039** | 213.4 | **7.72e18** | `runs/main/100M_adhoc/rg_100M/checkpoints/0000003344` ✅ ece 에서 fetch 완료 |
| 98.59M | 42.077 | 426.8 | 1.54e19 | `runs/small/cmp_100M/v4_root_greedy_ot/checkpoints/0000006688` |
| 98.59M | 84.154 | 853.6 | 3.09e19 | `runs/others/superseded/cmp_100M_ratio20/v4_root_greedy/checkpoints/0000013376` |
| 296.0M | **62.285** | 210.4 | **5.60e19** | `runs/main/300M/rg_300M_adhoc/checkpoints/0000009900` ✅ ece 에서 fetch 완료 |
| 679.6M | 68.891 | 101.4 | 1.41e20 | `main/main/others/760M/bpebyte_root_greedy_760M_bpp101/checkpoints/0000029200` |
| 679.6M | **142.974** | 210.4 | **2.93e20** | `main/main/760M/rg_760M/checkpoints/0000060600` |

`cmp_100M_ratio20` 은 **2단계 cosine** 이라 @6688 도 완전 감쇠 endpoint 이지만, D 가
`cmp_100M_ot@6688` 과 동일(42.077 GB)하므로 fit 에는 하나만 넣고 나머지 하나는 §7.2 의
**anneal 보너스 통제쌍**으로 쓴다 (같은 아키텍처·`seed: 777`·`data.seed: 42`, LR 스케줄만 다름).

### link pool = **≤760M 전 체크포인트, de-dup 후 ~18점**

de-dup 규칙: 한 런 안에서 D 가 직전 채택점의 1.15× 이상 차이 날 때만 채택, 단 각 런의
최종(annealed) 체크포인트는 무조건 채택.

- 99M: `ablation@1672`, `cmp_g10@3344`, `cmp_100M_ot@6688`, `ratio20@{6688, 8360, 10032, 11704, 13376}` → **8**
- 296M: `adhoc@9900` → **1**
- 680M: `bpp101@{8760, 17520, 24000, 28000, 29200}`, `rg_760M@{18180, 36360, 54540, 60600}` → **9**

**총 ~18점 / 독립 런 7개.** 불확실성의 유효 표본수는 18도 25도 아니고 **런 7개**다 — 논문에 명시.

### held-out = 1.3B (`main/main/1.3B/bpebyte_br_greedy_root_1.3B`)

| step | D (GB) | C (FLOPs) | lr% | 역할 |
|---|---|---|---|---|
| 54000 / 108000 / 162000 | 84.94 / 169.87 / 254.81 | 3.05 / 6.10 / 9.15e20 | mid | **보조 패널** (§6.2) |
| **180000** | **283.115** | **1.017e21** | 0.010 | **헤드라인 held-out** |

### 검증된 사실

- **경계 스킴이 전부 동일**: `{bpe_br: 1@1}`, `bpe_online: true`, `greedy`, `root`,
  `committed_view false`, `context_prefix 0`, `vocab_size 258`. 아키텍처
  `[512,768]/[3,10]` (100M) · `[512,1280]/[3,13]` (300M) · `[512,1536]/[3,24]` (760M) ·
  `[512,2048]/[3,25]` (1.3B). byte 모델 평가의 단골 교란인 경계-스킴 차이가 **없다**.
- ece 의 `cmp_g10/{v4_root_greedy, rg_300M}` 은 로컬 `runs/main/{100M_adhoc/rg_100M,
  300M/rg_300M_adhoc}` 과 마지막 step·total_tokens·loss 가 완전히 동일 — 같은 런이다.
  그래서 가중치를 그 메타데이터 옆에 받았고, 로드·아키텍처·regex 검증까지 마쳤다.
- 100M step-6688 두 체크포인트는 크기·스텝이 같지만 **md5 가 다르다** → 통제쌍.
- `runs/r10lr_collected/*/checkpoints/*` 는 8 KB **빈 스텁**(가중치 삭제됨).

---

## 3. FLOPs 축 — "측정"이 아니라 **해석적** 값

`apps/aunet/train.py:608` 이 `FLOPS = wps × model.flops_per_token()` 로 로깅하므로
`median(speed/FLOPS ÷ speed/wps)` 는 측정값이 아니라 **config 만의 함수인 해석적 상수**다
(모든 스텝에서 bit-identical). `6ND` 는 계층형 byte 모델을 1.5–2.2× 과대평가하므로 이 값을
쓰되 "analytic FLOPs/byte" 라 부른다.

| N | FLOPs/byte | 비고 |
|---|---|---|
| 98.59M | 3.669e8 | `total_flops.md` 의 추정식 `46·N^0.866` 은 3.85e8 로 **4.7 % 높다** |
| 296.0M | 8.984e8 | 추정식 9.98e8 로 **10 % 높다** |
| 679.6M | 2.049e9 | |
| 1.300B | 3.593e9 | |

→ **추정식 `fpt(N)=46·N^0.866` 은 폐기하고 측정된 4개 값을 쓴다.**

`flops_per_token` 은 상위 스테이지 길이에 `max_seqlens`(예산 패치 수)를 쓰므로 실제 패치가
더 짧으면 C 는 상한이다. 사다리 전체에 일관되게 적용되어 상대 비교에서는 상쇄되지만
절대 ZFLOP 수치는 이 성질을 물려받는다.

### 외삽 거리 — 정직하게

- fit 구간 compute: 3.86e18 → 2.93e20 = **75.9× (1.88 decade)**
- 헤드라인 held-out: 1.017e21 = 최대 fit 점 대비 **3.47× (0.54 decade)**, N 은 1.91×, D 는 1.98×

> **Delphi 의 헤드라인은 "2 decade 외삽"이다. 우리는 그 1/4 decade 를 한다.
> 그들의 프레이밍을 빌려오지 않는다.** ±0.5 % → ±4 % 로 넓어지는 CI 그림도 재현 불가 —
> 대신 §6.4 의 중첩 fit-set 버전을 만든다.

---

## 4. Soft metric — 정확한 정의

### 4.1 추출

`harness.log_samples: true` → `simple_evaluate` 반환 dict 의 `"samples"` 키가
`{task: [record]}` 이고 `apps/aunet/eval.py:800` 이 dict 전체를 덤프하므로 `results.json`
안에 들어온다. 실물 확인 완료(`.../cmp_100M_ratio20/v4_root_greedy/eval_hs_samples/results.json`,
41 MB, hellaswag 10042건).

각 레코드: `doc_id, doc, target, arguments[(ctx, continuation)], resps,
filtered_resps[[loglik, is_greedy]], acc, acc_norm`.

**즉시 compact 테이블로 변환**하고 원본은 gzip 보관. 컬럼:
`doc_id, K, gold, ll[K](float64), b_cont[K], b_choice[K], c_choice[K], is_greedy[K]`.
원본은 체크포인트당 ~58 MB × 25 ≈ 1.45 GB, compact 은 전체 ~24 MB.

**길이 정의 — 정확히 이대로**:
- `b_cont[c] = len(arguments[c][1].encode())` — 모델이 실제로 채점한 바이트, `target_delimiter`(`" "`) **포함** (`task.py:1390`)
- `b_choice[c] = len(choice.encode())` — 구분자 **제외**. 패치된 `acc_bytes` 가 쓰는 값 (`task.py:1495`)
- `c_choice[c] = len(choice)` — 문자 수. **`acc_norm` 이 쓰는 값** (`task.py:1494`)

구분자 1바이트 비대칭은 실재한다: ARC 는 continuation 중앙값이 18 B 라 정규화항의 5.5 % —
argmax 를 뒤집기 충분하다. **정규화항은 `b_choice`(또는 `c_choice`)를 쓴다** — soft metric 의
argmax 가 헤드라인 지표의 판정 규칙을 정확히 재현하도록. 이 불일치는 "고치지 말고" 기록한다
(고치면 대리하려는 지표와 분리된다). 태스크별로 `argmax(ll/b_choice) ≠ argmax(ll/c_choice)`
비율을 1회 검증하고 0.5 % 초과 시 `c_choice` 로 통일.

### 4.2 ⚠️ 순진한 per-byte softmax 는 **망가진다** — 반드시 길이 균등화

`softmax(ℓ_c / b_c)` 를 그대로 쓰면 **안 된다.** hellaswag 는 정답/오답의 per-byte rate 차이가
~0.03–0.1 nats/byte 라, `ℓ/b` 위의 softmax 는 거의 균등해지고 **모든 모델에서 X ≈ log 4** 가
된다. 지표가 사다리 전체에서 거의 상수가 되어 신호가 사라진다. (이것이 이 문서에서 가장
조용히 망가지기 쉬운 지점이다.)

**해법 — 문서별 길이 균등화 loglik**:

```
b̄_d = (1/K_d) Σ_c b_{d,c}                 # 그 문서의 기준 길이
ℓ̃_{d,c} = ℓ_{d,c} · (b̄_d / b_{d,c})       # argmax 는 보존, nat 스케일은 복원
p̃_{d,c} = softmax_c(ℓ̃_{d,·})
x_d = −log p̃_{d,gold}   (≥ 0 nats)
X_task = (1/n) Σ_d x_d                     # ← headline soft metric
```

모든 `z_c` 에 같은 양수를 곱하는 것이므로 `acc_norm` 의 argmax 는 그대로 보존되고, nat
스케일만 되살아난다. 문서 안 선택지 길이가 같으면 `b̄_d/b_c = 1` 이라 raw 형과 **정확히 일치**
한다 — WinoGrande(길이 분산 0)와 PIQA(중앙값 3 B)는 사실상 영향이 없고, hellaswag(중앙값
69 B)만 실제로 이 선택에 의존한다.

**`acc` 가 헤드라인인 태스크(winogrande, boolq)는 균등화 없이 raw `ℓ` 로 softmax.**
규칙: *soft metric 은 그 태스크의 헤드라인 지표가 쓰는 정규화를 그대로 쓴다.*

chance 수준은 `X_chance = E_d[log K_d]`, 바닥 `E ≥ 0` 은 벤치마크 레이블 노이즈.

**집계는 로그 공간(= x_d 의 평균).** 정답 확률의 기하평균의 음로그라 proper scoring rule 이고,
저장소가 이미 BPB 에 쓰는 가법 Chinchilla 형에 그대로 꽂힌다. 확률공간 평균
`p̄ = (1/n)Σ p̃_{d,gold}` 는 정확도처럼 **포화**해서 선형화 목적을 무너뜨린다 — 보조 진단으로만.

### 4.3 대안들을 기각/채택한 이유

| 후보 | 판정 |
|---|---|
| (a) raw `ℓ_g − logsumexp(ℓ_c)` | Delphi 가 말한 그 지표. **`acc` 태스크에는 이걸 쓴다.** 하지만 hellaswag 의 헤드라인은 `acc_norm` 이고 760M 에서 acc 43.5 vs acc_norm 52.2 로 8.7 pp 벌어진다 — `acc` 모양 대리지표로 `acc_norm` 을 매개하면 불필요한 link 노이즈가 낀다. 사전등록된 robustness 변형으로만. |
| (b) `softmax(ℓ_c/b_c)` | **망가짐. 사용 금지.** (§4.2) |
| (c) margin `ℓ_g − logsumexp_{c≠g} ℓ_c` | `logit(p̃_g)` 의 단조 재매개변수라 정보량은 (a)와 동일하지만 문서 평균이 proper scoring rule 이 아니고 꼬리가 두껍다. 주지표에서 기각. |
| (d) 정답 continuation 의 BPB | 오답을 완전히 버려서 정확도를 결정하는 판별 정보를 날린다. **baseline B5** 로 유지 — OLMo ladder 의 "task loss"(Bhagia et al. 2024)에 해당하는 선행연구 경쟁자다. |

**꼬리 가드**: (model, task) 마다 `x_d` 의 평균 / 1 % 절사평균 / 99 백분위를 보고,
`|X − X_trim1%| < 0.02 nats` 를 요구. 위반이면 소수의 파국적 문서가 지표를 끌고 있는 것이므로
fit 전에 조사한다.

내부 계산은 nats, 표에는 `X/ln2` (bits) 병기.

### 4.4 태스크별 함정 (전부 실물 확인)

| task | n | K 분포 | chance `E[1/K]` | `X_chance = E[log K]` | continuation bytes 중앙값 | 문서내 길이 분산(중앙값) | 헤드라인 |
|---|---|---|---|---|---|---|---|
| hellaswag | 10042 | 4 | 0.2500 | 1.3863 | 147 | **69** | acc_norm |
| arc_easy | 2376 | 4×2365, 3×7, 5×4 | 0.2502 | 1.3858 | 18 | 7 | acc_norm |
| arc_challenge | 1172 | 4×1165, 3×4, 5×3 | 0.2502 | 1.3859 | 28 | 8 | acc_norm |
| piqa | 1838 | 2 | 0.5000 | 0.6931 | 66 | **3** | acc_norm |
| winogrande | 1267 | 2 (multiple_input) | 0.5000 | 0.6931 | 22 | **0** | acc |
| boolq | 3270 | 2 | 0.5000 (다수결 **0.6217**) | 0.6931 | `" no"`=3 / `" yes"`=4 | 1 | acc |

- **ARC 의 가변 K**: 문서별 실제 `K_d` 로 `x_d` 계산, chance 는 `E_d[log K_d]`.
  교차태스크 비교용으로는 **excess 형 `x*_d = x_d − log K_d`** (chance = 0)를 함께 보고.
- **WinoGrande 는 `multiple_input`** (`doc_to_text` 가 int 반환, `task.py:822`): `target` 이
  문자열이고 `arguments[i][1]` 이 전부 동일하며 **context 쪽이 다르다**. 정답은
  `int(doc["answer"]) − 1`. 따라서 점수는 `log P(suffix|prefix_a)` vs `log P(suffix|prefix_b)`
  — **정답에 대한 사후확률이 아니라 문맥 간 우도비**다. 논문에 그렇게 쓴다. acc ≡ acc_norm.
- **BoolQ**: 레이블 사전분포가 **62.17 % yes** 이고 rg 정확도는 chance 아래로도 내려간다.
  yes/no 편향이 X 를 오염시키므로 **사전등록 보조지표**로 무조건부 정규화
  `ℓ_c − ℓ_c^uncond` (lm-eval 의 `acc_mutual_info` 기계, `task.py:1497-1508,1571-1578`;
  `aux_arguments` 경로로 요청 2배)를 함께 낸다. 내부 CV(§5.5)가 어느 쪽을 고르는지 보고.
  프롬프트 바이트는 중앙값 566 / p90 1012 / 최대 4847 로 8192 안 — **절단 교란 없음** (확인 완료).
- **`acc_bytes` 는 우리 6-bench 에서 출력되지 않는다** (`mmlu_text`/`global_piqa` 에만 등록).
  직접 계산하므로 무관.

### 4.5 정밀도 caveat (실측)

`generate.py:558` 이 bf16 logits 로 `-F.cross_entropy(...)` 를 계산하고 `eval.py` 가
`ll[p_len:].sum()` 으로 bf16 누적한다. 저장된 샘플이 그대로 보여준다 (`-121.5, -114.0,
-70.0, -77.0` — 0.25~0.5 배수).

| continuation | bf16 누적 | fp32 누적 |
|---|---|---|
| 40 B | ±0.071 nats | ±0.015 |
| 120 B | ±0.188 nats | ±0.025 |
| 300 B | ±0.575 nats | ±0.041 |

편향은 없어 1만건 평균에는 ≈0.003 nats 영향뿐이지만 **§4.6 의 목표 정밀도가 0.010–0.033
nats** 이므로 무시할 수 없다. `eval.py` 의 합을 `ll[p_len:].float().sum()` 으로 (선택적으로
`generate.py:558` 을 `x.float()` 로) 바꾼다. **게이트 대상 변경** — §7.1 게이트를 다시 돌려
정확도가 안 변하는 걸 확인한 뒤 채택.

### 4.6 사전 타당성 계산 (Go/No-Go) — 먼저 종이 위에서

바닥 고정 sigmoid 에서 `d â/dX = −k(1−p_c)σ(1−σ)`, 최대 `−k(1−p_c)/4`.
hellaswag 기준 실증 앵커: 정확도가 28.6 → 62.5 (Δ=0.339) 움직이는 동안 X 는 대략
1.37 → 1.02 nats (Δ≈−0.35) ⇒ **|dacc/dX| ≈ 0.95–1.1 pp / 0.01 nat**.

- 기존 3.3 pp 를 맞추려면 **|X̂ − X| ≲ 0.033 nats ≈ 상대 2.7 %** (Delphi 의 0.5 % 보다 5배 느슨)
- ≤1 pp "wow" 결과를 내려면 **≲ 0.010 nats ≈ 0.85 %** (Delphi 급 정밀도)

**판정: GO.** 단 사전 스윕 후 태스크별 실측 동적 범위로 재계산하고, 필요한 soft 정밀도가
재현성 바닥(§7.4, 0.003 nats)보다 낮게 나오는 태스크는 중단한다.

---

## 5. Fit / link / 누출 규칙

### 5.1 축 — 우선순위를 뒤집는다

`fpt` 이 config 만의 함수이므로 `C = fpt(N)·D` 는 **(N,D) 평면의 1차원 사영**이고, fit 점이
전부 한 궤적 위에 있을 때만 식별된다. 우리 점들은 그렇지 않다 — 99M 행이 고정 N 에서 D 를
8× 훑고, 680M 행에 D 가 둘 있다. 게다가 fpt 은 대략 `N^0.87` 이라 `C ∝ N^0.87 D` 이고,
C 단일축 fit 은 데이터가 지지하지 않는 N↔D 대체율을 암묵적으로 못박는다.

> **초기 계획의 "γ=10 rung 을 headline, off-ratio 를 robustness" 는 거꾸로다.**
> γ=10 rung 은 3점인데 포화 멱법칙은 3파라미터 — **자유도 0**. 완벽 보간이라 `E` 가 전혀
> 식별되지 않고 fit 품질에 대한 정보가 없다. (`reports/downstream_scaling.md` 의
> "L(1.3B)=0.605 예측, 실제 0.605" 도 4점 3파라미터라 근사-정확함이 검증이 아니다.)
> **off-ratio 점들이 하중을 받는 본체다.** 99M D-스윕과 680M@68.9 가 없으면 식별 자체가 안 된다.

**주 fit — 가법 Chinchilla 형, (N,D) 결합**

```
X_t(N,D) = E_t + A_t·N^(−α) + B_t·D^(−β)
```

anneal 끝난 ≤760M **7점**에, **α·β 는 6개 태스크에서 공유**, `(E_t, A_t, B_t)` 는 태스크별.
파라미터 2 + 18 = 20, 데이터 6×7 = 42 ⇒ **자유도 22** (태스크별 자유지수는 7점 5파라미터 = 자유도 2로 너무 얇다).

레버리지 출처:
- **β** ← 99M 행 (annealed 4점, D 8× 범위) + 680M 쌍 (2.08×)
- **α** ← **준-iso-D 삼중 99M@42.1 / 296M@62.3 / 680M@68.9** — 비슷한 D 에서 N 이 3단계.
  우리가 가진 IsoFLOP slice 에 가장 가까운 것이고, 이 설계가 구제되는 이유다.
- **E_t** 가 약한 파라미터 — §5.3 에서 별도 처리.

풀링을 위해 태스크별로 표준화한다 (1차 자유지수 pass 의 잔차 SD `σ̂_t` 로 각 태스크 잔차를
나눔). 안 하면 `log X` 범위가 큰 hellaswag 가 공유 지수를 지배한다.
공유 `(α, β)` 는 같은 7점의 **BPB** Chinchilla fit 지수와 비교한다 — 크게 어긋나면
downstream 표면이 단순히 LM 품질을 따라가는 게 아니라는 뜻이니 숨기지 말고 보고.

**C 는 그림의 x축용으로만** 쓰고, C 단일축 fit 은 γ=10 ray 위에서만 식별된다고 명시.

### 5.2 손실·파라미터화·가중치

Hoffmann 레시피대로 **로그공간 + log-sum-exp 파라미터화** (raw 최소제곱보다 조건수가 훨씬 좋다):

```
min Σ_i ρ_δ( LSE(e, a − α·log N_i, b − β·log D_i) − log X_i )
E = e^e, A = e^a, B = e^b   (양수성 자동), ρ_δ = Huber, δ = 1e-3
```

**멀티스타트** (Hoffmann 그리드): `e ∈ {−1,−0.5,0,0.5,1}`, `a,b ∈ {0,5,10,15,20}`,
`α,β ∈ {0,0.5,1,1.5,2}` (625 시작). 경계 `α,β ∈ [0,2]`, `E ∈ (0, 0.95·X_chance,t]`.
단위는 N 을 1e6 params, D 를 GB 로 두어 `a,b` 를 O(1) 로.

> **⚠️ 저장소 선례인 N-가중치(`plot_ratio10_ladder.py` 의 `w ∝ N`)를 그대로 쓰면 안 된다.**
> ① X 의 측정 노이즈는 `n_doc` 에서 오지 N 에서 오지 않는다 — 680M 점이 99M 점보다 6.9배
> 신뢰할 이유가 없다. ② N-가중은 99M 행을 6.9× 낮춰 잡는데, **그 행이 β 를 식별하는 유일한
> 정보**다. ③ N 이 3단계뿐이라 유효 표본수가 2로 붕괴한다.
> **→ 로그공간 균등 가중치.** N-가중 fit 은 민감도로만 보고한다.
> 문서 단위 SEM 역분산 가중도 쓰지 않는다 (n≥1172 라 SEM 이 0.005–0.02 nats 로 모델 간
> 산포에 파묻히고, 파라미터 SE 만 과신하게 만든다).

Delphi 는 식을 주지 않는다. "asymptotic power law" 를 `y = y_∞ + A x^(−α)` 로 읽고 그
(N,D) 일반화(Hoffmann 2022 Eq.4)를 채택했음을 **우리 해석이라고 명시**한다.

### 5.3 바닥 `E_t` — 가장 약한 지점

7점으로는 간신히 식별된다. 셋 다 한다:
1. `E_t` 자유 (경계 §5.2) — **주 결과**
2. `E_t ≡ 0` — 변형
3. **예측을 `E_t` 위에서 profile**: `[0, 0.95·X_chance]` 그리드에서 `E_t` 를 고정하고 나머지를
   재적합, `X̂_t(1.3B)` 와 `â_t(1.3B)` 를 `E_t` 에 대해 플롯. **이 그림 하나가 전체 프로젝트에서
   가장 정보량이 큰 민감도 산출물**이다 — 예측의 얼마가 데이터이고 얼마가 바닥 사전분포인지
   독자에게 그대로 보여준다. 이 구간의 폭을 보고 불확실성에 포함.

### 5.4 Stage 2 — soft → hard

```
â_t(X) = p_c,t + (1 − p_c,t)·σ( k_t·(x0_t − X) ),   k_t > 0
```

천장은 **1.0 고정** (우리 정확도 최대가 74 % — 자유 천장은 식별 안 되고 발산한다).
바닥 `p_c,t`:

| task | `p_c` |
|---|---|
| hellaswag / arc_easy / arc_challenge | 0.2500 / 0.2502 / 0.2502 (고정) |
| piqa / winogrande | 0.5000 (고정) |
| **boolq** | **[0.45, 0.65] 자유** |

**바닥 고정이 필수인 이유**: rg 99M ARC-C 는 22.6 % 로 chance(25.02 %) **아래**,
296M BoolQ 는 47.3 %. 자유바닥 4파라미터 로지스틱은 하한을 20 % 아래에 놓고 기울기를 노이즈에
맞춰 휘어버린다. chance 점의 작은 음수 잔차는 binomial 노이즈 안이니 "고치지" 않는다.
**BoolQ 만 예외** — "항상 yes" 가 62.17 % 이므로 실효 바닥이 0.5 가 아니다. 자유로 두고
민감도를 보고하며, §4.4 의 mutual-info 변형이 이를 정직하게 만드는지 확인한다.

**적합은 quasi-binomial MLE** (`n_t` 가중 + `p(1−p)` 분산 가중이 자동으로 들어간다;
퍼센트포인트 최소제곱보다 엄밀히 낫다). 과산포 `φ̂ = χ²_Pearson/dof` 를 추정해 SE 를 `√φ̂` 로
부풀린다 (모델 간 산포가 binomial 을 크게 넘을 것). **단, link 의 모든 CI 는 §6.3 의 모델 단위
재표집에서 나와야 한다** — 18개 점은 같은 문서를 상관된 18개 모델이 채점한 것이라 binomial n 은
올바른 *가중치*이지만 완전히 틀린 *불확실성* 근거다.

link family 민감도: Gadre et al. 2024 의 지수감쇠형 `Err = ε·exp(k(X−E))` 를 최소한 하나 더
돌린다 (인용 전 원문 파라미터화 확인).

### 5.5 누출 규칙 (절대) + 모델 선택

**1.3B 런에서 파생된 어떤 것도 Stage 1/2 에 들어가지 않는다.** soft 점수, 정확도, BPB,
`E` 선택, 초기화 선택 — 그리고 사람들이 실제로 어기는 것: **어떤 soft-metric 변형이나 함수형을
보고할지 고르는 데 쓰는 것.**

기계적 강제:
1. compact 테이블을 ≤760M 과 1.3B 로 **디렉토리를 분리**해 만든다
2. fit 스크립트는 명시적 `--holdout` 리스트를 받고, **예측을 디스크에 쓰기 전에는 holdout
   디렉토리를 읽기를 거부**한다
3. **사전등록 파일**(soft-metric 변형·함수형·가중치·바닥 정책·held-out 정의·주지표)을
   1.3B 테이블을 읽기 **전에** 커밋하고 해시+타임스탬프를 기록

`p_c,t` 와 `X_chance,t` 는 데이터셋 속성이지 모델 속성이 아니므로 예외.

**모델 선택은 fit 집합 내부 CV 로만**: 99M+296M 5점으로 적합 → 680M annealed 2점 예측,
`mean|ΔX|` (Stage 1) 과 `mean|Δacc|` (합성)으로 점수. 내부 held-out 이 2점뿐이라 거친 선택만
가능하니, **모든 선택지에 기본값을 사전등록**하고 내부 CV 가 그 2점의 binomial SE 를 넘는
차이로 대안을 선호할 때만 바꾼다.

### 5.6 확장 (권장) — 교차 패밀리 link pool

같은 패밀리로 sigmoid 를 fit 하면 두 단계가 데이터를 공유해 오차가 상관되고, byte 모델 특유의
버릇이 그대로 박힌다. Delphi 의 요점은 이 매핑이 **패밀리 무관한 관측적 맵**이라는 것이다.
저장소에 Llama / AU-Net / hybrid 의 100M–760M 체크포인트가 이미 있으므로 **학습 비용 0** 으로
되찾을 수 있다: **≤760M 교차패밀리 풀에 sigmoid 를 fit 하고 BPEByte-rg 1.3B 로 검정.**
모든 패밀리의 1.3B 를 풀에서 제외한다. 교차패밀리 sigmoid 가 동일패밀리만큼 잘 맞으면
주 결과보다 훨씬 강한 주장이 된다.

---

## 6. Held-out 검증과 불확실성

### 6.1 무엇을 무엇에 대고 채점하나

held-out 점마다 ① Stage 1 의 `X̂_t` 를 측정 `X_t` 에, ② `â_t = sigmoid(X̂_t)` 를 측정
헤드라인 정확도에 대고 채점. **헤드라인은 180000 / 283.1 GB annealed rung 하나.**

태스크별 + macro-average 로 보고:
- soft 잔차: **nats(절대)**, **X 대비 %**(Delphi 단위), 그리고 정직한 것 —
  **정규화 잔차** `ρ_t = |X̂_t − X_t| / (X_chance,t − min_fit X_t)`, 즉 *사다리에서 실제로 달성한
  진전 대비* 비율. Delphi 의 "0.5 % 이내"는 동적 범위가 ~25 % 뿐인 양에 대한 상대값이라
  값의 0.5 % ≈ 동적 범위의 2 % 다. **상대값만 보고하면 그 수사적 속임수를 반복하는 것이므로
  ρ 를 항상 병기한다.**
- hard 잔차: **pp**, 부호 있는 값과 절대값
- MAE, RMSE, **부호 있는 평균 오차** — 기존 baseline 의 실패 양상이 체계적 **−3 pp 과소예측**
  이므로 주장은 "크기 감소"가 아니라 **"편향 제거"** 다.

**macro-average binomial 노이즈 바닥 ≈ 0.44 pp** (태스크별 SE: HS 0.50, ARC-E 1.02,
ARC-C 1.36, PIQA 1.05, WinoG 1.40, BoolQ 0.87). **0.6 pp 미만 개선은 측정 불가**임을 선언.

### 6.2 나머지 세 1.3B 체크포인트 — 동등한 held-out 이 아니다

mid-cosine 이다. Stage 1 은 annealed endpoint 로 적합됐으므로 그 D 에서의 *annealed* X 를
예측하고, 그건 mid-cosine 모델이 실제 달성한 값보다 체계적으로 좋다. annealed 법칙으로
예측하면 가짜 overshoot 이 만들어진다.

→ **"mid-schedule trajectory" 별도 패널**로 라벨링하고, 편향의 부호를 미리 명시한 뒤 크기는
99M 통제쌍(`cmp_100M_ot@6688` annealed vs `ratio20@6688` mid, 동일 seed·데이터 순서·D=42.077 GB)
에서 추정해 보고한다. 680M 에서 같은 통제쌍을 만들려면 `rg_760M@29200` 이 필요한데 존재하지
않는다. 가장 가까운 680M 통제는 *스케줄 위치* 쌍 `bpp101@17520`(cosine 60 % 지점) vs
`rg_760M@18180`(30 % 지점).

### 6.3 부트스트랩 — 정확히 무엇을 재표집하나

노이즈원이 셋이다. **셋을 함께** 재표집하되 각각 단독 결과도 보고한다 (분해가 단일 구간보다
훨씬 유용).

1. **문서 표집** — 태스크당 문서 인덱스 multiset 을 **하나** 뽑아 **모든 모델과 두 단계에
   동일하게** 적용. 모델별로 따로 뽑으면 모델 간 상관이 파괴돼 구간이 부풀고, Stage 1/2 에
   다른 multiset 을 쓰면 §5.4 가 경고한 공유 데이터 구조가 깨진다. **가장 자주 틀리는 지점.**
2. **Stage 1 fit 점 노이즈 — case bootstrap 이 아니라 wild residual bootstrap.**
   7점에서 case bootstrap 은 자주 랭크 결손이 난다 (99M 4점이 다 빠지면 β 미식별, N 3단계 중
   2개가 빠지면 α 미식별). 대신 한 번 적합해 로그공간 잔차 `r_i` 를 얻고
   `log X_i^(b) = log X̂_i + ε_i r_i` (Rademacher `ε_i ∈ {−1,+1}`) 로 **설계를 고정한 채** 재적합.
3. **Stage 2 풀 노이즈 — 런 단위 block bootstrap.** **런 7개**를 복원추출하고 뽑힌 런의 채택
   체크포인트를 통째로 가져간다. 런 내 상관을 존중하는 유일한 재표집. 블록 7개라 거칠고
   보수적이다. 체크포인트 단위(~18점) 부트스트랩도 낙관적 하한으로 함께 보고하고, 진실이
   둘 사이에 있다고 명시.

**합성**: 복제 `b = 1..2000` 마다 셋을 함께 뽑아 두 단계를 재적합하고 held-out 에서
`X̂_t^(b)`, `â_t^(b)` 를 낸다. CI = 백분위 [2.5, 97.5]. 네 구간(결합 + 각 원인 단독)을 보고.
저장소 관례(`downstream_scaling.md` 의 LOO 밴드)에 맞춰 **leave-one-point-out** 과
**leave-one-N-level-out** 도 함께.

### 6.4 "외삽할수록 CI 가 넓어진다" 그림 — 우리가 실제로 만들 수 있는 것

**(a) fit 집합 고정, held-out 거리 변화**: 1.3B 4점이 C 기준 **1.04× / 2.08× / 3.13× / 3.47×**
(N 외삽은 1.91× 고정). §6.2 의 스케줄 caveat 적용.

**(b) 중첩 fit 집합, 타겟 고정** — 1.3B@283 rung 을 매번 예측:
- ≤680M (7점): `C_max = 2.93e20` → **3.47×**
- ≤296M (5점): `C_max = 5.60e19` → **18.2×**
- ≤99M (4점): **N 축 퇴화 — α 미식별. 버린다. 숫자를 보고하지 않는다.**

중첩 스윕에서는 자유도 회계를 정직하게 맞추려 `E_t ≡ 0` 으로 고정(4 자유 파라미터, ≤296M 은
자유도 1). `|Δâ|` 와 CI 반폭을 외삽 배율(로그 x축)에 대해 플롯. 2 rung 은 얇지만 진짜이고,
Delphi 그림의 정직한 버전이다.

### 6.5 캘리브레이션 — 반드시 보고

held-out 예측이 4 D × 6 task = **24개** (annealed rung 만 보면 6개). 명목 95 % 구간의
**실제 포함률**을 보고한다. MAE 3 pp 에 잘 캘리브레이션된 ±4 pp 구간이, MAE 3 pp 에 ±1 pp
구간보다 **더 나은 과학**이다. scaling law 논문들이 거의 안 하는데 여기선 거의 공짜다.

---

## 7. 사전 게이트 — 어떤 숫자든 믿기 전에 통과해야 한다

통과 여부와 무관하게 결과를 전부 보고한다.

**7.1 정확성 게이트 (2회, ~15분) — 스윕 전 필수.**
작업 트리에 eval 경로 **미커밋 변경 904줄**이 있다 (`regex_cutting.py` 404, `hierarchical.py`
452, `generate.py` 71, `byte_trie.py` 11). 매니페스트에 `git rev-parse HEAD` 와
`sha256(git diff)` 를 기록하고 아래를 먼저 통과시킨다.

- **게이트 A** — `runs/others/superseded/cmp_100M_ratio20/v4_root_greedy/checkpoints/0000006688`
  vs 같은 런의 `eval_r20_allbench/results.json` (전체 데이터셋, `limit: null`):
  hellaswag 29.39/30.38 · arc_easy 38.55/35.61 · piqa 61.32/59.63 · arc_challenge 19.71/23.46 (acc/acc_norm)
- **게이트 B** — `main/main/others/760M/bpebyte_root_greedy_760M_bpp101/checkpoints/0000029200`
  vs `evals_5bench/results.json` (boolq·winogrande 포함):
  hellaswag 40.57/45.78 · arc_easy 56.48/51.05 · boolq 45.47 · piqa 69.80/69.10 · winogrande 54.62
- **기준 ±0.5 pp.** A 가 1 pp 넘게 어긋나면 다른 걸 건드리기 전에 작업 트리 변경을
  `git stash` 하고 재실행. A 통과·B 실패면 boolq/winogrande 특유 문제 → `multiple_input` 경로 점검.

**7.2 soft metric 단조성 — 프로젝트의 중심 주장.**
태스크별로 γ=10 rung (99M@21 → 296M@62.3 → 680M@143) 과 99M D-스윕을 따라 `X_t` 가 단조
감소하는지 확인. **BoolQ 의 X 가 단조인데 정확도가 비단조라면 우리 사다리에서 Delphi 의 핵심
현상을 재현한 것이고, 그게 헤드라인 그림이다.** X 도 비단조라면 그 태스크에서는 전제가
무너진 것이므로 그렇게 쓴다.

**7.3 mid-cosine 붕괴 검정.** ~18개 풀 점의 `a_t` vs `X_t` 를 annealed/mid-cosine 색으로 플롯.
**같은 곡선 위에 있으면** link 가 레시피에 강건하고 Stage 2 에 중간 체크포인트를 쓰는 것이
정당화된다. 눈에 띄게 갈라지면 풀을 annealed 로 제한해야 하고 (sigmoid 에 7점만 남는다)
그 사실을 보고한다. 99M 통제쌍의 `ΔX`, `Δa` 로 정량화.

**7.4 평가 레짐 동질성.** 저장소에 평가시 경계 플래그가 잔뜩 있다
(`force_bpe_online_mode`, `answer_greedy_loglikelihood`, `committed_answer_loglikelihood`,
`atomic_answer_loglikelihood`, `offline_question_loglikelihood`, `root_stats_answer_loglikelihood`, …
`apps/aunet/eval.py:120-190`). 기존 eval 디렉토리들은 수개월에 걸쳐 서로 다른 플래그로
만들어졌다 (§1 의 300M 표 불일치가 그 증거). **6개 벤치를 하나의 고정 config 로 전 체크포인트에
재실행**하고, 모든 `eval_meta.json` 에 대해 `bpe_online=true, mode="greedy",
placement="root", committed_view=false, context_prefix=0` 및 모든 경계 플래그가 기본 null/false
임을 프로그램으로 assert 한다. 복제할 정본:
`main/main/1.3B/bpebyte_br_greedy_root_1.3B/eval_confirm_4bench/config.yaml` +
`tasks: [hellaswag, arc_easy, arc_challenge, piqa, winogrande, boolq]`, `num_fewshot: null`,
`limit: null`, `log_samples: true`, `bootstrap_iters: 1000`, `dtype: bf16`, 시드 전부 기본값,
**체크포인트당 6개 태스크를 단일 호출로** (fewshot RNG·문서 순서 동일 보장).

**7.5 수치 재현성.** 한 체크포인트를 서로 다른 world size / batch size 로 두 번 채점해
`|ΔX_t| < 0.003 nats` 를 요구. 동적 패칭 byte 모델은 배치에 민감할 수 있고, §4.6 에서
0.01 nats ≈ 1 pp 이므로 0.003 nats 불안정성만으로 목표 정밀도의 1/3 을 먹는다.
실패하면 world size·batch size 를 스윕 전체에 고정하고 재실행. **재현성을 그 이하로
보이지 못한 파이프라인에서 Delphi 급 0.5 % 정밀도를 주장할 수 없다.**

**7.6 샘플 완전성.** `doc_iterator` 가 `islice(rank, limit, world_size)` 로 샤딩하므로 doc id 는
합집합에서 유일하지만, `doc_id` 로 dedup 해 개수가 공표값(10042/2376/1172/1838/1267/3270)과
같은지 assert.

**7.7 태스크별 SNR.** `SNR_t = (X_t^{99M@10.5} − X_t^{680M@143}) / SEM(X_t)`. **20 미만이면**
5파라미터 표면을 지탱할 수 없다. fit 풀에서 정확도 범위가 binomial SE 의 3배 미만이면
**hard-metric fit 에 정보가 없는 태스크로 선언**하고 점 추정 대신 "chance ± fit 불확실성"으로
보고한다. **WinoGrande 가 유력 후보** (296M→680M 이 0.1 pp, SE 1.40 pp).

**7.8 이상치 감사** (§4.3): 평균 vs 1 % 절사평균이 0.02 nats 이내.
**7.9 정규화항 일치** (§4.1): `argmax(ll/b_choice)` vs `argmax(ll/c_choice)` 불일치 < 0.5 %.
**7.10 N 회계**: 98.59M / 296.0M / 679.6M / 1.300B 가 같은 카운터에서 나왔는지 확인.

---

## 8. 이 작업을 정당화하는 비교

### 8.1 기존 3.3 pp 는 그대로는 유효한 baseline 이 아니다

`reports/downstream_scaling.md` 의 MAE 3.3 pp 는 **4개 태스크**(winogrande·boolq 없음),
**4점 N 단일축** 사다리, 다른 fit 집합에서 나왔다. **동일 분할(fit = annealed ≤760M 7점,
예측 = 1.3B@283.1)과 동일 6개 태스크로 재실행**해야 한다. 재실행 baseline 은 3.3 pp 보다
**나빠질 것으로 예상**한다 — BoolQ 가 비단조라 어떤 단조 hard-metric 법칙으로도 못 맞추기 때문.
그게 바로 요점이다.

### 8.2 baseline 목록 (전부 동일 분할)

| id | 방법 |
|---|---|
| **B0** | **carry-forward: `â_t(1.3B) = a_t(680M@143)`** |
| B1 | 직접 hard 법칙 (N,D): `a_t = C_t − A_t N^(−α) − B_t D^(−β)` |
| B2 | logit 선형 직접: `logit(a_t) = c0 + c1 log N + c2 log D` |
| B3 | loss 매개 (기존 방법의 2D 일반화): BPB(N,D) Chinchilla → `logit(a_t) = a + b·BPB` |
| **B4** | **우리 것**: `X_t(N,D)` Chinchilla → 바닥고정 sigmoid → `a_t` |
| B5 | gold-only soft metric (정답 continuation BPB, 오답 폐기) = OLMo ladder "task loss" |

**B0 는 필수다.** N 1.9× 외삽에서 순진한 carry-forward 가 의외로 강하고, 논문들이 습관적으로
빼먹는다. **B4 가 B0 를 못 이기면 결과가 없는 것이다.**

### 8.3 개선의 원인을 귀속시키는 2×2 — 결정적 실험

기존 방법의 실패는 체계적 −3 pp 과소예측이었고 Stage 1 은 loss 를 거의 정확히 맞췄다
(0.605 vs 0.605). 즉 실패는 **Stage 2 = link** 에 있었다. 그런데 link 에는 독립적인 두 부분이
있다 — **입력**(BPB vs 태스크별 X)과 **함수형**(logit 선형 vs 바닥고정 sigmoid).
둘을 한꺼번에 바꾸면 무엇이 중요했는지 알 수 없다. 동일 분할에서 2×2 를 전부 돌린다:

| | 입력 = BPB | 입력 = `X_t` (soft) |
|---|---|---|
| **logit 선형 link** | B3 (기존 방법) | B3′ |
| **바닥고정 sigmoid link** | B3″ | **B4 (우리 것)** |

**B3″ ≈ B4 이면 soft metric 은 아무 기여도 안 했고 개선은 전부 sigmoid 덕분**이라는 뜻이다 —
훨씬 값싼 발견이고, 묻지 말고 보고해야 한다.

### 8.4 held-out 없이도 되는 fit-집합 진단

≤760M 풀만으로 각 후보 link 주변의 **정확도 잔차 SD** 를 동일 자유도에서 비교:
sigmoid(`X_t`) vs sigmoid(BPB) vs `log N` 직접. `X_t → a` 가 `BPB → a` 보다 조이지 못하면
held-out 이 뭐라 하든 soft metric 은 값을 못 한 것이다. 태스크별로 보고 — boolq /
arc_challenge / winogrande 에서 격차가 가장 클 것으로 예상.

### 8.5 차이의 검정

태스크별 + macro-average 로 부호 오차, |오차|, 그리고 **B4 와 각 baseline 의 |오차| 차이에 대한
paired bootstrap CI** 를 §6.3 의 *동일* 복제로 보고 (두 방법이 같은 문서 multiset·같은 fit 점을
보므로 pairing 이 유효하고 독립 구간 비교보다 훨씬 강력하다).
태스크가 6개뿐이므로 macro-average 에 p-value 는 **보고하지 않는다**.

---

## 9. 실행 순서

### Phase 0 — GPU 확보 (블로킹)

현재 A5000 4장이 사용자의 `scripts/probes/flores/blt_native_flores.py` 로 96–98 % 점유
(카드당 17.6 GB, 여유 6.9 GB). `r10lr_downstream.sh` 는 `MAX_USED_MB=2000` 을 하드코딩하고,
학습 잡이 올라간 GPU 에 평가를 얹었다가 *"254 MiB 요청, 244 MiB 여유"* 로 죽은 사례를 주석에
남겼다 (`max_tokens=16384` 의 `create_block_mask` 가 크다).
**4장 모두 `memory.used < 2000` MiB 가 될 때까지 스윕을 시작하지 않는다.**

### Phase 1 — ece fetch ✅ **완료**

`consolidated/` 만 받았다 (100M 1.2 G, 300M 3.4 G). `metrics.jsonl` 은 로컬 사본이 동일해 불필요.
`torch.load(map_location='meta')` 로 로드 검증 + 아키텍처·regex 확인 완료.
(`rsync` 를 ssh 로 태울 때 `ssh -n` 금지 — `r10lr_downstream.sh` 가 프로토콜이 조용히 깨진다고 기록.)

### Phase 2 — 정확성 게이트 (§7.1)

### Phase 3 — consolidation (CPU 전용, 병렬 가능)

`lingua/checkpoint.py:69-88` 의 `consolidate_checkpoints` 는 `dcp_to_torch_save` 호출이라
**GPU 를 쓰지 않는다** — Phase 2 와 동시 진행 가능. 박스는 64코어 / 251 GB RAM / 285 GB 여유라
3–4 병렬이 안전 (더 늘리면 NVMe 가 thrash).

**link pool 에 필요한 100M raw 4개(8360, 10032, 11704) + 필요시 추가만 consolidate 한다**
(각 ~1.2 GB, ~1분). 760M raw 4개(24000/28000/56000/60000) 중 **24000·28000 은 de-dup 규칙상
채택**되므로 consolidate 한다 (각 ~8 GB, 3–5분). 56000·60000 은 54540·60600 과 D 차이가
1.15× 미만이라 **버린다**.

**함정**: `consolidate_checkpoints` 는 `consolidated.pth` 존재 여부만 본다. 중간에 죽으면
**잘린 파일이 다음번에 조용히 로드된다.** 스윕 안의 암묵 경로에 맡기지 말고 미리 명시적으로
돌린 뒤 `torch.load(..., map_location='meta')` 로 텐서 수를 검증한다.

### Phase 4 — 스윕 (~2시간, 최악 4시간)

**실측 소요 시간** (`runs/bpebyte_br_greedy_root_1.3B_a100x4/stage_*`, 동일 아키텍처 1.3B,
`log_samples=true`, 전체 데이터셋, `mtime(results.json) − mtime(config.yaml)`):

| dump dir | tasks | docs | wall | 크기 |
|---|---|---|---|---|
| `stage_0000045000_avg5` | hs, arc_e, arc_c, piqa, wino | 16,695 | **2분 31초** | 49.9 MB |
| `stage_0000045000_avg7` | boolq + mmlu_text | 17,312 | **4분 05초** | 42.9 MB |
| `stage_0000045000` | 9-task (`*_gen` 포함) | 33k | **12.2시간** | 219.6 MB |

**6-bench 는 1.3B 에서 4 GPU 로 ~3.5분.** 사람들이 기억하는 12–24시간은 전적으로 byte 단위
자기회귀 디코딩이 필요한 `*_gen` + mmlu_text 때문이고, 우리 스윕에는 생성 태스크가 없다.
GPU 1장 레인 기준: 99M ~7분, 297M ~9분, 714M ~12분, 1.32B ~16분.
4레인 × CPU 경합 1.5–2× ⇒ **~25 체크포인트 ≈ 2시간.**
평가는 **CPU 바운드**다 (online greedy BPE 패칭이 시퀀스당 CPU; GPU util 0 % 근처).
프로세스당 OMP/MKL/RAYON 8스레드 제한(`apps/aunet/eval.py:6-9`) → 4레인이 64코어 중 32개 사용.

레코드당 실측 JSON 크기: hellaswag 4,087 B · boolq 2,474 · arc_c 1,699 · arc_e 1,558 ·
piqa 1,091 · winogrande 885 ⇒ **6-bench 합계 ~58 MB/체크포인트, 25개면 1.45 GB.** 문제 없음.

**실행 명령**

```bash
cd /mnt/ssd2/hyun2/AUNet/lingua
export LD_LIBRARY_PATH="$(ls -d .venv/lib/python3.12/site-packages/nvidia/*/lib | paste -sd: -):${LD_LIBRARY_PATH:-}"
export TMPDIR=/var/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ENABLE_INTRA_NODE_COMM=0
export TOKENIZER_PATH=/mnt/ssd2/hyun2/AUNet/tokenizer/llama3/tokenizer.model
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m torch.distributed.run \
  --nproc-per-node 1 --master-port $((29600 + GPU)) \
  -m apps.aunet.eval \
  config=apps/aunet/configs/delphi/eval_6bench_local.yaml \
  ckpt_dir="$CKPT" dump_dir="$OUT" > "$OUT/eval.log" 2>&1
```

**함정 (전부 소스로 확인)**

1. **`regex_bpe_tokenizer_path=` 를 절대 CLI 로 넘기지 않는다.** `apps/aunet/eval.py:617-628`
   이 `RegexPool(RegexArgs(strategy, bpe_tokenizer_path, bpe_tokenizer_kind))` 만 넘겨 재생성하고,
   `RegexArgs` 기본값이 `bpe_online=False / "bt" / "before_root"`
   (`apps/aunet/data/regex_cutting.py:108-120`) 라 우리 체크포인트의 `True/greedy/root` 가
   **에러 없이 조용히 초기화**된다 → chance 근처 결과. **`lingua/r10lr_eval.sh` 가 바로 이
   버그를 갖고 있으니 복사하지 말고**, 5일 뒤 수정본인 `r10lr_downstream.sh` 를 템플릿으로 쓴다.
2. **토크나이저 경로**: 대부분 `/NHNHOME/.../tokenizer/llama3/tokenizer.model` 을 굽고 있고
   (`cmp_100M_ratio20` 만 `/home/hwbae/...`), `generate.py` 의 `_resolve_bpe_tokenizer_path` 가
   경로에 `/tokenizer/` 가 있으면 `<AUNet-root>/tokenizer/...` 로 재-루팅하는 **fallback 을 이미
   갖고 있다** — 로컬에 파일이 있으므로 `TOKENIZER_PATH` 는 안전장치일 뿐 필수는 아니다.
3. **`validation: null`** 을 config 에 박는다. 없으면 `lingua/lingua/data.py:478` 이
   `assert n_chunks > 0` 보다 **먼저** `world_size % n_chunks` 를 하므로
   `ZeroDivisionError: integer modulo by zero` 로 죽는다.
4. **`--master-port` 를 레인마다 다르게.** `get_master_port` 는 `LOCAL_RANK` 가 있으면
   (= torchrun 하에서는 항상) `MASTER_PORT` 를 쓰므로 `torchrun --master-port` 로 충분하다.
   28805 충돌은 `LOCAL_RANK` 없이 맨몸 `python -m` 으로 띄울 때만.
5. **`harness.include_path` 는 넣지 않는다** — 6개 태스크 중 `lingua/eval_tasks/` 가 가리는
   것이 없음을 확인했다. 빼면 실패 지점이 하나 준다.
6. **`LD_LIBRARY_PATH`** 에 venv 의 `nvidia/*/lib` 가 없으면 `import torch` 가
   `libcusparseLt.so.0` 로 실패한다.
7. 평가 직후 **`verify_boundaries`** (`eval_meta.json` vs `params.json['data']['regex']`).
   불일치 행은 격리하고 평균에 넣지 않는다 — 과거 이 검사를 건너뛰어 한 라운드 결과가 통째로
   무효화된 적이 있다.

**새 config** `apps/aunet/configs/delphi/eval_6bench_local.yaml`:

```yaml
name: "delphi_6bench_local"
harness:
  log_samples: true
  limit: null
  num_fewshot: null
  bootstrap_iters: 1000
  tasks: [hellaswag, arc_easy, arc_challenge, piqa, winogrande, boolq]
validation: null
generator: {max_tokens: 16384, dtype: bf16}
```

### Phase 5 — 추출 / 압축 / 검증 (~10분)

dump 완료 직후: ① `verify_boundaries` → ② per-doc 테이블 생성 → ③ `gzip -9 results.json`
(58 MB → 10–15 MB; **지우지 말 것** — 감사 추적용이고 재실행은 10분이지만 provenance 재구성은 불가).

### Phase 6 — fit & 리포트

`scripts/downstream_scaling.py` 의 2단계 골격을 재사용하되 하드코딩 배열을 매니페스트로 바꾸고
Stage 1 의 x축을 N 에서 soft metric 표면으로 교체.

---

## 10. 산출물 / 스키마

```
reports/delphi/
  PREREGISTRATION.md                # 1.3B 테이블을 읽기 전에 커밋 (해시+타임스탬프)
  manifest.jsonl
  per_doc/le760M/<run>__<step>.jsonl    # ← fit/link 용
  per_doc/holdout_1p3b/<run>__<step>.jsonl  # ← 디렉토리 분리로 누출 기계적 차단
  soft_metrics.csv
  preflight_report.md               # §7 게이트 10개 전부
  logs/<run>__<step>.log
  summary.md
  figs/{monotonicity, collapse, E_profile, delphi_2panel, extrapolation_distance}.png

scripts/probes/delphi_sweep.sh
scripts/delphi_extract.py
scripts/delphi_fit.py               # --holdout 명시 필수, 예측 기록 전 holdout 읽기 거부
apps/aunet/configs/delphi/eval_6bench_local.yaml
```

평가 dump 는 저장소 관례대로 체크포인트 옆:
`<run_dir>/delphi_<step>/{results.json.gz, eval_meta.json, config.yaml, eval.log}`.

**`manifest.jsonl`**: `id, family, series, scale, N, step, D_bytes, flops_per_byte, C_flops,
lr_at_step, lr_frac_of_max, annealed, train_loss_at_step, ckpt_path, consolidated_source,
dump_dir, per_doc, regex{...}, boundary_check, git_head, worktree_diff_sha256,
role(fit|link_pool|holdout), eval_seconds, status`

**`per_doc/*.jsonl`** (문서당 1행, ~150 B):

```jsonc
{"task":"hellaswag","doc_id":0,"K":4,"gold":3,
 "ll":[-193.0,-203.0,-132.0,-141.0],   // filtered_resps[i][0], 자연로그
 "b_cont":[39,31,29,39],                // 구분자 포함 (모델이 채점한 것)
 "b_choice":[38,30,28,38],              // 구분자 제외 (acc_bytes)
 "c_choice":[38,30,28,38],              // 문자 수 (acc_norm)
 "is_greedy":[false,false,false,false],
 "acc":0.0,"acc_norm":1.0,              // lm-eval 자체 값 = self-check
 "multiple_input":false}
```

**추출기 불변식 (warn 이 아니라 assert)**
- `argmax(ll) == gold` ⟺ 기록된 `acc == 1.0`
- `argmax(ll / c_choice) == gold` ⟺ 기록된 `acc_norm == 1.0` (boolq·winogrande 는 `acc_norm` 없음)
- winogrande: `multiple_input=true`, `gold = int(doc["answer"]) − 1`, 길이는 `arguments[i][0]` 에서
- 길이 규칙:
  ```python
  if all(a[1] == arguments[0][1] for a in arguments):   # multiple_input
      choice_str = arguments[i][0]
  else:
      choice_str = arguments[i][1][1:]                   # 앞의 " " 제거
  ```

---

## 11. 드라이버 설계 (`scripts/probes/delphi_sweep.sh`)

`lingua/nb_lane.sh` (GPU당 레인 + `.DONE` 스킵) 와 `runs/_scripts/eval_scheduler.py` 를 따른다.

- **큐**: `manifest.jsonl` 에서 `status != "done"`
- **재개**: `dump_dir/results.json(.gz)` 이 있고 비어있지 않으면 스킵. 살아있는 `.running`
  마커도 스킵 — **`pgrep -f` 로 폴링 금지** (`r10lr_downstream.sh` 가 자기 명령줄에 매칭되는
  문제를 기록). 파일 기반으로만.
- **레인**: 4개, `CUDA_VISIBLE_DEVICES=$lane`, `--master-port $((29600+lane))`, `--nproc-per-node 1`.
  §7.5 를 위해 **world size·batch size 를 스윕 전체에 고정**한다.
- **순서**: 큰 것부터, 스케일을 레인 간 인터리브
- **사전 점검**: 매 launch 전 `nvidia-smi` 로 `used < 2000 MiB`
- **사후**: `verify_boundaries` → `delphi_extract.py` → `gzip -9` → 매니페스트 갱신
- **재시도**: `results.json` 없이 종료하면 최대 2회, 그 뒤 `status: "failed"`
- **detach**: `setsid nohup ... &`, pgid 를 `logs/sweep.pgid` 에 남기고 **프로세스 그룹으로만
  종료** — 이 박스에는 다른 잡이 돈다. `pkill` 금지.

---

## 12. 비용 요약

| Phase | 시간 | 디스크 Δ | 블로킹 |
|---|---|---|---|
| 0. GPU 확보 | 미정 (FLORES 4잡 실행 중) | 0 | **예** |
| 1. ece fetch | ✅ 완료 | +4.6 GB | — |
| 2. 정확성 게이트 2회 | ~15분 | +0.12 GB | **예** |
| 3. consolidate (100M 3개 + 760M 2개) | ~15분 (3-way) | +19.6 GB | 아니오 |
| 4. 스윕 ~25 ckpt × 6-bench, 4레인 | **~2시간** (최악 4시간) | +1.45 GB | — |
| 5. 추출 + gzip | ~10분 | −1.2 GB | — |
| 6. fit + 그림 | ~1일 (분석) | 미미 | — |
| **합계** | **GPU 확보 후 ~3시간 계산** | **+25 GB** (285 GB 여유) | |

---

## 13. 리포트에 명시할 caveat

1. **3.47× compute 외삽(0.54 decade)이지 2 decade 가 아니다.** Delphi 프레이밍을 빌리지 않는다.
2. fit 점들은 **IsoFLOP optima 가 아니다** — frontier 자체가 없다. 원문과의 가장 큰 구조적 차이.
3. **N 이 3단계뿐** (98.59M / 296.0M / 679.6M) — α 가 사실상 세 숫자로 결정된다.
   공유지수 풀링(§5.1)과 E profile(§5.3)은 완화책이지 해결책이 아니다.
4. **`E_t` 는 간신히 식별된다** — profile 그림이 그 공개다.
5. **WinoGrande 는 hard-metric 예측에 정보가 없을 수 있다** (296M→680M 0.1 pp vs SE 1.40 pp).
   macro-average 를 포함/제외 둘 다 보고. 단 **soft** 지표가 단조로 움직이면 정확도 예측이
   무의미해도 그건 방법에 대한 **긍정적** 결과이므로 그렇게 제시한다.
6. **BoolQ 의 바닥은 0.5 가 아니라 최대 0.62(다수결)** 이고 rg 는 296M 에서 0.5 **아래**다.
   자유바닥 + mutual-info 변형 + 민감도.
7. **macro-average 노이즈 바닥 ≈ 0.44 pp** — 0.6 pp 미만 개선은 측정 불가.
8. **Delphi 의 "0.5 %" 는 동적 범위가 제한된 양에 대한 상대값**이다. 항상 ρ(달성 진전 대비
   비율)를 병기.
9. **link 의 유효 표본수는 런 7개**이지 체크포인트 18–25개가 아니다.
   binomial-only / 체크포인트 단위 구간을 헤드라인으로 인용하지 않는다.
10. **동일 패밀리 sigmoid 풀**은 두 단계가 데이터를 공유하므로 joint bootstrap(§6.3)이
    선택이 아니라 **필수**다. §5.6 의 교차패밀리 확장이 잃은 주장을 되찾는다.
11. `flops_per_token` 이 상위 스테이지에 `max_seqlens`(예산 패치 수)를 쓰므로 **C 는 상한**이다.
12. **작업 트리에 eval 경로 미커밋 변경 904줄** — 매니페스트에 `git_head` +
    `worktree_diff_sha256` 필수 기록.
13. 한 패밀리 · 한 데이터 믹스 · 한 아키텍처다. 적합된 `α, β, k, x0` 가 어디로든 전이된다는
    근거는 여기에 없다.
