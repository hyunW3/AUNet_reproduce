# Patch-Lag Residual Updater — 코드 기준 검토 및 실행 계획

원안: `plans/patch_lag_residual_updater_experiment_plan.md` (이하 "원안")
대상 체크포인트: `main/main/1.3B/aunet2_1.3B/checkpoints/0000180000` (= 프로브 태그 `aunet_static`)
작성: 2026-08-30

---

## 0. 요약 (원안 대비 무엇을 바꾸는가)

원안의 방향(백본 동결 + 경량 residual updater + gate + teacher distillation)은 이 코드베이스에
그대로 매핑된다. 다만 **순서와 범위**를 바꾼다.

1. **먼저 상한(oracle)을 잰다.** 원안은 Stage 0을 "구현 검증"으로 시작하지만, updater를 만들기
   전에 "완벽한 fresh 글로벌 벡터가 있으면 BPB가 얼마나 내려가는가"를 재야 한다. 이 값이 작으면
   이후 모든 단계가 무의미하다. 이미 있는 측정치로 계산한 도달 가능 손실 질량은 **전체의 45.9%**
   (0.406 of 0.884 BPB)이고, 그중 실제로 회수 가능한 몫은 아직 미지수다 (§2).
2. **학습 전에 closed-form ridge probe로 linear head의 상한을 본다.** teacher residual을 캐시한
   뒤 `delta_target ≈ W·features` 를 최소제곱으로 풀면, 학습 없이 linear head의 R²/cosine 상한이
   나온다. Stage 1의 2×2 스윕(원안 §11)은 이 결과를 보고 결정한다.
3. **파라미터 0개 베이스라인(M0)을 추가한다.** `delta = trans_down(down_norm(h_t)) − p_j`
   (현재 바이트로 새로 pool한 벡터 − 현재 패치 벡터). trunk 재계산 없이 "내용 주입"만 하는
   무학습 대조군으로, updater가 회수하는 것이 *내용*인지 *trunk 연산*인지 분리한다.
4. **Linear-attention updater(원안 §5.2)를 intra-patch causal attention으로 교체한다.**
   패치 길이가 짧고(평균 ~4.8 B, `max_pos=16`에서 wrap) 패치 경계마다 state reset이 필요한데,
   레포의 GLA 경로(`apps/aunet/linear_attn.py`, `fla.ops.gla.chunk_gla`)는 임의 위치 reset을
   지원하지 않는다. 패치 블록 대각 causal mask 한 장이 더 싸고 정확하다.
5. **Full-prefix 입력 범위(원안 §6.2)는 우선순위를 낮춘다.** `g_{p-1}`은 이미 trunk를 통과한
   벡터라 prefix 정보를 담고 있고, local hidden `res` 역시 512 B sliding window를 본다.
   current-patch-only가 실패했을 때만 본다.
6. **LoRA(원안 §7.2/7.3)는 Stage 4로 미룬다.** 레포에 LoRA 구현이 없다
   (`mla_kv_lora_rank`는 MLA latent이지 adapter가 아니다). Stage 1–3에서 이득이 확인된 뒤에만
   구현한다.
7. **"무해성"은 구조적으로 보장된다.** updater는 read staleness ≥ 1인 위치에만 붙으므로,
   staleness 0 위치(전체 바이트의 20.5%, **전체 손실의 54.1%**)의 예측은 비트 단위로 불변이다.
   회귀 위험은 나머지 45.9%에 국한된다 — 이 사실을 리포트의 sanity check로 쓴다.

---

## 0.1 실행 결과 (2026-08-30 ~ 09-01, ece-agpu11 A100 · snu55 A5000) — **완료, 음성**

| 단계 | 상태 | 결과 |
|---|---|---|
| Stage −1 oracle | 완료 | oracle(= Scratchpad) **+0.6172 BPB 악화**. prev1 +0.596 / look1 +0.749도 동일 수준 → 문제는 최신성이 아니라 **정렬**. 사전 중단 기준을 악화 방향으로 충족 |
| Stage 0 / §8.1 distillation | 완료 (부록 A) | teacher residual은 **예측 가능**하다 (cos 0.625→0.710, R²=0.29). 그런데 잘 맞출수록 task가 나빠진다: distill-only **+0.081**, task+repr **+0.0108** (task-only는 −0.0023). 폐기 사유는 "학습 불가"가 아니라 **"방향이 틀림"** |
| Stage 1 (task loss) | 완료 | M1 −0.0018 / M2 −0.0023 / M3 −0.0033 BPB (전부 유의하지만 상대 0.2~0.4%) |
| **C1 대조군** | 완료 | **현재 단어 정보가 0인 head가 M3와 동일한 −0.0033** (M3−C1 = −0.00004, CI [−0.00008, +0.00001], per-byte 상관 0.986) → **이득은 patch lag와 무관한 용량 효과** |
| positive control | 완료 | 정답 바이트를 주면 stale BPB 0.559 → 0.067 → 배선·대역폭 정상 |
| 도메인 전이 | 완료 | zh **+0.0025 (악화)**, code −0.0005. 영어 전용 적합 |
| 주입 지점 (global vs local) | 완료 | local(up-projection 이후, byte residual 옆)이 같은 이득을 **파라미터 3.2배·섭동 40%·게이트 50배** 적게 냄. 천장은 동일 |
| 용량 vs 정보 | 완료 | 입력만 다른 3-way에서 Δ가 **파라미터 수를 그대로 따라감** (0.53/1.05/1.58 M ⇒ −0.0020/−0.0024/−0.0026) |
| `patch_offset` 특징 추가 | 완료 | 효과 없음 (−0.0028 → −0.0028). frozen `IndexedMatMul`이 이미 오프셋별 전용 행렬 16.8 M로 쓰고 있음 |
| downstream (HS·ARC-C·BoolQ) | 완료 | updater 넷 모두 McNemar p ≥ 0.29. 유의하게 움직인 건 BPB가 나쁜 D2뿐이고 그마저 양방향 |
| **BPEByte-rg 재현** | 완료 | Stage −1 구조 동일(oracle +0.72 / prev1 +0.69 / look1 +0.85), Stage 1 treatment ≡ control 동률 재현 → **결론이 패칭 방식에 의존하지 않음** |
| Stage 2–4 (gate/objective/LoRA) | **착수 안 함** | 정보와 무관한 −0.0028 위에 쌓는 것이므로 근거 없음 |

**결론: frozen 계층형 바이트 모델에서 patch lag는 회수 가능한 손실이 아니다** (AU-Net word 패칭과
BPEByte-rg BPE-root 패칭 양쪽에서 확인). 전체 리포트는
`reports/patch_lag/summary.md`. 아래 §1–§9는 실행 전 원안이며, 위 표가 실제 결과다.

산출물: `lingua/apps/aunet/patch_lag.py`(teacher/oracle/updater), `scripts/probes/patch_lag_oracle.py`,
`patch_lag_verify.py`(T1–T4 정합성 테스트), `train_updater.py`, `report_patch_lag.py`,
`reports/patch_lag/*.npz` · `*.log` · `summary.md`.

---

## 1. 이 레포에서 patch lag가 정확히 무엇인가

`aunet2_1.3B`: 2-stage, local dim 512 (enc 3 / dec 3 layers, 512 B sliding window),
trunk dim 2048 × 25 layers, byte seqlen 8192, patch seqlen 3200, `pooling_type =
simple_indexed_matmul`, 패치 규칙 `regex.strategy = {"word1": "1@1"}` (word 단위).

- **down** (`hierarchical.py:384`): 패치 j의 글로벌 입력 `p_j = trans_down(down_norm(h_enc[start_j]))`.
  즉 패치 벡터는 **패치의 첫 바이트에서 pool된 encoder hidden 하나**다. 그 바이트 이후의
  현재 단어 내용은 글로벌 경로에 전혀 들어가지 않는다.
- **up** (`hierarchical.py:407`, repeat 분기 445–465): `repeat_idx = cumsum(mask)`로 바이트 t가
  자기 패치의 trunk 출력을 gather하고, `position = (t − start) % 16`을 `IndexedMatMul`
  (`W[16, 2048, 512]`, 16.8M params)의 인덱스로 써서 local 공간으로 되돌린다.
- 따라서 바이트 t가 읽는 글로벌 벡터의 **read staleness = t − start(patch(t))** 바이트다.
  staleness 0은 패치 첫 바이트에서만 발생한다.

원안의 `g_{p-1}` ↔ 여기서는 "자기 패치의 벡터인데 패치 시작 시점에 pool된 것"으로 읽으면 된다
(원안의 표기와 인덱스만 다르고 의미는 동일: 글로벌 경로가 현재 패치 내용만큼 뒤처져 있다).

주의: 예측 위치 인덱스를 헷갈리기 쉽다. `global_ablation.py`의 `pos_in_patch`는 **예측되는
바이트**의 패치 내 오프셋 r이고, 그 예측을 하는 **입력 바이트**의 read staleness는 r ≥ 1일 때
r−1, r = 0일 때 (이전 패치 길이 − 1)이다. 아래 통계는 전부 read staleness 기준으로 다시 집계했다.

### 1.1 예시 (실제 held-out window 0, `reports/bpb_windows.jsonl` idx 0)

텍스트: `Discipling with Joy!\n\nJesus was the ultimate disciple maker! ...`
`word1` 패치 경계(`|`가 패치 시작): `Jesus| was| the| ultimate| disciple| maker|!|`

**패치별로 글로벌 벡터가 무엇을 보았는가** — `p_j`는 패치의 *첫* 바이트(= 앞의 공백)에서 pool된다:

| 패치 | pool된 바이트 | 그 벡터가 본 텍스트(끝) | 이 벡터를 읽는 입력 바이트 | staleness |
|---|---|---|---|---|
| `" was"` | 27 | `…g with Joy!\n\nJesus ` | 27–30 | 0–3 |
| `" the"` | 31 | `…th Joy!\n\nJesus was ` | 31–34 | 0–3 |
| `" ultimate"` | 35 | `…oy!\n\nJesus was the ` | 35–43 | 0–8 |
| `" disciple"` | 44 | `…s was the ultimate ` | 44–52 | 0–8 |
| `" maker"` | 53 | `… ultimate disciple ` | 53–58 | 0–5 |
| `"!"` | 59 | `…ate disciple maker!` | 59–59 | 0 |

즉 `" ultimate"` 패치의 글로벌 벡터는 `"…Jesus was the "`까지만 담고 있고, **`u`,`l`,`t`,`i`,`m`,`a`,`t`,`e`
어느 글자도 담고 있지 않다.** 그런데 그 단어를 쓰는 동안 바이트 35–43은 전부 이 하나의 2048-d 벡터를
읽는다 (차이는 `IndexedMatMul`의 오프셋 인덱스 0–8뿐).

**바이트 단위 타임라인** (BPB는 실측, `aunet_static.npz`):

| 예측 바이트 | 입력 바이트 | 읽는 벡터 | 그 벡터가 못 본 것 | staleness | BPB full | BPB local_only |
|---|---|---|---|---|---|---|
| `u` | `" "`(35) | `p(" ultimate")` | — | 0 | 2.54 | 8.40 |
| `l` | `u`(36) | 동일 | `u` | 1 | 0.02 | 6.06 |
| `t` | `l`(37) | 동일 | `ul` | 2 | 0.00 | 0.03 |
| `i` | `t`(38) | 동일 | `ult` | 3 | 0.00 | 1.91 |
| … | … | 동일 | … | … | … | … |
| `" "` | `e`(43) | 동일 | `ultimate` | 8 | 0.02 | 0.27 |
| `d` | `" "`(44) | `p(" disciple")` | — | 0 | 0.92 | 5.14 |

읽는 법:

- **fresh 벡터를 받는 자리는 단어당 정확히 하나**, 그리고 그 자리는 하필 *단어의 첫 글자를
  예측하는 자리*다 (여기 `u`: BPB 2.54). 아키텍처가 신선도를 가장 비싼 바이트에 쓰고 있다.
- lag가 있는 자리(staleness ≥ 1)는 단어 내부 continuation이라 이미 BPB가 0에 가깝다
  (`l`,`t`,`i`,… = 0.00–0.02). §2의 45.9% 손실 질량이 넓게 흩어져 있다는 뜻이다.
- 글로벌 경로가 못 보는 것은 **"지금 쓰고 있는 단어" 딱 그것뿐**이다. 이전 단어들은
  `p_j`가 pool된 공백 바이트의 encoder hidden(512 B sliding window)에 이미 들어 있다.
- 그리고 그 "지금 쓰는 단어"는 **local decoder는 보고 있다**(3층 + 512 B window + 오프셋별
  `IndexedMatMul`). 따라서 진짜 결손은 "현재 부분 단어 × 512 B 밖 문맥"의 상호작용, 즉 trunk가
  해야만 하는 long-range 연산으로 국한된다. updater가 회수할 수 있는 몫이 작을 수 있는 이유이자,
  Stage −1 oracle을 먼저 재야 하는 이유다.

---

## 2. 이미 있는 측정치로 본 문제 크기 (`reports/global_ablation/aunet_static.npz`)

영어 held-out 320 windows / 1,228,796 bytes, teacher-forced, `full` 모드 기준 BPB 0.8844.

| read staleness (B) | 바이트 비중 | BPB | 전체 손실 중 비중 |
|---|---|---|---|
| 0 (fresh) | 20.5% | 2.337 | **54.1%** |
| 1 | 17.7% | 0.907 | 18.1% |
| 2 | 16.4% | 0.576 | 10.7% |
| 3 | 13.4% | 0.360 | 5.5% |
| 4 | 10.1% | 0.307 | 3.5% |
| 5–7 | 16.1% | ~0.31 | 5.5% |
| 8+ | 6.0% | 0.393 | 2.6% |
| **≥1 (updater 사정권)** | **79.5%** | **0.511** | **45.9%** (0.406 bits/byte) |

읽는 법:

- 손실의 절반 이상(54.1%)은 **staleness 0**, 즉 단어의 첫 글자를 예측하는 자리에 몰려 있고
  이 자리는 이미 fresh 벡터를 받는다. updater가 손댈 수 없고, 손대서도 안 된다.
- updater가 건드릴 수 있는 전체 손실은 0.406 bits/byte이며, 이는 "그 손실을 전부 없앴을 때"의
  물리적 상한이다. 현실적 이득은 이보다 훨씬 작다 — 그래서 §3의 oracle 측정이 필요하다.
- 참고로 `delay1`(모든 바이트가 한 패치 더 뒤를 읽음)은 BPB +1.370으로 `local_only`(+1.406)와
  거의 같다. **글로벌 벡터를 조금만 잘못 바꿔도 모델은 붕괴한다**는 뜻이고, gate/zero-init와
  "실제 벡터가 fresh인 위치는 절대 건드리지 않는다"는 원안의 규칙이 옳다는 강한 근거다.

---

## 3. 단계별 계획

### Stage −1. Oracle (= Scratchpad) 상한 측정 — **가장 먼저, 학습 없음**

원안의 B1(Scratchpad)과 B2(Oracle upper bound)는 이 아키텍처에서 **같은 것**이다: 바이트 t에서
`p̃(t) = trans_down(down_norm(h_enc[t]))`를 현재 패치 슬롯에 끼워 trunk를 다시 태운 출력이
teacher이자 도달 상한이다. 하나로 합쳐 측정한다.

**효율적 계산 (prefix-KV trick).** 오프셋 r마다 전 패치의 partial 벡터를 한 번에 처리한다:

1. 표준 trunk forward 1회 → 각 layer의 K/V 캐시(패치 시퀀스).
2. 오프셋 r ∈ {1..15}마다, 입력이 `p̃_j(start_j + r)`인 **query-only** stream을 25 layer에 흘린다.
   슬롯 j의 query는 (a) 캐시된 K/V 중 위치 < j (strict causal), (b) 자기 자신의 K/V 만 본다.
   softmax는 두 항을 concat해서 한 번에 정규화한다 (~15줄짜리 커스텀 attention; RoPE 위치는 j로 동일).
3. 그 오프셋의 teacher 벡터를 up()에 넣고 decoder까지 태워, staleness == r인 바이트의 NLL만 취한다.

비용 ≈ window당 full forward의 16배. 320 windows면 A5000 1장에서 수십 분.

측정 모드(전부 `global_ablation.py`의 `Ablation`과 같은 방식으로 추가):

| 모드 | 정의 | 질문 |
|---|---|---|
| `oracle_pos_r` | teacher 벡터 + `IndexedMatMul` position은 원래대로 r | 진짜 상한 |
| `oracle_pos_0` | teacher 벡터 + position 0 | position/벡터 페어링이 confound인지 |
| `M0_freshpool` | trunk 없이 `p̃(t)`의 up-projection만 교체 (무학습, 0 params) | "내용 주입"만으로 얼마나 되나 |

**중단 기준:** `max(oracle_pos_r, oracle_pos_0)`의 전체 BPB 개선이 **< 0.010 bits/byte**(상대 1.1%)
이거나, staleness ≥ 1 구간 BPB 개선이 < 3% 상대이면 **여기서 멈추고 음성 결과로 기록한다**
(`reports/patch_lag/summary.md` + `reports/ThingsToMake.md` 항목).

산출물: `reports/patch_lag/oracle.npz` (모드별 per-byte NLL + staleness 메타), `summary.md` 표.

### Stage 0. Teacher residual의 선형 예측 가능성 (ridge probe) — 학습 없음

Stage −1이 통과하면, teacher를 만들 때 (`h_enc[t]` 512-d, `up_norm(g_read)` 2048-d,
`up_norm(g_teacher)` 2048-d, staleness r)를 서브샘플로 캐시한다 (窗당 ~256 위치 × 2k windows
≈ 0.5M 샘플 ≈ 4 GB fp16).

- 특징 후보: `last = h_t`, `mean = 패치 시작~t 평균`, `mean+last`, `+ g_read`.
- `delta_target = up_norm(g_teacher) − up_norm(g_read)`에 대해 오프셋 r별/전체로
  ridge closed-form 해를 구하고 **R², cosine, ‖delta‖ 설명률**을 본다.
- 이게 원안 Stage 1의 "linear로 충분한가"에 대한 답을 GPU-시간 없이 준다.

**분기 기준:** cosine ≥ 0.6 → linear head로 Stage 1 진행. 0.3–0.6 → intra-patch attention head를
같이 준비. < 0.3 → 선형 residual 가설 자체를 재검토(패치 lag가 trunk 연산에 의존한다는 뜻).

### Stage 1. Updater 학습 (head only)

고정: current-patch only, mean+last pooling, global scalar gate(logit −3 초기화), residual
projection zero-init, distill + task 결합 손실, 백본 완전 동결, staleness 0 위치 skip.

비교 축(원안 Stage 1의 2×2를 축소):

| ID | head | 입력 요약 |
|---|---|---|
| M0 | (무학습) fresh pool 차분 | — |
| M1 | Linear (`[mean;last] → 2048`, ~2.1M params) | current patch |
| M2 | Intra-patch causal attention 1 layer (d=256, ~1M params) | current patch |
| M1f | Linear | full prefix (cumulative mean) — 후순위 |

### Stage 2. Gate / 요약 어블레이션
Stage 1 최상위 모델에 대해서만: gate ∈ {fixed 1, global scalar, input-dependent scalar,
vector}, 요약 ∈ {last, mean, mean+last, +g_read}. 원안 §11 Stage 2와 동일하되 대상은 1개 모델.

### Stage 3. Objective 어블레이션
distill only / task only / combined (원안 §8). combined의 λ는 1개 값만 튜닝한다.

### Stage 4. LoRA (조건부)
Stage 1–3에서 유의미한 이득이 확인된 경우에만. 부착 위치는 `transitions[0].trans_up`,
`transitions[0].indexed_linear`, decoder block의 Q/V (r ∈ {4, 8, 16}).

---

## 4. 구현 설계

### 4.1 새 파일

```
lingua/apps/aunet/patch_lag.py     # PatchLagUpdater / attach_updater / teacher_globals
scripts/probes/patch_lag_oracle.py # Stage -1 (oracle / M0 / teacher 캐시 덤프)
scripts/probes/train_updater.py    # Stage 1-3 단일 GPU 학습 루프
scripts/probes/report_patch_lag.py # staleness별 표 + paired bootstrap CI
```

기존 파일 수정은 **두 곳뿐**:

- `hierarchical.py`: `SimpleTransition.__init__`에 `self.patch_lag_updater = None` 슬롯 추가,
  `up()`의 두 분기(학습 462행 근처 / 생성 425행 근처)에서 gather 직후 훅 호출. updater가 None이면
  기존 코드와 비트 동일 — 실행 중인 학습 파이프라인에 영향 없음.
- `global_ablation.py`: `MODES`에 `oracle_*`, `M0_freshpool`, `updater:<path>` 추가.

`train.py`는 건드리지 않는다 (프로덕션 런의 dataloader/forward 시그니처 위험 회피).

### 4.2 주입 지점 (정확히)

```python
# hierarchical.py:462 근처, repeat 분기 (학습/티처포싱)
x = self.up_norm(x)                                   # [B,P,2048]
x = x.gather(1, repeat_idx.unsqueeze(-1).expand(-1,-1,self.dim_out))   # [B,T,2048]
if self.patch_lag_updater is not None:                # <-- 여기
    raw_pos = self.position - idx.gather(1, repeat_idx)   # mod 이전의 진짜 staleness
    x = self.patch_lag_updater(x, res, raw_pos)           # res = encoder hidden [B,T,512]
return self.trans_up(x) + self.indexed_linear(x, position)
```

- RMSNorm은 position-wise라 `gather(up_norm(x)) == up_norm(gather(x))`. 따라서 **post-norm 공간**에
  주입하고 teacher도 `up_norm(g_teacher)`로 정의하면 스케일이 맞는다.
- `res`는 encoder 출력(`hierarchical.py:702`에서 push, `up()`의 3번째 인자)이라 현재 패치 바이트의
  local hidden이 이미 손 안에 있다. 추가 forward 없음.
- 마스크: `m = (raw_pos >= 1)`. `raw_pos == 0`이면 delta를 그대로 0으로 만든다 (mod 16 wrap 때문에
  `position`이 아니라 `raw_pos`를 써야 한다 — staleness 16이 0으로 alias되는 버그 방지).

### 4.3 Updater 모듈

```python
class PatchLagUpdater(nn.Module):        # dim_local=512, dim_global=2048
    # summary: cumsum 트릭으로 패치 시작~t의 mean을 O(T)로 계산 (루프 없음, compile 친화)
    #   c = res.cumsum(1); start_c = c.gather(1, start_idx)   # start_idx = t - raw_pos
    #   mean = (c - start_c + res.gather(start_idx)) / (raw_pos + 1)
    # delta = W_delta @ LN([mean; last])          W_delta zero-init
    # alpha = sigmoid(a)  (global scalar, a init -3 → alpha ~ 0.047)
    # return x + m * alpha * delta
```

intra-patch attention 변형(M2)은 같은 인터페이스에 1-layer causal attention을 쓰되,
attention mask를 `same_patch(q,k) & (k <= q)` 블록 대각으로 준다 (flex_attention 또는
패치 최대 길이로 잘라낸 sdpa). GLA/`fla` 경로는 쓰지 않는다 (패치별 state reset 미지원).

### 4.4 Teacher 계산

`patch_lag.py:teacher_globals(model, ids, lm, offsets)` — §3 Stage −1의 prefix-KV 절차.
검증 항목:

- 오프셋 0에서 teacher == 표준 trunk 출력과 **비트 동일** (구현 정당성 회귀 테스트).
- causality: 슬롯 j의 query가 위치 > j를 절대 보지 않음 (마스크 단위 테스트).
- `p̃_j(start_j)`가 `down()`의 결과와 동일 (pool 경로 일치).

---

## 5. 학습 레시피

- **하드웨어**: 로컬 `gpusvr0908`의 A5000 24 GB 1장 (GPU 0/1 유휴 확인). 다른 노드는
  기존 GPU 사용 정책을 따른다.
- **메모리**: 1.3B bf16 = 2.6 GB + 동결 백본(옵티마이저 state 없음). task loss는 백본을 통과하는
  backward가 필요하므로 `bs=1, seqlen 4096`부터 시작해 맞춘다.
  `selective_activation_checkpointing`은 금지(AU-Net in-place 변형과 충돌, 설정 파일 주석 참조).
- **2-phase 학습**: (a) distill-only — 캐시된 teacher로 head만 학습, 백본 backward 없음 → 매우 저렴.
  (b) combined — (a)의 head로 초기화 후 낮은 LR로 task loss 포함.
- **데이터**: `data/dclm_baseline_1.0_2shards_shuffled` (397 GB, 로컬). eval window와 겹치지 않는
  chunk를 쓴다 (`scripts/probes/build_windows.py`가 쓴 소스와 대조 확인 필수).
- **예산**: head 파라미터가 ~2M이므로 0.5–2 B bytes면 충분할 것으로 본다. 먼저 0.2 B로
  learning-curve를 보고 늘린다.
- **모니터링** (원안 §9 그대로): `mean/std(alpha)`, `‖delta‖`, `‖alpha·delta‖/‖g_read‖`,
  `cos(g_hat, g_teacher)`, staleness별 CE.
- **주의**: 백본 파라미터만 `requires_grad=False`로 두고, `g_hat` 이후 그래프를 `no_grad`로
  감싸지 않는다 (원안 §9의 지적이 정확하다).

---

## 6. 평가

**1차 지표 (모두 기존 harness 재사용, 숫자 직접 비교 가능)**

- `reports/bpb_windows.jsonl` (영어 320 windows)에서 staleness별 BPB + **paired bootstrap CI**
  (window 단위 resample, `scripts/probes/bootstrap_ci_report.py`).
- Recovery Ratio = (BPB_method − BPB_full) / (BPB_oracle − BPB_full), staleness bin별로 보고.
- **무해성 체크**: staleness 0 위치의 NLL이 `full`과 비트 동일한지 assert (구조상 보장, 회귀 테스트로 고정).
- Representation: `cos(g_hat, g_teacher)`, `MSE`, 오프셋 r에 따른 추이.

**2차 지표**

- 다른 도메인 window: `reports/bpb_windows_zh.jsonl`, `reports/windows/{te,code,despace_*}`
  — 패치 길이 분포가 다른 영역에서 무너지지 않는지.
- Downstream: `apps/aunet/eval.py`의 5-bench 세트 (Stage 1 우승 모델에만).
- 검색성 진단: NIAH / copy 계열(`scripts/probes/mk_niah.py`)에서 oracle 이득이 어디에 몰리는지.
  oracle 이득이 "retrieval" 쪽이면 linear head로는 회수 불가라는 신호다.
- 효율: 추가 params / FLOPs per byte / decode latency (`apps/aunet/bench/bench_decode.py`),
  Scratchpad(= trunk 1회 추가/byte) 대비 절감률.

---

## 7. 리스크와 대응

| 리스크 | 근거 | 대응 |
|---|---|---|
| 상한 자체가 작다 | 손실의 54.1%가 fresh 위치에 몰려 있음 | Stage −1을 먼저, 중단 기준 명시 |
| 글로벌 벡터 섭동에 모델이 극도로 민감 | `delay1` = +1.37 BPB ≈ `local_only` | zero-init + 작은 gate + staleness 0 skip + 무해성 assert |
| linear head가 trunk의 cross-patch 연산을 흉내낼 수 없음 | teacher는 25-layer attention의 출력 | Stage 0 ridge probe로 사전 판정, M0로 "내용 vs 연산" 분리 |
| decoder가 이미 local 바이트를 봄 → 이득 중복 | dec 3층 + 512 B window + per-offset IndexedMatMul | oracle이 이미 이 중복을 반영한 상한이므로 Stage −1이 곧 답 |
| teacher 구현 버그 | 커스텀 attention | 오프셋 0 비트 동일 테스트 + causality 테스트 |
| teacher 캐시 용량 | 위치당 2048×2 B | 전량 캐시하지 않고 서브샘플(ridge용) + on-the-fly(학습용) |

---

## 8. 일정 (단일 GPU 기준, 대략)

| 단계 | 작업 | 소요 |
|---|---|---|
| 0 | `patch_lag.py` teacher + oracle 모드 구현/검증 | 1.5–2 d |
| 1 | Stage −1 oracle 측정 + 리포트 | 0.5 d (GPU 수십 분) |
| 2 | Stage 0 ridge probe | 0.5 d |
| 3 | Stage 1 학습 (M0/M1/M2) + 평가 | 2–3 d |
| 4 | Stage 2–3 어블레이션 | 2 d |
| 5 | Stage 4 LoRA (조건부) | +2 d |

Stage −1/0에서 중단되면 총 3일 이내에 "왜 안 되는지"까지 포함한 음성 결과가 남는다.

---

## 9. 체크리스트

- [ ] `SimpleTransition`에 updater 훅 추가 (None이면 비트 동일 — 회귀 테스트로 고정)
- [ ] `raw_pos`(mod 이전) 기반 staleness 마스크
- [ ] `teacher_globals` prefix-KV 구현 + 오프셋 0 비트 동일 테스트 + causality 테스트
- [ ] `oracle_pos_r` / `oracle_pos_0` / `M0_freshpool` 모드 및 리포트
- [ ] **중단 기준 판정** (개선 < 0.010 BPB → 종료 및 기록)
- [ ] teacher 서브샘플 캐시 + ridge probe (R²/cosine)
- [ ] Linear head / intra-patch attention head, gate 3종
- [ ] distill / task / combined 손실
- [ ] staleness별 BPB + paired bootstrap CI 리포트
- [ ] 무해성 assert (staleness 0 위치 NLL 불변)
- [ ] decode 경로(`generate.py` up() 호출)에 updater 연결 + latency 측정
- [ ] 도메인 일반화(zh/te/code/despace) 확인
