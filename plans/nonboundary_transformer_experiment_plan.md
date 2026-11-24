# Non-Boundary Transformer `T_N` — 실행 계획

원안: `brainstorming/bpebyte_boundary_nonboundary_experiment.md`
선행: `plans/patch_lag_residual_updater_implementation_plan.md`, `reports/patch_lag/summary.md`
노드: **ece-agpu18** (A100-80GB ×8, GPU **0,5,6,7만** 사용)
작성: 2026-09-07

---

## 0. 요약 — 원안 대비 무엇을 바꾸는가

원안의 구조(경계 트랜스포머 `T_B` + 패치 내부 전용 `T_N` + mean fusion)는 그대로 간다.
바꾸는 것은 **arm 구성**이다. 이 레포는 이미 같은 가설을 두 번 쳐 봤고, 그 결과가 어떤
대조군이 필수인지를 정해 준다.

1. **Frozen 백본에서는 죽은 가설이다.** 1.3B 동결 체크포인트에 사후 이식하면 oracle(= Scratchpad)
   조차 **+0.617 BPB 악화**했고, 경량 head는 −0.0026에서 포화했으며, **현재 패치 정보가 0인
   대조군(C1)이 treatment와 −0.00004 차이로 동률**이었다. 문제는 정보가 아니라 정렬이었다.
2. **함께 학습하면 부호가 뒤집힌다.** 같은 개입을 처음부터 같이 학습한 100M 블록에서
   `scratchpad1` −0.0116, 50M updater −0.0197. 그래서 이 실험(= 처음부터 공동 학습)은
   할 가치가 있다.
3. **그런데 아직 답이 없는 두 질문이 있고, 그게 이 실험의 진짜 축이다.**
   `reports/patch_lag/sp_eval/*.json`에 있으나 `summary.md`에는 아직 반영되지 않은 snu20 블록:

   | arm (snu20 블록, 2,000 steps) | 총 params | BPB | Δ vs base |
   |---|---|---|---|
   | base | 98.6 M | 1.2592 | — |
   | `l3o_50m` — 50M updater head | 149.3 M | 1.2405 | −0.0187 |
   | `l3o_50m_gread` — **정보 0** 대조군 (같은 50M) | 149.3 M | 1.2544 | −0.0048 |
   | `base_150m` — **같은 파라미터를 trunk 깊이에** ([3,17]) | 148.2 M | **1.2384** | **−0.0208** |

   읽는 법: 현재-패치 정보는 공동 학습에서 실제로 값어치가 있다(−0.0187 vs −0.0048 = **−0.0139**).
   frozen에서와 달리 정보 효과가 살아 있다. **하지만 같은 파라미터를 그냥 trunk 깊이에 쓰면
   −0.0208로 더 낫다.** 즉 "patch lag를 고치는 것"이 "모델을 키우는 것"을 아직 못 이겼다.

**따라서 이 실험이 답해야 하는 질문은 "T_N이 baseline을 이기는가"가 아니라
"T_N이 *같은 예산의 더 깊은 trunk*를 이기는가"이다.** arm 표(§3)는 그 질문에 맞춰 짰다.

원안 §9의 3-arm(base / 2L / 3L)에 대조군 둘을 추가하고, 원안 Phase B의 `No-anchor`를
후순위가 아니라 **1차 웨이브**에 넣는다 — frozen에서 정확히 그 대조군이 결론을 뒤집었기 때문이다.

---

## 1. 구조와 구현

### 1.1 코드 위치

| 파일 | 내용 |
|---|---|
| `lingua/apps/aunet/nonboundary.py` | `NonBoundaryTransformer` / `NonBoundaryBlock` (신규) |
| `lingua/apps/aunet/hierarchical.py` | `nb_*` args · `_nonboundary_trunk_and_up` · init · FSDP plan |
| `lingua/apps/aunet/test_nonboundary.py` | 정합성 테스트 5종 (§1.4) |
| `lingua/apps/aunet/configs/nb/_gen.py` | arm config 생성기 (**YAML 직접 수정 금지**) |
| `lingua/nb_train.sh` | 한 arm = 한 GPU 쌍 |

### 1.2 전방 경로

`글로벌 경로는 손대지 않는다.` trunk는 여전히 패치당 한 항목만 보고, `T_N`의 출력은 trunk로
되돌아가지 않는다 (원안 §4의 `feed_into_next_boundary: false`). 바뀌는 것은 **패치 내부 바이트가
내려오는 길에 무엇을 읽는가** 뿐이다.

```
q_{i,j} = e_{i,j} + W_BN · RMSNorm(B_i)          # 경계 조건화 (anchor)
N_{i,j} = T_N(q ; 현재 패치의 KV만)               # causal, 경계마다 리셋
G_{i,j} = (B_i + W_NB · RMSNorm(N_{i,j})) / 2     # 오프셋 1..15 (mean fusion)
G_{i,0} = B_i                                     # 경계는 정확한 벡터를 그대로
```

`B_i = g_read`는 그 바이트가 원래 읽던 trunk 출력이고, `e_{i,j} = res`는 encoder 출력이다.
둘 다 이미 손 안에 있으므로 **추가 forward는 없다**.

### 1.3 두 가지 설계상의 성질

**(a) step 0 == baseline, 정확히.** `W_NB`(`out_proj`)를 zero-init하면 비경계 위치에서
`G = 0.5·B_i`가 되는데, 바로 다음 연산이 `up_norm`(RMSNorm)이라 **균일한 0.5는 상쇄된다.**
따라서 모든 arm이 완전히 같은 모델에서 출발하고, 차이는 전부 학습이 찾아낸 것이다.
`init_weights`에서 `non_boundary`를 **맨 마지막에** 초기화하므로 백본 가중치 draw 순서도 동일하다.

**(b) FLOPs가 바이트 선형으로 유지된다.** 모든 projection(wq/wk/wv/wo/FFN)은 T개 바이트 위치에서
정확히 한 번씩 돈다. 패치 블록으로 모으는 것은 attention 뿐이고, 길이 16짜리 attention은 무시할
수준이다. 패치 블록에서 projection까지 돌리는 순진한 구현은 같은 일에 3.4배를 쓴다.

**attention scoping**: 바이트를 `[B, P, 16, ...]` 블록으로 gather한다 — 블록 `p`의 슬롯 `o`는
바이트 `start_p + o`. 그 안에서 평범한 causal SDPA를 돌리면 슬롯 `o`는 슬롯 `0..o`,
즉 `start_p..start_p+o` = **현재 패치의 지금까지**만 본다. `is_causal` 외의 마스크가 필요 없다.
패치 끝을 넘어간 슬롯은 계산되지만 아무도 읽지 않고, 실제 바이트가 쓰는 슬롯이 그것을 볼 수는
없다(causality). `ctx=16`은 `IndexedMatMul`의 `max_pos`가 wrap하는 지점과 같게 맞춘 것이다.
오프셋 ≥ 16인 바이트는 `G = B_i`로 baseline과 동일하게 둔다.

**위치 정보**: 패치 내 오프셋 0..15의 학습된 임베딩을 `T_N` 입력에 더한다. RoPE 대신 절대 오프셋을
쓰는 이유는 모델이 이미 `IndexedMatMul`에서 같은 인덱스를 쓰고 있고, 16칸짜리 창에서는
절대 위치가 곧 상대 위치이기 때문이다.

### 1.4 정합성 테스트 (`test_nonboundary.py`, 전부 PASS)

| 테스트 | 결과 |
|---|---|
| `nb_layers=0`이 기존 코드와 동일 경로 | 구조상 보장 (dispatch가 gate) |
| `nb_layers ∈ {1,2,3}` 및 no-anchor가 **init에서 baseline과 동일** | max\|Δ\| = 2.6e−05 (fp32 RMSNorm 반올림) |
| `W_NB ≠ 0`이면 모델이 실제로 움직인다 | max\|Δ\| = 1.47 |
| causal: t+1 이후를 바꿔도 0..t 출력 불변 | max\|Δ\| = 0 |
| patch-scoped: 이전 패치 바이트를 바꿔도 현재 패치 `T_N` 출력 불변 | max\|Δ\| = 0 |
| 경계 바이트(오프셋 0)는 `g_read`를 비트 그대로 유지 | 정확히 일치 |
| 블록 인덱스 왕복: 슬롯 (p,o) ↔ 바이트 `start_p+o` | 전 바이트 일치 |

---

## 2. 학습 레시피

`r10lr_rgqkmuon_100M_m10` — 현재 이 레포의 100M 최고 arm — 을 그대로 쓴다. 원안 §2와 동일.

- BPEByte-rg (llama3 online greedy, `bpe_online_placement: root`), 바이트 seqlen 8192
- local 512×3 / trunk 768×10, `simple_indexed_matmul`, QK-norm
- Muon lr 1.0e-2, muon/adamw ratio 0.1, wd 0.1, clip 0.2, WSD, decay 0.1, seed 777
- **글로벌 배치 768 seq × 8192 B = 6.29 MB/step 고정** (2 GPU × bs 8 × grad_acc 48)
- 데이터 `dclm_baseline_1.0_2shards_shuffled` (ece-agpu18 로컬)

> **world_size를 arm 사이에서 절대 바꾸지 않는다.** `find_and_sanitize_chunks`는 `n_chunks >
> world_size`일 때 앞에서 `world_size`개만 취한다 — 2 GPU면 chunk 00–01, 4 GPU면 00–03을 본다.
> world_size가 다른 두 run은 **다른 데이터로 학습된 것**이라 BPB를 직접 비교할 수 없다.
> 그래서 모든 arm이 2 GPU × grad_acc 48이다.

**예산: 3,344 steps × 6.29 MB = 21.0 GB** (기록된 `rgqkmuon_m10` 런과 동일).

데이터도 같다: `find_and_sanitize_chunks`가 world_size 2에서 고르는 두 청크가 chunk **00, 01**로,
r10lr 스윕이 쓰던 `R10LR_CHUNKS`와 정확히 같은 파일이다 (확인함). 차이는 rank→chunk 배정뿐이다
(4랭크가 2청크를 나눠 갖던 것이 2랭크 1:1이 됨) — 시퀀스 패킹 순서는 달라지지만 데이터 풀은 같다.

> **주의 — 1.1044와 1.1384는 같은 지표가 아니다.** r10lr 장부의 1.1044는
> `eval_val_bpb.yaml`(학습 청크 옆의 `*.val.jsonl`, 200 문서)에서 나온 값이고, 이 실험이 쓰는
> 1.1384는 `reports/bpb_windows.jsonl`(영어 320 windows) 프로토콜이다. 두 숫자를 직접 빼면 안 된다.

**외부 검증은 포기했다 (2026-09-07).** `eval_val_bpb.yaml`을 `nb_base`에 돌려 봤으나
`nll_per_byte = −2.066` → **2.98 bits/byte**가 나왔다. 이 트리에서 harness 호출이 잘못 구성된
것이고(패칭 설정이 checkpoint에서 제대로 실려 오지 않은 것으로 보인다), 모델 문제가 아니다:

| 근거 | 값 |
|---|---|
| `nb_base` 최종 학습 loss (step 3340) | 0.7746 nats/byte = **1.117 bits/byte** |
| 320-window held-out BPB | **1.1384** |
| downstream (HS 31.3 · ARC-E 35.9 · BoolQ 60.6) | 전부 우연 이상 |

학습 loss와 held-out BPB의 간격(+0.021)이 정확히 예상 크기다.
`runs/nb/_valbpb_INVALID_harness_misconfigured/` 산출물은 **읽지 말 것**.
이 실험의 결론은 5개 arm 사이의 320-window 비교에만 의존하고, 그 안에서는 모든 arm이
같은 스코어러·같은 windows·같은 데이터를 쓴다.

---

## 3. Arm 표

전부 같은 생성기(`configs/nb/_gen.py`)에서 나오므로, YAML 사이의 차이는 아래 열이 전부다.

| # | arm | `layers` | `nb_layers` | anchor | 이 arm이 답하는 질문 |
|---|---|---|---|---|---|
| 1 | `nb_base` | [3, 10] | 0 | — | 이 블록의 기준점 |
| 2 | `nb_2l` | [3, 10] | 2 | ✔ | 원안의 주 제안. `T_N` = 512 × 2L |
| 3 | `nb_2l_noanchor` | [3, 10] | 2 | ✘ | **이득이 patch lag 때문인가, 그냥 추가 바이트 연산 때문인가** |
| 4 | `nb_3l` | [3, 10] | 3 | ✔ | 필요한 updater 용량 (원안 RQ4) |
| 5 | `nb_base_deep` | **[3, 11]** | 0 | — | **같은 파라미터를 trunk 깊이에 쓰면?** (§0의 미해결 질문) |

**실측 파라미터** (5개 config 전부 인스턴스화해 확인, FSDP grouping plan 항목도 전부 실제 모듈로 해석됨):

| arm | 총 params | vs `nb_base` | 그중 `T_N` |
|---|---|---|---|
| `nb_base` | 98.59 M | — | — |
| `nb_2l` | **106.21 M** | +7.61 M | 7.61 M |
| `nb_2l_noanchor` | 105.81 M | +7.22 M | 7.22 M (`W_BN`+`boundary_norm` 0.39 M만 빠짐) |
| `nb_3l` | 109.62 M | +11.02 M | 11.02 M |
| `nb_base_deep` | **105.67 M** | +7.08 M | — (trunk 11층) |

→ `nb_base_deep` 105.67 M vs `nb_2l` 106.21 M — **0.5% 이내로 파라미터가 맞는다.**
**FLOPs는 맞지 않는다** — trunk 층은 패치 위치(~1,750)에서, `T_N`은 바이트 위치(8,192)에서 돈다
(§4.1의 +7% vs +67%). 두 수치를 §5.4에서 나란히 보고한다.

### 3.1 추가된 arm (2026-09-08, 1차 결과 이후)

1차 5개 arm의 결과(`reports/nb/summary.md`)가 두 가지 새 질문을 만들었다.

| # | arm | anchor | 무엇을 답하는가 |
|---|---|---|---|
| 6 | `nb_2l_const` | 학습된 768-d **상수** | anchor가 "패치마다 달라지는 신호"라서 듣는가, 그냥 입력 슬롯 하나라서 듣는가 |
| 7 | `nb_2l_detach` | `B_i.detach()` | 내용은 그대로, **gradient 경로만** 제거 |
| 8 | `nb_base_s778` | — (seed 778 반복) | 이 블록의 **학습 run-to-run σ** |

`const`는 `boundary_norm`/`boundary_proj`를 그대로 두고 입력만 상수로 바꾼다 — 파라미터가
`nb_2l`(106.209 M)과 **+768로 정확히 맞는다**(106.210 M). 상수를 `x`에 바로 더하면 다음 norm이
흡수해 버리고 파라미터도 안 맞으므로 그렇게 하지 않았다.

읽는 법:

- `const ≈ detach ≈ boundary` → 파라미터/입력 슬롯 효과. 내용도 gradient도 무관.
- `const ≈ noanchor` ≪ `detach ≈ boundary` → **패치 내용이 원인** (원안 가설 지지).
- `const ≈ detach ≈ noanchor` ≪ `boundary` → **gradient 경로가 원인** (아키텍처가 아니라 최적화).

**`nb_base_s778`이 왜 필요한가.** §5.1의 paired bootstrap CI는 *같은 체크포인트를 평가할 때의*
노이즈만 잰다. 같은 config를 다른 seed로 **학습**했을 때의 분산은 거기 들어 있지 않다.
`nb_base_deep`의 −0.0035 같은 작은 Δ는 그 값을 모르면 유의하다고 말할 수 없다.
한계: top-level seed만 바꾸고 `data.seed`는 42 그대로이므로 (레포가 `rg/wsd20s778`에서 쓴 것과
같은 방식, σ ≈ 0.012와 비교 가능) 이렇게 잰 σ는 **하한**이다.

`nb_2l_noanchor`는 파라미터와 FLOPs가 `nb_2l`과 **정확히** 같고 (`W_BN`과 `boundary_norm`
0.39 M만 빠진다), 입력에서 `B_i`만 사라진다. frozen 실험에서 이 형태의 대조군이 treatment와
동률이었으므로, `nb_2l − nb_2l_noanchor`가 이 실험의 **1차 결과값**이다.

---

## 4. 실행 계획 (ece-agpu18, GPU 0,5,6,7)

사용자 지시: **GPU 0,5,6,7만 사용한다.** 다른 GPU(1,2,3,4)는 다른 사용자/작업이 쓰고 있거나
사용 금지다. 여유 VRAM은 허가가 아니다 (`gpu-node-usage-policy`).

### 4.1 실측 throughput (2 × A100-80GB, step 20 → 40, compile on)

| arm | s/step | vs base | peak mem/GPU | 3,344 steps |
|---|---|---|---|---|
| `nb_base` | **9.33** | 1.00x | 17.6 GB | **8.7 h** |
| `nb_2l` | **15.56** | **1.67x** | 27.5 GB | **14.5 h** |
| `nb_base_deep` (추정) | ~10.0 | 1.07x | ~18 GB | ~9.3 h |
| `nb_3l` (추정) | ~18.7 | 2.00x | ~32 GB | ~17.4 h |

**`T_N`은 "lightweight"가 아니다.** d=512 층 하나가 바이트 위치 8,192에서 도는 비용은
step당 약 3.1 s로, baseline 전체 step(9.33 s)의 1/3이다. 2층이면 **+67% wall-clock**이고
파라미터는 +7.7%(7.61 M)에 불과하다 — 원안이 "cheap refresh"라고 부른 것의 실제 가격이며,
§5.4에서 정면으로 보고할 숫자다. (구현 자체는 최적이다: 측정된 3.1 s/층이 d=512 층의
fwd+bwd FLOPs에서 계산한 값과 일치한다. 비용은 gather 오버헤드가 아니라 연산이다.)

이 때문에 `nb_base_deep`과의 비교는 파라미터로는 공정하지만 **연산으로는 `T_N`에게 매우 불리하다**
— 같은 7 M을 trunk에 쓰면 +7%, `T_N`에 쓰면 +67%다. 결과 해석에 반드시 붙인다.

### 4.2 레인 배치

| 레인 | GPU | 큐 | 누적 |
|---|---|---|---|
| A | 0,5 | `nb_base` (8.7 h) → `nb_2l_noanchor` (14.5 h) → `nb_base_deep` (9.3 h) | ~32.5 h |
| B | 6,7 | `nb_2l` (14.5 h) → `nb_3l` (17.4 h) | ~31.9 h |

읽을 수 있게 되는 시점:

- **t ≈ 14.5 h** — `nb_base` + `nb_2l`. "`T_N`이 baseline을 이기는가"
- **t ≈ 23 h** — `+ nb_2l_noanchor`. **결정적 대조군**, "이득이 patch lag에서 오는가"
- **t ≈ 32 h** — `+ nb_3l`, `+ nb_base_deep`. 용량 사다리와 "trunk에 쓰는 게 낫지 않은가"

중간 판독은 기다릴 필요가 없다 — 롤링 체크포인트(500 step)가 항상 하나 있으므로 학습 중에도
그 시점 모델을 평가할 수 있다.

### 4.3 운영 규칙

- 사용자 지시: **GPU 0,5,6,7만.** 다른 카드는 비어 보여도 건드리지 않는다
  (`gpu-node-usage-policy`: 여유 VRAM은 허가가 아니다).
- 각 run은 `setsid`로 띄우고 pid를 `runs/nb/_logs/<arm>.pid`, 레인 pgid를
  `lane_<gpus>.pgid`에 남긴다. **중단은 그 pid로만** — 이 계정에 다른 사람 작업이 있을 수 있어
  `pkill` 금지.
- 체크포인트 `every: 500`, `keep: 1` → 중단 시 손실 < 500 step (~1.3 h). `train.py`가 자동
  resume하므로 레인 스크립트를 다시 띄우면 이어서 간다.
- 30–60분마다 GPU 점유 재확인. 우리 4장 밖으로 번지거나 다른 사용자가 들어오면 즉시 정지.

## 5. 평가

원안 §11–§13. 전부 기존 harness 재사용이라 숫자가 선행 결과와 직접 비교된다.

### 5.1 1차: held-out BPB
`reports/bpb_windows.jsonl` 영어 320 windows / 1,228,796 bytes,
`scripts/probes/eval_sp_models.py` 그대로. window 단위 paired bootstrap CI
(`scripts/probes/bootstrap_ci_report.py`).

### 5.2 핵심 진단: patch-offset 조건부 BPB
바이트를 자기 패치 내 오프셋 `r`로 나눠 `BPB(r)`을 보고하고, `ΔBPB(r) = BPB_base(r) − BPB_ours(r)`
를 그린다. 가설이 맞다면 **`r`이 커질수록 Δ가 커져야** 한다.

- `r = 0`은 이미 fresh 벡터를 받는 자리이고 `T_N`이 구조적으로 손대지 않으므로
  **`r = 0`의 BPB는 arm 사이에서 (학습 차이를 빼면) 움직일 이유가 없다** — sanity check.
- 참고로 frozen 1.3B word-patching에서는 손실의 **54.1%가 `r = 0`에 몰려** 있었다.
  BPEByte-rg에서 같은 분포를 먼저 확인하고 표에 넣는다.

### 5.3 patch-length 조건부 BPB
`P = 1, 2, 3, ≥4`로 나눈다. **`P = 1` 패치에는 비경계 위치가 없으므로 base와 차이가 없어야 한다**
(원안 §12의 internal control). 긴 패치에서 이득이 커지면 patch-lag 가설을 직접 지지한다.

### 5.4 compute
추가 params · FLOPs/byte · 학습 throughput(bytes/s) · peak memory ·
persistent KV(패치) vs temporary KV(패치 내부) 크기. `nb_base_deep`과 나란히 놓는다.

### 5.5 downstream
HellaSwag / ARC-E / BoolQ. **단, 100M·짧은 예산에서 이 셋은 arm을 분해하지 못한다**는 것이
선행 블록에서 확인됐다(BPB 0.0008 차이인 두 arm의 avg3가 3.7 pt 벌어짐). 참고치로만 싣는다.

---

## 6. 판정 기준

### 성공 (원안 §14보다 엄격하게)
1. `nb_2l < nb_base` (paired bootstrap CI가 0을 넘지 않음), **그리고**
2. **`nb_2l < nb_2l_noanchor`** — 이득이 경계 조건화된 refresh에서 온다, **그리고**
3. `ΔBPB(r)`이 `r`에 대해 증가, `r = 0`에서는 평평, **그리고**
4. 긴 패치에서 이득이 크고 `P = 1`에서는 없음.

2번이 없으면 1·3·4가 있어도 "바이트 연산을 더 넣으면 좋아진다"는 이미 아는 사실의 재확인이다.

### 조기 중단
- `nb_2l ≈ nb_2l_noanchor` (Δ의 CI가 0을 포함)이면 **`nb_3l`을 취소한다.** 용량 사다리를
  정보와 무관한 축 위에 쌓는 것이기 때문이다. 대신 그 슬롯을 `nb_base_deep`에 준다.
- `nb_2l`이 `nb_base`보다 나쁘면 즉시 정지하고 원인부터 (step 0 동일성은 테스트로 고정돼 있으므로
  학습 불안정 / gate 폭주 / fusion 스케일을 먼저 본다).

### 해석 규칙
`nb_base_deep ≤ nb_2l`이면 — 즉 같은 파라미터의 더 깊은 trunk가 `T_N`만큼 좋으면 —
결론은 "patch lag는 실재하지만 **가장 싼 해법이 아니다**"이지 "patch lag가 없다"가 아니다.
snu20 블록(§0)이 이미 그 방향을 가리키고 있고, 이 실험은 *구조화된* `T_N`이 그 격차를
뒤집는지를 본다.

---

## 7. 범위 밖 (원안 §15–§17 중 미룸)

- 1B 확장 (원안 Phase D) — 100M에서 §6의 2번을 통과한 뒤에.
- `feed_into_next_boundary: true` (Consolidation), learned gate, residual-sum fusion —
  원안 Phase B. 1차 결과 이후.
- compute-matched 비교 (원안 Phase C) — §5.4의 FLOPs 수치를 먼저 확보하고 설계한다.
