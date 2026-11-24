# Plan — Compression Stability under Distribution Shift (motivation experiment)

**작성일**: 2026-08-10 · **상태**: 실행됨 (결과는 `reports/compression_stability/summary.md`)
**대상 논문**: `paper_overleaf/` (BPEByte)

> **실행 중 계획과 달라진 점 (2026-08-10)**
>
> 1. **F1이 발동했다.** Telugu에서 BPEByte CR=1.57 (I=1.55), random에서 1.39 (I=1.72).
>    "모든 분포에서 유계"는 **거짓**이므로 §3의 주장을 철회하고 §7 복구 arm 기반 주장으로 교체.
> 2. **`random` 도메인을 임의 바이트 → 균일 printable ASCII로 변경.** 임의 바이트는 유효
>    UTF-8이 아니라 subword arm(`str` 인코딩)이 측정 불가가 되어 5개 arm 중 하나가 빠진다.
> 3. **BLT의 θ는 `--calibrate_bps`로 영어에서 직접 적합**했다. 저장돼 있던
>    `theta_monotonic_4.5.txt`(0.4598)는 이 윈도우 프로토콜에서 10.84 B/patch를 내어 재사용 불가.
>    설정 2종: design(영어 4.5 B/patch → θ=0.2392), released-equivalent(영어 1.2 → θ=−0.3185).
>    **released arm은 게이트의 독립 검증이 아니다** — 목표값 1.2가 곧 공개값이므로 정의상 일치한다.
> 4. **`ENTROPY_DEVICE=cuda`를 스크립트에서 명시.** `EntropyModel`은 dataloader fork 안전성
>    때문에 CPU가 기본값이라, 그대로 두면 스윕이 ~100× 느려진다.
> 5. **H-Net arm은 이 호스트에서 실행 불가.** `code_appendix/eval_suite/paths.py`가 익명화된
>    placeholder 경로이고 `hnet`+`mamba-ssm` env가 없다 (실제 env는 ece-agpu18).
>    계획의 fallback대로 `tab:zh_stats`의 EN/ZH 2점만 provenance를 붙여 표시하고,
>    나머지는 `summary.md`의 **Coverage gaps** 절에 누락으로 명시한다.
> 6. **`lingua/.venv`의 torch가 `libcusparseLt.so.0`를 못 찾는다.** 실행 시
>    `LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib` 필요.

---

## 1. Context — 왜 이걸 하는가

논문 intro는 *"기존 patching 기법은 learned parser의 latency나 heuristic rule의 커버리지
한계를 갖는다"* 고 주장하지만, **그 주장을 parser 자체에서 직접 측정한 실험이 본문에 없다.**
근거가 전부 downstream 숫자인데 두 가지 문제가 있다:

1. BLT/H-Net은 arch·data가 달라 confounded (`5-experiments.tex`에서 저자도 명시).
2. `reports_afterAAAIsub/54k_gate_and_significance.md` — matched 3모델의 downstream 격차는
   전부 통계적 tie (Avg5 +0.46pp, 유의 필요폭 1.36pp).

동시에, **정확히 필요한 측정이 이미 존재하는데 컴파일된 PDF에 없다.**
`sections/appendix_aux.tex`는 어디에서도 `\input`되지 않아
`tab:zh_stats`·`tab:niah_variants`·`tab:decode_latency`·`tab:ext_retrieval`·`tab:blt_probes`가
통째로 빠져 있다 (`paper_results/03_consistency_issues.md` #9).

`tab:zh_stats`의 bytes/patch:

| Parser | English | Chinese | 거동 |
|---|---:|---:|---|
| Transformer (Llama-3 BPE) | 4.6 ±2.6 | 3.4 ±1.2 | 완만 |
| AUNet (whitespace regex) | 4.9 ±2.9 | **20.3 ±19.1** (max 156) | 위로 폭발 |
| BLT (entropy patcher) | **1.2 ±0.9** | 1.1 ±0.5 | 영어에서조차 압축 없음 |
| H-Net (learned router) | 4.2 ±2.7 | **1.1 ±0.4** | 아래로 붕괴 |
| **BPEByte (OnlineBPE)** | 4.6 ±2.6 | **2.8 ±1.6** | 유계 |

**양방향 실패다.** heuristic parser는 위로 터지고(절 전체가 한 벡터 → 정보 병목),
learned parser는 아래로 무너진다(1.1 B/patch = global module이 byte resolution으로 동작 →
계층 구조의 연산 이득이 0). 여기서 파생되는 관찰 하나가 서사를 바꾼다:

> BLT는 **영어에서도 1.2 B/patch**다. released threshold 1.335에서 DCLM 압축이 1.13×
> (`tab:throughput_1b` 캡션). 즉 BLT의 Noise/Despace 우위(16.1 / 6.8)는 *더 나은 patcher*가
> 아니라 **거의 patching을 하지 않기 때문**이고, 대가가 latency 1315 ms(전체 최하위)와
> S-NIAH 0.43이다.

이 계획은 2열 표(EN/ZH)를 **distribution-shift 스윕**으로 확장해 본문 motivation 실험으로
승격시킨다. **신규 학습 0.**

### 왜 이 축이 credible한가 — 도메인마다 무너지는 parser가 다르다

"우리 빼고 전부 망가진다"가 아니다. 각 도메인이 서로 다른 parser에 적대적이다:

| 도메인 | 적대 대상 | 이유 |
|---|---|---|
| Chinese | AU-Net | 구분자 없음 → whitespace 규칙 붕괴 |
| **Telugu** | **Llama BPE, BPEByte** | **vocab 근거 없음 (아래 §2)** |
| Code | AU-Net, (Llama) | 공백 규약이 자연어와 다름 |
| Random bytes | BLT, H-Net | 학습 분포 밖 → 매 바이트 경계 |
| Despaced EN | AU-Net | 논문 Despace 열의 parser-level 원인 |

BPEByte가 적대 케이스를 갖는다는 점이 이 실험을 방어 가능하게 만든다.

---

## 2. Telugu — 확인 완료 (2026-08-10)

`tokenizer/llama3/tokenizer.model` (128,000 엔트리) 직접 검사:

```
script        entries w/ script   complete-char entries
Telugu                      39                       0     ← 완전 문자 0개
Devanagari                1051                    1012
Hangul                    2283                    2246
CJK                       4455                    4387
Cyrillic                  6516                    6510
Arabic                    3787                    3782

llama3 fertility (bytes/token, 로컬 val 셋)
  English (DCLM)   4.546
  Chinese (wiki-zh) 3.284
  Telugu (wiki-te)  1.569   ← 문자당 ~2토큰 (Telugu 문자 = UTF-8 3B) → byte fallback
```

**Telugu 완전 문자를 담은 vocab 엔트리가 0개**다 (39개는 lead byte만 걸리는 partial fragment).
Chinese는 CJK 엔트리 4,387개가 있어 OnlineBPE가 2.8 B/patch를 유지했지만, Telugu에는
그 근거가 없다. 따라서 **Telugu는 BPEByte 자신에 대한 적대적 케이스**이고, 결과는 미지수다.

동시에 Telugu는 **공백으로 단어를 구분**하므로 AU-Net은 여기서 정상 동작할 것으로 예상된다.
이 비대칭이 §1 마지막 표의 핵심이다.

검사 스크립트는 재현용으로 `scripts/probes/vocab_script_coverage.py`로 커밋한다
(현재 scratchpad `check_telugu.py`).

---

## 3. 주장과 반증 조건 (사전 등록)

**주장.** OnlineBPE는 평가한 모든 분포에서 patch 길이를 유계 대역 안에 유지하는 유일한 parser다.
heuristic parser는 구분자가 사라지면 위로, learned parser는 학습 분포를 벗어나면 아래로 벗어난다.

**주 지표.** 도메인 $d$에서의 불안정성
$$I_d = \left|\log_2 \frac{\mathrm{CR}_d}{\mathrm{CR}_{\text{en}}}\right|,\qquad
\mathrm{CR}_d = \frac{\text{total bytes}}{\text{total patches}}$$
양방향을 동일하게 벌점. 보고: $\max_d I_d$ 와 $\overline{I_d}$.

**반증 조건 (하나라도 걸리면 주장 축소).**

- **F1** — Telugu에서 OnlineBPE의 $I \ge 1.0$ (CR ≤ 2.3 또는 ≥ 9.2). 이 경우 "모든 분포에서
  유계"를 철회하고 *"vocab이 커버하는 스크립트 안에서 유계, 밖에서는 §7의 vocab augment로 복구"*
  로 축소. §7이 이 시나리오의 대비책이다.
- **F2** — §5의 재현 게이트(EN/ZH가 `tab:zh_stats`를 ±0.2 B/patch 안에서 재현) 실패 →
  측정 스크립트 버그. 해결 전까지 신규 도메인 수치를 신뢰하지 않는다.
- **F3** — 어떤 도메인에서든 BLT 또는 H-Net의 $\max_d I_d$가 BPEByte보다 작게 나옴 →
  "learned chunker가 OOD에서 붕괴한다"는 서술을 데이터에 맞게 다시 씀.

**하지 않을 것.** `tab:niah_variants`를 parser 증거로 쓰지 않는다. 저자 본인의 결론이
*"The S-NIAH-3 collapse is not a segmentation or patch-size effect"* 이고, 32-random-digits에서는
BPEByte가 지므로(0.32 vs 0.92) 인용하면 자기 반증이 된다.

---

## 4. 산출물

1. **본문 그림** `figures/compression_stability.pdf` (2패널, 1열 폭)
   - (a) 도메인별 CR (bytes/patch), log y, parser 5선 + ±std 리본, "쓸 만한 압축" 대역 음영
   - (b) 도메인 × parser 불안정성 $I_d$ 그룹 막대
2. **본문/appendix 표** `table_appendix/compression_stability.tex` —
   도메인별 mean±std / max / p99 / %len-1 patches / %bytes in len>16 patches
3. **원본 데이터** `reports/compression_stability/{measurements.jsonl, summary.md}`
4. **텍스트 수정** — §5에 motivation 문단 추가, abstract/intro의 overclaim 교정 (§8)
5. **`appendix_aux.tex` 미컴파일 문제 처리** (§8)

---

## 5. Phase 1 — 코퍼스와 윈도우

기존 프로토콜을 그대로 따른다: **비중첩, ≤3,840 byte 윈도우, UTF-8 문자 경계 정렬.**
기존 파일 `reports/bpb_windows.jsonl` (EN, 320개 × 3,840B),
`reports/bpb_windows_zh.jsonl` (ZH, 241개)이 기준이고 재사용한다.

**신규 스크립트** `scripts/probes/build_windows.py` — jsonl 코퍼스 → 윈도우 jsonl
(`{"idx", "n_bytes", "text"}` 스키마, 기존 파일과 동일).

| 도메인 | 소스 | 출력 |
|---|---|---|
| English (기준) | *기존* | `reports/bpb_windows.jsonl` |
| Chinese | *기존* | `reports/bpb_windows_zh.jsonl` |
| Telugu | `data/te_bpb/wiki_te/wiki.val.jsonl` | `reports/windows/te.jsonl` |
| Code (py) | `data/code_bpb/py/code.val.jsonl` | `reports/windows/code_py.jsonl` |
| Code (multi) | `data/code_multi/{c,go,java,js,rust}/…` | `reports/windows/code_multi.jsonl` |
| Despaced EN | EN 윈도우에서 합성, $p \in \{0, .25, .5, .75, 1.0\}$ | `reports/windows/despace_p{NN}.jsonl` |
| Random bytes | 합성 (uniform 0–255, seed 777) | `reports/windows/random.jsonl` |

Despace는 논문 Despace 열과 동일한 규칙(intra-line space 제거)을 써야 하므로
`lingua/apps/aunet/eval_despace_mc.py`의 제거 함수를 import해 재사용한다 — 재구현 금지.

목표 규모: 도메인당 ≥240 윈도우(≈0.9 MB). Telugu val이 1.19 MB이므로 충족.

---

## 6. Phase 2 — parser별 경계 측정

**신규 스크립트** `scripts/probes/compression_stability.py`.
입력: 윈도우 jsonl + parser 이름 → `reports/compression_stability/measurements.jsonl`에
`{parser, domain, window_idx, n_bytes, n_patches, patch_lengths_hist}` 추가.

**집계 규약은 `tab:zh_stats` 캡션을 그대로 따른다: "shared boundary byte counted once."**
이걸 어기면 재현 게이트(§5)를 통과하지 못한다.

### Arm 1–3 — 로컬 CPU, GPU 불필요

`viz_parse_boundaries.py`의 parser 구성 방식을 그대로 재사용:

```python
from apps.aunet.data.regex_cutting import RegexPool, RegexArgs
from lingua.tokenizer import TikTokenTokenizer

TOK = "/mnt/ssd2/hyun2/AUNet/tokenizer/llama3/tokenizer.model"

# AU-Net: whitespace/word pooling
aunet = RegexPool(RegexArgs(strategy={"word1": "1@1"}, bpe_tokenizer_path=TOK))

# BPEByte: OnlineBPE (greedy, root placement — 논문 main 설정)
bpebyte = RegexPool(RegexArgs(strategy={"bpe_br": "1@1"}, bpe_tokenizer_path=TOK,
                              bpe_online=True, bpe_online_mode="greedy",
                              bpe_online_placement="root"))

# Llama: subword 토큰 경계 (TikTokenTokenizer.encode_ordinary offsets)
```

`RegexPool.get_levels_mask(list_of_bytes)` → level mask에서 경계 인덱스 추출.

**대량 윈도우 처리**: `precompute_segcache.py`의 병렬 워커 패턴을 참고. 총 ~2,500 윈도우 ×
3 parser면 단일 프로세스로도 수 분.

### Arm 4 — BLT entropy patcher (GPU, 로컬 구현 사용)

레포 내 `lingua/apps/aunet/data/entropy_patch.py`를 쓴다 (HF BLT 모델 불필요):

```python
RegexArgs(..., bpe_online=True, bpe_online_mode="entropy",
          entropy_model_path=<consolidated dir>, entropy_rule="monotonic",
          entropy_theta=<θ>, bpe_online_placement="root")
```

**θ 설정이 이 arm의 최대 쟁점.** 두 값을 모두 보고한다:
- **released**: BLT 공개 threshold 1.335 (`reports/verify_blt_1335/`에 재검증 있음) — 논문
  `tab:zh_stats`의 1.2 B/patch가 이 설정
- **design**: 설계 목표 4.5 B/patch에 해당하는 2.85

released만 쓰면 "BLT는 압축을 안 한다"가 threshold 선택 탓으로 반박된다. 둘 다 보이면
*"공개 설정에서는 압축이 없고, 압축하도록 맞추면 OOD에서 무너진다"* 를 둘 다 말할 수 있다.
**θ는 도메인마다 재캘리브레이션하지 않는다** — 재캘리브레이션 자체가 heuristic parser가
비판받는 per-domain tuning이기 때문. 이 결정을 캡션에 명시할 것.

entropy model: `lingua/apps/main/configs/entropy_byte_50M.yaml` 학습본 또는
`facebook/blt-entropy`. 둘 중 무엇을 썼는지 기록. 사전 계산 캐시가 필요하면
`precompute_entropy_boundaries.py` 사용.

### Arm 5 — H-Net router (GPU, **신규 작업 필요**)

`code_appendix/eval_suite/hnet_eval/harness.py`에는 `_score_batch` / `generate_until`만 있고
**경계 추출 훅이 없다.** `hnet` 패키지의 routing module이 내는 boundary probability /
`b_pred`를 forward hook으로 가로채 `--mode boundaries`를 추가해야 한다.

**Fallback** (훅이 어려울 경우): `tab:zh_stats`의 H-Net EN 4.2 / ZH 1.1은 이미 측정돼 있으므로
그 두 점만 쓰고 신규 도메인은 "not measured"로 남긴다. 이 arm 때문에 전체를 막지 않는다.

---

## 7. Phase 3 — Telugu 복구 arm (F1 대비책이자 §5.3 강화)

`RegexArgs`에 이미 있는 필드:

```python
bpe_augment_tokenizer_path   # 예: tokenizer/bloom
bpe_augment_unicode_range    # 예: "0C00-0C7F"
```

(현재 모든 config에서 `null`.) Telugu를 커버하는 BLOOM vocab에서 U+0C00–U+0C7F 엔트리만
llama3 trie에 이식하고 CR을 재측정한다.

**이 arm의 논지**: OnlineBPE는 커버리지 실패에 대해 **스칼라 하나짜리 복구책**을 갖는다.
AU-Net에는 없고(공백이 없으면 규칙 자체가 정의되지 않음), BLT/H-Net에는 chunker 재학습이
필요하다. §5.3의 *"one scalar knob"* 주장을 서술에서 측정으로 바꾼다.

측정: 증강 전/후 Telugu CR, 그리고 **영어 CR이 변하지 않았음**(증강이 기존 도메인을
망가뜨리지 않음) 확인.

---

## 8. Phase 4 — 논문 통합

### 8.1 그림·표

`scripts/probes/plot_compression_stability.py` — 색·마커는 `viz_tradeoff.py` 규약
(`Llama #0072B2 / AU-Net #E69F00 / BPEByte #009E73`) 준수, BLT/H-Net 색 신규 지정.
`reports/parse/tradeoff_draw.py`처럼 **값을 하드코딩하지 말고** `measurements.jsonl`에서
읽는다 (`03_consistency_issues.md` #16이 지적한 문제 반복 금지).

### 8.2 텍스트

- `5-experiments.tex` §"Limitations of Heuristic Parsing"을 §"Where Parsing Breaks"로 확장.
  현재의 despace 파싱 예시(정성적)를 이 정량 결과 뒤에 배치.
- **abstract / intro / conclusion의 `"degrades the least among all evaluated models"` 교정.**
  현재 거짓이다 — PBP는 AUNet 0.01 / BLT·H-Net 0.00 < BPEByte 0.05, Noise·Despace는 BLT가
  앞선다 (`03_consistency_issues.md` #6). *"among the matched models"* 로 축소하고, 대신
  본 실험이 지지하는 **더 강한 문장**으로 교체:
  > "the only parser whose patch length stays bounded across every evaluated distribution"
- BLT 문단에 압축률 관찰 추가 — 공개 설정에서 1.2 B/patch이므로 robustness와 latency 1315 ms가
  같은 원인에서 나온다는 점.

### 8.3 `appendix_aux.tex` 결정 (선행 조건)

이 계획은 `tab:zh_stats`를 근거로 삼는데 그 표가 PDF에 없다. 셋 중 하나를 골라야 한다:

1. `appendix.tex`에서 `appendix_aux.tex`를 `\input` — 누락된 5개 표가 모두 복귀
2. `tab:zh_stats`만 `appendix.tex`로 이동 (지면 제약 시)
3. 신규 `compression_stability` 표가 `tab:zh_stats`를 대체·흡수 — **권장**.
   신규 표가 EN/ZH를 포함하므로 중복이고, `appendix_aux.tex`의 나머지는 별도 판단.

어느 쪽이든 `appendix.tex` 프로즈가 `appendix_aux.tex` 내용을 참조하는 부분이 있으므로
(`03_consistency_issues.md` #9) 함께 점검.

---

## 9. 검증

**게이트 1 — 재현 (다른 모든 것보다 먼저).**
신규 스크립트로 EN·ZH를 측정해 `tab:zh_stats`를 재현:

| Parser | EN 기대 | ZH 기대 |
|---|---:|---:|
| Transformer | 4.6 ±2.6 | 3.4 ±1.2 |
| AUNet | 4.9 ±2.9 | 20.3 ±19.1 |
| BPEByte | 4.6 ±2.6 | 2.8 ±1.6 |
| BLT (released θ) | 1.2 ±0.9 | 1.1 ±0.5 |
| H-Net | 4.2 ±2.7 | 1.1 ±0.4 |

허용 오차 ±0.2 B/patch. **실패 시 신규 도메인 측정을 진행하지 않는다** (F2).
독립 교차검증: Llama arm의 EN CR은 §2에서 tiktoken으로 직접 잰 4.546 B/token과 일치해야 한다.

**게이트 2 — 정합성.** 신규 despace $p{=}1.0$ 지점의 AU-Net CR이 `figure_appendix/despace_grid`의
AU-Net 붕괴와 같은 방향인지, 그리고 논문 Despace −22.4 pt와 정성적으로 맞는지.

**게이트 3 — 위생.** 각 parser × 도메인에서 (a) patch 경계의 합이 총 바이트 수와 일치,
(b) 모든 patch 길이 ≥ 1, (c) 첫 경계가 index 0.

**실행 확인**:
```bash
cd /mnt/ssd2/hyun2/AUNet/lingua && source .venv/bin/activate
python ../scripts/probes/compression_stability.py --gate-reproduce   # 게이트 1만
python ../scripts/probes/compression_stability.py --all              # 전체 스윕
python ../scripts/probes/plot_compression_stability.py
```

---

## 10. 비용과 순서

| Phase | 작업 | 비용 |
|---|---|---|
| 0 | `vocab_script_coverage.py` 커밋 (§2, 이미 실행됨) | 분 |
| 1 | `build_windows.py` + 도메인 윈도우 생성 | ~1 h, CPU |
| 2 | 게이트 1 (EN/ZH 재현, arm 1–3) | ~1 h, CPU |
| 3 | arm 1–3 전체 스윕 | ~1 h, CPU |
| 4 | arm 4 (BLT entropy, θ 2종) | ~3 h, GPU 1장 |
| 5 | arm 5 (H-Net 훅) — 실패 시 fallback | ~4 h, GPU 1장 |
| 6 | Telugu 복구 arm (§7) | ~2 h, CPU |
| 7 | 그림·표·논문 텍스트 (§8) | ~4 h |

**신규 학습 0. GPU 총 ~7시간 (1장).** DNA/HG38은 이 호스트에 원본이 없으므로
(`paper_results/sources/README.md` §1) 이번 스윕에서 제외한다.

Phase 1–3만 끝나도 (CPU, 반나절) 논문의 핵심 주장 — heuristic은 위로, learned는 아래로,
OnlineBPE만 유계 — 중 AU-Net·Llama·BPEByte 축은 완성된다. arm 4–5는 그 위에 얹는 것이다.
