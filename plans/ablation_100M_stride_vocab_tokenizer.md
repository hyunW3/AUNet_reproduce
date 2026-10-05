# Plan — 100M ablation: Fixed-stride · Vocabulary size · Tokenizer source

**작성일** 2026-09-29 · **상태** 설계 (학습 미실행)
**관련** `reports/leaderboard_100M.md`, `plans/plan_vocab_prior.md`, `paper_overleaf/sections/6-conclusion.tex:16`

---

## 0. 현황 확인 (로컬 snu55 + ece-agpu11)

세 ablation 모두 **100M 영어(DCLM)에서 학습된 run이 없다.**

| 축 | 상태 | 가장 가까운 기존 run |
|---|---|---|
| **A. Fixed-stride** | 없음. 코드 knob도 없음 (`regex_cutting.py:427` 전략은 `word/pretok/punct/char/bpe*`뿐) | level-2 static 그룹핑만 있음: `superword_l3/baseline_static2word` (BPB 1.094), `static2subword` (0.999), `armC_static2subword_rg` (1.099), `r10lr_rgl3_100M_*` (`bpe_static2`). byte-stride는 DNA appendix(stride-3, ~30M, 공저자 보유, repo에 없음)뿐 |
| **B. Vocab size** | 없음. 모든 BPEByte/Llama run이 llama3 128K | `reports_afterAAAIsub/bpebyte3_level2.md` (L2 vocab 4k/32k/128k parse 통계만, 학습 X), `plan_vocab_prior.md` (설계만) |
| **C. Tokenizer source** | **부분** — 중국어만 | `runs/zh/100M_21G/scratch/rg_zh_{llama3,qwen2}` (53,504 step; cloze 27.3 vs 27.2, 거의 chance). 영어 GPT-2/Qwen2/Gemma 등 학습 run 없음. 1.3B eval-time swap(`PoC/tokenizer_swap_all_results.md`)은 재학습 없는 swap이라 별개 |

논문 conclusion의 주석 처리된 future-work 문단이 B/C 공백을 직접 인정하고 있음.

---

## 1. 무엇을 보이려는가

리뷰어 질문 세 개에 대한 답:

1. **"BPE 경계가 정말 필요한가? 같은 압축률의 고정 stride면 되지 않나?"** → A.
   BPEByte(root_greedy)는 평균 4.54 B/patch. 같은 평균 patch 길이의 fixed stride와 BPB/downstream 비교.
   여러 stride를 돌리면 **BPB–(bytes/patch) 곡선**이 나오고, 이 곡선이 "내용 무관 경계"의 frontier 역할을 함.
2. **"128K llama3 vocab에 과적합된 결과 아닌가? vocab 크기에 얼마나 민감한가?"** → B.
3. **"llama3라는 특정 tokenizer 덕분 아닌가?"** → C.

B·C의 모든 점을 A의 stride 곡선과 같은 (B/patch, BPB) 평면에 올리는 것이 핵심 figure.
→ "같은 compression에서 content-aware 경계가 stride 대비 얼마나 이득인가"를 한 장으로 보여줌.

---

## 2. 측정한 사전 통계 (DCLM chunk.14, 300 docs, 1.68 MB, greedy byte-trie)

**Vocab size (llama3 rank-prefix 절단)** — single byte는 rank 0–255이므로 앞 V개 rank만 남겨도 유효한 BPE vocab (nested family).

| V | 8K | 16K | 32K | 64K | 100K | 128K (현 baseline) |
|---|---:|---:|---:|---:|---:|---:|
| B/patch | 3.49 | 3.87 | 4.18 | 4.43 | 4.54 | 4.54 |

→ 100K 이상은 영어에서 차이 없음(100K–128K는 다국어 추가분 ≈ cl100k 경계). **64K 이하만 의미 있음.**

**Tokenizer source** (online greedy-trie B/patch / offline 실제 tokenizer B/tok)

| tokenizer | vocab | trie | offline |
|---|---:|---:|---:|
| llama3 (baseline) | 128K | 4.54 | 4.55 |
| GPT-2 | 50K | 4.36 | 4.30 |
| Qwen2 | 152K | 4.42 | 4.41 |
| Gemma (SentencePiece) | 256K | 4.29 | 4.39 |
| BLOOM | 251K | 4.50 | 4.52 |
| StarCoder2 (code) | 49K | 3.87 | 3.81 |
| llama3_superbpe | 200K | 6.07 | 6.44 |

→ 자연어 tokenizer들은 영어에서 4.3–4.5 B/patch로 **압축률이 거의 같음** → C는 "몇 개로 자르나"가 아니라 "어디서 자르나"의 비교가 됨 (좋은 통제).
StarCoder2(도메인 불일치)와 SuperBPE(더 거친 경계)는 각각 도메인·granularity 대조군.

---

## 3. 공통 레시피 (leaderboard_100M과 동일 → 기존 행과 바로 비교 가능)

`runs/poc/portable_aunetlaw/configs/bpebyte_rg_100M.yaml` 그대로:
dims [512,768], layers [3,10], head_dims [64,128], 98.59M params · seq 8192 · global batch 48 ·
**53,504 step (21.04 GB, γ≈10.4)** · LR 3.4e-3, warmup 3000, cosine min 0.01, wd 0.1, clip 0.2 · seed 777 · DCLM 2 shards.
`max_seqlens: [-1, 3200]` (8192 B / 3200 = 2.56 B/patch 하한 → stride ≥3이면 수정 불필요).

기준점 (재사용, 새로 안 돌림): BPEByte root_greedy 1.079 · AU-Net word 1.082 · BLT 1.102 · Llama 1.053.

변경하는 것은 **`data.regex` 블록뿐**:

| 축 | override |
|---|---|
| A | `strategy: {stride: k@1}`, `bpe_online: false`, `bpe_tokenizer_path: null` |
| B | `bpe_tokenizer_path: tokenizer/llama3_trunc/llama3_V{8,16,32,64}k.model` (나머지 rg 설정 동일) |
| C | `bpe_tokenizer_path: tokenizer/<name>/tokenizer.json` (HF; `bpe_tokenizer_kind` 자동감지) — zh qwen2 run과 같은 경로라 코드 변경 없음 |

---

## 4. Arms

### A. Fixed-stride (신규 코드 필요, §6)

| arm | stride | 목적 |
|---|---:|---|
| **A-s4** (P0) | 4 | rg(4.54)보다 약간 세밀 |
| **A-s5** (P0) | 5 | rg보다 약간 거침 → s4/s5가 rg를 사이에 둠 (보간으로 iso-compression 비교) |
| A-s3 (P1) | 3 | 곡선의 세밀 끝 (global compute ≈1.5×) |
| A-s6 (P1) | 6 | SuperBPE(6.07)와 iso-compression 짝 |
| A-s8 (P1) | 8 | 거친 끝 |

### B. Vocab size (llama3 rank-prefix 절단, online greedy root)

| arm | V | B/patch |
|---|---:|---:|
| **B-8k** (P0) | 8,192 | 3.49 |
| **B-32k** (P0) | 32,768 | 4.18 |
| B-16k (P1) | 16,384 | 3.87 |
| B-64k (P1) | 65,536 | 4.43 |
| (128K) | 기존 `lb_rg_100M` | 4.54 |

rank-prefix 절단을 쓰는 이유: 같은 merge 순서의 prefix라 "크기만" 바뀜 (새 BPE를 학습하면 데이터·알고리즘 차이가 섞임).

### C. Tokenizer source (online greedy root, 영어 DCLM)

| arm | tokenizer | 역할 |
|---|---|---|
| **C-gpt2** (P0) | GPT-2 50K | 고전 영어 BPE, 다른 pretokenizer regex |
| **C-qwen2** (P0) | Qwen2 152K | 다른 vendor, 대형 다국어 — zh run과 짝 |
| C-gemma (P1) | Gemma 256K SentencePiece | 다른 알고리즘 (SP, `▁` metaspace) |
| C-starcoder2 (P1) | StarCoder2 49K | 도메인 불일치 대조군 |
| C-superbpe (P2) | llama3_superbpe 200K를 **L1으로** | whitespace-crossing, 거친 경계 (6.07) |

### Noise floor

| arm | 내용 |
|---|---|
| **N-rg-s778** (P0) | baseline rg seed 778 |
| N-rg-s779 (P1) | seed 779 |

leaderboard 상위권 차이가 0.003–0.04 BPB이므로, seed 분산 없이는 B/C의 작은 차이를 해석할 수 없음.
(`~/AUNet_nb/runs/nb/nb_base_s778` 선례 있음.)

### Optional (P2): subword baseline vocab 민감도
Llama 100M을 32K vocab으로 1 run. embedding 파라미터가 바뀌므로(128K×768 ≈ 98M → 25M) non-embedding 동일로 맞추고 따로 보고.
"byte 모델은 vocab에 둔감, subword는 민감" 대비가 나오면 논문 메시지에 도움.

**총계**: P0 = 7 arms, P1 = +8, P2 = +2 (최대 17).

---

## 5. 평가

1. **Held-out BPB** (DCLM val) — 학습 loss proxy가 아닌 held-out. 추가로 기존 `data/{code_bpb,flores_bpb,zh_bpb,te_bpb}`:
   tokenizer 축(C)은 도메인/언어 BPB에서 차이가 날 가능성이 큼.
2. **Downstream** — leaderboard와 같은 full-dataset 0-shot 6-bench (HS/ARC-E/ARC-C/BoolQ/PIQA/Wino), HS/ARC-E/PIQA 평균을 주 지표로.
3. **효율 지표** — 실측 B/patch, global FLOPs/byte, 학습 throughput. stride 곡선 위에서 비교하려면 필수.
4. **Figure**: x = B/patch, y = held-out BPB. stride 점들(선), vocab 점들(llama3 family), tokenizer 점들, AU-Net word/BLT 기존 점.
   주 비교 = 같은 x에서 **stride 곡선 대비 수직 거리**.

### 판정 기준 (미리 고정)
- **A**: iso-compression에서 rg − stride(보간) ≤ −0.02 BPB 이고 seed 노이즈의 3× 이상 → "BPE 경계가 내용적으로 의미 있음".
  그렇지 않으면 논문 주장을 "compression이 대부분"으로 조정해야 함 (**가장 중요한 리스크 체크**).
- **B**: 32K–128K 구간에서 |ΔBPB| ≤ seed 노이즈 → "vocab 크기에 robust". 8K에서의 저하는 압축률 감소(3.49)로 설명되는지 stride 곡선으로 확인.
- **C**: GPT-2/Qwen2/Gemma가 llama3와 seed 노이즈 이내 → "특정 tokenizer 의존 아님".

---

## 6. 필요한 구현

1. **stride 전략** (`lingua/apps/aunet/data/regex_cutting.py`)
   - `RegexPool.__init__` (~:437 `char` 분기 옆): `elif strategy.startswith("stride"): self.patterns.append((None, None))`
   - `str_offset` (~:1345): `k = int(self.strategy[strat].split('@')[0])`; boundary 목록 생성.
   - ⚠ `str_offset`은 **문자(codepoint) offset**을 반환하고 뒤에서 byte로 매핑됨. DCLM은 대부분 ASCII라 거의 같지만,
     정확한 byte stride를 원하면 byte 단계(mask 생성)에서 `range(k-1, n_bytes, k)`로 넣는 편이 맞음 → byte-exact로 구현 권장.
   - 단위 테스트: 임의 텍스트에서 평균 B/patch == k, BOS 처리, 문서 경계(eos 뒤 리셋) 확인.
   - stride는 자명하게 causal이라 online/offline 구분 불필요 → generation 경로도 그대로.
2. **절단 vocab 파일**: `tokenizer/llama3_trunc/llama3_V{8,16,32,64}k.model` = llama3 `tokenizer.model`의 앞 V줄
   (single byte가 rank 0–255라 유효). `TikTokenTokenizer`가 special token id를 128000+로 잡는지 확인. byte 모델 입력은 bytes라 vocab은 trie 구성에만 쓰임.
3. **config 생성**: `r10lr_gen_configs.py` 방식으로 `runs/poc/portable_aunetlaw/ablation_svt/` 아래 arm별 yaml 생성,
   driver는 `run_zh_100M_21G.sh` 패턴(`run <name> <yaml> <entry> <bs> ... <tokenizer>`)을 복사.
4. **smoke test**: 각 arm 200 step — 로그의 `B/patch` 통계가 §2 값과 맞는지, level-1 patch 수가 3200을 넘지 않는지 확인.

---

## 7. 자원 / 일정

- 비용: `plan_vocab_prior.md` 기준 law 레시피 **≈17 GPU-h/arm (A100/B200급)**. 4×A5000 슬롯이면 대략 2.5–3배.
- P0 7 arms ≈ 120 GPU-h(A100) → 4-GPU 슬롯 2개면 약 2–3일. P0+P1 15 arms ≈ 255 GPU-h.
- 노드 (2026-09-29 확인):
  - ece-agpu11: 8×A100 전부 다른 사용자(flame.train) 사용 중 → **현재 불가**
  - ece-agpu18: 사용자 지정 GPU 0,5,6,7만
  - r10lr 노드(info107a/b, info103, snu55 GPU 2,3): r10lr 레시피용; law 레시피는 batch 48로 재계산 필요
  - 실제 배치는 사용자 확인 후 결정 (메모리의 노드 정책 준수)
- 순서: (1) stride 코드 + 단위테스트 + 절단 vocab 파일 → (2) 전 arm smoke 200 step → (3) P0 7 arms → (4) A 판정 확인 후 P1.
  **A-s4/A-s5를 가장 먼저** 돌림 (논문 주장에 대한 가장 큰 리스크).

---

## 8. 진행 (2026-09-30)

P0 중 사용자 지정 4개 arm을 ece-agpu18 GPU 5,6에서 순차 학습 중. 당시 ece-agpu11은 8장 모두 타 사용자가 점유해 사용하지 않음.
arm: `stride4p57` (BPEByte 4.566 B/patch에 맞춘 고정 stride), `rg_llama3_V32k`, `rg_gpt2`, `rg_qwen2`.
설계 대비 변경: stride를 4/5 두 개 대신 **4.57 하나**로 줄임 (5/4 교대로 BPEByte와 iso-compression). seed 반복(N-rg-s778)은 이번 범위에서 제외.
재현 스크립트, 사전 검증, 상태 확인 명령: `scripts/ablation_svt/README.md`.

## 9. 결과 — train BPB (2026-10-02, 4 arm 모두 53,504 step 완료)

leaderboard와 같은 방식: 마지막 20 logged step `loss/out` 평균 ÷ ln2 (`scripts/ablation_svt/bpb_svt.py`).

| arm | BPB | Δ vs rg |
|---|---:|---:|
| baseline `lb_rg_100M` (llama3 128K) | 1.0794 | — |
| `stride4p57` | 1.1338 | **+0.054** |
| `rg_llama3_V32k` | 1.0849 | +0.0055 |
| `rg_gpt2` | 1.0843 | +0.0049 |
| `rg_qwen2` | 1.0842 | +0.0048 |
| (ref) `lb_aunet_100M` | 1.0821 | +0.0027 |

- stride는 같은 압축률에서 +0.054로 명확히 나쁨 → 경계의 내용(content)이 중요함 (판정 기준 −0.02 충족).
- V32k / GPT-2 / Qwen2는 셋 다 1.084로 모여 있고 baseline과 차이는 +0.005.
  ⚠ baseline은 07-09 코드, 새 arm은 09-30 snapshot 코드로 학습됨 → 이 0.005가 tokenizer 차이인지 코드/seed 차이인지 아직 구분 불가.
  해결: snapshot 코드로 baseline rg를 full step 재학습해 같은 조건의 기준점을 만들 것.
- arm당 wall-clock ~8 h (step 시간 중앙값 0.27 s 기준 연산만은 ~4 h; 차이의 원인은 미확인).

## 10. 후속 (2026-10-04)

1. **baseline 재학습** `rg_snapshot_repro`: snapshot 코드, `lb_rg_100M`과 동일 HP. ece-agpu18 GPU 5,6이 비면 자동 시작 (현재 8장 모두 타 사용자).
2. **평가**: `svt_downstream_queue.sh`가 arm별로 (a) 6-bench 0-shot full downstream, (b) held-out BPB
   (`scripts/probes/bpb_windows_local.py`, held-out DCLM 320 windows, 1,228,802 B)를 snapshot 코드로 순차 실행.
   대상: stride4p57, rg_llama3_V32k, rg_gpt2, rg_qwen2, lb_rg_100M(재채점), rg_snapshot_repro.
   ece-agpu18(GPU 0,7)과 ece-agpu11(전체)에서 빈 GPU를 기다리는 중.
   - 사고 기록: 첫 실행 때 `free_gpu`가 `pipefail` + `grep -q` SIGPIPE 때문에 사용 중인 GPU를 비어 있다고 판단해
     타 사용자 GPU 0에 eval 프로세스 3개를 띄움. 셋 다 import 단계(`gen_mc_helpers` 누락)에서 10–17초 만에 종료되어
     CUDA 메모리 할당 전이었음. 함수 수정 후 두 노드에서 "빈 GPU 없음" 반환 확인, `eval_tasks` symlink 추가.
3. **parser ablation 현황**

   | arm | 상태 | train BPB |
   |---|---|---:|
   | OnlineBPE (rg, llama3 128K) | `lb_rg_100M` 있음 (+ 재학습 대기) | 1.079 |
   | fixed-stride matched-CR | `stride4p57` 완료 | 1.134 |
   | random trie matched-CR | **구현·config 준비 완료, 학습 미실행** (`rg_randtrie_mcr`) | — |
   | AU-Net (word) | `lb_aunet_100M` 있음 (CR 미매칭) | 1.082 |

   **random trie 정의** (`scripts/ablation_svt/build_random_trie.py`): DCLM chunk.00에서 **균일 랜덤 byte 위치**의
   n-gram을 llama3 multi-byte 토큰 길이 분포로 뽑아 vocab을 구성 (corpus 빈도는 반영, BPE merge 통계·단어 정렬은 없음).
   256개 single byte 포함, tiktoken `.model`로 저장 → 기존 online greedy root 파이프라인을 코드 변경 없이 사용.
   - online greedy는 trie dead-end에서 읽은 byte를 모두 commit하므로, patch 길이는 토큰 길이가 아니라 **trie 경로 집합**
     (= vocab 크기)이 결정. 토큰 길이 0.6–1.0배로는 5.13–5.31 B/patch로 거의 변하지 않고, vocab 크기로 보정:
     16K 3.89 · 32K 4.34 · 44K 4.56 · **44.5K 4.565** · 64K 4.82 · 128K 5.31.
   - 실제 `tokenize()` 파이프라인 측정: **4.567 B/patch** (baseline 4.566). 예: `The qu|ick |bro|wn f|ox |ju|mps| over the |...`
   - 주의: matched-CR을 위해 vocab 크기는 44.5K로 baseline(128K)과 다름 → 비교 대상은 "같은 압축률의 다른 경계"이고 vocab 크기는 통제되지 않음.

## 11. 결과 — downstream · held-out BPB (2026-10-04)

실행 위치: `rg_snapshot_repro`는 GPU 5,6에서 12:42 자동 시작. `rg_randtrie_mcr`는 사용자 지시로 ece-agpu18 **GPU 3,4**에서 시작 (port 29772).

**Downstream** (6-bench 0-shot full, snapshot 코드). `lb_rg_100M` 재채점이 7월 결과와 소수점까지 동일 → eval 코드 일관.

| arm | HS | ARC-E | ARC-C | BoolQ | PIQA | Wino | HS/AE/PI | all-6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lb_rg_100M` | 30.7 | 34.6 | 23.5 | 39.1 | 59.4 | 50.2 | 41.59 | 39.60 |
| `stride4p57` | 30.0 | 34.6 | 24.4 | 41.7 | 58.4 | 49.8 | 40.99 | 39.81 |
| `rg_llama3_V32k` | 31.0 | 33.7 | 24.1 | 61.7 | 60.1 | 50.9 | 41.59 | 43.58 |
| `rg_gpt2` | 30.9 | 32.3 | 22.9 | 41.5 | 58.4 | 51.3 | 40.53 | 39.55 |
| `rg_qwen2` | 31.1 | 34.3 | 23.4 | 58.5 | 60.1 | 50.7 | 41.85 | 43.02 |

BoolQ가 39–62로 크게 흔들림 (다수 클래스 ≈62% 근처의 노이즈) → all-6보다 HS/AE/PI 평균이 신뢰할 만함. 차이는 모두 ±1pt 이내.

**Held-out BPB** — `scripts/ablation_svt/heldout_fmha.py` (학습 forward, `attn_impl="fmha"`; held-out DCLM 320 windows, 1,228,802 B, 8192 packing):

| arm | held-out BPB | Δ vs rg | B/patch |
|---|---:|---:|---:|
| `lb_rg_100M` | 1.1058 | — | 4.58 |
| `stride4p57` | 1.1563 | **+0.050** | 4.56 |
| `rg_llama3_V32k` | 1.1066 | +0.0008 | 4.25 |
| `rg_gpt2` | 1.1057 | −0.0001 | 4.41 |
| `rg_qwen2` | 1.1049 | −0.0009 | 4.50 |

→ train BPB의 +0.005 차이는 held-out에서 사라짐. vocab 32K·GPT-2·Qwen2 모두 baseline과 ±0.001. stride만 +0.05.

**⚠ 채점 경로 문제 (미해결)**: `scripts/probes/bpb_windows_local.py`(generator 경로)는 이 100M checkpoint들에서
2.8–3.5 BPB를 냄 (baseline 2.91). 같은 텍스트를 학습 forward로 채점하면 `fmha` 1.35, `sdpa` 4.1
(`dbg_bpb.py`; held-out과 학습 데이터 모두 같은 경향 → 데이터가 아니라 채점 경로 문제).
downstream도 같은 generator를 쓰므로 **100M downstream 절대값이 영향을 받았을 가능성**이 있음.
accuracy는 chance보다 높게 나오지만 정확한 영향은 미확인. 1.3B에서는 같은 harness가 0.908을 냈음.

## 12. baseline 재학습 결과 (2026-10-04 21:05 완료)

`rg_snapshot_repro` = snapshot 코드로 `lb_rg_100M`과 동일 설정 재학습 (llama3 128K, seed 777).

| | train BPB | held-out BPB (fmha) | HS/AE/PI | all-6 |
|---|---:|---:|---:|---:|
| `lb_rg_100M` (07-09 코드) | 1.0794 | 1.1058 | 41.59 | 39.60 |
| `rg_snapshot_repro` (09-30 코드) | **1.0854** | 1.1065 | 41.96 | 39.70 |

→ train BPB +0.005는 **코드 snapshot 차이**였음: 같은 코드 기준 baseline 1.0854 vs V32k 1.0849 · GPT-2 1.0843 · Qwen2 1.0842.
held-out 기준 run 간 노이즈 ≈ 0.001. vocab/tokenizer arm은 같은 코드 baseline과 train·held-out 모두 ±0.001 이내.
stride만 train +0.048 (1.1338 vs 1.0854), held-out +0.050.

## 13. 100M parser ablation — 최종 (2026-10-05)

모든 run: 98.6M, 53,504 step, 21 GB DCLM, seed 777. held-out = `heldout_fmha.py` (320 windows, 1.23 MB). downstream = 6-bench 0-shot full.

| parser | 학습 코드 | B/patch (held-out) | train BPB | held-out BPB | HS/AE/PI |
|---|---|---:|---:|---:|---:|
| OnlineBPE (llama3 128K) `rg_snapshot_repro` | 09-30 | 4.58 | 1.0854 | 1.1065 | 41.96 |
| OnlineBPE (llama3 128K) `lb_rg_100M` | 07-09 | 4.58 | 1.0794 | 1.1058 | 41.59 |
| fixed-stride matched-CR `stride4p57` | 09-30 | 4.56 | 1.1338 | 1.1563 | 40.99 |
| random trie matched-CR `rg_randtrie_mcr` | 09-30 | 4.64 | 1.1337 | 1.1575 | 41.06 |
| AU-Net (word) `lb_aunet_100M` | 07-09 | 4.89 | 1.0821 | 1.1107 | 42.32 |

- **random trie ≈ fixed stride**: train 1.1337 vs 1.1338, held-out 1.1575 vs 1.1563. corpus 빈도로 뽑은 n-gram 사전으로 greedy 파싱해도
  내용 무관 stride보다 낫지 않음 → OnlineBPE의 이득(held-out −0.05)은 "사전 기반 greedy 파싱"이 아니라 **BPE merge 통계가 주는 경계**에서 옴.
- AU-Net(word)은 같은 코드의 OnlineBPE 대비 held-out +0.005 (1.1107 vs 1.1058).
- vocab/tokenizer 대체(§11–12): 32K·GPT-2·Qwen2 모두 같은 코드 baseline 대비 ±0.001.
- downstream은 전 arm ±1pt 이내로 구분력 없음. generator 채점 경로 문제(§11)는 미해결.
