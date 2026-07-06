# Hybrid Prefill/Decode Tokenization (BPEByte)

> **Revision note (2026-07-03).** Revised after a 3-aspect review (methodology / experiment
> design / implementation). The headline changes: (1) **Full BPB is leak-contaminated** and is
> demoted to a secondary diagnostic — all three prefill candidates are non-causal, so their
> Full-BPB numbers are not a valid cross-method ranking; (2) the grid gains two **anchor
> controls** (`C_online`, `C_offline`) plus a fixed external reference, without which no hybrid
> cell is interpretable; (3) the "no correction needed downstream" guarantee holds **only for
> B1**; (4) statistics (seeds + CIs), a training **budget** (≥ ratio-40), and a numeric
> **promotion gate** are now required — this closes the standing "no CIs" audit item; (5) the
> interface is made **byte-native** (AU-Net consumes byte ids + per-byte `level_mask`, not
> subword tokens); (6) **P2 "leaf parsing" is undefined in the codebase and is deferred**. See
> §8 for the execution plan.

## 1. Overview

A feature that splits each sequence into a **prefill region** and a **decode region**, applying a
different tokenization/parsing strategy to each.

- **Prefill region**: the full text is available, so offline (non-causal) parsing is possible
- **Decode region**: must mimic autoregressive generation, so it is fixed to **onlineBPE
  (root parsing + no backtracking)** — this is the project's verified `root_greedy` mode
  (`bpe_online_mode="greedy"`, placement `root`), the one segmentation the repo has confirmed
  is **0% causal-leak** and byte-for-byte reproducible by a streaming decoder.

```
|<---------------- context (length N) --------------->|
|<----- prefill ----->|<---------- decode ---------->|
   offlineBPE or            onlineBPE
   longest match            (root + no_bt)  ← leak-free, the ONLY scored/primary region
        ↑
    boundary b (static or dynamic at training time)
```

**Framing (important).** Because the decode region is *identical* across all prefill candidates,
on the primary (decode-only) metric this experiment is a **conditioning / curriculum** choice, not
a tokenization-fidelity win. Expect P1/P3 deltas on Partial BPB to be **small**; power the
experiment accordingly (§4, §8).

---

## 2. Components

### 2.1 Prefill Candidates

| ID | Method | Parsing | Status |
|----|--------|---------|--------|
| `P1` | offlineBPE | root placement | **ready** — reuses `_offline_levels_mask` + `root` placement shift |
| `P2` | offlineBPE | leaf placement | **DEFERRED** — "leaf parsing" has no definition/symbol in the codebase; must be specified before it can be run (see §5.2) |
| `P3` | longest match | backtracking (`bt`) | **ready** — reuses `ByteTrie.longest_match_with_backtracking` (`mode="bt"`) |

> **Leak caveat.** All prefill candidates are **non-causal**: P3 is the canonical backtracking
> longest-match (`bt`) — the mode whose NLL the repo measured as leaky (1.3B: `bt` 0.787 vs
> leak-free `root_greedy` 0.860, Δ≈0.073 BPB); P1/P2 are offline BPE whose merges over the full
> prefix are likewise non-causal (appending bytes can move earlier boundaries). Their NLL is
> **optimistically biased, and biased by a different amount per method.** This is why the prefill
> region must never contribute to a headline model-quality ranking (§3.4).

### 2.2 Decode (fixed)

- **onlineBPE**: root parsing + no backtracking (`root+no_bt`) = `root_greedy`.
- Tokens in the decode region are determined online, left to right; once a boundary is committed
  it is never changed by subsequent bytes. The byte-trie **restarts at root** at the region start,
  so greedy is *memoryless at the boundary* and **cannot** merge across the prefill/decode split
  (this is a correctness guarantee by construction, not just something unit tests check).

### 2.3 Prefill/Decode Boundary Strategies (training time)

| ID | Strategy | Description | Downstream-eval safe? |
|----|----------|-------------|-----------------------|
| `B1` | Dynamic `[0, N]` | `b ~ Uniform(0, N)` | **Yes** — only B1 covers arbitrary eval boundaries |
| `B2` | Static `N/2` | always `b = N/2` | No — OOD for arbitrary question lengths |
| `B3` | Dynamic `[N/3, 2N/3]` | `b ~ Uniform(N/3, 2N/3)` | Partially — OOD outside the middle band |

> `b` is sampled as a **byte offset**, then snapped to a token boundary by default (see 3.2).
> **B1 is the default for any run that will be evaluated with likelihood-based downstream tasks**
> (§5.1): its training boundary distribution is the only one that covers a question that ends
> anywhere in the sequence.

---

## 3. Implementation Design

### 3.1 Interface (byte-native)

AU-Net/BPEByte consumes **byte ids + a per-byte integer `level_mask`** (the pooling level), *not*
subword token ids. The interface is therefore byte-indexed throughout.

```python
class HybridTokenizer:
    def __init__(
        self,
        bpe_model,
        prefill_mode: str,      # "offline_leaf" (offline-BPE before_root) | "offline_root" (offline-BPE root) | "longest_bt"
        boundary_mode: str,     # "uniform_full" (B1) | "static_half" (B2) | "uniform_mid" (B3)
        seed: int | None = None,
    ): ...

    def encode_train(self, text_bytes: bytes) -> HybridEncoding:
        """Training: sample boundary b (seeded, per-doc), snap, encode prefill/decode separately."""

    def encode_eval(self, text_bytes: bytes, boundary: int) -> HybridEncoding:
        """Evaluation: boundary specified explicitly (byte offset)."""


@dataclass
class HybridEncoding:
    byte_ids: list[int]        # the raw byte stream the model consumes (special ids >=256 peeled for BPB)
    level_mask: list[int]      # per-byte pooling level — the actual training signal
    labels_level_mask: list[int]  # label-shifted level_mask (level_mask[1:]) — aligns region to the PREDICTED byte
    region_mask: list[int]     # PER-BYTE: 0 = prefill, 1 = decode
    boundary_byte_idx: int     # byte offset where the decode region starts (post-snap)
    bpb_count_mask: list[int]  # 1 = counts toward the raw-byte BPB denominator (0 for special ids >=256)
```

Notes: `boundary_token_idx` from the earlier draft is dropped (there are no subword tokens in a
byte model). `region_mask` and `bpb_count_mask` are **per byte** so they co-index with `level_mask`.

### 3.2 Boundary Handling Rules

1. Sample `b` in byte units (seeded per document; see §8 for RNG threading).
2. Encode `text[:b]` with the prefill parser. If the last prefill token would cross `b`,
   **snap to the token boundary** (default at *training*: pull backward — keep only tokens fully
   before `b`).
3. Define the decode region as **`byte[snap:]`** (disjoint from prefill = `byte[:snap]`). Encode
   it with onlineBPE (`root_greedy`) starting fresh at `snap` — the trie restarts at root, so no
   token merges across the snap point.
4. Every byte is assigned to exactly one region; no byte is duplicated or dropped (unit tests
   required). This is guaranteed by construction because prefill = `byte[:snap]`, decode =
   `byte[snap:]` are a disjoint partition.

**Edge cases to watch**
- `b = 0` (all decode) / `b = N` (all prefill) — possible under B1. Exclude near-degenerate
  boundaries (`b` within a few tokens of 0 or N) from **Partial-BPB aggregation**: a near-empty
  decode region gives a tiny, high-variance denominator.
- `b` falling in the middle of a multi-byte (UTF-8) character → allowed (byte-level model);
  snapping to a prefill *token* boundary already lands on a byte boundary.
- Prefill and decode share the same vocab/merge table (only the parsing method differs).
- **Seg-cache purity:** the existing segmentation cache memoizes on `(seg_sig, mode, tuple(byte))`,
  assuming boundaries are a pure function of bytes+mode. A random per-doc `b` **breaks that
  assumption** — the cache key must include `b` (and boundary_mode/seed), or caching must be
  disabled for hybrid mode. Failing to do this yields silently stale masks (§8, risk #1).

### 3.3 Training Pipeline Integration

- The data loader samples a boundary per document (seeded) → produces a `HybridEncoding`, and
  emits `region_mask` as an **extra per-byte view row**, exactly mirroring the existing
  delayed-mask "3rd view" (`committed_patch_idx`) plumbing (dataloader row → `pack_tokens` vstack →
  `train.py` → model kwarg).
- `region_mask` must be **label-shifted** identically to `labels_level_mask[:, :-1] =
  level_mask[:, 1:]`, so a byte's region label aligns to the *predicted* byte, not the input byte.
- Loss is applied over the whole sequence, but **per-region losses are logged separately**
  (prefill loss / decode loss) for analysis.

### 3.4 Evaluation (Bits per Byte, BPB)

Standardize on **BPB** — token counts differ across parsing methods, so token-level PPL is not
comparable. Compute both metrics in a **single forward pass** using `region_mask`; the denominator
is the **raw byte count** of the region (special ids ≥256 excluded via `bpb_count_mask`).

| Metric | Definition | Role |
|--------|------------|------|
| **Partial BPB** | NLL over the decode region (`root_greedy` span) / decode bytes | **PRIMARY** — leak-free; the actual generation regime |
| **Full BPB** | total NLL over the whole sequence / total bytes | **SECONDARY DIAGNOSTIC ONLY** — leak-contaminated (includes the non-causal prefill), differently biased per prefill method; **must not** be used to rank methods or select combos to scale |

**Baselines / anchor controls (required — see §4).** Report every hybrid cell as the **fraction of
the `C_online → C_offline` gap it recovers**, not as a bare BPB.

- Evaluate with a **common eval-boundary grid** `b/N ∈ {0.25, 0.5, 0.75}` applied identically to
  all models. **Never** evaluate a model only at its own training boundary (evaluating B2 at N/2 is
  train-on-the-test-condition and flatters B2).
- Report the **boundary sweep curve** (Partial BPB vs `b/N`) as the discriminator between B1/B2/B3.
- Report **tokens-per-byte (compression) alongside BPB** as a covariate: P1/P3 emit different
  bytes/token, which changes the number of loss terms and the effective token-context at a fixed
  byte boundary. A BPB gap that is really a compression gap must not be read as a modeling win;
  prefer comparisons at **matched byte-context**.

---

## 4. Experiment Design: grid + controls

**Scientific question:** does hybrid prefill/decode training improve the *decode-regime* (Partial)
BPB and downstream likelihood over the naive extremes, and which prefill/boundary combination is
best? This is only answerable relative to the two extremes, so the grid **must** include them.

### 4.1 Anchor controls (bound the axis)

| ID | Prefill | Decode | Purpose |
|----|---------|--------|---------|
| `C_online` | onlineBPE (`root_greedy`) | onlineBPE (`root_greedy`) | leak-free floor; = current inference regime; answers Open-Q "how much does no_bt cost" |
| `C_offline` | offlineBPE everywhere | offlineBPE (eval-only) | leaky upper bound (do NOT report as a final claim) |
| `C_ref` | — | — | fixed external reference (Llama subword or plain byte), as every repo study carries |

### 4.2 Hybrid grid (P2 deferred → 2×3, run-2x cells blocked)

| | `B1` uniform [0,N] | `B2` static N/2 | `B3` uniform [N/3, 2N/3] |
|---|---|---|---|
| `P1` offlineBPE-root | run-11 | run-12 | run-13 |
| `P2` offlineBPE-leaf | *(deferred)* | *(deferred)* | *(deferred)* |
| `P3` longestMatch-bt  | run-31 | run-32 | run-33 |

- Decode is shared across all runs: onlineBPE (`root_greedy`).
- Shared conditions: same vocab/merge table, same data, same step count, **same seed set + ≥3
  seeds on the promoted subset**.
- Metrics per run:
  - **Partial BPB** (decode region only) — **primary**, reported as mean ± bootstrap-CI over docs.
  - **Full BPB** — secondary diagnostic (leak-contaminated; never the selector).
  - Prefill-region loss — reference only.
  - Tokens/byte compression ratio (per prefill method) — covariate.
  - Partial-BPB **boundary-sweep curve**.

### 4.3 Budget & promotion rule

- **Small-scale budget:** the repo's own 1672-step (ratio-5) run is documented as *too
  undertrained to rank schemes*, so the small-scale floor is **ratio-40 (13,376 steps ≈ 84B
  bytes)** on the 100M arch used for the ablations. Do not promote on ratio-5 numbers.
- **Promotion gate (numeric):** advance a config to larger scale **only if** its Partial BPB beats
  `C_online` by **> 2× its 95% CI** at ratio-40. Full BPB and (at 100M, near-chance) downstream
  accuracy are **not** promotion gates — BPB is.

---

## 5. Discussion Points

### 5.1 Likelihood-Based Downstream Tasks — ⚠️ Resolved *only for B1*

**Policy: question (context) = prefill method, answer candidate (continuation) = onlineBPE.
Boundary handling is exactly the same as training** — *provided the training boundary distribution
covers the eval boundary*. This holds for **B1** only. B2 (static N/2) and B3 (mid-band) never
train the online-restart offsets that arbitrary question lengths produce, so the "no correction
needed" guarantee **fails** for them: use B1 for downstream-eval runs.

Procedure (identical **backward** snap to training, per §3.2):

1. Encode the question text with the prefill parser.
2. **Snap backward** at the question/answer boundary — keep only prefill (question) tokens fully
   before the boundary; this is the *same* snap rule as training. The residual tail bytes of the
   straddling question token therefore fall into the onlineBPE (answer) region, exactly as in
   training.
3. Encode from the snapped point with onlineBPE (`root_greedy`).
4. Likelihood = sum of token NLLs over the byte span of the answer candidate. Because the snap is
   backward, the first online token may **straddle** the Q/A boundary (contain question-tail +
   answer bytes); sum **from the first token that contains an answer-candidate byte** (spec
   default). Byte-normalize per existing convention using the answer-candidate byte set.

**Why backward (train/eval consistency, per §3.2 + §5.1).** Training snaps backward, so the tail
bytes of a straddling prefill token are re-parsed by the decode (`root_greedy`) parser. Eval must
reproduce the **same** pattern — snap backward too — so the tokenization-mismatch the model saw in
training is exactly what it sees downstream, and **no extra correction is needed**. The straddle
token's question-tail NLL is a small, systematic term present identically at train and eval; it is
the price of exact consistency (a forward snap would remove the straddle but break train/eval
symmetry — rejected). Reuse the existing `get_levels_mask_prefill` machinery for the split.

Implementation notes:
- Reuse the boundary/snap logic in `get_levels_mask_prefill` — do **not** reimplement.
- Downstream tasks: the repo standard is HellaSwag / ARC-E / ARC-C / PIQA. Note these are
  **near-chance at 100M**, so downstream cannot be the promotion gate — BPB is (§4.3).

### 5.2 Choosing the Prefill Method: offlineBPE vs longest match

- Decide empirically via the P1/P3 comparison — but **on Partial BPB and downstream, never on
  Full BPB** (which is leak-contaminated and would tend to pick the *leakiest* prefill).
- Hypothesis: if a prefill's token distribution is closer to the decode region's `root_greedy`,
  the train/inference mismatch shrinks. This is a *conditioning* effect; expect small deltas.
- **P2 "leaf parsing" is undefined.** There is no `leaf` symbol in the codebase (only root /
  before_root *placement* and greedy/bt/prefix_free/root_stats *modes*). Before P2 can run, define
  precisely what "leaf parsing" computes (candidate: boundaries at the leaves of the BPE merge tree
  = minimal-unit segmentation) and add it as an explicit mode. Until then P2 is deferred.

### 5.3 Boundary Split Strategy

- B1 (uniform full range): robust to any boundary position, and the **only** downstream-eval-safe
  strategy; cost is that extreme boundaries (all-prefill / all-decode) spend training signal on
  degenerate cases and raise Partial-BPB estimator variance (exclude near-degenerate `b` from
  aggregation, §3.2).
- B2 (static): simple, but overfits its boundary and is downstream-OOD.
- B3 (mid-band): a compromise; still OOD outside the band.
- Verify generalization via the boundary sweep at eval (§3.4), always with the eval boundary
  **decoupled** from the training boundary.

---

## 6. Implementation Checklist

- [x] `onlineBPE (root + no_bt)` encoder — **exists** (`root_greedy`: `ByteTrie.greedy_*` +
  `bpe_online_mode="greedy"`, placement `root`)
- [x] `offlineBPE` root/leaf — **root exists** (`_offline_levels_mask` + placement shift);
  **leaf deferred** (undefined)
- [x] `longest match (bt)` encoder — **exists** (`mode="bt"`)
- [ ] Boundary sampler (`B1`/`B2`/`B3`) + backward-snap + **seg-cache key fix** (include `b`/mode/seed)
- [ ] `region_mask` + `bpb_count_mask` as per-byte data views (mirror the committed 3rd-view) →
  `pack_tokens` → `train.py` → model, with **label-shift alignment**
- [ ] Per-region NLL out of the model forward + **Full BPB + Partial BPB in one pass** (raw-byte
  denominator, special ids ≥256 excluded)
- [ ] Baseline/control configs: `C_online`, `C_offline`, `C_ref`
- [ ] Downstream likelihood eval — question = prefill / answer = onlineBPE, **backward-snap
  (identical to training, §5.1)**, reusing `get_levels_mask_prefill`
- [ ] Unit tests: no dup/drop bytes, `b=0`/`b=N`, UTF-8 boundaries, cache-key correctness
- [ ] Grid configs: P1/P3 × B1/B2/B3 (6 cells) + 3 controls, at ratio-40 100M

---

## 7. Open Questions

1. Snap direction: **backward at BOTH train and eval** (resolved per §3.2 + §5.1 — train/eval
   consistency is the priority; the straddle-token question-tail NLL is accepted as a small,
   symmetric term). Open sub-question: does a forward-snap *variant* (train+eval both forward,
   still symmetric) measurably change results? Secondary ablation only.
2. Loss on decode region only vs whole sequence — does decode-only loss sharpen the primary metric?
3. Confirm the real inference map: user prompt = prefill (offline), generation = onlineBPE.
4. Quantify how no_bt affects compression / BPB — this is exactly the `C_online` control (§4.1).

---

## 8. Execution Plan

Phased so nothing multi-day launches before the new code is verified at tiny scale. Reuse is
maximal; net-new work is concentrated in three items (sampler, region view, dual-BPB).

### Phase 0 — Ground & map (done / in progress)
- 3-aspect review complete (methodology / experiment / implementation).
- Exact code-change map (file:line insertion points) produced against `regex_cutting.py`,
  `data.py`, `train.py`, `hierarchical.py`, `eval.py`.

### Phase 1 — Net-new code (reuse-first)
1. **Boundary sampler + snap** (new method on `RegexPool`): B1/B2/B3, seeded per-doc, backward
   snap, decode = `byte[snap:]`. **Seg-cache fix**: extend the cache key with `(boundary_mode, b,
   seed)` or bypass caching in hybrid mode. *(Highest silent-bug risk — see below.)*
2. **`region_mask` + `bpb_count_mask` per-byte views**: mirror the delayed-mask 3rd-view path in
   `data.py` (`pack_tokens`) and `train.py`; apply the same label-shift.
3. **Per-region NLL + dual BPB**: expose per-position CE from the model forward
   (`hierarchical.py`), reduce under `region_mask` with the raw-byte denominator (exclude ids ≥256)
   → Full BPB + Partial BPB in one pass. Log both, plus per-region loss.
4. **Configs**: clone the 100M root_greedy ablation config into 6 grid cells + `C_online` /
   `C_offline` / `C_ref`.

### Phase 2 — Verify at tiny scale (no multi-day compute yet)
- **Unit tests** (model on `test_greedy_root.py`): partition is exact (no dup/drop), `b=0`/`b=N`,
  UTF-8 mid-char, cache-key correctness (same bytes + different `b` ⇒ different mask).
- **Smoke forward pass**: a handful of docs, 1 GPU (or CPU), one grid cell — confirm the
  region_mask threads through and Full/Partial BPB come out finite and correctly partitioned
  (Full ≈ byte-weighted blend of the two regions).

### Phase 3 — Small-scale pilot (ratio-40, 100M)
- Launch `C_online` + one promising hybrid cell (e.g. `P1×B1`) first as the minimal informative
  slice, then fill the rest of the grid.
- **Machine/queue:** pick an idle-gated slot (the repo's existing orchestrators); do **not**
  preempt the standing B200 / entropy queues. Confirm availability before launch.
- Report Partial BPB (mean ± CI) vs `C_online`, boundary sweep, compression covariate.

### Phase 4 — Promote & scale
- Apply the §4.3 gate (Partial BPB beats `C_online` by > 2× CI at ratio-40). Scale only winners.

### Risk register (net-new items, most→least likely to hide a silent bug)
1. **Seg-cache purity under dynamic `b`** — stale masks are silent and would poison every metric.
   Mitigate: cache-key includes `b`; a unit test asserts different `b` ⇒ different mask.
2. **Per-region NLL label-shift & denominator** — off-by-one in the shift, or counting special ids
   in the byte denominator, biases BPB. Mitigate: assert Full BPB equals the byte-weighted blend of
   the two region BPBs on a synthetic batch.
3. **region_mask view threading** — must survive `pack_tokens` rewind/shift like the committed
   view. Mitigate: reuse the exact 3rd-view code path; test round-trip.
