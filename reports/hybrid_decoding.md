# Hybrid Prefill/Decode Tokenization — Results

Spec: [`PoC/hybrid_prefill_decode_tokenization_en.md`](../PoC/hybrid_prefill_decode_tokenization_en.md).
Code: `lingua/apps/aunet/data/regex_cutting.py` (`hybrid_levels_mask`), eval scorer
`lingua/apps/aunet/eval_hybrid_bpb.py`, tests `lingua/apps/aunet/test_hybrid_tokenization.py`.

## 0. Naming convention

Models are named **`hybrid(prefill / decode, boundary)`**:
- **prefill / decode placement:** `leaf` = before_root (boundary on the token's *last byte*),
  `root` = root placement (boundary at the token *start*). Decode is always `root_greedy` (root).
- **boundary** = training split-point strategy: **`(0,N)`** = Uniform(0,N) *(was B1)*, **`N/2`** =
  static *(was B2)*, **`(N/3,2N/3)`** = Uniform *(was B3)*.
- **original gr** = `root_greedy` everywhere, no prefill region — the not-hybrid baseline (old name:
  C-online). **offline gr (all)** = offline-BPE everywhere, leaky reference (old name: C-offline).

Old→new map (note the P1/P2 inversion — the default hybrid I trained used **before_root = leaf**):
C-online → **original gr**; the default "hybrid P1×B1" (before_root prefill) →
**hybrid(leaf/root, (0,N))**; "rootp" → **hybrid(root/root, …)**; P3 / bt (longest-match) dropped.

**Evaluation (question-tokenization)** is named `prompt / answer (prompt-mode / answer-mode)`, where
placement ∈ {root, before_root} and mode ∈ {online, offline}. The answer is always `root_greedy` =
**root (online)**, so only the prompt varies:
- **root/root (online/online)** = online root_greedy prompt — the *original root_greedy way* (native
  to **original gr**) *(was "greedy-Q")*.
- **before_root/root (offline/online)** = offline-BPE before_root prompt (native to
  **hybrid(leaf/root)**) *(was "offline-Q" / "leaf-Q")*.
- **root/root (offline/online)** = offline-BPE root prompt (native to **hybrid(root/root)**) *(was
  "offline-root-Q" / "root-Q")*.

## 1. What this is

Each sequence of N bytes is split at a byte offset **b** into two regions:

```
|<----- prefill [0, b) ----->|<----- decode [b, N) ----->|
   offline real-BPE (the           root_greedy (causal,
   "prompt", NON-causal)           the "generation")
   b/N = fraction that is prefill
```

- **Prefill** = the region tokenized offline (as a real prompt would be), non-causal.
- **Decode** = fixed to **root_greedy** (`bpe_online_mode=greedy`, placement `root`) — the project's
  verified 0%-leak, byte-for-byte-reproducible causal segmentation; the actual generation regime.
- **Backward snap** at the boundary (spec §3.2/§5.1): prefill tokens are kept whole; the straddling
  token's tail bytes fall into the decode region; **train and eval snap identically**.

**Question:** does training a model on this hybrid regime beat a plain root_greedy model on the
**decode** BPB — especially when the prompt is offline-tokenized (the realistic inference scenario)?

## 2. Models

- **original gr** (control): root_greedy everywhere (no prefill region) — the current inference regime.
- **hybrid(leaf/root, (0,N))**: prefill = offline real-BPE (P1), boundary b ~ Uniform(0, N) per doc (B1),
  decode = root_greedy. Identical arch/optim/data/budget to original gr; the **only** difference is the
  hybrid prefill region.
- **offline gr (all)** (reference, grid only): offline BPE everywhere — leaky upper bound, not deployable.

## 3. Eval methodology

Teacher-forced full-sequence byte scoring (`eval_hybrid_bpb.py`): one forward per (doc, boundary),
per-byte NLL split by region.

- **Partial / decode BPB** (PRIMARY) = Σ NLL over decode-region bytes / decode bytes / ln2. Leak-free.
- **Full BPB** (SECONDARY DIAGNOSTIC) = Σ NLL over the whole sequence / total bytes / ln2. Includes
  the non-causal prefill → leak-contaminated and **not** a valid cross-model ranking (see §6).
- **Boundary sweep** b/N ∈ {0, 0.25, 0.5, 0.75}, applied identically to every model. b/N=0 = pure
  greedy (fair, both models native).
- **Controls scored with `--force_hybrid`**: a root_greedy model gets the offline-prefix grid too,
  so its decode BPB is a matched anchor.
- **95% CIs** by bootstrap over documents (2000 resamples).
- Docs capped to **512 B** (= the byte-encoder sliding window) so the portable `sdpa` eval attention
  is exact; sequences padded to the native `seq_len` behind a sealing boundary. 400 val docs.

## 4. Results — 100M, ratio-10 pilot (single seed, no CIs)

**Decode BPB (primary):**

| b/N | original gr | hybrid(leaf/root, (0,N)) | Δ (hybrid − control) |
|-----|----------|--------------|----------------------|
| 0.00 (pure greedy) | 1.316 | **1.291** | −0.025 |
| 0.25 | 1.329 | **1.201** | −0.128 |
| 0.50 | 1.368 | **1.184** | −0.184 |
| 0.75 | 1.461 | **1.186** | −0.275 |

**Full BPB (diagnostic):** original gr 1.70 / 2.05 / 2.38 at b/N 0.25/0.5/0.75; hybrid 1.24 / 1.21 / 1.18.

**Read:** hybrid wins decode BPB at every boundary — marginal at pure greedy (−0.025), widening as
the offline prompt grows. Hybrid decode BPB is **flat** across b/N (robust to prompt tokenization);
original gr **degrades** (1.32→1.46) because it never trained on offline-parsed prompts. Raw:
`runs/poc/bpb_results.log`.

## 5. Results — 100M, ratio-40 with 3 seeds + CIs (the rigorous headline)

Campaign `runs/poc/campaign/` (ratio-40 = 13,376 steps; seeds 777/778/779; bootstrap-over-docs 95%
CIs). **Decode BPB, mean over 3 seeds** — CI = [min lo, max hi] across seeds:

| b/N | original gr | hybrid(leaf/root, (0,N)) | Δ (hybrid − control) | CI-separated? |
|-----|----------|--------------|----------------------|---------------|
| **0.00** (pure greedy) | 1.1520 [1.129, 1.176] | 1.1812 [1.158, 1.205] | **+0.029** | ~borderline (hybrid worse) |
| 0.25 | 1.1692 [1.147, 1.192] | **1.0936** [1.069, 1.118] | −0.076 | yes |
| 0.50 | 1.2139 [1.192, 1.236] | **1.0788** [1.054, 1.102] | −0.135 | **yes** (1.102 < 1.192) |
| 0.75 | 1.3147 [1.290, 1.339] | **1.0839** [1.057, 1.111] | −0.231 | **yes** |

**Seed variance is negligible** — e.g. original gr @ b/N=0.5 across seeds = 1.2143 / 1.2129 / 1.2145;
hybrid = 1.0795 / 1.0782 / 1.079 — so the CIs above are essentially eval-set (bootstrap) noise, and
seeds add almost nothing. The b/N>0 gaps are cleanly CI-separated.

**Two findings, and an honest correction to the pilot:**

1. **At pure greedy (b/N=0), hybrid is slightly WORSE (+0.029)** — a *sign flip* from the noisy
   ratio-10 pilot (which showed −0.025 within noise). With 4× training the plain-greedy control
   extracts more from a greedy-tokenized prompt; hybrid spends a little capacity on the prefill
   regime it also has to model. So there is a **small real cost at pure greedy**.
2. **Under an offline-tokenized prompt (b/N>0), hybrid wins large and CI-separated** (−0.08 → −0.14
   → −0.23 as the prompt grows). The margin is a touch smaller than ratio-10 (−0.135 vs −0.184 at
   b/N=0.5) but now rigorous.

Net rigorous story: **hybrid trades a small amount of pure-greedy BPB for a large, significant
advantage whenever the prompt is offline-tokenized** — the realistic inference regime. The
scale-ladder gate (hybrid CI below control at b/N=0.5) passes decisively.

## 6. Why Full BPB is only a diagnostic

`level_mask` sets the model's actual pooling boundaries, not just a region label. Under
`--force_hybrid`, the control's prefill region uses offline-BPE patch boundaries it **never trained
on** → high NLL there → Full BPB inflates with b/N (1.70→2.38). That penalizes the control for a
regime it was never meant to run — an eval artifact, not a modeling deficit. Decode BPB (in-regime,
leak-free) is the honest metric.

## 7. Scale ladder (gated, running)

`runs/poc/scale/` — original gr vs hybrid(leaf/root, (0,N)) at **300M → 760M → 1.3B**, each gated on the previous
(hybrid decode-BPB 95% CI clearly below the control's at b/N=0.5, else HALT). 760M/1.3B original gr
**reuse existing root_greedy checkpoints**; only the hybrid twin is trained. 1.3B hybrid (180k steps)
is the dominant cost, gated on 760M passing.

**300M rung — decode BPB [95% CI]** (gate passed → 760M training):

| b/N | original gr 300M | hybrid 300M | Δ |
|-----|---------------|-------------|-----|
| 0.00 | 1.1142 [1.092, 1.136] | 1.1436 [1.121, 1.166] | **+0.029** |
| 0.25 | 1.1402 [1.119, 1.162] | **1.0569** [1.034, 1.079] | −0.083 |
| 0.50 | 1.1947 [1.173, 1.216] | **1.0440** [1.020, 1.067] | −0.151 |
| 0.75 | 1.3129 [1.290, 1.335] | **1.0529** [1.028, 1.079] | −0.260 |

**The pattern holds and strengthens with scale.** At 300M, hybrid is again worse at pure greedy by
exactly **+0.029** (identical to 100M ratio-40) but wins by **more** under offline prompts (b/N=0.5:
−0.151 vs −0.135 at 100M; b/N=0.75: −0.260 vs −0.231), all CI-separated. So the trade — small
fixed pure-greedy cost, growing offline-prompt gain — is **consistent across 100M→300M**.

**decode-BPB(b/N=0.5) vs N so far:** original gr 1.214 (100M) → 1.195 (300M); hybrid 1.079 → 1.044.
760M/1.3B rungs pending (760M hybrid ~2–3 days). Downstream matrix (greedy/bt/offline-Q) runs per
rung too.

## 7b. Hybrid prefill-placement ablation — leaf (before_root) vs root (rootp)

The default hybrid trains its **prefill with before_root placement** (boundary on each token's last
byte = "leaf", config value `offline_leaf`) and decodes with root placement — a placement change at the
prompt→generation seam. The **rootp** variant makes it placement-consistent: **prefill = offline-BPE
with root placement** (config value `offline_root`), decode = root_greedy (already root). Each evaluated in its own native regime at
100M ratio-10 (rootp: offline-root prompt + online-root/greedy answer). `runs/poc/downstream/rootp/`.

**Decode BPB [95% CI]:**

| b/N | original gr | hybrid(leaf/root, (0,N)) | hybrid(root/root, (0,N)) |
|-----|-------------|--------------------------|--------------------------|
| 0.00 | 1.316 | 1.291 | 1.278 |
| 0.50 | 1.368 | 1.184 [1.156,1.203] | **1.180** [1.156,1.203] |
| 0.75 | 1.461 | 1.186 | 1.182 |

**Downstream (5-benchmark, limit 1000, native regime each):**

| model / tokenization | hellaswag | arc_easy | boolq | piqa | winogrande | **mean** |
|----------------------|-----------|----------|-------|------|------------|----------|
| original gr + root/root (online/online) | 0.320 | 0.295 | 0.376 | 0.561 | 0.510 | 0.412 |
| hybrid(leaf/root, (0,N)) + before_root/root (offline/online) | 0.331 | 0.308 | **0.421** | 0.581 | 0.486 | **0.425** |
| hybrid(root/root, (0,N)) + root/root (offline/online) | 0.340 | 0.294 | 0.375 | 0.568 | 0.499 | 0.415 |

**Finding: prefill placement is second-order.** On BPB, root ≈ leaf (1.180 vs 1.184 @ b/N=0.5,
fully overlapping CIs) — placement-consistency gives no gain, both crush original gr. On downstream,
hybrid(leaf/root) (0.425) edges hybrid(root/root) (0.415) and original gr (0.412), but the difference
is entirely **boolq** (leaf 0.421 vs root/original-gr ~0.375; the other four tasks are tied/near-chance)
— i.e. noise at this scale. So the **default leaf (before_root) prefill is (marginally) the better
config**, and the prefill→decode placement mismatch is not a defect worth "fixing."

## 8. Downstream task eval — 100M ratio-10

5-benchmark MC loglikelihood, **same protocol for both models** (`answer_greedy_loglikelihood`:
question=bt, answer=greedy — the repo standard for BPEByte online models), 1000 questions/task.
`runs/poc/downstream/`.

| task | original gr | hybrid(leaf/root, (0,N)) | Δ | random |
|------|----------|--------------|-----|--------|
| hellaswag | 0.319 | **0.326** | +0.007 | 0.25 |
| arc_easy | 0.296 | **0.306** | +0.010 | 0.25 |
| boolq | 0.376 | **0.382** | +0.006 | 0.50 |
| piqa | 0.562 | **0.569** | +0.007 | 0.50 |
| winogrande | 0.510 | **0.513** | +0.003 | 0.50 |
| **mean** | 0.413 | **0.419** | **+0.0066** | — |

**Read:** hybrid is marginally higher on **all 5** tasks (mean +0.66 pp). Each per-task Δ is *within*
noise (stderr ≈ 1.5 pp at n=1000), but the **consistent direction across 5 independent tasks** is
weakly suggestive (5/5, sign-test p ≈ 0.03) and agrees with the BPB result — hybrid training does
not hurt, and slightly helps, downstream. Absolute scores are **near-chance** (100M; boolq even sits
below its 0.50 baseline for both models), so downstream mainly confirms **no regression + parity or
slight gain**, not a strong effect. This matches the documented repo finding that 100M downstream is
near-chance; BPB is the sensitive metric.

### 8a. Offline-question downstream (faithful regime)

New eval mode `offline_question_loglikelihood` (`eval.py` + `get_levels_mask_prefill`): the MC
**question is tokenized offline** (the hybrid model's prefill regime), answer greedy — the downstream
analog of the BPB `--force_hybrid` comparison. Hybrid is now **in-regime**; original gr is **OOD** (it
never trained on offline-tokenized prompts). Same 5 tasks, limit 1000.

| task | original gr (OOD) | hybrid (native) | Δ |
|------|----------------|-----------------|-----|
| hellaswag | 0.302 | **0.331** | +0.029 |
| arc_easy | 0.286 | **0.308** | +0.022 |
| boolq | 0.376 | **0.421** | +0.045 |
| piqa | 0.571 | **0.581** | +0.010 |
| winogrande | 0.495 | 0.486 | −0.009 |
| **mean** | 0.406 | **0.425** | **+1.94 pp** |

**The offline-question shift is the key result** — it isolates exactly the effect the PoC predicts:

| | bt-question | offline-question | shift |
|---|---|---|---|
| **original gr mean** | 0.4126 | 0.4060 | **−0.66 pp** (offline prompt hurts it — OOD) |
| **hybrid mean** | 0.4192 | 0.4254 | **+0.62 pp** (offline prompt helps it — native) |
| **hybrid − original gr** | +0.66 pp | **+1.94 pp** | advantage ~**3×** larger under offline prompts |

Feeding an offline-tokenized prompt **hurts** original gr and **helps** hybrid — so the hybrid advantage
roughly triples (+0.66 → +1.94 pp), with boolq (+4.5 pp) and hellaswag (+2.9 pp) the largest gains
(winogrande, at its 0.50 chance level, is the lone noisy −0.9 pp). This mirrors the BPB result: the
hybrid model is genuinely more robust to — and benefits from — offline-tokenized prompts, the real
inference scenario. Absolute scores remain near-chance (100M), so treat magnitudes as directional.
`runs/poc/downstream/oq_*`.

### 8b. Full matrix — trained model × question tokenization

**Question** tokenization ∈ {greedy (a greedy model's native), bt (repo-standard), offline (hybrid's
native)}; **answer** greedy throughout. Rows = tasks; columns = 6 (model × question-tokenizer).

| task | orig-gr·greedyQ | orig-gr·btQ | orig-gr·offlineQ | hyb(leaf/root)·greedyQ | ·btQ | ·offlineQ |
|------|-----------:|--------:|-----------:|----------:|-------:|-----------:|
| hellaswag  | 0.320 | 0.319 | 0.302 | 0.323 | 0.326 | **0.331** |
| arc_easy   | 0.295 | 0.296 | 0.286 | 0.308 | 0.306 | **0.308** |
| boolq      | 0.376 | 0.376 | 0.376 | 0.381 | 0.382 | **0.421** |
| piqa       | 0.561 | 0.562 | 0.571 | 0.569 | 0.569 | **0.581** |
| winogrande | 0.510 | 0.510 | 0.495 | 0.515 | 0.513 | 0.486 |
| **MEAN**   | 0.4124 | 0.4126 | 0.4060 | 0.4192 | 0.4192 | **0.4254** |

Reading the means:

- **Each model peaks in its own training regime:** original gr at greedy/bt-Q (0.412), hybrid at
  offline-Q (0.425). Off-regime tokenization moves each the expected way — original gr **drops** under
  offline-Q (0.4126→0.4060), hybrid **rises** under offline-Q (0.4192→0.4254).
- **Hybrid ≥ original gr at every question-tokenization**, and by the most under offline-Q:
  greedy-Q +0.68 pp, bt-Q +0.66 pp, **offline-Q +1.94 pp**.
- **Best-vs-best (each in native regime):** hybrid (offline-Q) 0.4254 vs original gr (greedy-Q) 0.4124
  = **+1.30 pp**. So even comparing each model at its own best prompt tokenization, hybrid wins.
- greedy-Q ≈ bt-Q for both models (the prompt segmentation barely differs there); the real lever is
  **offline-Q**, which only the hybrid model was trained to exploit.

Near-chance at 100M (winogrande sits at its 0.50 baseline — its −0.9 pp offline-Q dip for hybrid is
noise); directional, to be re-checked at 300M+. `runs/poc/downstream/{gq_,,oq_}*`.

### 8c. Downstream vs scale — 100M ratio-40 & 300M (mean acc over 5 tasks, limit 1000)

Re-run at the **rigorous** budgets (100M ratio-40, not the ratio-10 pilot above) and extended to
300M, across the three question tokenizations. `runs/poc/downstream/scales/`.

| scale | root/root (online/online) — orig-gr → hyb(leaf/root), Δ | *bt (P3, dropped)* | before_root/root (offline/online), Δ |
|-------|--------------------------|----------|---------------|
| **100M** (r40) | 0.439 → 0.492 (**+0.053**) | 0.439 → 0.494 (+0.055) | 0.414 → 0.478 (**+0.064**) |
| **300M** | 0.448 → 0.455 (**+0.008**) | 0.444 → 0.456 (+0.012) | 0.425 → **0.497** (**+0.073**) |

**The key scaling signal:** the **offline-Q hybrid advantage grows with scale** (+0.064 → +0.073),
while the greedy/bt-Q advantage **collapses toward parity** (+0.053 → +0.008). So at 300M the two
models are ~even when the prompt is greedy/bt-tokenized (matching the BPB finding that hybrid pays a
small pure-greedy cost), but hybrid opens a **large, growing gap under an offline-tokenized prompt**
— its native, and the realistic inference, regime. This is the downstream analog of the BPB
scaling in §7, and both point the same way. (Downstream is above chance here — 300M offline-Q hybrid
0.497 — so these are more than directional.)

## 8d. Full design grid — all 7 cells (100M ratio-10), BPB (B200) + downstream (A100)

Complete grid over prefill-placement {leaf, root} × boundary {(0,N), N/2, (N/3,2N/3)} + `original gr`.
**BPB on B200; downstream re-run entirely on A100 (ece-agpu11) for hardware-consistent reproducibility.**
P3/bt dropped.

**Decode BPB [b/N sweep]:**

| model | 0.00 | 0.25 | 0.50 | 0.75 |
|-------|------|------|------|------|
| original gr | 1.316 | 1.329 | 1.368 | 1.461 |
| hybrid(leaf/root, (0,N)) | 1.291 | 1.201 | **1.184** | 1.186 |
| hybrid(leaf/root, N/2) | 1.308 | 1.213 | 1.193 | 1.193 |
| hybrid(leaf/root, (N/3,2N/3)) | 1.307 | 1.212 | 1.190 | 1.190 |
| hybrid(root/root, (0,N)) | 1.278 | 1.198 | **1.179** | 1.182 |
| hybrid(root/root, N/2) | 1.287 | 1.210 | 1.191 | 1.188 |
| hybrid(root/root, (N/3,2N/3)) | 1.289 | 1.207 | 1.188 | 1.186 |

**Downstream (5-benchmark mean, limit 1000, A100):** columns = eval `prompt/answer (mode)`.

| model | root/root (on/on) | before_root/root (off/on) | root/root (off/on) |
|-------|-------------------|---------------------------|--------------------|
| original gr | **0.400** | 0.400 | 0.395 |
| hybrid(leaf/root, (0,N)) | 0.441 | **0.441** | 0.417 |
| hybrid(leaf/root, N/2) | 0.442 | **0.442** | 0.444 |
| hybrid(leaf/root, (N/3,2N/3)) | 0.462 | **0.462** | 0.413 |
| hybrid(root/root, (0,N)) | 0.405 | 0.405 | **0.399** |
| hybrid(root/root, N/2) | 0.396 | 0.396 | **0.405** |
| hybrid(root/root, (N/3,2N/3)) | 0.400 | 0.400 | **0.407** |

(**bold** = native regime.) Findings, consistent BPB↔downstream:
- **Prefill placement is the main lever, and leaf (before_root) wins:** leaf hybrids native 0.441–0.462
  vs root hybrids 0.399–0.407 ≈ original gr 0.400. On BPB the placements tie (all ~1.18–1.19), but
  downstream favors leaf.
- **Boundary strategy (B1/B2/B3) is second-order** on both metrics (BPB within 0.01; downstream leaf
  0.441/0.442/0.462 — noise).
- **All hybrids beat original gr** on decode BPB under an offline prompt (~1.18 vs 1.37 @ b/N=0.5).
- Downstream: **greedy-Q ≡ leaf-Q accuracy on A100** (MC argmax insensitive to that question-tok
  distinction at near-chance); only root-Q shifts the seam enough to move a few. Code verified —
  distinct tokenizations, identical argmax. A100 numbers differ slightly from earlier B200 downstream
  (§8/§8b/§8c) — the hardware inconsistency this A100 re-run resolves. `runs/poc/downstream/fullmatrix/` (A100).

## 9. Caveats

- ratio-10 pilot: **single seed, no CIs**. The ratio-40 + 3-seed campaign is the rigorous version.
- The clear win is at b/N>0 (offline prompt); at b/N=0 (pure greedy, fair) it is marginal and, at
  ratio-40, within CI — so the headline claim is **robustness to offline-tokenized prompts**, the
  realistic inference scenario, not a large pure-greedy gain.
- 100M downstream MC is near-chance (documented repo finding) → BPB is the sensitive metric.
