# Performance summary per scale — byte-level LM families

Five families on the AU-Net/BPEByte backbone: **BPEByte v4 (root_greedy)**, **AU-Net**
(word patches), **Llama** (subword baseline), **Entropy patching (BLT-style)**, and
**ByteFlow** (lossy coding-rate patching). Metric conventions: **BPB** = bits-per-byte,
**lower is better**; **downstream** = mean accuracy, **higher is better**.

> ⚠️ **Read this first — three measurement regimes, not one.** Only rg / AU-Net / Llama
> have a rigorous per-scale ladder (§1). Entropy (§2) and ByteFlow (§3) exist **at 100M
> only** and were measured under **different protocols** (different budget, benchmark set,
> and metric). **Do not cross-compare raw numbers between §1, §2, §3** — compare only
> *within* each section's baseline. Larger-scale entropy/ByteFlow configs exist but are
> **not yet trained**.

---

## 1. Mature families — full ladder (rg / AU-Net / Llama)

Protocol: **ratio-10 (γ=10)** scaling ladder, identical DCLM data, held-out **BPB**,
N-weighted Huber fit. Source: `scaling_laws/scaling_laws_plan.md`, `reports/model_results_1.3B.md`,
`reports/model_results_760M.md`, `reports/bpb_ci_1.3B.md`.

### 1a. BPB (↓ better)

| Scale (N) | BPEByte rg | AU-Net | Llama | Notes |
|---|---|---|---|---|
| **100M** | 1.1958 | 1.1972 | *n/a* | Llama not trained at 100M (BPB) |
| **300M** | 1.0142 | 1.0157 | *n/a* | Llama not trained at 300M (BPB) |
| **760M** | 0.9098¹ | 0.9169 | 0.9234 | ¹fit; measured new-code rg 0.9256 |
| **1.3B** | **0.8578** [0.858, 0.860] | 0.8662 | **0.8404** | temporal CI only (not seed CI) |
| **1.3B @ iso-byte (283 GB)** | 0.860 | 0.866 | **0.839** | Llama edge widens at iso-byte |
| **7B** *(extrapolated)* | 0.764 | 0.779 | **0.728** | fit prediction, 5.4× past 1.3B |

**BPB ranking (measured):** **Llama < rg < AU-Net** at every real scale. The subword
baseline keeps a ~0.02 (iso-param) → ~0.02–0.04 (iso-byte) BPB edge; byte models pay raw
BPB for robustness (see §1c).

### 1b. Downstream accuracy (↑ better)

| Scale | rg (4-bench) | AU-Net (4-bench) | Llama (4-bench) | rg / AU-Net / Llama (5-bench, 0-shot) |
|---|---|---|---|---|
| **100M** | 35.5% | 35.9% | 40.3% | — |
| **300M** | 42.0% | 42.9% | 46.4% | — |
| **760M** | 52.6% | 52.8% | 55.7% | — |
| **1.3B** | 60.3% | 59.8% | 60.3% | 64.4% / 65.0% / **65.6%** |
| **7B** *(extrap.)* | **72.2%** | 70.9% | 70.2% | — |

**Downstream ranking:** Llama leads at small scale, but the **byte models' slope is
steeper** — the cluster converges at 1.3B (rg ≈ Llama ≈ 60.3% on 4-bench), and the fit
predicts **rg overtakes Llama by 7B** (72.2% vs 70.2%). At 1.3B, 5-shot already flips it
(rg 67.9% > Llama 67.4% > AU-Net 66.6%, `model_results_1.3B.md`).

### 1c. Why bytes lose BPB but win robustness (1.3B)

| | rg | AU-Net | Llama |
|---|---|---|---|
| PBP ΔBPC (boundary/prompt-cut sensitivity) | ~0 | ~0 | **+0.71** (high) |
| HellaSwag-Noise | 42.3% | 41.5% | 37.6% |
| CUTE (character tasks) | 59.3 (spell) | ~59 | 18.4% (low) |

Byte models are **cut-invariant** and far stronger on character/noise tasks; Llama is
boundary-fragile ("never end a prompt with a trailing space"). Source: `model_results_1.3B.md:542–629`.

---

## 2. Entropy patching (BLT-style) — 100M only

**Method.** A small 50M causal byte-LM scores next-byte entropy; a patch boundary opens on
an entropy spike (global `H>θ` or drift-robust monotonic `H−H_prev>θ`), placed at
patch-start (leak-free, like root_greedy). Target ~4.5 bytes/patch. Entropy model trained at
5×/10×/20× Chinchilla (LOW/MID/HIGH). Source: `plans/plan_entropy_patching.md`,
`reports/100M_ablation.md §v8`, `lingua/apps/aunet/data/entropy_patch.py`.

> Protocol here is the **100M ablation**: ratio-40 budget, **4-bench** (HellaSwag, ARC-Easy,
> PIQA, ARC-Challenge), **accuracy only (no BPB)**. Numbers are NOT comparable to §1's
> ratio-10 5-bench figures — compare only to the rg/prefix_free rows in *this* table.

| 100M boundary scheme | Downstream avg (4-bench) | Total FLOPs (EFLOP) |
|---|---|---|
| v6 prefix_free (trie) | **39.07** | 31 |
| **v4 root_greedy (trie)** | **38.76** | 31 |
| Entropy HIGH (20×) | 38.15 | 39 |
| Entropy MID (10×) | 38.13 | 39 |
| Entropy LOW (5×) | 37.33 | 39 |
| Meta BLT (100M, learned) | 36.98 | 48 |

**Finding:** entropy patching **underperforms trie root_greedy** by ~0.6 pt (38.13 vs
38.76) **and costs +26–55% more compute** (entropy-model precompute). Entropy-model budget
**plateaus at MID** (LOW→MID +0.8, MID→HIGH +0.02). Meta's larger learned BLT is *worse*
than our 50M entropy model.

**Larger scales:** 300M / 760M / 1.3B configs exist (`bpebyte_*entropy*.yaml`) but are
**NOT trained** — no data.

---

## 3. ByteFlow (lossy coding-rate patching) — 100M only

**Method.** Boundaries chosen model-side from the first encoder's hidden states by marginal
**coding rate** ΔRₜ (information gain): exact log-det (Sherman-Morrison) or L2 approx
(ΔRₜ ∝ ‖hₜ‖²); selection by causal EMA-quantile threshold, rolling/sliding top-k, or global
top-k (diagnostic). Source: `plans/plan_byteflow.md`, `lingua/apps/aunet/coding_rate.py`.

> Protocol here: **100M, ~1672 steps (~ratio-5), train-loss (per-byte CE, nats) as a BPB
> proxy, no downstream, no held-out BPB.** Numbers are NOT comparable to §1 or §2 — compare
> only to the rg row in *this* table.

| 100M mode (1672 steps) | Train loss (nats, rank0) ↓ |
|---|---|
| **BPEByte v4 root_greedy (peer)** | **0.931** |
| ByteFlow sliding_topk L=2048 (best) | 1.000 |
| ByteFlow logdet_quantile | 0.997 |
| ByteFlow l2_quantile (primary) | 1.007 |
| ByteFlow sliding_topk L=512 | 1.011 |
| ByteFlow exact log-det | 1.027 |

**Finding:** ByteFlow **underperforms rg by +0.06–0.07** at this budget; score fidelity
(exact vs L2) and selection method make **no significant difference** here. Attributed to
cold-start (random encoder at init), a **missing Canon layer** (~2 pts per paper ablation),
encoder depth 3 vs paper's 6, and heavy undertraining (1672 vs paper's ~1.95M steps) — i.e.
**not yet a faithful reproduction**.

**Paper's own 600M (reference, not ours):** global top-k BPB 0.86 / 50.9% downstream.

**Larger scales:** 300M / 760M / 1.3B configs + launch scripts exist but are **NOT trained** — no data.

---

## 4. Bottom line

- **Best raw BPB:** Llama (subword) at every real scale; edge widens at iso-byte, persists in the 7B fit.
- **Best downstream trajectory:** byte models (rg/AU-Net) — steeper slope, converge with Llama at 1.3B and the fit predicts **rg overtakes by 7B**; byte models dominate robustness/character tasks at all scales.
- **rg vs AU-Net:** near-identical (rg slightly better BPB, essentially tied downstream); rg is the leak-free byte reference.
- **Entropy patching:** implemented, but **loses to trie root_greedy at 100M** on accuracy while costing more compute; unproven above 100M.
- **ByteFlow:** implemented, **loses to root_greedy at 100M** on train-loss proxy; not a faithful repro yet; unproven above 100M.
- **Maturity:** only rg / AU-Net / Llama have a real scale ladder. **Entropy and ByteFlow are 100M-only** — their 300M→1.3B configs are staged but unrun.

### Sources
`scaling_laws/scaling_laws_plan.md` (ladder BPB + downstream, 7B fit) · `reports/model_results_1.3B.md`
(1.3B benchmarks + robustness) · `reports/model_results_760M.md` · `reports/bpb_ci_1.3B.md`,
`reports/bpb_compare_1.3B_275G.md` (1.3B CIs, iso-byte) · `reports/100M_ablation.md §v8`
(entropy vs trie) · `plans/plan_entropy_patching.md`, `lingua/apps/aunet/data/entropy_patch.py`
(entropy method) · `plans/plan_byteflow.md`, `lingua/apps/aunet/coding_rate.py` (ByteFlow).

_Caveats: 1.3B CIs are temporal (motion-block) bands, not seed CIs. Llama 100M/300M BPB not
measured (only downstream/fit). 7B row is a 5.4× extrapolation from a 4-point fit — hypothesis,
not measurement._
