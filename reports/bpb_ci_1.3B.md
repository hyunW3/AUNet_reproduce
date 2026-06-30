# 1.3B BPB — uncertainty bands without retraining (D9)

**Goal.** The headline 1.3B train-BPB numbers (root_greedy **0.860**, AU-Net2 **0.866**,
Llama 1.8B **0.839**) are single-run point estimates with no confidence interval, and the close
delta (~0.006) sits at the stated ±0.005 noise floor. A true cross-seed CI needs retraining
(infeasible). Below are the best *defensible* bands obtainable from `metrics.jsonl` / log analysis
only — and an explicit statement of what each does and does **not** capture.

Reproduce: `lingua/.venv/bin/python runs/bpb_ci_analysis.py`

---

## 1. BPB derivation (confirmed, matches the published pipeline)

Logged per step in `metrics.jsonl`: `loss/out` = `ce/out` = mean cross-entropy in **nats**
(`apps/aunet/train.py:561`). The plot script `runs/plot_bpb_all.py` converts:

- **byte models** (`loss/out` is nats/**byte**): `BPB = loss / ln2`
- **Llama** (`loss/out` is nats/**token**): `BPB = loss / (ln2 · 4.5483)`, where 4.5483 bytes/token
  is measured on 5000 DCLM docs.
- **EMA α=0.01** over raw per-step BPB; published number = EMA read at the **common 283.1 GB**
  iso-byte budget (the byte-model endpoint; Llama overshoots to 286.2 GB so it is read back at 283).

My script reproduces the published values exactly:

| model | run dir | EMA@283 GB BPB | raw last-step BPB |
|-------|---------|:--------------:|:-----------------:|
| BPEByte root_greedy | `bpebyte_br_greedy_root_1.3B` | **0.8603** | 0.8675 |
| AU-Net2 (pure byte) | `aunet2_1.3B` | **0.8657** | 0.8724 |
| Llama 1.8B subword | `llama_1.8B_paper` | **0.8383** | 0.8700 |
| BPEByte offline bt (leaky) | `bpebyte_br_bt_1.3B` | 0.7805 | 0.7886 |
| BPEByte online bt (leaky) | `bpebyte_br_bt_online_1.3B` | 0.7874 | 0.7958 |

Note the EMA−raw gap is ~**−0.007** for byte models and **−0.032** for Llama (Llama logs fewer,
noisier steps and its last raw step reads 0.87 — confirming the doc's warning that the raw last
step is not the curve value).

---

## 2. Band (a) — temporal / optimization noise (within one run)

Over the **last ~3 GB** of training (steady-state, post-LR-decay flat tail; N≈1900 logged steps
for byte runs, 630 for Llama):

| model | raw per-step sd | EMA-curve sd | EMA−raw gap |
|-------|:---------------:|:------------:|:-----------:|
| root_greedy | 0.0186 | 0.0012 | −0.0072 |
| AU-Net2 | 0.0190 | 0.0013 | −0.0067 |
| Llama 1.8B | 0.0167 | 0.0009 | −0.0316 |

The **single raw step** wobbles by ±~0.019 (large — a per-step minibatch estimate). The reported
number is not a single step, it is the EMA / tail-mean, so the right band on *that* quantity is the
CI of the tail mean, computed with a **moving-block bootstrap** (block=50 steps, 2000 resamples) to
respect step-to-step autocorrelation:

| model | tail-mean BPB | 95% CI (temporal) | half-width |
|-------|:-------------:|:-----------------:|:----------:|
| root_greedy | 0.8592 | [0.8583, 0.8600] | ±0.0008 |
| AU-Net2 | 0.8645 | [0.8637, 0.8654] | ±0.0008 |
| Llama 1.8B | 0.8389 | [0.8378, 0.8400] | ±0.0011 |

**Temporal band on the reported endpoint ≈ ±0.001 BPB.** What it captures: how much the *number*
moves due to minibatch/optimization noise within a single run at the endpoint. What it does **NOT**
capture: seed, data-order, or init variance (the run is one fixed seed/data shuffle).

---

## 3. Band (b) — data-sampling / validation bootstrap: **infeasible from saved artifacts**

A per-document bootstrap CI on a held-out validation BPB was the goal, but:

- The val loop (`apps/aunet/eval.py:eval_on_val`, lines 517–528) computes `nll_per_byte` per doc
  and then **averages before saving** — only the *mean* lands in `metrics.validation.jsonl`. The
  per-document array needed to resample is **not persisted**. Recovering it requires re-running the
  GPU val pass (disallowed here).
- The two head-to-head opponents have **no validation jsonl at all**: `aunet2_1.3B` and
  `llama_1.8B_paper` were never run through `eval_on_val`. So even the *mean* val BPB is unavailable
  for the comparison that matters.
- The val set that *does* exist (`dclm_baseline_1.0_2shards_shuffled`, `*.val.jsonl`) is a
  **train-source** split, not a held-out OOD set, and a single source — so a bootstrap over it would
  measure in-distribution sampling noise, not generalization.

For the record, the val BPB (= −nll_per_byte/ln2) that *is* logged @180k:
root_greedy **0.8245**, offline_bt 0.7816, online_bt 0.7850 (no AU-Net2 / Llama). These are lower
than the train-BPB endpoints (val docs are longer / full-context), and preserve the
root_greedy > bt ranking — but cannot be turned into a bootstrap CI or a 3-way comparison.

**Bottom line for (b):** not computable without a GPU re-eval that also adds val passes for
aunet2 and llama. Quantify what it would take: re-run `eval_on_val` for all 4 models with per-doc
`nll_per_byte` dumped (patch line 528 to keep the list), on a true held-out shard; then 1000×
per-doc resample → 95% CI. ~1 GPU-hour/model.

---

## 4. Head-to-head verdict under the available bands

Using the temporal moving-block CIs (band a):

| comparison | Δ BPB | temporal CIs | verdict (temporal only) |
|------------|:-----:|:------------:|:-----------------------:|
| root_greedy vs AU-Net2 | −0.0054 | [.858,.860] vs [.864,.865] | **disjoint** — survives temporal noise |
| root_greedy vs Llama 1.8B | +0.0220 | [.858,.860] vs [.838,.840] | **disjoint** — survives |
| AU-Net2 vs Llama 1.8B | +0.0274 | [.864,.865] vs [.838,.840] | **disjoint** — survives |

All three pairs separate under the temporal band — **but this is the weak claim.** The temporal
band (±0.001) is ~5× *tighter* than the stated ±0.005 noise floor precisely because it ignores the
dominant source of variance (seed / data order). So "CIs disjoint" here means only *"not explained
by within-run wobble,"* not *"reproducible across reruns."*

- **root_greedy (0.860) vs AU-Net2 (0.866):** Δ=0.0054. Survives temporal noise, but Δ ≈ the
  ±0.005 seed-floor → **NOT safely distinguishable**; treat as within plausible seed noise.
- **root_greedy / AU-Net2 vs Llama (Δ≈0.022–0.027):** ~4–5× the seed-floor → **robustly
  distinguishable**; the byte models having *higher* train BPB than the 1.8B subword baseline is a
  real ordering, not noise.

---

## 5. Caveats — neither band is a cross-seed CI

- **Band (a)** captures only minibatch/optimization wobble at the endpoint of *one* run. It is an
  *under*-estimate of true uncertainty and should not be quoted as "the CI."
- **Band (b)** (data-sampling) is **not computable** from saved artifacts (means-only val logs;
  missing val for 2 of 4 models).
- **Neither is a seed CI.** Run-to-run variance from random seed, data-shuffle order, and init is
  the term that actually governs whether a 0.006 gap is real, and it is **not measurable without
  reruns**. Rule of thumb at this scale: seed std on train BPB is typically 0.003–0.01 — i.e. of the
  *same order as the root_greedy↔AU-Net2 gap itself.*
- **To get a real seed CI:** retrain each model with ≥3 seeds (5 preferred) to the same 283 GB
  budget; report mean ± t-interval. At 1.3B / 180k steps that is ~3–5× the original compute per
  model — the reason it is deferred. A cheaper partial proxy: vary only the data-shuffle seed for
  the last ~10–20 GB from a shared checkpoint (captures data-order, not init/full-trajectory
  variance).

**One-line summary:** within-run temporal noise on the reported BPB is ≈ ±0.001 (moving-block
bootstrap); the root_greedy↔AU-Net2 0.006 gap exceeds that but sits at the ±0.005 seed-floor so it
is **not defensibly distinguishable**, whereas the ~0.022–0.027 gap to Llama 1.8B **is**. A true
cross-seed CI is unavailable without reruns (~3–5 seeds × full 1.3B training).
