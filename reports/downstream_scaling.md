# Downstream scaling law (AU-Net word ladder)

Two-stage: **scale N → loss L(N) → benchmark acc_norm**. Data = final train loss +
existing acc_norm (hellaswag/arc_easy/arc_challenge/piqa) at 100M/300M/760M/1.3B.
Script `scripts/downstream_scaling.py`; figure `reports/downstream_scaling.png`.

## Data
| scale | N | loss | hellaswag | arc_easy | arc_chall | piqa | avg |
|---|---|---|---|---|---|---|---|
| 100M | 98.6M | 0.807 | 28.5 | 33.7 | 23.3 | 58.3 | 36.0 |
| 300M | 296.7M | 0.709 | 37.2 | 43.9 | 25.6 | 64.9 | 42.9 |
| 760M | 714.4M | 0.647 | 52.4 | 55.9 | 31.7 | 71.1 | 52.8 |
| 1.3B | 1.324B | 0.605 | 59.1 | 65.7 | 37.9 | 74.2 | 59.2 |

## Stage 1 — L(N) = E + A·N^-α
E = 0.176 (irreducible), A = 9.51, α = 0.147. Predicts L(1.3B)=0.605 (actual 0.605).

## Stage 2 — acc_norm = logistic(loss),  logit(acc) = a + b·L
Per-task + average fits (see figure). Accuracy is smooth and monotone in loss.

## Hold-out validation (fit on ≤760M → predict 1.3B)
| task | actual | predicted | err |
|---|---|---|---|
| hellaswag | 59.1 | 56.6 | −2.5 |
| arc_easy | 65.7 | 60.5 | −5.2 |
| arc_challenge | 37.9 | 32.9 | −5.0 |
| piqa | 74.2 | 73.5 | −0.7 |
| AVERAGE | 59.2 | 55.9 | −3.3 |

**MAE 3.3 pp** across a 1.8× param / 0.85× loss extrapolation. Systematic ~3 pp
**undershoot** — accuracy rises slightly faster at the top end than logit-linear-in-loss
predicts. piqa (near saturation) predicts best; arc_challenge (hardest, still ~38%) worst.

## Extrapolation (full-data fit, illustrative)
- 3B → L≈0.556 → avg ≈ 63% (hella 66, arc_e 70, arc_c 40, piqa 77)
- 7B → L≈0.511 → avg ≈ 68% (hella 72, arc_e 76, arc_c 43, piqa 80)

## 7B estimate (full 4-point fit, incl. 1.3B)
Stage 1: L(7B) ≈ **0.51**. Stage 2 (logistic in loss) → acc_norm:

| task | point | leave-one-out | via log(N) alt |
|---|---|---|---|
| hellaswag | 72% | 66–78 | 77 |
| arc_easy | 76% | 69–81 | 81 |
| arc_challenge | 43% | 37–51 | 47 |
| piqa | 80% | 78–82 | 82 |
| **AVERAGE** | **68%** | **63–73** | **72** |

Two functional forms bracket the average at **68–72%**, and the hold-out undershoot
(~3 pp) suggests the loss-based point is conservative → best guess **avg ≈ 70%**.
Script `scripts/downstream_7B.py`, figure `reports/downstream_7B.png`.

## Both families (AU-Net byte vs Llama subword), trunk-matched
`scripts/downstream_families.py`, figure `reports/downstream_families.png`. Common
trunk-N axis; loss units differ (byte vs token) so Stage 2 is fit per family.

Data (avg acc_norm): Llama leads at small scale (100M: 40.4 vs 36.0) but AU-Net's
steeper slope closes it (760M: 52.8 vs 55.8; 1.3B: 59.2 vs 60.3).

7B estimate (point [leave-one-out band]):

| task | AU-Net | Llama |
|---|---|---|
| hellaswag | 72 [66–78] | 75 [57–79] |
| arc_easy | 76 [69–81] | 75 [62–80] |
| arc_challenge | 43 [37–51] | 43 [31–52] |
| piqa | 80 [78–81] | 80 [74–82] |
| **AVERAGE** | **68 [63–73]** | **68 [56–73]** |

Hold-out 1.3B MAE: AU-Net 3.3 pp, Llama 4.2 pp (Llama's ladder is noisier → wider
bands). Both **undershoot ~3–4 pp**, so best guess **≈70–71% avg** for either. Net:
**the two architectures are predicted to tie at 7B (~68–71% avg)** — Llama's small-scale
edge (subword efficiency on English MC) is erased by AU-Net's faster scaling.

## Method comparison vs scaling_laws_plan.md  (7B avg acc_norm)
`scripts/downstream_methods.py`, figure `reports/downstream_methods.png`.

| method | AU-Net | Llama | winner | rg (plan) |
|---|---|---|---|---|
| (A) loss-based two-stage (mine) | 68.1 | 68.5 | **Llama +0.5** | — |
| (B) direct logit–log₁₀N (mine) | 72.3 | 71.9 | AU-Net +0.4 | — |
| (C) N-weighted Huber (plan) | 70.9 | 70.2 | AU-Net +0.7 | 72.2 |

**Method (A) — the loss-based two-stage — is the one that changes the conclusion.**
It (i) sits ~2–4 pp *lower* and (ii) is the **only** method where Llama stays ahead of
AU-Net at 7B. Both N-direct methods — my logit–logN (B) and the plan's Huber (C) —
instead have **AU-Net (byte) overtaking Llama**, matching the plan's headline ("byte's
steeper slope carries rg slightly ahead of Llama by 7B"; rg=72.2 leads there).

**Why they diverge:** (A) routes accuracy *through loss* (N→L→acc); as N grows, loss
saturates toward its irreducible floor, so the acc gain per decade is compressed —
suppressing byte's late-scale catch-up and leaving Llama nominally ahead. (B)/(C) fit
acc *directly* against scale, preserving the steep per-decade slope, so AU-Net's larger
slope overtakes Llama. The link choice (loss-mediated vs direct-in-N) matters more than
the fit details — B (logit) and C (linear-in-acc, N-weighted Huber) agree in direction,
differing only ~1.4 pp in level.

**Bottom line:** all three land at ~68–72% and put AU-Net and Llama within ~1 pp of each
other — a statistical tie at this extrapolation. Whether byte "overtakes" subword at 7B
is method-dependent and inside the noise; only the loss-mediated route flips the sign.

## Caveats
- 7B is a **5× extrapolation** beyond the 1.3B (largest point) from only 4 points —
  treat as a ballpark, not a committed number.
- Assumes the 7B is trained at a **budget consistent with the ladder trend**; the
  single-variable L(N) conflates N and D, so heavier/lighter training shifts loss
  (and hence downstream). A proper L(N,D) needs more points.
- Only 4 points; Stage-1 mixes recipes (100/300/760M are γ=10, 1.3B is TPP≈218) —
  Stage 2 (loss→acc) is recipe-robust, Stage 1 less so.
- Hard top-1 acc_norm is noisy near chance (arc_challenge). For tighter low-signal
  prediction, refit Stage 2 on a **continuous surrogate** (log-lik margin of the
  correct choice) via `harness.log_samples=true`, then map surrogate→accuracy.
