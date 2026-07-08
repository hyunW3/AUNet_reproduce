# 300M parsing-rule comparison: monotonic vs global (HIGH entropy model, iso-data 3344 steps)

Scale-up of the 100M parsing-rule ablation. Fix the HIGH entropy model (byte_50M, 9200 steps / 20x
Chinchilla); vary the boundary RULE. Both 300M AU-Nets (dims [512,1280], layers [3,13], seq 8192,
296.66M params), trained on ece-agpu18 4xA100 (batch16 ga12 = 1.57M tok/step, compile off, entropy_gpu),
3344 steps (iso-data with 100M ratio-10; = ratio-3.3 for a 300M — see note).
- monotonic: boundary where dH_t = H_t - H_{t-1} > theta,  theta = 0.6173 (val-cal, 4.5 B/patch)
- global:    boundary where H_t > theta,                   theta = 1.4797 (val-cal, 4.5 B/patch)
Eval on ece-agpu11 (shared disk), limit-1000. HS/ARCe/PIQA/ARCc = acc_norm; BoolQ = acc. (gen_ll generation-framed variants dropped -- cloze scoring only.)

> **RATIO LABELING (important).** "ratio-N" = data-to-model ratio (tokens/param). These 300M runs are
> STEP-matched to the 100M runs (both ~1.57M tok/step; 1672 / 3344 steps = iso-DATA, ~2.6B / ~5.3B bytes),
> deliberate for an iso-data scale comparison. But 300M has **3.01x** the params (296.66M vs 98.59M), so at
> equal data it sits at a LOWER data/model ratio: the "ratio-5 (1672)" / "ratio-10 (3344)" columns below
> are, for a 300M, actually **ratio-1.7 / ratio-3.3** — NOT true ratio-5/10. A true 300M ratio-10 needs
> 3.01x the data ≈ **10,065 steps** (~15.8B bytes; a fresh cosine run — the 3344 run's LR already decayed
> to its floor, so it can't be resumed — being trained separately). Implication: the cross-scale BPB here
> is iso-DATA (same tokens), and the 300M is *under-trained for its size*; at true ratio-10 its BPB would
> drop further. The mono-vs-global comparison is unaffected — both 300M runs sit at the same ratio-3.3.

## Train BPB (mean last-40 loss/ln2) -- the primary signal

| rule       | ratio-5 (1672) | ratio-10 (3344) |
|------------|:--------------:|:---------------:|
| monotonic  | 1.219          | **1.131**       |
| global     | 1.230          | 1.144           |
| delta (g-m)| +0.011         | +0.013          |

**KEY FINDING: the 100M global-rule BPB edge INVERTS at 300M.**
- 100M (r10): global 1.230 < monotonic 1.242  -> global better by 0.012
- 300M (r10): monotonic 1.131 < global 1.144   -> MONOTONIC better by 0.013
The advantage of H>theta over dH>theta at equal 4.5 B/patch does NOT survive scaling; at 300M the
monotonic (surprise-jump) rule places boundaries marginally better for LM loss. Both improve ~0.09-0.10
BPB from 100M->300M at r10 (mono 1.242->1.131, global 1.230->1.144).

## Downstream (limit-1000)

| metric (r10, 3344)        | monotonic | global | note |
|---------------------------|:---------:|:------:|------|
| HellaSwag (acc_norm)      | 36.1      | 37.6   | global +1.5 |
| ARC-easy (acc_norm)       | 35.4      | 35.2   | tie |
| PIQA (acc_norm)           | 57.4      | 58.9   | global +1.5 |
| ARC-challenge (acc_norm)  | 22.8      | 22.2   | tie |
| BoolQ (acc)               | 60.0      | 39.0   | DEGENERATE (majority ~62; both off, global collapsed) |

| metric (r5, 1672)         | monotonic | global |
|---------------------------|:---------:|:------:|
| HellaSwag (acc_norm)      | 34.5      | 35.9   |
| ARC-easy (acc_norm)       | 33.3      | 32.4   |
| PIQA (acc_norm)           | 58.3      | 56.4   |
| ARC-challenge (acc_norm)  | 21.1      | 22.0   |
| BoolQ (acc)               | 60.0      | 43.3   |

## Verdict
- **BPB: monotonic wins at 300M** (1.131 < 1.144), reversing the 100M global edge -> the parsing-rule
  BPB ranking is scale-dependent and small (~1%); not a robust win for either rule.
- **Downstream: flat within noise.** Global edges HellaSwag/PIQA by ~1.5 at r10 but loses ARCe/BoolQ;
  BoolQ is degenerate (global=39 is below the ~62 majority = collapsed, not signal).
- Consistent with the 100M conclusion: at these budgets the boundary RULE (dH>t vs H>t) barely moves
  either BPB or downstream; the ~0.01 BPB gap flips sign with scale and is below run-to-run relevance.

## Full cross-scale table (cloze downstream tasks + BPB)

Cloze (standard loglikelihood continuation) scoring only -- the gen_ll generation-framed variants are
dropped. HIGH entropy model, val-cal, equal 4.5 B/patch, limit-1000. HS/ARC-e/PIQA/ARC-C = acc_norm %;
BoolQ = acc %. Avg4 = mean of the 4 non-degenerate acc_norm tasks (HS, ARC-e, PIQA, ARC-C); BoolQ is
listed but EXCLUDED from Avg4 (majority-class degenerate, ~62 baseline). BPB = mean loss/ln2 over the
last-40-step window (consistent method, 2026-07-07). Best BPB per (scale,budget) in bold.

| scale | rule      | budget | HS   | ARC-e | PIQA | ARC-C | BoolQ | Avg4 | BPB       |
|-------|-----------|--------|------|-------|------|-------|-------|------|-----------|
| 100M  | monotonic | r5     | 35.0 | 32.2  | 54.0 | 20.2  | 38.3  | 35.4 | **1.294** |
| 100M  | global    | r5     | 34.1 | 31.9  | 52.0 | 21.6  | 59.0  | 34.9 | 1.321     |
| 100M  | monotonic | r10    | 35.6 | 33.6  | 55.9 | 21.3  | 41.4  | 36.6 | 1.251     |
| 100M  | global    | r10    | 34.6 | 33.0  | 55.0 | 21.6  | 55.1  | 36.1 | **1.237** |
| 300M  | monotonic | r5     | 34.5 | 33.3  | 58.3 | 21.1  | 60.0  | 36.8 | **1.219** |
| 300M  | global    | r5     | 35.9 | 32.4  | 56.4 | 22.0  | 43.3  | 36.7 | 1.230     |
| 300M  | monotonic | r10    | 36.1 | 35.4  | 57.4 | 22.8  | 55.6  | 37.9 | **1.131** |
| 300M  | global    | r10    | 37.6 | 35.2  | 58.9 | 22.2  | 39.0  | 38.5 | 1.144     |

**Cloze downstream confirms no reliable rule winner, and does not track BPB.** By Avg4 monotonic edges
global at 100M (35.4 vs 34.9 r5; 36.6 vs 36.1 r10) but global edges monotonic at 300M/r10 (38.5 vs 37.9)
-- the OPPOSITE of the BPB ranking (global better at 100M/r10, mono better at 300M/r10). BoolQ swings on
majority-class noise (collapses to 39-43 in three cells) and is excluded from Avg4. Net: cloze downstream
is flat/noise-dominated at both scales; the ~1% BPB gap is the only consistent signal, and it flips with scale.

## Cross-scale summary (100M vs 300M) -- BPB recomputed consistently (last-40-step window, loss/out)

All 8 BPB values below use the SAME method (mean loss/ln2 over the last-40-step window at each ckpt),
recomputed 2026-07-07 so 100M and 300M are apples-to-apples. This shifts the 100M r10 figures up ~+0.008
vs the original r10-only write-up (mono 1.242->1.251, global 1.230->1.237) but preserves the ordering.

NOTE: the r5/r10 columns are STEP counts (1672/3344), STEP-matched across scales = iso-DATA. For the 300M
(3.01x params) these are true **ratio-1.7 / ratio-3.3**, not ratio-5/10. So "scale helps" below is at EQUAL
DATA; the 300M is under-trained for its size, and a true 300M ratio-10 (10,065 steps, in training) would
lower its BPB further still.

| scale | mono r5 | mono r10 | global r5 | global r10 | r10 winner        |
|-------|:-------:|:--------:|:---------:|:----------:|-------------------|
| 100M  | 1.294   | 1.251    | 1.321     | 1.237      | global (-0.014)   |
| 300M  | 1.219   | 1.131    | 1.230     | 1.144      | monotonic (-0.013)|

- **At ratio-5, monotonic wins at BOTH scales** (100M 1.294<1.321; 300M 1.219<1.230). The global rule
  only ever overtakes at **100M/ratio-10** -- and even there by ~0.014 -- then LOSES again at 300M/ratio-10.
- **So the global-rule BPB edge is fragile, not a scaling trend**: it appears in exactly one cell
  (100M/r10) and reverses with scale. The H>theta vs dH>theta ranking is not robust; the gap is ~1%.
- **Scale helps ~10x more than rule** (at EQUAL DATA): r10 BPB drops ~0.10-0.12 from 100M->300M
  (mono 1.251->1.131, global 1.237->1.144) vs the ~0.013 rule effect — and the 300M is only at ratio-3.3,
  so its true-ratio-10 BPB would be even lower.
- **Downstream flat within noise at both scales** -- no consistent rule winner; BoolQ
  degenerate. Only BPB is a reliable ranker, and it disagrees across scale/budget.
- NET: the boundary rule (entropy-jump vs entropy-level) is not a meaningful lever for AU-Net quality at
  these budgets. (100M source: runs/cmp_100M/ratio5_comparison.md; recompute from monoval/global
  metrics.jsonl on snu55/ece.)
