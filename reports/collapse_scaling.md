# AdamW weight-decay timescale (τ) across the scaling fleet

Analysis of the AdamW EMA timescale following Wang & Aitchison,
*"How to set AdamW's weight decay as you scale model and dataset size"*
([arXiv:2405.13698](https://arxiv.org/abs/2405.13698), OpenReview `3YKeB9R1g9`).

## Definitions

AdamW's decoupled update `w ← (1 − ηλ)·w − η·û` is an EMA of the normalized
update `û`, with timescale:

- **τ_iter = 1 / (ηλ)** — in optimizer steps.
- **τ = τ_iter / T = 1 / (ηλT) = B / (ηλD)** — dimensionless (fraction of the
  training run; equals *epochs* when the run is ~1 pass over the data).
  - `T = D / B` = total optimizer steps
  - `D` = total training tokens
  - `B` = tokens per optimizer step = `batch_size · seq_len · grad_acc · dp_replicate`
  - `η` = learning rate, `λ` = weight decay

Paper's empirically optimal range (Llama/StableLM, ~1 epoch):
**τ_initial ≈ 0.067–0.19**, **τ_final ≈ 0.67–3.1**.

### Schedule caveat

All runs use **cosine** decay with `lr_min_ratio = 0.01`, so η drops 100× over
training and τ_iter = 1/(ηλ) grows 100× from start to end. Hence the paper's
*initial* (peak-lr) vs *final* (min-lr) split; here **τ_final = 100 · τ_initial**.
The `initial` column below is the peak-lr timescale (directly comparable to the
paper's `τ_initial`).

## Runs

Constants across the fleet: **λ = 0.1**, `dp_replicate = 4`, `seq_len = 8192`.
`B` verified against logged `total_tokens` (760M: 2.36M tok/step × 60,600 =
142.97B ✓).

| run (checkpoint) | status | η (peak) | B tok/step | T steps | D tokens | τ_iter = 1/ηλ | **τ initial** | τ final (×100) |
|---|---|---|---|---|---|---|---|---|
| cmp_g10/aunet_760M | ✅ done (60,600) | 0.00165 | 2.36M | 60,600 | 143.0B | 6,061 | **0.100** | 10.0 |
| cmp_g10/rg_760M    | ▶ training (~11.5k) | 0.00165 | 2.36M | 60,600 | 143.0B | 6,061 | **0.100** | 10.0 |
| cmp_g10/aunet_300M | done (9,900)    | 0.0019  | 6.29M | 9,900  | 62.3B  | 5,263 | **0.532** | 53.2 |
| cmp_300M/aunet_300M| ▶ training (~220)   | 0.0019  | 6.29M | 6,688  | 42.1B  | 5,263 | **0.787** | 78.7 |
| cmp_100M/v4_root_greedy_ot | ▶ training (~6.68k) | 0.002 | 6.29M | 6,688 | 42.1B | 5,000 | **0.748** | 74.8 |
| cmp_g10/aunet_100M | done (3,344)    | 0.002   | 6.29M | 3,344  | 21.0B  | 5,000 | **1.495** | 149.5 |

Per-run `B` (tokens/step, from `batch_size · seq_len · grad_acc · dp_replicate`):
- 760M (aunet): 12·8192·6·4 = **2,359,296**
- 760M (rg):    24·8192·3·4 = **2,359,296**
- 300M:         48·8192·4·4 = **6,291,456**
- 100M:         96·8192·2·4 = **6,291,456**

## Finding: τ is not held constant — 0.10 → 1.50 (15× spread)

τ_iter = 1/(ηλ) is nearly pinned (5,000–6,061 steps) because λ is fixed at 0.1
and lr only mildly muP-scales, while T collapses 60,600 → 3,344. So τ = τ_iter/T
balloons on the short runs.

- **760M runs (τ_initial = 0.100) sit squarely in the paper's optimal band
  (0.067–0.19).** Well tuned — because `warmup = 0.1·T` makes τ_iter ≈ warmup here.
- **100M/300M runs are 3–15× above the band.** `cmp_g10/aunet_100M` at
  **τ = 1.50** has an EMA window *longer than the entire run* — weight decay never
  effectively bites and early updates are never forgotten (the paper's stated
  failure mode: "τ should not be much bigger than the number of training epochs").

## Flagship 1.3B + 760M runs

Same τ_iter = 1/(ηλ), τ_initial = τ_iter/steps, τ_final = 100·τ_initial.
`B` verified against logged `total_tokens` (1.3B: 1,572,864 × 180,000 = 283.12B ✓;
llama_1.8B: 1,048,576 × 60,000 = 62.91B ✓; llama_760M: 1,179,648 × 12,900 = 15.22B ✓).

| run | η (peak) | λ | B tok/step | T steps | D tokens | τ_iter=1/ηλ | **τ initial** | τ final |
|---|---|---|---|---|---|---|---|---|
| 1.3B/aunet2_1.3B                | 0.00165 | 0.1   | 1.57M | 180,000 | 283.1B | 6,061  | **0.034** | 3.37 |
| 1.3B/bpebyte_br_bt_1.3B         | 0.00165 | 0.1   | 1.57M | 180,000 | 283.1B | 6,061  | **0.034** | 3.37 |
| 1.3B/bpebyte_br_bt_online_1.3B  | 0.00165 | 0.1   | 1.57M | 180,000 | 283.1B | 6,061  | **0.034** | 3.37 |
| 1.3B/bpebyte_br_greedy_root_1.3B| 0.00165 | 0.1   | 1.57M | 180,000 | 283.1B | 6,061  | **0.034** | 3.37 |
| 1.3B/llama_1.8B_paper           | 0.003   | 0.033 | 1.05M | 60,000  | 62.9B  | 10,101 | **0.168** | 16.84 |
| 760M/aunet2_760M                | 0.00165 | 0.1   | 2.36M | 29,200  | 68.9B  | 6,061  | **0.208** | 20.76 |
| 760M/bpebyte_root_greedy_760M   | 0.00165 | 0.1   | 2.36M | 29,200  | 68.9B  | 6,061  | **0.208** | 20.76 |
| 760M/llama_760M                 | 0.0056  | 0.1   | 1.18M | 12,900  | 15.2B  | 1,786  | **0.138** | 13.84 |

Per-run `B` (`batch_size · seq_len · grad_acc · dp_replicate`):
- 1.3B (aunet/br_bt): 12·8192·4·4 = **1,572,864**; (br_greedy_root): 24·8192·2·4 = **1,572,864**
- 1.3B/llama_1.8B: 4·4096·8·8 = **1,048,576**
- 760M (aunet2): 12·8192·6·4 = **2,359,296**; (rg): 24·8192·3·4 = **2,359,296**
- 760M/llama: 16·2048·9·4 = **1,179,648**

### Observations at scale

- **The flagship 1.3B AU-Net/bpebyte runs sit at τ_initial = 0.034 — *below* the
  paper's optimal band (0.067–0.19).** This is the *opposite* failure from the
  small runs: the 180k-step horizon makes the EMA window too short relative to the
  run, so weight decay is (mildly) too aggressive and early updates are forgotten
  too fast. All four 1.3B variants share this — they use the fixed lr=0.00165 /
  wd=0.1 recipe with T pushed to 180k.
- **The two Llama runs land in-band** (1.8B: 0.168, 760M: 0.138) — because they
  use *different* hyperparameters (llama_1.8B: λ=0.033, η=0.003; llama_760M:
  η=0.0056), not the AU-Net recipe. These look independently tuned.
- **The 760M AU-Net runs come in two flavors:** `aunet2_760M` (29.2k steps,
  **τ=0.208**, just above band) vs `cmp_g10/aunet_760M` (60.6k steps, **τ=0.100**,
  in-band). Same lr/wd — the 2× longer horizon is the entire difference.

### Fleet-wide picture

Across everything, **τ_initial spans 0.034 → 1.50 — a ~44× range** with no iso-τ
discipline. Ordered:

```
1.3B aunet/bpebyte  0.034   ← below band (too aggressive)
cmp_g10 760M        0.100   ← in band ✓
760M llama          0.138   ← in band ✓
1.3B llama_1.8B     0.168   ← in band ✓
760M aunet2         0.208   ← just above
cmp_g10 300M        0.532   ← above
cmp_100M v4         0.748   ← above
cmp_300M            0.787   ← above
cmp_g10 100M        1.495   ← EMA window > whole run
```

The in-band cluster (0.10–0.17) is exactly the runs whose (η, λ, T) happen to line
up; the AU-Net fixed-recipe runs drift both ways as T changes — short T → τ too
big (100M), long T → τ too small (1.3B).

### Implication for the matched comparison

Weight-decay timescale is a **confound** between size buckets: the 760M gets a
15× shorter effective EMA window than the 100M. To make the smaller runs iso-τ
with the 760M's 0.10, raise ηλ — e.g. `aunet_100M` needs τ_iter ≈ 0.10·3344 ≈ 334
steps → ηλ ≈ 3.0e-3, i.e. **λ ≈ 1.5** at η = 0.002 (or a compensating lr change),
not 0.1.

## Experiment: does the 1.3B τ collapse at 100M?

At 1.3B, τ=0.034 was **benign** — loss fell smoothly (aunet 6.02→0.60, bpebyte
5.99→0.55; grad norms ~0.03–0.05, stable). Question: does the *same* τ collapse a
100M model (a scale-dependent WD pathology)?

**Setup** (ece-agpu18, 4×A100-80GB, GPUs 0–3, `runs/collapse_tau/`):
2×2 design, all runs `steps=6688`, `lr=2e-3`, `warmup=400`, `bs=16`, `grad_acc=2`,
1 GPU each, identical except model × λ. τ matched via weight decay (the only
feasible knob — matching via run length needs T≈147k steps).

| run | model | config | λ | τ = 1/(ηλT) |
|---|---|---|---|---|
| bpebyte_tau0034 | BPEByte (root_greedy) | bpebyte_100M_v4_root_greedy | 2.2 | **0.034** (=1.3B) |
| bpebyte_tau0748 | BPEByte (root_greedy) | bpebyte_100M_v4_root_greedy | 0.1 | 0.748 (baseline) |
| aunet_tau0034   | AU-Net (word)         | r20_aunet_orig              | 2.2 | **0.034** (=1.3B) |
| aunet_tau0748   | AU-Net (word)         | r20_aunet_orig              | 0.1 | 0.748 (baseline) |

Both 100M (dims [512,768], layers [3,10]). Launched via
`runs/collapse_tau/run_collapse_tau.sh` (nohup). Collapse signature to watch:
loss/out diverging or plateauing high, grad_norm spiking, vs the λ=0.1 baseline.

**Result:** _(in progress — see `runs/collapse_tau/*/train.log` on ece)_

Interim @ step 510 — **degradation, not catastrophic collapse:**

| run | τ | loss @510 | grad |
|---|---|---|---|
| bpebyte_tau0034 | 0.034 | 1.408 | 0.335 |
| bpebyte_tau0748 | 0.748 | 1.211 | 0.171 |
| aunet_tau0034   | 0.034 | 1.410 | 0.320 |
| aunet_tau0748   | 0.748 | 1.221 | 0.165 |

Low-τ runs are ~0.19 nats worse with ~2× higher grad norm (aggressive-WD
signature), but still descending (3.57→1.41) — no divergence/NaN. Both model
families near-identical.

Mid-run trajectory (loss/out):

| step | bpe τ0.034 | bpe base | aunet τ0.034 | aunet base | grad low-τ / base |
|---|---|---|---|---|---|
| 510  | 1.41 | 1.21 | 1.41 | 1.22 | 0.33 / 0.17 |
| 1810 | 1.30 | 1.06 | 1.30 | 1.07 | 0.30 / 0.08 |
| 3410 | 1.148 | 0.929 | 1.115 | 0.998 | 0.55 / 0.055 |

**Verdict (through step 3410): persistent degradation, NOT catastrophic collapse.**
- Low-τ keeps descending smoothly (1.61→1.15); no divergence/NaN/plateau-blowup.
- Loss gap holds ~0.1–0.2 nats — stable offset, *not* progressively widening.
- Low-τ grad norm sits ~10× higher than baseline (0.5 vs 0.05) for the whole run,
  loss noticeably noisier — the aggressive-WD high-tension equilibrium (weights
  held small, gradients large relative to them), but stable.

This matches the paper's framing (too-low τ is *suboptimal*, not blow-up-inducing)
and mirrors the 1.3B, which was also fine at τ=0.034 — the low-τ regime is a smooth
performance hit at both scales, not a scale-dependent collapse.

**Final (step 6688, smoothed final loss/out):**

| run | τ | L_final | Δ vs baseline |
|---|---|---|---|
| aunet base | 0.75 | 0.937 | — |
| aunet low  | 0.034 | 0.991 | +0.054 |
| bpe base   | 0.75 | 0.920 | — |
| bpe low    | 0.034 | 1.021 | +0.101 |

Confirmed: **stable degradation, no collapse.** Low-τ costs ~0.05 (AU-Net) / ~0.10
(BPEByte) nats and runs at ~10× grad norm, but trains to a healthy optimum.

**Downstream eval (acc_norm; hellaswag/arc_easy/arc_challenge/piqa):**

| run | τ | hellaswag | arc_easy | arc_chall | piqa | avg |
|---|---|---|---|---|---|---|
| aunet base | 0.75 | 27.2 | 30.3 | 21.5 | 54.1 | **33.3** |
| aunet low  | 0.034 | 24.9 | 27.0 | 25.4 | 52.6 | **32.5** |
| bpe base   | 0.75 | 26.9 | 30.3 | 20.2 | 55.1 | **33.1** |
| bpe low    | 0.034 | 24.2 | 27.7 | 25.7 | 51.6 | **32.3** |

Near chance (expected for 100M byte models); low-τ is uniformly ~0.8 pp worse, no
downstream collapse. At matched τ the two architectures nearly coincide (base 33.3
vs 33.1; low 32.5 vs 32.3) — same architecture-invariant, τ-dependent pattern seen
in the loss and the normalized TLC.

## Normalized training-loss curves (TLC) — the paper's "collapse"

Following 2509.25087: normalize each loss curve by its final loss,
ℓ(t̂)=L(t̂·T)/L_final vs t̂=t/T. If runs share the same normalized shape they
"collapse" onto one curve. Collapse residual = mean pointwise spread across a group
over t̂∈[0.2,1] (post-warmup). Script `scripts/normalize_tlc.py`; figures
`reports/tlc_scale_ladder.png`, `reports/tlc_collapse_tau.png`.

**Result 1 — matched-τ collapse across architectures (`tlc_collapse_tau.png`).**
At the SAME τ, the 100M AU-Net and BPEByte normalized curves lie almost exactly on
top of each other: residual **0.004** (baseline τ=0.75) and **0.017** (low τ=0.034).
Architecture is irrelevant to the normalized shape. But baseline vs low-τ have
visibly DIFFERENT shapes (low-τ sits higher through mid-training, descends steeper
at the end) — **τ deforms the TLC**, exactly the paper's claim.

**Result 2 — the scale ladder does NOT collapse (`tlc_scale_ladder.png`).**
The AU-Net γ=10 ladder (100M/300M/760M, iso-TPP) has residual **0.055** and clearly
fans out early: 100M stays high, 760M drops fast. Reason: the ladder is iso-TPP but
NOT iso-τ (100M τ=1.50, 300M τ=0.53, 760M τ=0.10 — see table above). Since τ sets the
curve shape, a τ-varying ladder can't collapse.

**Result 3 — 1.3B overlaid on 100M at matched τ (`tlc_100M_vs_1p3B.png`).**
The 1.3B (aunet2 + bpebyte_br_bt) run at τ=0.034 — the SAME τ as the 100M low-τ
runs — so if τ alone drove collapse they should overlay. They do NOT: the
across-scale matched-τ residual is **0.084**, ~as large as the 100M τ-gap itself
(0.75 vs 0.034 = 0.097). The 1.3B curves drop much faster and flatten by t̂≈0.1,
while the 100M curves descend gradually. Cause: **TPP mismatch** — the 1.3B is at
TPP≈218 (heavily overtrained) vs the 100M collapse_tau at TPP≈17.5. Per the paper's
Eq. 9 the normalized TLC depends on *both* τ *and* the overtraining factor v (=TPP
ratio); large v drops fast then flattens, exactly the 1.3B shape. So matched τ with
mismatched TPP does not collapse.

**Result 4 — full recipe matched → across-scale collapse CONFIRMED (`tlc_six_curve.png`).**
Trained 100M AU-Net(word) + BPEByte(root_greedy) with the 1.3B's EXACT recipe:
η=1.65e-3, λ=0.1, cosine over 180k steps, warmup 10k, decay to 1% — so τ=0.0337 and
the LR schedule are identical to the 1.3B, and TPP=209 (batch shrunk to 14 seq to hold
it). Six curves, all τ=0.034:

| group | TPP | residual vs 1.3B |
|---|---|---|
| 100M new (matched TPP) | 209 | **0.006** |
| 100M old (collapse_tau) | 18 | 0.073 |

The matched-TPP 100M curves land **on top of** the 1.3B (residual 0.006 — a 14×
improvement over the TPP-mismatched 0.084 from Result 3), across a 13× scale gap and
both architectures. Fixing the last mismatch (TPP) closes the collapse. Final losses:
100M new 0.756 (both families) vs 1.3B 0.595/0.599 — absolute loss differs (smaller
model) but the *normalized* curve is identical.

**Overall conclusion:** normalized TLC = f(τ, TPP, schedule), independent of model
size and architecture. Collapse holds when and only when all three are matched:
matched τ alone → architecture-invariance but not across-scale (Result 3); matched
τ+TPP+schedule → full across-scale collapse (Result 4). The production γ=10 ladder
fails (Result 2) only because τ drifts across it.

On our models, matched-τ curves
collapse to residual ~0.004 regardless of architecture (AU-Net vs BPEByte), while the
production scale ladder fails to collapse purely because τ drifts 15× across it. To
get predictable/collapsing TLCs across scales, hold τ constant (Δλ per scale, per the
"Implication for the matched comparison" above), not just TPP.

## Reproduction

- Configs: `runs/*/config.yaml` (`optim.lr`, `optim.weight_decay`, `steps`,
  `data.batch_size`, `data.seq_len`, `grad_acc_steps`, `distributed.dp_replicate`).
- Token check: `optim/total_tokens` in `runs/*/metrics.jsonl`.
- τ_iter = 1/(lr·weight_decay); τ_initial = τ_iter/steps; τ_final = τ_initial/lr_min_ratio.
