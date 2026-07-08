# 100M ratio-5 comparison (2.3B tokens) — Llama vs original AU-Net

Full dataset (hellaswag 10042, arc_easy 2376). acc_norm (%). Final ckpts.
Llama: cmp_100M/llama_100M ckpt 2200 (apps.main eval).
AU-Net: cmp_100M/aunet_orig_100M ckpt 1672, word1 whitespace boundaries (apps.aunet eval).

| Model                          | HellaSwag | ARC-easy |
|--------------------------------|-----------|----------|
| Llama (subword)                | 28.8      | 40.6     |
| original AU-Net (word1)        | 27.8      | 30.4     |
| v2_online (BPEByte, DEPRECATED)| 27.7      | 31.1     |
| random                         | 25.0      | 25.0     |

Notes:
- v2_online is a BPEByte ablation variant (bpe_br online bt), NOT AU-Net. It was wrongly
  used as the AU-Net side initially; original AU-Net (word1) is the correct baseline.
- All near chance on HellaSwag -> 100M @ ratio-5 too undertrained to discriminate.
  Only signal above noise: Llama ARC-easy 40.6 vs ~30 for byte models. Don't over-read;
  ratio-20 sweep is the discriminating run.
  (gen_ll generation-framed columns dropped — cloze scoring only; they were at chance/dead.)

Artifacts:
  runs/cmp_100M/llama_100M/eval/results.json
  runs/cmp_100M/aunet_orig_100M/evals_full/results.json   (full; the limit-150 evals_full was overwritten)
  configs: apps/main/configs/eval_llama_100M.yaml, apps/aunet/configs/aunet_orig_100M.yaml

## ratio-5 downstream — all 5 boundary schemes (entropy budget study, 2026-06-27)

100M AU-Net, ratio-5 (1672 steps, 10.5B bytes), final ckpt 0000001672. acc_norm (%) for HellaSwag/ARC-easy.
Random ≈ 25 (HS 4-way, ARC-e ~4-way). (gen_ll generation-framed variants dropped — cloze scoring only.)
train-BPB recomputed 2026-07-08 with the consistent last-40-step-window method (all runs on snu55/B200).

| scheme            | HellaSwag | ARC-easy | eval n         | train-BPB |
|-------------------|-----------|----------|----------------|-----------|
| aunet (word)      | 27.79     | 30.39    | full 10042/2376| 1.327     |
| root_greedy (BPE) | 27.49     | 29.97    | full           | 1.321     |
| entropy-LOW (5×)  | 34.5±1.5  | 31.2±1.5 | 1000           | 1.389     |
| entropy-MID (10×) | 35.0±1.5  | 33.5±1.5 | 1000           | 1.387     |
| entropy-HIGH (20×)| 34.2±1.5  | 30.3±1.5 | 1000           | 1.386     |

**IMPORTANT subset caveat.** word/root_greedy are FULL-dataset; entropy LOW/MID/HIGH are **limit-1000**
(first 1000 docs) — entropy eval is CPU-bound (per-request boundary computation, GPU idle) so full timed out
on A5000. The 1000-doc subset is NOT the same set as the full eval, so the entropy-vs-word gap (entropy HS
~34-35 vs word 27.8) is **largely a subset-difficulty artifact, NOT evidence entropy beats word**. Do not
compare entropy rows to word/root rows directly. The only clean comparison is LOW vs MID vs HIGH (identical
1000-subset).

**Findings.**
- Within-study (same subset): LOW/MID/HIGH HellaSwag 34.5 / 35.0 / 34.2 and ARC-easy 31.2 / 33.5 / 30.3 are
  FLAT — all within ~1 stderr (±1.5). **Entropy-model budget (5×→10×→20×) produces NO systematic downstream
  separation** at ratio-5, mirroring the flat train-BPB (1.389 / 1.387 / 1.386).
- Consistent with the headline: at ratio-5 (1672 steps) the 100M byte models are too undertrained for
  downstream to discriminate; the budget signal lives in train-BPB, not these benchmarks. A fair word-vs-
  entropy readout needs same-subset eval and/or the ratio-20+ budget (per repo convention, ratio-20=6688
  steps is the minimum "real read").

Artifacts: entropy snu55 `runs/aunet_100M_entropy_{low,mid,high}/evals_r5_final/results.json` (limit-1000);
word/root B200 `runs/cmp_100M/aunet_orig_100M/evals_full` + `runs/ablation_100M/v4_root_greedy/evals_full` (full).

### Extra tasks — PIQA / ARC-Challenge / BoolQ (all 5 schemes, ALL limit-1000, SAME subset)

Unlike the HS/ARC-easy block above (word/root full vs entropy limit-1000), here ALL FIVE schemes were run at
limit-1000 on the identical first-1000 subset, so these columns are directly comparable across schemes.
PIQA/ARC-C = acc_norm; BoolQ = acc. Random: PIQA ≈ 50, ARC-C ≈ 25, BoolQ ≈ 62 (majority-class). n=1000 (±~1.5%).

| scheme            | PIQA (an) | ARC-C (an) | BoolQ (acc) | train-BPB |
|-------------------|-----------|------------|-------------|-----------|
| aunet (word)      | 54.3      | 22.4       | 43.6        | 1.327     |
| root_greedy (BPE) | 53.9      | 20.4       | 38.1        | 1.321     |
| entropy-LOW (5×)  | 52.5      | 21.0       | 62.2        | 1.389     |
| entropy-MID (10×) | 53.7      | 21.4       | 41.5        | 1.387     |
| entropy-HIGH (20×)| 52.7      | 21.5       | 60.0        | 1.386     |

**All at chance.** PIQA 52.5-54.3 ≈ random 50; ARC-C 20.4-22.4 ≈ (slightly below) random 25; BoolQ swings
38-62 with no pattern = majority-class noise (62 = always-true). Entropy LOW/MID/HIGH are flat within ±1.5%
on every task (PIQA 52.5/53.7/52.7, ARC-C 21.0/21.4/21.5, BoolQ 62.2/41.5/60.0 — BoolQ scatter is noise,
not a trend). **No entropy-budget downstream separation on any of the 5 benchmarks**, consistent with the
flat train-BPB (1.389/1.387/1.386) and the ratio-5 "too short to rank" verdict.

Artifacts: entropy snu55 `runs/aunet_100M_entropy_{low,mid,high}/evals_r5_extra/results.json`;
word/root B200 `runs/cmp_100M/aunet_orig_100M/evals_r5_extra` + `runs/ablation_100M/v4_root_greedy/evals_r5_extra`.

## Methods note — entropy θ calibration validated on held-out data

The BLT entropy-patch threshold θ (per entropy model, monotonic rule, target 4.5 bytes/patch) was
calibrated on the *training* shards. A held-out check (2026-06-29) confirms this is not a leakage
confound: recalibrating each model's θ on the disjoint `dclm…val.jsonl` shard (excluded from
entropy-model training — the loader globs only `*.chunk.*.jsonl`) gives essentially the same θ, and
the train-calibrated θ holds the 4.5 B/patch target on held-out text.

| entropy model | θ_train | θ_val | θ gap | bytes/patch on val @ θ_train | mean-H gap (val−train) |
|---------------|---------|-------|-------|------------------------------|------------------------|
| LOW (5×)      | 0.6709  | 0.6723| 0.0015| 4.495                        | −0.0042                |
| MID (10×)     | 0.6408  | 0.6428| 0.0020| 4.494                        | −0.0045                |
| HIGH (20×)    | 0.6142  | 0.6173| 0.0031| 4.490                        | −0.0047                |

(monotonic rule, nats, bf16 forward matching the cache build, n=2000 docs each; train=chunk.00.)
θ moves only 0.2–0.5 %; the mean per-byte entropy gap is tiny and *negative* (val slightly easier →
no memorization); train-θ on held-out val gives 4.49 B/patch (0.1–0.2 % off target). The θ-gap grows
weakly with budget (the "more training → more fit" signature) but at ~0.1 % in B/patch is far too
small to perturb the flat downstream results above — no cross-model patch-rate confound. Caveat: the
train↔val θ difference is *smaller* than the quantile's doc-sampling noise, so which docs you sample
matters more than train-vs-val. Full writeup: `runs/cmp_100M/calibration_train_vs_val.md`.

## ratio-10 budget downstream (entropy LOW/MID/HIGH, limit-1000)

Same 100M AU-Nets, entropy-model budget LOW(5×)/MID(10×)/HIGH(20×), but trained to **ratio-10 (3344
steps, 2× the ratio-5 budget)**. Downstream limit-1000 (same subset/method as the ratio-5 table).
acc_norm % for HS/ARC-e/PIQA/ARC-C; acc % for BoolQ. (gen_ll generation-framed variants dropped — cloze
scoring only.) train-BPB kept on the original mean-last-40 method here: the r10 LOW/HIGH metrics live only
on snu30 (off-limits) and can't be recomputed, so all three rows stay same-method for a valid 3-way
comparison. For cross-scale reference, MID recomputes to **1.248** under the consistent last-40-step window
(~+0.010); LOW/HIGH would shift alike, so the flat ordering is unchanged.

| entropy model | HS | ARC-e | PIQA | ARC-C | BoolQ | train-BPB |
|---------------|------|-------|------|-------|-------|-----------|
| LOW  (5×)     | 34.9 | 33.8  | 57.5 | 20.7  | 44.7  | 1.239 |
| MID  (10×)    | 35.8 | 33.8  | 56.5 | 22.1  | 56.8  | 1.238 |
| HIGH (20×)    | 34.4 | 35.3  | 56.5 | 21.5  | 60.1  | 1.243 |

ratio-5 baseline for reference (limit-1000): HS 34.5/–/34.2, ARC-e 31.2/–/30.3, PIQA 52.5/–/52.7,
ARC-C 21.0/–/21.5, BoolQ 62.2/–/60.0; r5 train-BPB LOW 1.389 / HIGH 1.386.

**Conclusion — entropy-model budget STILL does not transfer downstream at ratio-10 (all 3 rows).** LOW,
MID and HIGH are flat within noise on every non-degenerate task: HS 34.9/35.8/34.4, ARC-e 33.8/33.8/35.3,
PIQA 57.5/56.5/56.5, ARC-C 20.7/22.1/21.5; BoolQ (44.7/56.8/60.1) is majority-class noise. Crucially the **train-BPB is monotone-flat: LOW 1.239 ≈ MID 1.238
≈ HIGH 1.243 (total spread 0.005), and it does NOT decrease with entropy-model budget** — a 4× better
entropy LM (LOW→HIGH) buys nothing on the AU-Net, even with 2× the training. Doubling the AU-Net budget
(ratio-5→10) lifted absolute scores modestly (ARC-e +3–5, PIQA +4–5, BPB −0.15) as expected from more
training, but LOW/MID/HIGH stay indistinguishable. Artifacts: HIGH/LOW `runs/aunet_100M_entropy_{high,low}_r10/checkpoints/0000003344/evals_r10_{main,extra}`;
MID snu55 `runs/aunet_100M_entropy_mid_r10/checkpoints/0000003344/evals_{main,extra}`.

## Parsing-rule ablation: monotonic-val vs global-val (HIGH, ratio-5 + ratio-10)

Fixed entropy model (HIGH, 20×) and calibration source (DCLM **val**, target 4.5 B/patch); vary only the
BLT boundary RULE. **monotonic**: open a patch where dH_t = H_t − H_{t-1} > θ (θ=0.6173). **global**: where
H_t > θ (θ=1.4797). Both calibrated to the same 4.5 B/patch, so equal patch *count* but different
*placement*. 100M AU-Net, downstream limit-1000, **cloze scoring only** (gen_ll generation-framed variants
dropped): acc_norm % for HS/ARC-e/PIQA/ARC-C; acc % for BoolQ. train-BPB = mean `loss`/ln2 over the
last-40-step window at each budget's checkpoint (recomputed
2026-07-07 for BOTH r5+r10 with the SAME method as the 300M study, for cross-scale consistency; this
shifts the earlier r10-only figures up ~+0.008 — mono 1.242→1.251, global 1.230→1.237 — but preserves the
ordering. See cross-scale note in `runs/cmp_300M/parsing_rule_comparison_300M.md`).

| variant (HIGH, val-cal)         | budget   | HS   | ARC-e | PIQA | ARC-C | BoolQ | BPB       |
|---------------------------------|----------|------|-------|------|-------|-------|-----------|
| monotonic (dH>θ, θ0.6173)       | ratio-5  | 35.0 | 32.2  | 54.0 | 20.2  | 38.3  | 1.294     |
| monotonic (dH>θ)                | ratio-10 | 35.6 | 33.6  | 55.9 | 21.3  | 41.4  | **1.251** |
| global (H>θ, θ1.4797)           | ratio-5  | 34.1 | 31.9  | 52.0 | 21.6  | 59.0  | 1.321     |
| global (H>θ)                    | ratio-10 | 34.6 | 33.0  | 55.0 | 21.6  | 55.1  | **1.237** |
| monotonic-train (θ0.6083, x-ref)| ratio-10 | 34.4 | 35.3  | 56.5 | 21.5  | 60.1  | 1.243†    |

† x-ref r10 BPB carried from the original write-up; that run's `metrics.jsonl` on snu55 only reaches step
1670 (r5 = 1.386 by the consistent method), so its r10 could not be re-derived here — left as originally reported.

**Conclusion — the parsing RULE barely moves downstream, but global has a small consistent BPB edge.**
Downstream, monotonic vs global are flat/mixed within noise at both budgets: at ratio-10 HS 35.6 vs 34.6,
ARC-e 33.6 vs 33.0, PIQA 55.9 vs 55.0, ARC-C 21.3 vs 21.6; BoolQ swings
(41.4 vs 55.1) on majority-class noise — no consistent downstream winner (as expected: 100M is too
undertrained to rank downstream, same story as the budget study). The one real signal is **train-BPB:
global 1.237 < monotonic 1.251 at ratio-10 (−0.014, ~1%)** — i.e. opening patches on the raw-entropy level
(H>θ) places boundaries marginally better for the AU-Net's LM loss than the entropy-jump rule (dH>θ), at
equal 4.5 B/patch. **But this global edge is budget-specific: at ratio-5 monotonic is better (1.294 <
1.321), and global only overtakes at ratio-10.** The monotonic-train cross-reference (θ0.6083) r10 1.243 ≈
monotonic-val 1.251, reconfirming train-vs-val calibration is immaterial (< the rule effect). Net ordering
(r10 BPB): global-val 1.237 < monotonic-val 1.251. Caveat: BoolQ degeneracy and the
100M/undertrained regime mean only BPB is a reliable ranker here; the global-rule BPB edge is worth
re-testing at larger scale/budget — **which the 300M study did, and it REVERSES (monotonic wins 1.131 <
1.144); see `runs/cmp_300M/parsing_rule_comparison_300M.md`.** Artifacts: snu55 `runs/aunet_100M_entropy_high_{monoval,global}_r10/checkpoints/{0000001672,0000003344}/evals_{main,extra}`.

### Full cross-scale table — all downstream tasks + BPB (100M & 300M)

Cloze (standard loglikelihood continuation) scoring only — the gen_ll generation-framed variants are
dropped. HIGH entropy model, val-cal, equal 4.5 B/patch, limit-1000. HS/ARC-e/PIQA/ARC-C = acc_norm %;
BoolQ = acc %. Avg4 = mean of the 4 non-degenerate acc_norm tasks (HS, ARC-e, PIQA, ARC-C); BoolQ is
listed but EXCLUDED from Avg4 (majority-class degenerate, ~62 baseline). BPB = mean `loss`/ln2 over the
last-40-step window (consistent method, recomputed 2026-07-07). Best BPB per (scale, budget) in bold.
**budget = STEP count (r5=1672, r10=3344), STEP-matched across scales = iso-DATA.** ratio ≡ data/model, so
for the 300M (3.01× params: 296.66M vs 98.59M) these are true **ratio-1.7 / ratio-3.3**, NOT ratio-5/10 —
the 300M rows are data-matched to 100M but under-trained for their size (a true 300M ratio-10 = 10,065
steps, in training). So the cross-scale BPB drop is measured at equal data, not equal ratio.

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

**Cloze downstream confirms no reliable rule winner, and does NOT track BPB.** By Avg4 monotonic edges
global at 100M (35.4 vs 34.9 r5; 36.6 vs 36.1 r10) but global edges monotonic at 300M/r10 (38.5 vs 37.9) —
the OPPOSITE of the BPB ranking (global better at 100M/r10, mono better at 300M/r10). BoolQ swings on
majority-class noise (collapses to 39–43 in three cells) and is excluded from Avg4. Net: cloze downstream
is flat/noise-dominated at both scales; the ~1% BPB gap is the only consistent signal, and it flips with scale.
