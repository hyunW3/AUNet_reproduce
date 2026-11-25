# Robustness by perturbation region (2026-11-25)

Each robustness axis is split by the region that gets perturbed:

- **Context**: only the context is perturbed (answer options clean).
- **Options**: only the answer options are perturbed.
- **Both**: context and options (the paper setting, `tab:main_13b`).

Each cell is the mean, over HellaSwag / ARC-E / ARC-C / PIQA, of the signed Δ = perturbed − clean (pp). Metric is
acc_norm (PIQA acc), limit 2000, perturbation seed 1234. The largest drop per model × axis is in **bold**.

| Model | Noise Ctx | Noise Opt | Noise Both | Typo Ctx | Typo Opt | Typo Both | Despace Ctx | Despace Opt | Despace Both | Leet Ctx | Leet Opt | Leet Both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama (BPE) | -10.0 | **-20.8** | -20.7 | -6.1 | -17.6 | **-18.8** | -7.5 | -13.6 | **-17.4** | -15.3 | -24.3 | **-24.8** |
| AU-Net | -7.8 | **-17.3** | -16.2 | -4.6 | -16.4 | **-16.8** | -16.0 | -18.5 | **-23.3** | -16.3 | -24.0 | **-24.6** |
| BPEByte | -7.7 | **-16.9** | -16.2 | -4.2 | -16.2 | **-16.6** | -7.9 | -13.8 | **-16.7** | -14.8 | -23.2 | **-24.3** |
| BLT θ=1.335 | -8.9 | -16.0 | **-17.4** | -4.6 | -16.3 | **-16.8** | -6.1 | -8.3 | **-8.5** | -14.3 | -22.8 | **-24.1** |
| BLT θ=1.609 | -8.3 | -15.5 | **-16.9** | -3.9 | -15.5 | **-15.9** | -4.8 | **-8.5** | -8.1 | -12.7 | -21.5 | **-23.3** |
| H-Net | -7.9 | **-17.6** | -16.6 | -4.8 | -16.3 | **-17.0** | -8.1 | -13.4 | **-16.4** | -14.6 | -22.9 | **-24.1** |

Files in this folder:

- `robust_region.md`: the table above plus per-task Δ.
- `robust_region.json`: clean and perturbed accuracy per task, plus mean |Δ| (the paper's main-table aggregate).
- `robust_region.tex`: LaTeX table (`tab:robust_region`, caption on top).

Regenerate with `python scripts/probes/robust_region_table.py` (branch `worktree-region-split-typoleet`).

## What is new and what is reused

| Axis | Context | Options | Both |
|---|---|---|---|
| Noise (5 strategies) | paper runs (`_prompt`) | paper runs (`_completion`) | paper runs (`_both`) |
| Despace (100 %) | paper runs (`despace`) | paper runs (`despaceans`) | paper runs (`despaceall`) |
| Typo (4 ops × char/word) | **new** `<task>_typoctx` | **new** `<task>_typoopt` | paper `typoboth` runs |
| Leet (NL-Augmenter) | **new** run of `nla_leet` | **new** `nla_leet_opt` | paper `nla_leet_both` runs |

How the new runs relate to the paper runs:

- **Typo seeds.** Typo context and options reuse the typoboth seeds (`lingua/apps/aunet/eval_typo_ds.py`, region
  argument). The context of typoctx and the options of typoopt are byte-identical to the matching half of typoboth.
  This was checked on 200 docs × 8 strategies × 4 tasks.
- **Leet variants.** `nla_leet_opt` is the option half of `nla_leet_both`, and `nla_leet` is its context half.
- **Configs.** Each model uses the config behind its paper number. For BPEByte that is
  `runs/robustness_paper4task_bpebyte/cfg` (official generator, no vocab_norm).
- **Clean baselines.** The clean scores inside each new run match the paper both-region runs exactly. The one
  exception is H-Net, at ±0.2 pp from batch-8 noise. Leet runs start from a copy of the paper cache, so their clean
  scores are the paper's.

Where things live:

- Raw runs are in `runs/robustness_typoleet_region/`, made by `scripts/probes/run_typoleet_region.sh`.
- 13 BLT jobs ran on info107 through `run_typoleet_region_remote.sh`. In a smoke check, the info107 results were
  bit-identical to local ones.

AU-Net Typo-options was run differently:

- The 4-task lingua eval hit a CUDA illegal memory access in flex_attention `create_block_mask` autotune. This
  happened 3 times, including once with a fresh inductor cache.
- It was therefore run as one eval per task (`trio_typoopt_aunet_split/`) and the results were merged.
- Its clean baselines differ from the 4-task runs by at most 0.3 pp (ARC-C 36.2 vs 35.9, PIQA 73.1 vs 73.0). The
  likely cause is batch packing.

## Reading

**Options dominate.** For every model, the options-only drop is 92–107 % of the both-region drop on all three of
Noise, Typo (94–98 %) and Leet (92–98 %). The context-only drop is much smaller:

| Axis | Context-only drop as % of Both |
|---|---:|
| Typo | 25–32 % |
| Noise | 47–51 % |
| Leet | 55–66 % |

Scoring compares the log-likelihood of the corrupted options, so corrupting the options is what hurts.

**Typo.**

- Context-only drop: BLT θ1.609 −3.9, BPEByte −4.2, AU-Net and BLT θ1.335 −4.6, H-Net −4.8, Llama −6.1.
- Llama has the largest drop in all three regions.
- The byte models are within about 1 pp of each other.

**Leet.**

- Context-only drop, in increasing order: BLT θ1.609 −12.7, BLT θ1.335 −14.3, H-Net −14.6, BPEByte −14.8,
  Llama −15.3, AU-Net −16.3.
- In the both region, all models sit between −23.3 and −24.8.

**Despace.** This is the only axis where the perturbed region changes the model ranking.

- BLT is barely affected by option despacing (−8.3 and −8.5, against −13.4 to −18.5 for the other models).
- AU-Net is the outlier on context despacing (−16.0).
