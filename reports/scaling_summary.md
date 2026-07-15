# Downstream accuracy vs scale — per-benchmark scaling figures

Companion to the figures `scaling_<benchmark>.png` (one per benchmark) and the grid
`scaling_all_benchmarks.png`. Four-way comparison on the AU-Net/BPEByte backbone:
**Llama** (subword baseline), **AU-Net (word patches)**, **BPEByte-rg** (online root_greedy
byte patches), and **Hybrid** (offline-leaf prefill + B3 boundary + online root_greedy decode).

- **x** = model scale (non-embedding params, log axis): 100M → 300M → 760M → 1.3B, with
  **total training FLOPs** shown on a second tick line beneath each scale:

  | scale | 100M | 300M | 760M | 1.3B |
  |---|---|---|---|---|
  | total training compute C | 3.4×10¹⁸ | 2.9×10¹⁹ | 1.6×10²⁰ | **5.5×10²⁰** |

  **C = (6·N_non-embed + 6·d·L·S)·D_tokens** — the linear + attention compute on the trunk
  **token** count (D_tokens = bytes ÷ 4.56), per `information_per_scale.md`. 1.3B = 5.5×10²⁰,
  matching the paper anchor (~5e20). byte and Llama are ~**compute-matched** here (token-budget
  matched), so the shared axis uses the byte-family value. *(This replaces the earlier
  FLOP/byte × bytes measure from `total_flops.md`, which was ~2× larger — it counts the byte
  encoder/decoder over all raw bytes, not the standard scaling-law C.)*
- **y** = accuracy (%), **0-shot**. Metric = **`acc_norm`** for HellaSwag / ARC-Easy /
  ARC-Challenge / PIQA, **`acc`** for BoolQ / WinoGrande (they have no `acc_norm`).
- Data + per-cell provenance: `scaling_data.csv`. Regenerate with
  `lingua/portable_aunetlaw/{collect_scaling.py, plot_scaling.py}`.

> ⚠️ **Read this first — Hybrid shows only the budget-matched leaf/B3 points (100M + 1.3B).**
> The p1b1 300M and the incomplete p1b1 760M were dropped (not budget-matched). What remains:
>
> | scale | hybrid run | variant | total bytes | vs rg | state |
> |---|---|---|---|---|---|
> | 100M | `lb_hybrid_100M` | leaf/B3 | **21.0 GB** | = rg 21.0 GB, **same model [512,768]** | full |
> | 1.3B | `hybrid_1p3B_leaf_B3` | leaf/B3 | 283 GB target | = rg 283 GB | training (~92%); eval @108k so far |
>
> So Hybrid-100M **is** comparable to rg-100M — same model, same 21 GB of data — differing only
> in global batch (48 vs 768) and the hybrid tokenization. On that footing Hybrid-100M beats
> rg-100M clearly (HS 39.3 vs 28.6). The 1.3B point is still a milestone (dashed) until the run
> finishes. p1b1 = the older `offline_root` prefill + `uniform_full` boundary; leaf/B3 =
> `offline_leaf` prefill + `uniform_mid` boundary (the ablation winner).

---

## Per-benchmark tables (0-shot; blank = never evaluated at that scale)

### HellaSwag — `acc_norm` (↑)
| Scale | Llama | AU-Net (word) | BPEByte-rg | Hybrid |
|---|---|---|---|---|
| 100M | 33.8 | 30.7 | 30.0 | — |
| 300M | 42.4 | 37.2 | 37.0 | — |
| 760M | 50.0 | 46.7 | 45.8 | 44.4 `@8.8k`¹ |
| 1.3B | 62.2 | **62.6** | 62.5 | 50.3 `@108k`² |

### ARC-Easy — `acc_norm` (↑)
| Scale | Llama | AU-Net (word) | BPEByte-rg | Hybrid |
|---|---|---|---|---|
| 100M | 44.0 | 36.2 | 34.8 | — |
| 300M | 48.8 | 43.9 | 41.8 | — |
| 760M | 55.7 | 53.4 | 51.1 | 38.0 `@8.8k`¹ |
| 1.3B | 65.5 | 65.7 | **66.8** | 56.1 `@108k`² |

### ARC-Challenge — `acc_norm` (↑)
| Scale | Llama | AU-Net (word) | BPEByte-rg | Hybrid |
|---|---|---|---|---|
| 100M | 25.1 | 24.1 | 24.1 | — |
| 300M | 26.0 | 25.6 | 24.8 | — |
| 760M | 29.8 | 29.4 | 29.3 | — |
| 1.3B | 35.3 | 36.5 | **37.5** | — |

### PIQA — `acc_norm` (↑)
| Scale | Llama | AU-Net (word) | BPEByte-rg | Hybrid |
|---|---|---|---|---|
| 100M | 64.5 | 60.9 | 59.0 | — |
| 300M | 68.3 | 64.9 | 64.3 | — |
| 760M | 70.9 | 69.2 | 69.1 | — |
| 1.3B | **75.3** | 74.2 | 74.3 | — |

### BoolQ — `acc` (↑)
| Scale | Llama | AU-Net (word) | BPEByte-rg | Hybrid |
|---|---|---|---|---|
| 100M | — | 43.6 | 38.1 | — |
| 300M | — | — | — | — |
| 760M | **58.4** | 49.9 | 45.5 | — |
| 1.3B | **63.5** | 61.1 | 62.0 | — |

### WinoGrande — `acc` (↑)
| Scale | Llama | AU-Net (word) | BPEByte-rg | Hybrid |
|---|---|---|---|---|
| 100M | — | — | — | — |
| 300M | — | — | — | — |
| 760M | 54.0 | **55.6** | 54.6 | — |
| 1.3B | **61.6** | 61.5 | 61.1 | — |

¹ Hybrid 760M = *p1b1* variant, step **8.76k** — very early, not a fair 760M end-state.
² Hybrid 1.3B = *leaf/B3* variant, step **108k** (~60% of 180k), not final.

---

## Provenance

| Series | Source (per run) | Checkpoint |
|---|---|---|
| Llama / AU-Net / BPEByte-rg @ 760M, 1.3B | `evals_5bench/` (HS/ARC-E/BoolQ/PIQA/Wino) + `evals_mmlu_arcc/` (ARC-C) | final |
| … @ 100M | `small/cmp_100M/*/eval_scaling/` + `evals_r5_extra/` (BoolQ) | final |
| … @ 300M | `small/cmp_g10/*/eval_scaling/` | final |
| Hybrid @ 760M / 1.3B | `runs/poc/scale/hybrid_{760M_p1b1,1p3B_leaf_B3}/evals/<step>/` | **milestone** |

Excluded by `collect_scaling.py`: typo/noise robustness, few-shot, generation, question-mode
wrappers (`*Q`, `gq_`, `oq_` — the ~0.13 undercount from `CORRECTED_A100_RESULTS.md`),
old-code/old-seg reruns, and `llama_1B_dm10_not_target`. Byte 0-shot evals here are the
canonical B200 `evals_5bench` (correct online-greedy tokenization).

## Readings

- **Llama leads at small scale but the byte models close the gap by 1.3B.** On HS/ARC-E/PIQA,
  Llama is clearly ahead at 100M–760M; by 1.3B AU-Net and BPEByte-rg **match or pass** it
  (ARC-E: rg 66.8 > Llama 65.5; HS: all three ≈ 62; ARC-C: rg 37.5 > Llama 35.3).
- **AU-Net (word) ≈ BPEByte-rg** at every scale on almost every benchmark — the patching
  method barely moves downstream accuracy once matched on data + budget.
- **Llama keeps an edge on BoolQ** (63.5 vs ~61–62 at 1.3B) and stays ahead on PIQA (75.3).
- **BPEByte-rg lags on BoolQ at 760M** (45.5, below chance) but recovers to 62.0 at 1.3B.

## Checkpoint / path per model variant

Every figure point traces to a run under `runs/` (this NHNHOME node) and a checkpoint step.
`ckpt` = step of the checkpoint the eval loaded (`checkpoints/<step>/consolidated`).

| Model | Scale | Run path (rel. to `runs/`) | ckpt step | Eval source subdir | Notes |
|---|---|---|---|---|---|
| Llama | 100M | `small/cmp_100M/llama_100M_ot` | 8800 | `eval_scaling` | |
| Llama | 300M | `small/cmp_g10/llama_300M` | — **eval-only** | `eval_scaling` | no `checkpoints/` → BoolQ/Wino unfillable |
| Llama | 760M | `760M/llama_760M` | 12900 | `evals_5bench` + `evals_mmlu_arcc` | |
| Llama | 1.3B | `1.3B/llama_1.8B_paper` | 60000 | `evals_5bench` + `evals_mmlu_arcc` | 1.8B-param paper model (iso-data 1.3B bucket) |
| AU-Net (word) | 100M | `small/cmp_100M/aunet_100M_ot` (BoolQ: `…/aunet_orig_100M`) | 6688 | `eval_scaling` (+ `evals_r5_extra`) | |
| AU-Net (word) | 300M | `small/cmp_g10/aunet_300M` | 9900 | `eval_scaling` | has checkpoint → fillable |
| AU-Net (word) | 760M | `760M/aunet2_760M` | 29200 | `evals_5bench` + `evals_mmlu_arcc` | |
| AU-Net (word) | 1.3B | `1.3B/aunet2_1.3B` | 180000 | `evals_5bench` + `evals_mmlu_arcc` | |
| BPEByte-rg | 100M | `small/cmp_100M/v4_root_greedy_ot` (BoolQ: `ablation_100M/v4_root_greedy`) | 6688 | `eval_scaling` (+ `evals_r5_extra`) | |
| BPEByte-rg | 300M | `small/cmp_g10/rg_300M` | — **eval-only** | `eval_scaling` | no `checkpoints/` → BoolQ/Wino unfillable |
| BPEByte-rg | 760M | `760M/bpebyte_root_greedy_760M` | 29200 | `evals_5bench` + `evals_mmlu_arcc` | |
| BPEByte-rg | 1.3B | `1.3B/bpebyte_br_greedy_root_1.3B` | 180000 | `evals_5bench` + `evals_mmlu_arcc` | |
| Hybrid | 760M | `poc/scale/hybrid_760M_p1b1` | 8760 (eval) | `evals/0000008760` | **p1b1** variant, ~15% trained |
| Hybrid | 1.3B | `poc/scale/hybrid_1p3B_leaf_B3` | 108000 (eval); **160000** latest | `evals/0000108000` | **leaf/B3**; run now at 160k |

Byte-online runs (rg, hybrid) must be evaluated with `force_bpe_online_mode=greedy` (+ the
llama3 tokenizer) or they are undercounted — see `CORRECTED_A100_RESULTS.md`.

## BPB vs scale (`scaling_bpb.png`, ↓ better)

Bits-per-byte on held-out DCLM, byte-normalised (Llama's per-token loss ÷ ln2 ÷ 4.5483 so
it is comparable per byte). Computed from each run's `metrics.jsonl` (tail-mean of `loss/out`
over the last 200 logged steps) — the **same runs** as the accuracy tables. Data: `scaling_bpb.csv`.

| scale | Llama | AU-Net (word) | BPEByte-rg | Hybrid |
|---|---|---|---|---|
| 100M | **1.040** | 1.114 | 1.125 | — |
| 300M | **0.968** | 1.016 | 1.014 | — |
| 760M | **0.919** | 0.944 | 0.940 | 1.009 `@9k`¹ |
| 1.3B | **0.840** | 0.866 | 0.858 | 0.851 `@153k`² |

- **Llama has the lowest BPB at every scale** — subword predicts fewer, higher-information units,
  so per-byte cross-entropy is lower even though (see accuracy tables) it does *not* dominate
  downstream by 1.3B.
- **AU-Net (word) ≈ BPEByte-rg** on BPB too; rg edges ahead at 760M/1.3B.
- **Hybrid 1.3B BPB 0.851 (@153k, ~85%) already undercuts rg's 0.858** — a positive early signal,
  consistent with the CORRECTED_A100 downstream read. (760M hybrid is the early p1b1 milestone.)
- ⚠️ At 100M/300M the byte and Llama runs use *different* raw byte budgets (Llama cmp runs vs the
  byte cmp ladder), so cross-model BPB at those two scales is **indicative, not iso-budget**.

## Filling the gaps — `fill_missing_evals.py`

`lingua/portable_aunetlaw/fill_missing_evals.py` computes the gap grid (4 models × 4 scales × 6
benchmarks − what's already in `scaling_data.csv`), maps each gap to its run + latest checkpoint,
routes to the right eval app (byte-online adds `force_bpe_online_mode=greedy`), and prints the
commands. Default is a **dry-run feasibility report**; `--run` launches them. Re-run
`collect_scaling.py` / `collect_bpb.py` + the plotters afterward.

Current status: **7 fillable** (checkpoint present), **3 blocked** (no checkpoint):

| status | cells |
|---|---|
| fillable | aunet 100M (Wino); aunet 300M (BoolQ,Wino); rg 100M (Wino); llama 100M (BoolQ,Wino); hybrid 760M & 1.3B (ARC-C,PIQA,BoolQ,Wino); hybrid 300M (all 6) |
| blocked | **rg 300M** (BoolQ,Wino) & **llama 300M** (BoolQ,Wino) — `cmp_g10` dirs are eval-only, no checkpoints; **hybrid 100M** — run doesn't exist |

> Note on the blocked cells: `cmp_g10/rg_300M` and `cmp_g10/llama_300M` hold only eval outputs
> (no `checkpoints/`), so BoolQ/WinoGrande at 300M can't be filled without re-locating or
> re-training those checkpoints. Hybrid needs its full-scale runs (the `bpebyte_rg_hybrid_leaf`
> ladder) to replace the milestone/variant-mixed points.

### On `c_online_300M` (a second rg-300M run — deliberately NOT used)
`runs/poc/scale/c_online_300M/` is also a valid BPEByte root_greedy 300M run, but it is a
**different, smaller budget** (42 GB / 142 bytes-per-param, LR 1.9e-3 / GB 768) than the 300M
comparison set (`cmp_g10`: 62 GB / 210 bpp). Using it would make the 300M column **not
budget-matched** across models, so the figures use the budget-matched `cmp_g10/rg_300M` instead.
(`c_online_300M` also has no downstream evals and is not the portable *aunet-law* recipe.)

## Known gaps (not zeros — never evaluated)

- **BoolQ / WinoGrande have no 100M/300M points** for most models (rg/llama 300M blocked — see above).
- **Hybrid**: HS + ARC-E only, milestone + mixed-variant (see the warning). Re-run the two
  scripts once its full-scale evals land to complete the picture.
