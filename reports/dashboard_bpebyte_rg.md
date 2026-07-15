# BPEByte root_greedy (rg) — dashboard

The **BPEByte-rg** family: online, 0-leak, byte-for-byte-reproducible `root_greedy` segmentation
(`bpe_online_mode=greedy`, placement `root`). This is the deployable byte-tokenization baseline and the
**control** for the hybrid ablation (see [`dashboard_bpebyte_hybrid.md`](dashboard_bpebyte_hybrid.md)).

Protocol: ratio-10 (γ=10) scaling ladder, identical DCLM data. **BPB** = held-out bits/byte (↓ better);
**downstream** = 0-shot mean accuracy (↑ better), `acc_norm` for HS/ARC-E/ARC-C/PIQA, `acc` for BoolQ/Wino.
Sources: `scaling_laws/scaling_laws_plan.md`, `reports/model_results_{760M,1.3B}.md`, `reports/bpb_ci_1.3B.md`,
`reports/performance_summary_per_scale.md`.

## Scaling ladder

| Scale (N) | Held-out BPB | Downstream 4-bench | Downstream 5-bench | Checkpoint |
|---|---:|---:|---:|---|
| **100M** | 1.1958 | 35.5% | — | `runs/small/cmp_g10/v4_root_greedy` / `lb_rg_100M` |
| **300M** | 1.0142 | 42.0% | — | `runs/300M/…` / `lb_rg_300M` |
| **760M** | 0.9098¹ | 52.6% | — | `runs/760M/bpebyte_root_greedy_760M` |
| **1.3B** | **0.8578** [0.858, 0.860]² | 60.3% | 64.4% | `runs/1.3B/bpebyte_br_greedy_root_1.3B` (180k) |
| **7B** *(extrap.)* | 0.764 | **72.2%** | — | fit prediction |

¹ fit; measured new-code rg = 0.9256. ² temporal CI only (not seed CI).

## rg vs the other mature families (same ladder)

| | 100M | 300M | 760M | 1.3B |
|---|---|---|---|---|
| **BPB** rg / AU-Net / Llama | 1.196 / 1.197 / — | 1.014 / 1.016 / — | 0.910 / 0.917 / 0.923 | **0.858** / 0.866 / **0.840** |
| **4-bench** rg / AU-Net / Llama | 35.5 / 35.9 / 40.3 | 42.0 / 42.9 / 46.4 | 52.6 / 52.8 / 55.7 | 60.3 / 59.8 / 60.3 |

- **BPB ranking: Llama < rg < AU-Net** at every real scale (subword keeps a ~0.02 iso-param edge).
- **Downstream: Llama leads small, byte slope is steeper** — the cluster converges at 1.3B (rg ≈ Llama ≈ 60.3%
  on 4-bench; 5-shot already flips it: rg 67.9 > Llama 67.4), and the fit predicts **rg overtakes Llama by 7B**.
- **rg's robustness edge (1.3B):** prompt-cut ΔBPC ~0 (Llama +0.71), HellaSwag-Noise 42.3 (Llama 37.6), CUTE
  spelling 59.3 (Llama 18.4) — bytes pay BPB but win character/robustness tasks.

## 1.3B training trajectory (held-out downstream, `leaderboard_1B.md`)

| train % | HS | ARC-E | BoolQ | PIQA | Wino | 5-bench | BPB train-log | BPB 512-cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.453 | 0.471 | 0.435 | 0.669 | 0.530 | 0.512 | 0.965 | 1.065 |
| 60% | 0.488 | 0.576 | 0.527 | 0.701 | 0.575 | 0.573 | 0.935 | 1.022 |
| 100% | 0.573 | 0.674 | 0.635 | 0.739 | 0.612 | **0.647** | 0.853 | 0.933 |

**Eval note:** ece/A100 byte downstream requires `force_bpe_online_mode=greedy` (a bare tokenizer override runs
offline and tanks online byte models) — see `leaderboard_1B.md` / `leaderboard_100M.md` notes.

## Detailed reports
- Per-scale: `model_results_1.3B.md`, `model_results_760M.md`, `bpb_ci_1.3B.md`, `leaderboard_100M.md`.
- Cross-family scaling: `performance_summary_per_scale.md`, `scaling_summary.md`, `scaling_downstream_table.md`.
- Chinese cloze: `zh_cloze_{100M,300M,1B}.md`.
