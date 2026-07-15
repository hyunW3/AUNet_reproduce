# State tracking & recall — in-context probes of frozen 1.3B checkpoints

Twelve synthetic probes run **in-context (no training)** on the frozen 1.3B
checkpoints, teacher-forced greedy exact-match. Three model families, all
token-budget matched (iso-byte):

| tag | checkpoint | family / pooling rule |
|---|---|---|
| **Llama** | `llama_1.8B_paper` | subword — flat BPE (128k) |
| **AUNet** | `aunet2_1.3B` | byte → **static word** pooling |
| **BPEByte rg** | `bpebyte_br_greedy_root_1.3B` | byte → **greedy-root BPE** boundaries |

`byte (bt)` (`bpebyte_br_bt_1.3B`, before-root BPE boundaries) is also run for the
pooling-rule comparisons but omitted from the headline tables.

The one knob that moves everything is the **pooling rule**: BPEByte rg
(content-adaptive boundaries) leads on recall/copy and easy–moderate state
tracking; AUNet's static word-pooling is the consistent weak spot; and **every**
family walls out on S5 permutation and ≥2-hop Variable Tracking.

---

## Full results

Bold = best of the three; *(chance)* marks metrics with a random floor.

| regime | benchmark | metric | chance | Llama | AUNet | **BPEByte rg** |
|---|---|---|---|---|---|---|
| State tracking | FFLM dense | read acc (T=512) | 0.50 | 0.90 | 0.91 | **0.92** |
| | FFLM in-dist | read acc | 0.50 | **0.74** | 0.68 | 0.73 |
| | FFLM sparse-OOD | read acc | 0.50 | 0.77 | 0.70 | **0.80** |
| | S5 permutation | read acc | 0.20 | 0.23 | 0.23 | 0.25 |
| | Dyck-3 brackets | close-pred acc | 0.33 | 0.52 | 0.54 | **0.67** |
| Recall / copy | S-NIAH-1 noise+num | exact (mean len) | — | 0.93 | **1.00** | **1.00** |
| | S-NIAH-2 essay+num | exact (mean len) | — | 0.70 | 0.98 | **0.99** |
| | S-NIAH-3 essay+**UUID** | exact (mean len) | — | 0.16 | 0.52 | **1.00** |
| | MK-NIAH K=1 | exact (2 KB) | — | 0.45 | 0.98 | **1.00** |
| | MK-NIAH K=8 | exact (2 KB) | — | 0.18 | 0.30 | **0.33** |
| | Var. Tracking 1-hop | exact | — | **1.00** | 0.35 | 0.75 |
| | Var. Tracking 2-hop | exact | — | **0.20** | 0.03 | 0.18 |

## Base vs. hard — does the advantage survive escalation?

↗ improved · ↘ dropped · → flat.

| benchmark | base → hard | Llama | AUNet | BPEByte rg |
|---|---|---|---|---|
| S5 permutation | 40ev/2-shot → 60ev/4-shot | 0.23→0.23 | 0.23→0.22 | 0.25↗0.28 |
| Dyck brackets | k3/depth6 → k4/depth10 | 0.52↘0.27 | 0.54↘0.35 | 0.67↘0.52 |
| MK-NIAH K=8 | 2 KB → 4 KB | 0.18↘0.10 | 0.30→0.28 | 0.33↗0.50 |
| Var. Tracking 2-hop | 3 → 6 chains | 0.20↘0.13 | 0.03→0.03 | 0.18↘0.10 |

The byte advantage holds under deeper Dyck and more needles; S5 stays at chance
and multi-hop VT stays collapsed for all.

---

## Per-benchmark detail

### FFLM (flip-flop, read accuracy, T=512)
| model | dense | in-dist | sparse-OOD |
|---|---|---|---|
| Llama | 0.904 | 0.735 | 0.772 |
| AUNet | 0.908 | 0.679 | 0.695 |
| BPEByte rg | 0.924 | 0.728 | 0.796 |
| byte (bt) | 0.931 | 0.702 | 0.774 |

Recency-glitch (acc when nearest distractor agrees vs disagrees): AUNet +0.125
(worst), BPEByte rg +0.093, byte(bt) +0.054, Llama +0.044.

### S-NIAH exact-match by context length (bytes)
| task | model | 512 | 1024 | 2048 | 4096 | 6144 |
|---|---|---|---|---|---|---|
| S1 noise+num | Llama | 0.85 | 0.85 | 1.00 | 1.00 | 0.95 |
| | AUNet | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| | BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| S2 essay+num | Llama | 0.90 | 0.70 | 0.65 | 0.65 | 0.60 |
| | AUNet | 1.00 | 1.00 | 0.95 | 1.00 | 0.95 |
| | BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
| S3 essay+UUID | Llama | 0.15 | 0.15 | 0.05 | 0.25 | 0.20 |
| | AUNet | 0.85 | 0.60 | 0.45 | 0.30 | 0.40 |
| | BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

S-NIAH-3 verified real by an absent-needle control (BPEByte rg exact 0.00 with
needle removed vs 1.00 with needle).

### MK-NIAH exact-match by #needles K (2 KB)
| model | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| Llama | 0.45 | 0.28 | 0.33 | 0.18 |
| AUNet | 0.98 | 0.73 | 0.43 | 0.30 |
| BPEByte rg | 1.00 | 0.90 | 0.73 | 0.33 |

Hard (4 KB): K=4/8/16 → Llama 0.05/0.10/0.10 · AUNet 0.38/0.28/0.20 · BPEByte rg 0.60/0.50/0.40.

**Pressure-test grid (depth × context, to 8 KB)** — `../niah/mk_grid.png`. Within
window, BPEByte rg is the most uniform, AUNet shows the "lost-in-the-middle" dip,
Llama is weakest and collapses past 4 KB. At **8 KB body the byte models are
out-of-window** → those cells are **N/A**: byte cap = **8192 bytes** (an 8k-body
prompt is ~8.6 KB). **Llama's window is 4096 tokens**, so its 8k column *is*
in-window (~2.1 k tokens) and shows a genuine collapse (mostly red). Takeaway:
byte-level's fixed **byte** budget caps usable context earlier than Llama's token
budget. Note: the naive over-window score was a truncation artifact (empty graded
slice → vacuous pass), caught by an absent-needle control and now guarded in
`score()` (empty ⇒ miss); `mk_grid.py` marks a cell N/A if any sample exceeds
`WINDOW` (aunet 8192 B, subword 4096 tok).

### Variable Tracking exact-match by #hops (3 chains)
| model | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Llama | 1.00 | 0.20 | 0.20 | 0.13 |
| AUNet | 0.35 | 0.03 | 0.05 | 0.00 |
| BPEByte rg | 0.75 | 0.18 | 0.15 | 0.03 |

### S5 / Dyck (base & hard)
| task | Llama | AUNet | BPEByte rg | byte(bt) | chance |
|---|---|---|---|---|---|
| S5 base (40ev/2-shot) | 0.230 | 0.229 | 0.252 | 0.232 | 0.20 |
| S5 hard (60ev/4-shot) | 0.234 | 0.222 | 0.276 | 0.248 | 0.20 |
| Dyck-3 base (depth6) | 0.520 | 0.540 | 0.673 | 0.690 | 0.33 |
| Dyck-4 hard (depth10) | 0.274 | 0.354 | 0.523 | 0.558 | 0.25 |

---

## Figures
- `../fflm/fflm_read_accuracy.png` — FFLM read accuracy by regime
- `../fflm/fflm_acc_vs_distance.png` — U-shaped accuracy vs dependency distance
- `../fflm/recency_glitch.md` — recency-copy error analysis
- `../fflm/segmentation_viz.png` / `_sparse.png` — how each model chunks FFLM
- `../niah/niah_exact_vs_length_S3.png` — UUID copy vs context length
- `../niah/mk_grid.png` — MK-NIAH pressure-test grid (depth × context)
- `statetrack_overview.png` — S5 / Dyck / MK / VT battery
- `statetrack_base_vs_hard.png` — base vs hard, all four tasks

## Method
Frozen 1.3B checkpoints, in-context (no fine-tuning). Each read/answer is scored
by teacher-forced greedy: the model's argmax must reproduce the answer token(s)
verbatim (== it would greedily generate them). Tokenization-agnostic across
subword / byte families. Probe code: `scripts/fflm/`, `scripts/niah/`,
`scripts/probes/`. Modest sample counts (n=40–2000 depending on task); scale for
tighter CIs.
