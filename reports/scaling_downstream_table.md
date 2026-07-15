# Downstream accuracy vs scale — two recipes (0-shot)

Families: **Llama** (BPE), **AU-Net** (word-patch), **BPEByte rg** (online root_greedy), **BPEByte
hybrid** (offline-leaf prefill) in two eval regimes — **leafQ** (offline-leaf question) / **greedyQ**
(native online-root-greedy question). Within each scale the byte models (rg/AU-Net/hybrid) are
**configuration-identical** (same trunk dims/layers, LR, batch, steps, budget) and differ **only** in
the segmentation scheme; Llama is the subword baseline at a matched **data budget**.

Two HP recipes are tabulated separately (both trained at the same per-scale byte budget):
- **AU-Net-law** — the paper's fit: small global batch + high LR, many steps (100M: 48 seq/step ×
  53504; 1.3B: 192 × 180000).
- **ad-hoc / γ10** — large batch, few steps (100M: 768 seq/step × 3344).

## AU-Net-law recipe

100M byte budget **21.0B** (bs12×ga2×dp2, 53504 steps); 1.3B **283B** (180000 steps). 300M-law is
still filling (see notes).

| Scale | Family | HS | ARC-E | ARC-C | PIQA | BoolQ | WinoG | MMLU-txt | **Avg3** | **Avg-all** |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **100M** | Llama | 33.2 | 45.1 | 23.8 | 64.0 | 54.9 | 50.6 | — | **47.4** | **45.3** |
|  | AU-Net | 31.8 | 36.0 | 24.4 | 59.1 | 54.8 | 49.6 | — | **42.3** | **42.6** |
|  | BPEByte rg | 30.7 | 34.6 | 23.5 | 59.4 | 39.1 | 50.2 | — | **41.6** | **39.6** |
|  | BPEByte hybrid (leafQ) | 31.5 | 34.8 | 24.1 | 61.3 | 53.5 | 50.7 | — | **42.5** | **42.7** |
|  | BPEByte hybrid (greedyQ) | 31.6 | 33.3 | 22.6 | 60.9 | 58.3 | 51.6 | — | **41.9** | **43.1** |
| **300M** | Llama | 39.3 | 48.0 | 24.7 | 67.4 | 59.2 | 52.2 | — | **51.6** | **48.5** |
|  | AU-Net | ⟳ | ⟳ | ⟳ | ⟳ | ⟳ | ⟳ | — | ⟳ | ⟳ |
|  | BPEByte rg | ⟳ | ⟳ | ⟳ | ⟳ | ⟳ | ⟳ | — | ⟳ | ⟳ |
|  | BPEByte hybrid (leafQ) | 40.6 | 42.3 | 25.7 | 64.9 | 57.2 | 52.8 | — | **49.3** | **47.3** |
|  | BPEByte hybrid (greedyQ) | 39.7 | 38.5 | 25.3 | 65.6 | 54.6 | 50.4 | — | **47.9** | **45.7** |
| **760M** | BPEByte hybrid (leafQ) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | — | ⏳ | ⏳ |
|  | BPEByte hybrid (greedyQ) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | — | ⏳ | ⏳ |
| **1.3B** | Llama | 62.2 | 65.5 | 35.3 | 75.3 | 63.5 | 61.6 | 31.9 | **67.7** | **56.5** |
|  | AU-Net | 62.6 | 65.7 | 36.5 | 74.2 | 61.1 | 61.5 | 32.2 | **67.5** | **56.3** |
|  | BPEByte rg | 62.5 | 66.8 | 37.5 | 74.3 | 62.0 | 61.1 | 32.3 | **67.9** | **56.6** |
|  | BPEByte hybrid (leafQ) | 62.0 | 58.8 | 35.8 | 74.2 | 64.3 | 61.4 | 30.7 | **65.0** | **55.3** |
|  | BPEByte hybrid (greedyQ) | 62.6 | 65.0 | 37.0 | 73.8 | 63.6 | 61.5 | 32.0 | **67.1** | **56.5** |

## Ad-hoc / γ10 recipe

100M budget **21.0B** (bs96×ga2×dp4, 3344 steps); 300M **62.3B** (768 seq/step × 9900). 760M = the
`cmp_g10` ratio-matched run.

| Scale | Family | HS | ARC-E | ARC-C | PIQA | BoolQ | WinoG | **Avg3** | **Avg-all** |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **100M** | Llama | 31.3 | 42.1 | 23.7 | 64.2 | 61.7 | 50.6 | **45.9** | **45.6** |
|  | AU-Net | 28.5 | 33.7 | 23.3 | 58.3 | 56.7 | 49.7 | **40.2** | **41.7** |
|  | BPEByte rg | 28.6 | 32.6 | 22.6 | 58.1 | 55.7 | 50.0 | **39.8** | **41.3** |
|  | BPEByte hybrid (leafQ) | 28.2 | 32.7 | 25.2 | 58.7 | 46.2 | 51.7 | **39.9** | **40.5** |
|  | BPEByte hybrid (greedyQ) | 28.5 | 31.8 | 21.2 | 58.5 | 54.0 | 51.6 | **39.6** | **40.9** |
| **300M** | Llama | 42.4 | 48.8 | 26.0 | 68.3 | 55.8 | 50.2 | **53.2** | **48.6** |
|  | AU-Net | 37.2 | 43.9 | 25.6 | 64.9 | 50.7 | 50.3 | **48.7** | **45.4** |
|  | BPEByte rg | 37.0 | 41.8 | 24.8 | 64.3 | 47.3 | 54.8 | **47.7** | **45.0** |
|  | BPEByte hybrid (leafQ) | 36.0 | 38.0 | 26.1 | 64.3 | 53.3 | 48.6 | **46.1** | **44.4** |
|  | BPEByte hybrid (greedyQ) | 35.0 | 36.9 | 25.0 | 64.0 | 60.0 | 50.9 | **45.3** | **45.3** |
| **760M** | Llama | 55.7 | 62.1 | 31.2 | 73.6 | 59.8 | 58.5 | **63.8** | **56.8** |
|  | AU-Net | 52.4 | 55.7 | 31.7 | 71.2 | 58.3 | 55.3 | **59.8** | **54.1** |
|  | BPEByte rg | 52.2 | 55.4 | 31.5 | 71.4 | 53.1 | 54.9 | **59.7** | **53.1** |

## Notes
- **Metric/shots:** HS/ARC-E/ARC-C/PIQA = `acc_norm`; BoolQ/WinoG/MMLU-text = `acc`; all **0-shot**,
  full-set. **Avg3** = mean(HS, ARC-E, PIQA); **Avg-all** = mean of all benchmark columns present.
- **Law recipe beats γ10 by ~1.5–2 pt at small scale** (rg 100M 41.6 vs 39.8 Avg3; aunet 42.3 vs 40.2;
  hybrid-greedyQ 41.9 vs 39.6) — the paper's "small-batch fixes small-model over-batching" effect,
  reproduced here for every family incl. the hybrid.
- **Identical byte config:** at each scale rg/AU-Net/hybrid share dims [512,768]@100M / [512,1280]@300M /
  larger@1.3B, LR, batch, steps, budget — differing only in `data.regex` (rg=online-greedy-root,
  AU-Net=word, hybrid=offline-leaf-prefill). Llama = subword baseline, matched data budget.
- **300M-law (⟳) is filling:** Llama (17836, complete) + hybrid (`hybrid_300M`@120752, complete) are in;
  **rg/AU-Net are resuming on ece GPU 7** (`rgllaw`/`aunetllaw` from their 40128 checkpoints → 80256).
  ⚠️ Budget caveat: law-300M rg/aunet target **42B** (80256×64 seq/step) while the law-300M hybrid ran
  **63B** (120752) — not iso-budget within the law-300M row (the γ10-300M row *is* iso-budget at 62.3B).
- **Hybrid checkpoints:** 100M-law `hybrid_100M`@53504, 100M-γ10 `hybrid_100M_g10`@3344, 300M-law
  `hybrid_300M`@120752, 300M-γ10 `hybrid_300M_g10`@9900, 1.3B `hybrid_1p3B_leaf_B3`@180000, 760M-law
  `hybrid_760M`@193867 (**⏳ gated: trains after 02:00 when all B200 GPUs idle**, `hybrid_760M_aunetlaw.yaml`,
  152 GB / full AU-Net-law budget — leafQ+greedyQ auto-eval on completion).
- **760M-γ10 = ratio-matched cmp_g10 (~210 bpp, 60600 steps), full-sample all 6 tasks (2026-07-14):** now
  `runs/760M`; **single-lineage** (the earlier BoolQ/WinoG 101-bpp main-lineage borrow is gone —
  rg BoolQ 45.5→53.1, AU-Net 49.9→58.3). The old undertrained 101-bpp run is preserved as
  `runs/760M_bpp101`. At 760M **only the hybrid uses the law recipe** (⏳ above); rg/AU-Net/Llama at 760M
  are **γ10-only** — so the 760M-law rows are hybrid-only.
- **Provenance:** law 100M rg/AU-Net/Llama = ece `lb_*_100M@53504` (scp'd to `runs/100M`); γ10 100M =
  `runs/100M_adhoc`; law-300M eval = ece `eval_law300M_full`.
