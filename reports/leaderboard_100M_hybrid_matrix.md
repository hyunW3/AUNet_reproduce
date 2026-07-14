
# 100M hybrid ablation — full matrix (AU-Net-law recipe)

> **Recipe: AU-Net-law** (global batch 48, 53,504 steps, LR 3.4e-3, warmup 3000) — the same 21 GB /
> γ10.4 budget as the ad-hoc matrix above, but the scaling-law HP recipe (Videau 2025 §2.3). This is the
> **law-recipe twin** of the ad-hoc grid: same 10 cells, evaluated identically (agpu18 GPU2,3, `eval_g23/`).
> Completed 2026-07-11 (survived a disk-quota incident + 2 bt restarts). All 10 cells trained to 53,504.

Downstream = 5-bench mean (HS/ARC-E/PIQA `acc_norm` + BoolQ/Wino `acc`, limit 1000). BPB = held-out
`eval_hybrid_bpb` 512-cap @ b/N=0.5. Train-BPB = final `loss/out ÷ ln2` (noisy single-step proxy).

| Trained model | Train-BPB | Decode@0.5 | Full@0.5 | root-online-gr | leaf-offline | root-offline | leaf-online-bt |
|---|---:|---:|---:|---:|---:|---:|---:|
| **original gr** (control) | 1.139 | 1.194 | 1.789 | **0.414** | 0.414 | 0.406 | 0.414 |
| **leaf · (0,N)** | 1.101 | 1.095 | 1.125 | 0.474 | **0.474** | 0.447 | 0.474 |
| **leaf · N/2** | **1.092** | 1.100 | 1.130 | 0.493 | **0.493** | 0.481 | 0.493 |
| **leaf · (N/3,2N/3)** | 1.097 | 1.098 | 1.126 | 0.487 | **0.487** | 0.488 | 0.487 |
| **bt · (0,N)** | 1.105 | 1.097 | 1.135 | 0.467 | 0.467 | 0.483 | **0.467** |
| **bt · N/2** | 1.095 | 1.100 | 1.137 | 0.484 | 0.484 | 0.430 | **0.484** |
| **bt · (N/3,2N/3)** | 1.100 | **1.094** | 1.129 | 0.483 | 0.483 | 0.495 | **0.483** |
| **root · (0,N)** | 1.136 | 1.078 | 1.167 | 0.458 | 0.458 | **0.450** | 0.458 |
| **root · N/2** | 1.135 | 1.079 | 1.168 | 0.413 | 0.413 | **0.416** | 0.413 |
| **root · (N/3,2N/3)** | 1.098 | **1.078** | 1.167 | 0.414 | 0.414 | **0.418** | 0.414 |

Bold DS cell = native regime (rg→root-online-gr, leaf→leaf-offline, bt→leaf-online-bt, root→root-offline).

## Law-recipe findings

- **The greedyQ≡leafQ≡btQ collapse holds identically** — for every cell the three before_root question
  columns (root-online-gr / leaf-offline / leaf-online-bt) are byte-identical; only root-offline differs.
  Same invariance as the ad-hoc recipe.
- **Law ≫ ad-hoc, as expected (~0.05–0.10 BPB better):** control Train-BPB 1.139 vs 1.239 (−0.10);
  leaf/bt ~1.10 vs ~1.13; Decode BPB all hybrids ~1.08–1.10 vs the ad-hoc ~1.18–1.19 (−0.09); downstream
  native ~+0.03–0.05 (leaf_half 0.493 vs ad-hoc leaf ~0.44). Confirms the law recipe's iso-budget gain.
- **Decode-BPB placement REVERSES vs ad-hoc:** on the ad-hoc recipe all hybrids tied (~1.18) on decode;
  on the law recipe **root prefill is the *best* on Decode BPB (1.078) < leaf/bt (1.094–1.100)** — a clean
  ~0.02 separation. But root is **worst on downstream** (rootQ 0.42–0.45 vs leaf/bt 0.47–0.49) and worst on
  Full-BPB (1.167 vs leaf/bt ~1.13). So root prefill trades downstream for a slightly lower leak-free decode BPB.
- **Downstream, native regime:** leaf ≈ bt (0.47–0.49) > root (0.41–0.46) > control (0.414) — same
  ordering as ad-hoc, sharper. **Best single cell: leaf · N/2 = 0.493** (5-bench), leaf/bt·(N/3,2N/3) ≈ 0.456 (3-bench).

**Net (law recipe):** before_root prefill (leaf ≈ bt) still wins downstream; **leaf · N/2** is the top
config. The one new twist vs ad-hoc: **root prefill gives the lowest leak-free Decode BPB** under the law
recipe, though it doesn't translate to downstream. Boundary rule remains second-order within a family.

## Downstream sorted by HS/ARC-E/PIQA 3-bench (native regime, acc_norm)

Dropping the near-chance BoolQ/WinoGrande from the 5-bench mean sharpens the ranking. Sorted best-first,
with both BPB measures alongside:

| # | Model (native) | HS | ARC-E | PIQA | **3-bench avg** | Decode-BPB@0.5 | Full-BPB@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | leaf · (N/3,2N/3) | 39.8 | 35.9 | 61.2 | **45.6** | 1.098 | 1.126 |
| 2 | bt · (N/3,2N/3) | 40.2 | 36.5 | 60.0 | **45.6** | 1.094 | 1.129 |
| 3 | leaf · N/2 | 40.0 | 36.7 | 59.5 | **45.4** | 1.100 | 1.130 |
| 4 | bt · (0,N) | 39.5 | 36.1 | 59.9 | **45.2** | 1.097 | 1.135 |
| 5 | leaf · (0,N) | 38.9 | 35.4 | 59.9 | **44.7** | 1.095 | 1.125 |
| 6 | bt · N/2 | 38.8 | 34.3 | 60.6 | **44.6** | 1.100 | 1.137 |
| 7 | control (gr) | 37.0 | 30.6 | 53.9 | **40.5** | 1.194 | 1.789 |
| 8 | root · N/2 | 36.3 | 30.8 | 53.2 | **40.1** | 1.079 | 1.168 |
| 9 | root · (N/3,2N/3) | 36.7 | 31.1 | 51.4 | **39.7** | 1.078 | 1.167 |
| 10 | root · (0,N) | 35.6 | 29.6 | 51.6 | **38.9** | 1.078 | 1.167 |

**Downstream ⊥ Decode-BPB:** the top-6 by downstream are all leaf/bt (before_root, 44.6–45.6) yet carry the
*highest* Decode-BPB (~1.094–1.100); the bottom-3 are all root (≈ control) yet have the *lowest* Decode-BPB
(~1.078). So lower leak-free Decode BPB does **not** predict better downstream here — root wins decode-BPB but
loses downstream. **Full-BPB tracks downstream correctly** (leaf/bt ~1.13 < root ~1.17 < control 1.79).

## 300M scale — leaf·MID vs leaf·N/2 vs bt·N/2

The three best 100M configs trained at **300M** (AU-Net-law, 120,752 steps, global batch 64) and evaluated
identically (native regime, 4-Q-mode + decode BPB, `eval_300M/`). leaf·MID = the scale-ladder `hybrid_300M`
re-evaluated with the same protocol; leaf·N/2 and bt·N/2 newly trained (ece-agpu11 GPU2-5, 2026-07-14).

| Model (native) | HS | ARC-E | PIQA | **3-bench** | 5-bench | Decode-BPB@0.5 | Full-BPB@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| leaf · MID (N/3,2N/3) | 46.4 | 44.6 | 67.1 | **52.7** | 53.7 | 0.996 | 1.020 |
| leaf · N/2 | 46.2 | 44.8 | 67.4 | **52.8** | 53.0 | 0.993 | 1.018 |
| bt · N/2 | 45.1 | 44.0 | 64.8 | **51.3** | 52.8 | 0.992 | 1.026 |

**vs 100M (native 3-bench):** leaf·MID 45.6→**52.7** (+7.1), leaf·N/2 45.4→**52.8** (+7.4), bt·N/2 44.6→**51.3** (+6.7).

**Findings:**
- **leaf·N/2 ≈ leaf·MID at 300M** (3-bench 52.8 vs 52.7 — tied within noise), both **> bt·N/2 (51.3)**. The
  100M ordering holds and the leaf-vs-bt gap (~1.5 pt) is stable across scale; the N/2-vs-MID boundary choice
  stays second-order within the leaf family (a wash).
- **All three scale ~+7 pts** (100M→300M) — the hybrid downstream gains persist with scale.
- **Decode BPB tightens to ~0.99 for all three** (0.992–0.996, bt marginally lowest, within noise); Full-BPB
  leaf ~1.018–1.020 < bt 1.026. (The 5-bench leaf·MID 53.7 > leaf·N/2 53.0 edge is a WinoGrande artifact —
  53.0 vs 50.0, i.e. near-chance noise; the 3-bench has them tied.)

**Net (300M):** **leaf prefill** (either boundary) is the pick; **bt trails by ~1.5 pt** downstream. leaf·N/2
and leaf·MID are indistinguishable (~52.8 3-bench), consistent with the boundary being second-order at 100M.


---
# 100M hybrid ablation — full matrix

> **Recipe: AD-HOC** (global batch 768, 3,344 steps, LR 2.0e-3, warmup 100) — the
> `runs/small/poc/campaign/` family. **NOT** the AU-Net scaling-law recipe used in
> [`leaderboard_100M.md`](leaderboard_100M.md) (batch 48, 53,504 steps, LR 3.4e-3). Same 21 GB / γ10.4
> budget, but the law recipe runs ~0.08 BPB / +3–4 downstream pts better, so **absolute numbers here
> are not comparable to `leaderboard_100M.md`** — only the *relative* ablation (all 10 rows share this
> recipe) is. A law-recipe version of this ablation is in progress (see foot of doc).

_100M (98.59M params), ratio ≈10.4 (21.04 GB), single seed 777. Boundary: `(0,N)`=Uniform-full (B1),
`N/2`=static-half (B2), `(N/3,2N/3)`=Uniform-mid (B3). Prefill placement: **leaf**=before_root (offline
real-BPE), **bt**=before_root (online backtracking/longest-match), **root**=offline-BPE root placement.
Decode is always root_greedy._

Rows = trained model. Columns = evaluation method. Downstream = 5-bench mean (HS/ARC-E/BoolQ/PIQA/Wino,
`acc`, limit 1000), ece A100. BPB = held-out `eval_hybrid_bpb`, 512-cap, b/N=0.5, native prefill each.

| Trained model | Train-BPB¹ | Decode BPB @0.5² [95% CI] | Full BPB @0.5² | DS root-online-gr³ | DS leaf-offline³ | DS root-offline³ | DS leaf-online-bt³ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **original gr** (control) | 1.239 | 1.368 [1.346, 1.390] | 2.048 | **0.400** | 0.400 | 0.395 | 0.400 |
| **leaf · (0,N)** | **1.123** | 1.184 [1.160, 1.208] | 1.210 | 0.441 | **0.441** | 0.417 | 0.441 |
| **leaf · N/2** | 1.130 | 1.193 [1.169, 1.217] | 1.216 | 0.442 | **0.442** | 0.444 | 0.442 |
| **leaf · (N/3,2N/3)** | 1.124 | 1.190 [1.166, 1.214] | 1.212 | 0.462 | **0.462** | 0.413 | 0.462 |
| **bt · (0,N)** | 1.127 | 1.182 [1.158, 1.205] | 1.215 | 0.470 | 0.470 | 0.463 | **0.470** |
| **bt · N/2** | 1.128 | 1.185 [1.161, 1.209] | 1.213 | 0.438 | 0.438 | 0.425 | **0.438** |
| **bt · (N/3,2N/3)** | 1.129 | 1.188 [1.165, 1.211] | 1.218 | 0.444 | 0.444 | 0.466 | **0.444** |
| **root · (0,N)** | 1.187 | 1.180 [1.156, 1.203] | 1.268 | 0.405 | 0.405 | **0.399** | 0.405 |
| **root · N/2** | 1.193 | 1.191 [1.167, 1.215] | 1.274 | 0.396 | 0.396 | **0.405** | 0.396 |
| **root · (N/3,2N/3)** | 1.191 | 1.188 [1.165, 1.212] | 1.278 | 0.400 | 0.400 | **0.407** | 0.400 |

Bold DS cell = model's native question regime (rg→greedy, leaf→leaf, bt→bt, root→root).

**Footnotes.** ¹ final-window `loss/out ÷ ln2` @ step 3340 (train proxy; `leaderboard_100M_adhoc.md`).
² native `eval_hybrid_bpb`, 512-cap, b/N=0.5, each model at its TRAINING prefill; leak-free Decode
(primary) / leak-contaminated Full (diagnostic). ³ 5-bench mean, ece A100, `matrix_v2/`.

## Finding 1 — the three before_root question modes collapse

**root-online-gr ≡ leaf-offline ≡ leaf-online-bt for every model** (byte-identical), and only **root-offline** differs. The MC
loglikelihood scores the *answer* (always greedy); the question tokenization only changes the answer's
conditioning **through its placement**, not its segmentation method. All of greedy / offline-leaf / bt
are **before_root** placements → identical answer NLL; offline-root (root-offline) shifts the boundaries → a
different number. So the requested **leaf-online-bt column is numerically the before_root-question column** — it's
not a new signal, it's the same signal, which is itself the result.

## Finding 2 — placement is the only axis that moves anything

- **Train-BPB & Full-BPB cleanly split placement:** before_root (leaf 1.123–1.130, bt 1.127–1.129) sits
  ~0.06 below root (1.187–1.193); Full-BPB leaf/bt ~1.21 vs root ~1.27. **leaf ≈ bt** everywhere —
  offline real-BPE and online backtracking are interchangeable.
- **Decode BPB ties across the whole grid:** every hybrid is 1.18–1.19 with fully overlapping CIs
  (placement and boundary are within noise), and all are ~0.18 below the control (1.368). The decode
  regime is root_greedy for all, so this is expected — the win is entirely "trained-on-a-prefill-region"
  vs not.
- **Downstream, native regime:** bt·(0,N) **0.470** is the best single cell, leaf·(N/3,2N/3) **0.462**
  next; both before_root families beat root (~0.40) and the control (0.400). Near-chance at 100M, so
  weakly suggestive, but it agrees with BPB.

**Net:** the winning axis is **before_root prefill (leaf or bt)**; boundary rule and leaf-vs-bt are
second-order. Best configs: **bt·(0,N)** and **leaf·(N/3,2N/3)**.
