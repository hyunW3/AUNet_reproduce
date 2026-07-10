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
