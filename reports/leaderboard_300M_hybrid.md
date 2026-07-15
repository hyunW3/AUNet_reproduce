# 300M hybrid scale — leaf·MID vs leaf·N/2 vs bt·N/2

_Completed 2026-07-14. AU-Net-law recipe (120,752 steps, global batch 64, LR 2.9e-3), ~63 B tokens (γ10.4).
leaf·N/2 and bt·N/2 newly trained (ece-agpu11 GPU2-5); leaf·MID = the scale-ladder `hybrid_300M`, re-evaluated
with the identical protocol. All evaluated in native regime (4 Q-modes + native decode/Full BPB, `eval_300M/`)._

Downstream = HS/ARC-E/PIQA `acc_norm` + BoolQ/Wino `acc`, limit 1000. ±95% CI from lm-eval per-benchmark
stderrs (SE of the mean); BPB 95% CI = bootstrap over docs (`eval_hybrid_bpb`, 2000 resamples, 512-cap @ b/N=0.5).

| Model (native) | HS | ARC-E | PIQA | **3-bench ±95%** | 5-bench ±95% | Decode-BPB@0.5 [95%] | Full-BPB@0.5 [95%] |
|---|---:|---:|---:|---:|---:|---:|---:|
| leaf · MID (N/3,2N/3) | 46.4 | 44.6 | 67.1 | **52.7 ±1.7** | 53.7 ±1.4 | 0.996 [0.974, 1.018] | 1.020 [1.000, 1.039] |
| leaf · N/2 | 46.2 | 44.8 | 67.4 | **52.8 ±1.7** | 53.0 ±1.4 | 0.993 [0.970, 1.014] | 1.018 [0.998, 1.038] |
| bt · N/2 | 45.1 | 44.0 | 64.8 | **51.3 ±1.8** | 52.8 ±1.4 | 0.992 [0.970, 1.014] | 1.026 [1.006, 1.045] |

**vs 100M (native 3-bench):** leaf·MID 45.6→**52.7** (+7.1), leaf·N/2 45.4→**52.8** (+7.4), bt·N/2 44.6→**51.3** (+6.7).

## Findings

- **All three are statistically tied at 300M.** The 3-bench CIs (±1.7–1.8) overlap heavily (52.8 / 52.7 / 51.3),
  so the ~1.5 pt leaf-over-bt gap is **within noise** at limit=1000. Decode BPB (0.992–0.996, CIs [0.970–1.018])
  and Full BPB (1.018–1.026) are likewise fully overlapping. The 100M leaf≥bt trend does **not** reach significance
  at 300M, and N/2 ≈ MID is a wash.
- **The scale gain is real: ~+7 pts (100M→300M)** for all three — this jump is large relative to the ±1.7 CI,
  unlike the between-config gaps. The hybrid downstream advantage persists with scale.
- **Decode BPB tightened to ~0.99** (from ~1.10 at 100M). The 5-bench leaf·MID 53.7 > leaf·N/2 53.0 edge is a
  WinoGrande near-chance artifact (53.0 vs 50.0); the 3-bench has them tied.

**Net:** the hybrid gain **scales**, but leaf-vs-bt and N/2-vs-MID are indistinguishable at limit=1000 — separating
them would need full test sets (tighter CIs) or multiple seeds, not more training.

## Checkpoints (all @ step 120752)

On **ece-agpu11** (shared `/home/hwbae` NFS):

| Config | Path | Size |
|---|---|---|
| leaf·N/2 | `runs/poc/portable_aunetlaw/lb_hyb_leaf_half_300M/checkpoints/0000120752` | 10 G (sharded + consolidated) |
| bt·N/2 | `runs/poc/portable_aunetlaw/lb_hyb_bt_half_300M/checkpoints/0000120752` | 10 G (sharded + consolidated) |
| leaf·MID | `runs/poc/portable_aunetlaw/lb_hyb_leaf_mid_300M/checkpoints/0000120752` | 3.4 G (consolidated-only copy) |

leaf·MID's original (full sharded) is on **B200**: `runs/300M/hybrid_300M/checkpoints/0000120752`.
Configs: `runs/poc/portable_aunetlaw/ablation_cfgs/lb_hyb_{leaf_half,bt_half}_300M.yaml`.

See the 100M ablation this scales from: [`leaderboard_100M_hybrid_matrix.md`](leaderboard_100M_hybrid_matrix.md).
