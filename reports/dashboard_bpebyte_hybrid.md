# BPEByte hybrid — dashboard (all BPEByte variants)

All BPEByte variants in one view: the **rg control** (pure `root_greedy`, [`dashboard_bpebyte_rg.md`](dashboard_bpebyte_rg.md))
vs the **hybrid prefill/decode** family. Hybrid = an offline/online **prefill region** (leak-tolerant, non-causal)
followed by a `root_greedy` **decode region** (0-leak, causal) — the decode part stays byte-reproducible while
the prefill part gets BPE-like compression.

- **Placement:** `leaf` = before_root, offline real-BPE · `bt` = before_root, online backtracking/longest-match ·
  `root` = offline-BPE at root placement. Decode is always `root_greedy`.
- **Boundary** (prefill region as fraction of doc length N): `(0,N)`=full · `N/2`=static-half · `(N/3,2N/3)`=mid.
- **Notation** ({placement}-{mode}): root-online-gr (=greedyQ) · leaf-offline (=leafQ) · root-offline (=rootQ) ·
  leaf-online-bt (=btQ). Downstream MC scores the answer greedily, so question tokenization matters **only via
  placement** → the three before_root modes (root-online-gr ≡ leaf-offline ≡ leaf-online-bt) are **byte-identical**;
  only root-offline differs. This *collapse* holds at every scale and recipe.
- **BPB split:** `Decode@0.5` = leak-free, primary (region past b/N=0.5) · `Full@0.5` = leak-contaminated diagnostic.

Details: [`leaderboard_100M_hybrid_matrix.md`](leaderboard_100M_hybrid_matrix.md) (10-cell law + ad-hoc matrices),
[`leaderboard_300M_hybrid.md`](leaderboard_300M_hybrid.md).

## Headline — hybrid vs rg control (native regime, AU-Net-law recipe)

| Scale | Variant | 3-bench (HS/ARC-E/PIQA) | Decode-BPB@0.5 | Full-BPB@0.5 |
|---|---|---:|---:|---:|
| **100M** | **rg (control)** | 40.5 | 1.194 | 1.789 |
| 100M | leaf · (N/3,2N/3) | **45.6** | 1.098 | 1.126 |
| 100M | bt · (N/3,2N/3) | **45.6** | 1.094 | 1.129 |
| 100M | leaf · N/2 | 45.4 | 1.100 | 1.130 |
| 100M | root · N/2 | 40.1 | **1.079** | 1.168 |
| **300M** | leaf · N/2 | **52.8 ±1.7** | 0.993 | 1.018 |
| 300M | leaf · MID | 52.7 ±1.7 | 0.996 | 1.020 |
| 300M | bt · N/2 | 51.3 ±1.8 | 0.992 | 1.026 |

*(100M CIs ≈ ±1.7 by the same estimator; the leaf/bt cluster clears rg/root but the within-cluster gaps are noise.)*

## What the ablation says

1. **Hybrid prefill beats the rg control on downstream by ~+5 pts at 100M** (leaf/bt 44.6–45.6 vs rg 40.5)
   and collapses **Full-BPB** from 1.79 → ~1.13 — the model trained with a prefill region simply handles the
   non-causal region far better. The gain **scales**: all three 300M configs jump ~+7 pts (100M→300M) to 51–53.
2. **Placement is the only axis that moves anything.** `leaf ≈ bt` everywhere (offline real-BPE ≈ online
   backtracking are interchangeable); boundary rule is second-order. `root` prefill lands with the control on
   downstream (~40) despite the lowest leak-free **Decode-BPB** (1.078–1.079).
3. **Decode-BPB ⊥ downstream (a warning).** On the law recipe, root prefill wins Decode-BPB but *loses*
   downstream; the leaf/bt winners carry the *highest* Decode-BPB. **Full-BPB tracks downstream** correctly
   (leaf/bt ~1.13 < root ~1.17 < control 1.79). Don't rank hybrids by leak-free decode BPB alone.
4. **At 300M the config choice is statistically tied** (limit=1000, 3-bench CIs ±1.7–1.8 overlap): leaf·N/2
   52.8, leaf·MID 52.7, bt·N/2 51.3 are indistinguishable, and Decode-BPB all ~0.99. Only the **+7 pt scale
   gain** is robust; separating leaf-vs-bt / N/2-vs-MID needs full test sets or seeds, not more training.

## Recipe note

Two recipes at the same γ10.4 token budget: **AU-Net-law** (batch 48, 53,504 steps @100M — the numbers above)
and **ad-hoc** (batch 768, 3,344 steps). Law runs ~0.05–0.10 BPB / +3–4 downstream pts better; the ad-hoc
10-cell matrix (same *relative* ordering, worse absolutes) is the bottom half of `leaderboard_100M_hybrid_matrix.md`.

## Checkpoints
- **100M** (law, 10 cells): `runs/small/…/lb_hyb_{leaf,bt,root}_{full,half,mid}_100M/checkpoints/0000053504`;
  configs `runs/poc/portable_aunetlaw/ablation_cfgs/`.
- **300M**: leaf·N/2 `ece:~/AUNet/runs/poc/portable_aunetlaw/lb_hyb_leaf_half_300M/checkpoints/0000120752`;
  bt·N/2 `…/lb_hyb_bt_half_300M/…`; leaf·MID `…/lb_hyb_leaf_mid_300M/…` (copy of B200 `runs/300M/hybrid_300M`).
