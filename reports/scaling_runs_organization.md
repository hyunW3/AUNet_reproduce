# runs/ organization — main / leaderboard / others (2026-07-12 reorg)

`runs/` is now bucketed by intent. **All old paths still work** (compat symlinks); the collector
is location-agnostic (globs `**/results*.json`, classifies by path regex), so moving is safe.

```
runs/
  main/        100M 300M 760M 1.3B   — canonical 4-method ladder (feeds BOTH figures & leaderboards)
  leaderboard/ 1B, 100M, 100M_law, hybrid_* — SYMLINK-VIEW into main/ + small/poc/portable_aunetlaw
  others/      superseded/ warmstart/ screening/ byteflow/ entropy/ ablations/ collapse/ poc/
  small/       THE REAL STORE for many exploratory runs (left in place — its internal relative
               symlinks back to the per-scale dirs would break if moved). others/* symlinks into it.
  _data _infra _logs _plots _reports _scripts   — support (unchanged)
  <compat symlinks>  100M 300M 760M 1.3B poc cmp_g10 + per-model aliases (aunet2_1.3B …) -> new homes
```

**Non-obvious gotchas (learned the hard way):**
- Many "top-level dirs" were **relative-symlink aliases into `small/`** (`poc`, `cmp_g10`, `cmp_100M`,
  `warm`, `bpebyte_screen`, `entropy_model`, `ablation_100M`, `leakfree_100M`, …). `du -sh x/` FOLLOWS
  them, so they masqueraded as real 25–225 GB dirs. Real data lives once in `small/`.
- `collect_scaling.py` glob(`**`) **follows directory symlinks** → compat symlinks would double-count.
  Fixed by deduping on `os.path.realpath` in the collector.
- Intermediate `…/stage_<step>/evals_5bench/<bench>/results.json` dumps carried `evals_5bench` in-path
  but dodged the milestone penalty and were out-ranking the FINAL evals (rg-1.3B arc_easy 46→67).
  Fixed by adding `/stage_\d` to the collector EXCLUDE. Figures now reproduce the committed CSV exactly.

---

# runs/ organization — per-scale folders (100M / 300M / 760M / 1.3B)

The 100M and 300M comparison runs were aggregated into `runs/100M/` and `runs/300M/`
(mirroring `runs/1.3B/` and `runs/760M/`). Sources were **moved** and a **symlink** left at
the old location, so existing paths keep working.

## What moved (the cmp_g10 constant-ratio ladder)

| New location | Was | ckpt | evals |
|---|---|---|---|
| `runs/100M/aunet_100M` | `small/cmp_g10/aunet_100M` | 3344 | eval_scaling, eval_5shot |
| `runs/100M/llama_100M` | `small/cmp_g10/llama_100M` | — (eval-only) | eval_scaling, eval_5shot |
| `runs/100M/rg_100M` | `small/cmp_g10/v4_root_greedy` | — (eval-only) | eval_scaling, eval_5shot |
| `runs/300M/aunet_300M` | `small/cmp_g10/aunet_300M` | 9900 | eval_scaling (+boolq), eval_5shot |
| `runs/300M/llama_300M` | `small/cmp_g10/llama_300M` | — (ckpt on **ece**) | eval_scaling, eval_5shot |
| `runs/300M/rg_300M` | `small/cmp_g10/rg_300M` | — (ckpt on **ece**) | eval_scaling, eval_5shot |

`small/cmp_g10/*` now points back via relative symlinks. The cmp_g10 **760M** runs
(`aunet_760M`, `llama_760M`, `rg_760M`) were left in place (760M already organized).

## ⚠️ Overlapped models — which config differs

Two provenance seams the figures stitch across. **Read before comparing across scales.**

### 1. 100M source SWAPPED: cmp_100M → cmp_g10 (this is a real number change)
The figures previously used `small/cmp_100M/*_ot` at 100M; they now use the cmp_g10 ladder.
Different **data-to-model ratio**, so 100M accuracies dropped ~2–3 pts (and now match the
canonical BPB ladder, e.g. rg-100M BPB 1.1958 == `performance_summary_per_scale.md §1a`).

| 100M run | ratio | budget | steps | bytes/param | HS acc_norm |
|---|---|---|---|---|---|
| old `cmp_100M/*_ot` (dropped) | ~20 | 42 GB | 6688 | 427 | rg 30.0 / aunet 30.7 / llama 33.8 |
| **new `runs/100M/*` (cmp_g10)** | **~10** | **21 GB** | **3344** | **213** | rg 28.6 / aunet 28.5 / llama 31.3 |

The collector now excludes `cmp_100m` + `ablation_100m` so only the cmp_g10 ladder is used.

### 2. 100M/300M (cmp_g10) vs 760M/1.3B (main runs) — different lineage & ratio
`runs/100M` + `runs/300M` are the cmp_g10 **constant-ratio (~210 bytes/param)** AU-Net-law
ladder. `runs/760M` + `runs/1.3B` are the **standalone main runs** — a *different* lineage at
*different* ratios:

| scale | run used | dim (trunk) | bytes/param | note |
|---|---|---|---|---|
| 100M | cmp_g10 `runs/100M/*` | 768 | 213 | ratio-10 ladder |
| 300M | cmp_g10 `runs/300M/*` | 1280 | 210 | ratio-10 ladder |
| 760M | main `760M/{aunet2,bpebyte_root_greedy,llama}_760M` | 1536/2048 | **101** | main run, **NOT** the cmp_g10 ratio |
| 1.3B | main `1.3B/{aunet2_1.3B, bpebyte_br_greedy_root_1.3B, llama_1.8B_paper}` | 2048 | 218 | main run |

So the 100M→300M segment is one internally-matched ladder; the 760M point in particular sits
at a **lower ratio (101 bpp)** than the 100M/300M (210 bpp). cmp_g10 *does* have ratio-matched
760M runs (`small/cmp_g10/{aunet,llama,rg}_760M`, 143 GB / 210 bpp) if a fully self-consistent
100M→760M cmp_g10 ladder is wanted later — they were not swapped in here.

## Not moved (deliberately)
- `poc/scale/c_online_300M` — a *second* rg-300M run at a **different budget** (42 GB / 142 bpp);
  not the cmp_g10 point (see `scaling_summary.md`).
- `poc/scale/hybrid_300M_p1b1`, hybrid milestone runs — the p1b1 variant; hybrid is tracked
  separately (canonical leaf/B3 hybrid 100M lives on ece as `lb_hybrid_100M`).
- `small/ablation_100M/*` — the boundary-scheme ablation (v1–v5), a different experiment.
