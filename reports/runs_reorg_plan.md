# runs/ reorganization plan — main-figure / leaderboard / others

**Goal:** group `runs/` (1.5 TB, ~35 top-level dirs + ~60 loose files) into three intents —
**main-figure** runs, **leaderboard** runs, and **others** — without breaking the paths that
`collect_scaling.py` / `scaling_data.csv` / the leaderboard `.md` files depend on.

## The overlap problem (decides the whole layout)
Main figures (`scaling_*.png`, `downstream_*.png`) and the leaderboards read the **same**
canonical per-scale model runs — 1.3B alone is 489 GB. We must **not** duplicate them.
**Resolution:** canonical runs live **once** under `main/<scale>/`; `leaderboard/` holds the
leaderboard *tables* + **symlinks** into `main/` + the few leaderboard-only runs; `figures/`
holds only the scripts/CSVs/PNGs (no checkpoints).

## Proposed layout
```
runs/
  main/                      # canonical 4-method × 4-scale ladder (source of figures AND leaderboards)
    100M/  {aunet_100M, llama_100M, rg_100M, hybrid_100M}
    300M/  {aunet_300M, llama_300M, rg_300M}        # hybrid_300M = p1b1, dropped from figures → others
    760M/  {aunet2_760M, bpebyte_root_greedy_760M, llama_760M}
    1.3B/  {aunet2_1.3B, bpebyte_br_greedy_root_1.3B, llama_1.8B_paper, hybrid_1p3B_leaf_B3}
  figures/                   # figure generation only — NO checkpoints
    scripts/ (collect_scaling.py, plot_scaling.py, plot_ratio10_ladder.py)
    data/    (scaling_data.csv, scaling_bpb.csv)   # PNGs stay in reports/
  leaderboard/               # leaderboard tables + eval-only aggregates + leaderboard-specific runs
    1B/            -> symlinks into main/1.3B + poc_ece_1p3b (hybrid 1.3B milestone evals)
    100M_law/      poc/portable_aunetlaw               (AU-Net-law 100M variants)
    100M_hybrid_matrix/  100M_adhoc/                   (hybrid-matrix + ad-hoc 100M tables)
  others/                    # exploratory / superseded / infra
    superseded/    cmp_100M_ratio20(88G), cmp_100M(25G), *_oldcode/_oldseg, cmp_g10/*_760M alt-ladder(225G)
    ablations/     ablation_100M, small/ablation_100M, leakfree_100M, pcut_rescore
    byteflow/      byteflow_100M*, byteflow_300M_paperK
    collapse/      collapse_tau_metrics
    entropy/       aunet_300M_entropy_{mono,global}_r10, aunet_100M_entropy_low, entropy_model
    warmstart/     tokwarm500, small/tokwarm, warm
    screening/     bpebyte_screen, superword_l3
    poc/           poc/ (hybrid PoC + scale, minus portable_aunetlaw), _hybrid_smoke, small/poc
    chinese/       1.3B/bpebyte_1.3B_zh_continue_{llama3,qwen2}
    bt_variants/   1.3B/bpebyte_br_bt_{,committed_,online_}1.3B, llama_1B_dm10_not_target
  _infra/ _logs/ _plots/ _scripts/ _data/ _reports/   # keep as-is (support)
  _loose/            # sweep the ~60 root watch_*.sh/.log, *.png, done-markers here (or delete)
```

## Top-level mapping (every current entry → destination)
| Current | Size | → Bucket |
|---|---|---|
| `1.3B/aunet2_1.3B`, `.../bpebyte_br_greedy_root_1.3B`, `.../llama_1.8B_paper`, `.../hybrid_1p3B_leaf_B3` | ~part of 489G | **main/1.3B** |
| `1.3B/poc_ece_1p3b` | — | **leaderboard/1B** |
| `1.3B/bpebyte_1.3B_zh_continue_*` | — | others/chinese |
| `1.3B/bpebyte_br_bt_*`, `.../*_oldseg`, `llama_1B_dm10_not_target` | — | others/bt_variants, others/superseded |
| `760M/{aunet2_760M, bpebyte_root_greedy_760M, llama_760M}` | ~part of 313G | **main/760M** |
| `760M/bpebyte_root_greedy_760M_oldcode_*` | — | others/superseded |
| `300M/{aunet_300M, llama_300M, rg_300M}` | 20G | **main/300M** |
| `300M/hybrid_300M` | — | others/poc (p1b1, dropped) |
| `100M/{aunet_100M, llama_100M, rg_100M, hybrid_100M}` | 4.5G | **main/100M** |
| `cmp_g10/` (llama/rg/aunet_760M alt-ladder + 100M/300M copies) | 225G | others/superseded (alt 210-bpp ladder, not in figures) |
| `poc/portable_aunetlaw` | part of 98G | **leaderboard/100M_law** |
| `poc/` (rest: scale, hybrid PoC) | ~98G | others/poc |
| `cmp_100M_ratio20` | 88G | others/superseded |
| `tokwarm500`, `small/tokwarm`, `warm` | 81G+ | others/warmstart |
| `bpebyte_screen`, `superword_l3` | 58G | others/screening |
| `cmp_100M` | 25G | others/superseded |
| `aunet_300M_entropy_{mono,global}_r10`, `aunet_100M_entropy_low`, `entropy_model`, `small/entropy_model` | 44G | others/entropy |
| `small/` (ablation_100M, cmp_100M, cmp_300M, cmp_g10 symlinks, leakfree, pcut, poc, _smoke_online) | 15G | dissolve → others/* per item |
| `ablation_100M` | 12G | others/ablations |
| `byteflow_*` (6 dirs) | 16G | others/byteflow |
| `leakfree_100M` | 3.4G | others/ablations |
| `_infra _logs _plots _scripts _data _reports collapse_tau_metrics pcut_rescore _hybrid_smoke __pycache__` | ~5G | keep as support (collapse→others/collapse) |
| ~60 loose root files (`watch_*.{sh,log}`, `ratio10_*.png`, `*_done`, `*.py`, `.rsync_done`) | 11M | `_loose/` or delete (PNGs already in reports/) |

## Path-compatibility (must do alongside any move)
The prior 100M/300M reorg (see `scaling_runs_organization.md`) already set the pattern:
**move, then leave a relative symlink at the old path.** Concretely:
1. `collect_scaling.py` reads `100M/aunet_100M/…`, `small/cmp_g10/aunet_300M/…`, `aunet2_1.3B/…`
   (cwd-relative). After moving into `main/<scale>/`, leave symlinks
   `runs/100M -> main/100M`, `runs/1.3B -> main/1.3B`, etc. → CSV + scripts keep resolving.
2. Leaderboard `.md` paths (`runs/1.3B/…`, `runs/poc/portable_aunetlaw/`) resolve the same way.
3. After the dust settles, optionally update `collect_scaling.py` base paths and drop the symlinks.

## STATUS: EXECUTED 2026-07-12
Decisions taken: **physical move + compat symlinks**; superseded → `others/superseded/`; loose files
sorted (scripts→`_scripts/`, logs→`_logs/`, unique `ratio10_*.png`→`reports/`, zero-byte markers deleted).
Outcome: **0 broken symlinks**, collector figures **byte-identical** before/after (safety gate passed),
no data moved or lost. Two collector patches applied (`/stage_\d` EXCLUDE + realpath dedup). Key surprise:
most top-level "dirs" were symlink-aliases into `small/`, which is the real store and was left in place.
See the header of `scaling_runs_organization.md` for the final layout + gotchas.

## Open decisions (original — now resolved above)
- **A. Execution style** — physical `mv` + compat-symlinks (clean tree, some risk / must verify
  the collector still runs), **vs.** non-destructive: keep canonical dirs where they are and add
  `main/ leaderboard/ figures/` as **symlink-only views** (zero path risk, less "clean").
- **B. Superseded heavy dirs** (`cmp_100M_ratio20` 88G, `cmp_100M` 25G, `cmp_g10/*_760M` 225G ≈ 340G) —
  move to `others/superseded/`, **or** delete (they're dropped from all current figures/tables;
  cmp_100M_ratio20 was just re-synced from ece today), **or** keep untouched.
- **C. Loose root files** — sweep into `_loose/` vs delete the stale `watch_*`/`*_done` markers.
```
