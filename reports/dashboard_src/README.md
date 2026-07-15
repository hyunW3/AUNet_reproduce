# Scaling dashboard — generator sources & reproduce chain

Self-contained HTML dashboard for the AU-Net / BPEByte / Llama tokenization scaling study
(downstream accuracy, bits-per-byte, training curves, robustness, 100M parsing ablation,
configuration). The published page is `reports/scaling_dashboard.html` — a single inline
Artifact (JSON + CSS + SVG, no external assets).

**These scripts fully regenerate the dashboard.** Regenerating from a clean checkout yields
byte-identical HTML (verified via md5).

## Reproduce chain

Run in this order from the repo root (`ROOT` is hard-coded in each script). Each step's outputs
are the next steps' inputs; the final `build_dashboard.py` consumes all the JSON/CSV/MD artifacts.

```
# 1. downstream accuracy table  (reads runs/**/evals*/results.json)
python reports/dashboard_src/assemble_data.py      # -> downstream_data.json, scaling_data.csv

# 2. config + training BPB curves  (reads runs/**/config.yaml, metrics.jsonl, scaling_bpb.csv)
python reports/dashboard_src/extract_dash.py       # -> model_config.json, training_bpb_curves.json

# 3. robustness — PBP (acc/acc_norm) + noise  (reads evals_pbp/pbp_mc_full/, downstream_data.json)
python reports/dashboard_src/extract_robust.py     # -> robustness.json

# 4. despace — prompt space-strip, 6 benchmarks  (reads evals_pbp/despace_mc_{full,more}/)
python reports/dashboard_src/extract_despace.py    # -> despace.json

# 4b. despace on NIAH retrieval  (reads reports/niah/despace/results_S{1,2,3}.jsonl)
python reports/dashboard_src/extract_niah_despace.py  # -> niah_despace.json

# 5. (optional) standalone markdown downstream table
python reports/dashboard_src/build_table.py        # -> scaling_downstream_table.md

# 6. assemble the page  (reads all of the above + parsing_ablation_100M.json, leaderboard_100M.md, scaling_bpb.csv)
python reports/dashboard_src/build_dashboard.py    # -> scaling_dashboard.html
```

`plot_from_json.py` is a side utility (per-benchmark PNGs from `downstream_data.json`); not part
of the HTML build.

## Script → output map

| script | writes | key inputs |
|---|---|---|
| `assemble_data.py`   | `downstream_data.json`, `scaling_data.csv` | `runs/**/evals*/results.json` |
| `extract_dash.py`    | `model_config.json`, `training_bpb_curves.json` | `runs/**/config.yaml`, `metrics.jsonl`, `scaling_bpb.csv` |
| `extract_robust.py`  | `robustness.json` | `runs/main/1.3B/*/evals_pbp/pbp_mc_full/`, `evals_noise/`, `downstream_data.json` |
| `extract_despace.py` | `despace.json` | `runs/main/1.3B/*/evals_pbp/despace_mc_{full,more}/` |
| `extract_niah_despace.py` | `niah_despace.json` | `reports/niah/despace/results_S{1,2,3}.jsonl` |
| `build_table.py`     | `scaling_downstream_table.md` | `downstream_data.json` |
| `build_dashboard.py` | `scaling_dashboard.html` | every JSON/CSV/MD above + `parsing_ablation_100M.json`, `leaderboard_100M.md` |

## Where the raw eval results come from

The robustness/despace extractors read eval dumps produced by the `lingua` submodule harnesses
(sentinels placed in `harness.tasks`):

- **PBP-mc** (`pbp_mc` sentinel, `apps/aunet/eval_pbp_mc.py`) — boundary-shift; scored on `acc`
  (cut-invariant) with `acc_norm` as a secondary, artifact-prone metric.
- **Despace** (`despace_mc` sentinel, `apps/aunet/eval_despace_mc.py`) — strip ALL prompt spaces
  ("I have a boy" → "Ihaveaboy") over arc_easy/arc_challenge/piqa/boolq/hellaswag/winogrande;
  `acc_norm` is the primary metric (matches the downstream table). Env `DESPACE_TASKS` selects a
  subset (used to split the 6 benchmarks across `despace_mc_full` + `despace_mc_more`).

- **Despace-NIAH** (`scripts/niah/niah_probe.py --despace`) — verbatim retrieval (RULER S-NIAH-1/2/3)
  with all prompt spaces stripped; headline `exact_match` (per the clean NIAH reports), `tok_frac`
  diagnostic. Inference-only over consolidated 1.3B checkpoints; results in `reports/niah/despace/`.

Byte families run through `apps.aunet.eval` (HierarchicalTransformer); Llama through
`apps.main.eval` (LMTransformer) — the two harnesses share the sentinel modules.
