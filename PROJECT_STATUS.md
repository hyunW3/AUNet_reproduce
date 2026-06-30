# Project status — consolidated tracker

**Last updated: 2026-06-30.** Single source of truth for status/tasks. Consolidates and supersedes:
`status.md` (1.3B repro), `TODO_260618.md` (eval comparison), `status_300M.md` (300M scope),
`260611_tasks.md` (ablation/resume pipeline). Those four are safe to delete once this is in place.

**What this project is.** Byte-level LM study on the AU-Net hierarchical backbone: compare
**subword (Llama)**, **AU-Net word/whitespace patches**, and **BPEByte** (BPE-trie boundaries on raw
bytes, leak-free causal) at matched compute across **100M / 300M / 760M / 1.3B**, plus a robustness
suite (PBP, typos, CUTE, noise) and a multilingual extension. Origin target: reproduce arXiv
2506.14761 (AU-Net2 1.3B; HellaSwag 64.2 / ARC-E 64.4).

**Repository docs map** (markdown organized 2026-06-30):

| dir | contents |
|---|---|
| `reports/` | results & plots: `model_results_*`, `100M_ablation`, `bpb_*`, `evaluation_results`, `b4_pilot_estimate`, `*.png` |
| `plans/` | plans: `plan`, `plan_byteflow`, `plan_entropy_patching`, `paper_outline.html` |
| `methods/` | method specs: `BPEByte_root_greedy_method`, `BPEByte_with_merge_rule` |
| `reviews/` | critical reviews: `review_blt` |
| `notes/` | explainers: `evaluation_flow`, `benchmark_example`, `Prompt_boundary_problem_example`, `c5_fewshot_truncation` |
| `PoC/` | proof-of-concepts & ideas: `PoC_*`, `idea_oov_eval`, `prefix_vocab_poc`, `trunk_warmup_poc` |
| `scaling_laws/` · `viz/` | scaling study · animations |

Root keeps only `README.md`, `READMD_kor.md`, this tracker, and `TRAINING_SCRIPTS.md`.

---

## 1. Current state (2026-06-30)

| Track | State |
|---|---|
| 1.3B AU-Net2 reproduction | ✅ done (HS 62.6 / ARC-E 66.5; HS ~0.6 below paper band) |
| 1.3B 4-model comparison (Llama/AU-Net2/online-bt/root_greedy) | ✅ done → `reports/model_results_1.3B.md` |
| 1.3B BPEByte root_greedy (new code, 180k) | ⏳ finishing (was 162k/180k on 06-28) — **verify if complete** |
| 760M matched-scale (4 models, new vs old code) | ✅ done → `reports/model_results_760M.md` |
| 100M boundary-scheme ablation (ratio-40) | ✅ done → `reports/100M_ablation.md` |
| 300M 4-model comparison | 📋 planned (scope below); ece 8×A100 |
| **Multilingual B4 pilot** (en/fi/zh/code, 4 families, 100M r40) | ⏸️ **PAUSED 2026-06-30** (user). Training done (final losses anomalous ~0.025, unverified); downstream eval + fertility figure + 760M follow-on NOT proceeding. Data/configs kept on ece for resume → `reports/b4_pilot_estimate.md` |
| Down-pool ablation (start vs last_byte) | ✅ done; last_byte −0.062 BPB but **leak-shaped → dropped** → `PoC/PoC_down_pooling.md` |
| Scaling laws / FLOPs | 📋 plan committed → `scaling_laws/` |
| Paper outline + acceptance plan | ✅ draft → `plans/paper_outline.html` |

**Immediate next:** (1) verify the multilingual pilot's anomalous loss; (2) build per-language
held-out BPB eval → fertility-slope figure; (3) confirm 1.3B new-code run finished + eval.

---

## 2. Headline results

### 1.3B — 4-model (0-shot full unless noted)
| Bench | BPEByte bt | BPEByte hybrid-greedy | AU-Net2 | Llama 1.8B |
|---|---|---|---|---|
| HellaSwag | 0.643 | 0.640 | 0.627 | 0.622 |
| ARC-Easy | 0.652 | 0.641 | 0.657 | 0.654 |
| BoolQ | 0.639 | 0.636 | 0.611 | 0.633 |
| PIQA | 0.741 | 0.738 | 0.744 | 0.749 |
| WinoGrande | 0.617 | 0.616 | 0.611 | 0.611 |

- **5-shot mean: root_greedy 67.9 > Llama 67.4 > AU-Net2 66.6** — byte overtakes subword by 5-shot.
- **CUTE (char): AU-Net2 23.9 / root_greedy 20.6 > Llama 18.4.** **PBP: byte ΔBPC≈0 vs Llama +0.71.**
- Leak cost: leaky `bt` is +0.2–1.1 pt over leak-free greedy (max on ARC-E). Honest byte # = greedy.
- Byte MBPP/HumanEval = N/A (online byte `generate_until` times out on long code-gen).

### 760M — BPB (bits/byte, lower=better)
`llama 0.9234 < BPEByte-new 0.9256 < BPEByte-old(4ed607e) 0.9275 < AU-Net-word 0.9284`.
New code vs old: **ΔBPB −0.0019** (new better) → plot `reports/bpebyte_newold_bpb.png`, `reports/bpb_compare.png`.

### 100M ratio-40 (84B bytes) — boundary-scheme avg (HS/ARC-E/PIQA/ARC-C acc_norm)
`llama 42.7 ≫ v6-root 39.1 ≈ v6 39.1 > v4(root_greedy) 38.8 > v7 38.7 > aunet-word 38.7 > v1 37.2`.
Every leak-free byte scheme ties/beats whitespace; subword leads by ~3.6. Warm-start does not help.

### Down-pool ablation (100M ratio-5, English) — ⚠️ leak-suspect, dropped
A0 start BPB 1.3413 vs A1 last_byte 1.2789 (**Δ −0.062**). Matches the ~0.07 BPB leak magnitude
exactly; leak probe never run → dropped pending audit. See `PoC/PoC_down_pooling.md` §7.

---

## 3. 300M comparison — planned scope (from status_300M.md)

Test the scaling claim: leak-free+causal penalty is within-noise at 100M but +4 at 1.3B — where does
300M land? Four models, byte trunk ≈296M (`dims [512,1280]`, `layers [3,13]`), Llama dim1280/15L:
1. AU-Net word · 2. Llama subword · 3. BPEByte greedy_root (leak-free) · 4. committed_view v1 (leak-free).
**Budget:** Option A iso-token = 42B bytes (6688 steps; Llama 8800) — extend to 84B if leak gap is
ambiguous. **Cost:** ~455 B200-GPU-hr serial, or ~2 days on ece 8×A100 (2 GPU/model concurrent).
Eval: 0-shot HS/ARC-E/ARC-C/PIQA + leak-gap column. Status: orch on ece, idle-gated.

---

## 4. Infrastructure & gotchas

- **Hosts:** B200 (4×, 180 GB, `/NHNHOME/.../AUNet`) — primary training. **ece-agpu11 / ece-agpu18**
  (8×A100-80GB each, user `hwbae`, `/home/hwbae/AUNet`, **shared NFS `ece-nfs3:/DATA/home`**, direct
  internet) — overflow/parallel + multilingual pilot. **snu55** (RTX A5000, cu121 env
  `aunet_eval_cu121`, NCCL_P2P_DISABLE) — eval offload + entropy-model training.
- **A100 memory:** B200 byte configs (bs96) OOM on 80 GB → use **bs24·ga8** (≡ bs96·ga2, preserves
  6.29M bytes/opt-step). Llama bs32·ga4 seq2048 fits.
- **Per-step time (current code, compile):** B200 ~0.33 s; A100 ~0.51 s (≈1.5× B200, *not* the stale
  ~4.1 s in old docs). 100M ratio-40 ≈ 1.9 h/byte-model.
- **Leak discipline:** boundary/pooling gains that look too good at low budget (before_root, last_byte)
  are usually **future-byte leaks**. Always confirm 0% leak with `probe_root_causal.py` before trusting.
- **Pipeline scripts** live git-untracked in the `lingua` submodule → a `git clean` there wipes them
  (has happened; back up). Eval intermediates (`requests.json`, `grouped_requests*.json`) are generated.
- **Byte generation:** loglikelihood never rolls back; only long autoregressive code-gen times out
  (O(n²) re-prefill). `online-bt` CUTE = 0.0 (degenerate decode).
- Commits: **no `Co-Authored-By` trailer** (user preference).

---

## 5. Source-file provenance (now folded in)

- **`status.md`** (2026-06-21) — 1.3B AU-Net2 repro step log (env→data→config→train 180k→eval),
  B200/torch-2.7 fixes (FA3 off, DTensor-safe clip, INTRA_NODE_COMM=0), 4-model comparison, eval
  infra (seg-cache, parallel precompute), 760M queued, Phase-3 hyperparam screen.
- **`TODO_260618.md`** (2026-06-18→21) — BPEByte vs AU-Net2 vs Llama eval comparison, leak-free
  tokenization study (bt/hybrid-greedy/full-greedy), 0/5/10-shot sweeps, ARC-C/MMLU/MBPP/HEval/CUTE,
  PBP + PhonologyBench-G2P + HellaSwag-Noise, eval-efficiency overhaul.
- **`status_300M.md`** (2026-06-20) — 300M scope: calibrated architecture, iso-token vs iso-ratio
  budget, GPU-hours, eval plan, config-generation next-actions.
- **`260611_tasks.md`** (2026-06-11) — ablation eval + v5_distilled retrain + 1.3B resume pipeline,
  B200/snu55 host split (online-boundary decode stalls), watchdog, snu55 cu121 env-build recipe.
