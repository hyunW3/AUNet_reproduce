# B4 multilingual/code pilot — 100M cost/time/performance (MEASURED)

_2026-06-28. Real 100M calibration runs on **ece-agpu11** (8×A100-80GB, used GPUs 4–7), current
`lingua` code (commit e500b7d), torch 2.7+cu128, `torch.compile`, bf16, FSDP full_shard, async
data. B200 box was unavailable (live 1.3B run at step 162k/180k, ~12 GB free/GPU)._

## What was run
Short steady-state calibrations on existing English DCLM (multilingual data not yet staged — see
Blockers). Each fixed 3 hardcoded B200 abs-paths (`data.root_dir`, `data.regex.bpe_tokenizer_path` /
`data.tokenizer.path`) and, for byte models, reduced micro-batch to fit 80 GB (`bs24 ga8`, which
**preserves the native 1.57 M bytes/opt-step**, so per-step time still extrapolates).

| model | config | A100 s/opt-step | mem (reserved) | note |
|---|---|---:|---:|---|
| **BPEByte root_greedy** | bs24·ga8, seq8192 | **0.508** | 39.7 GiB | measured, loss 3.31→2.47 over 50 steps |
| **Llama subword** | bs32·ga4, seq2048 | **0.515** | (fits) | measured, data_load ≈0.002 s |
| **AU-Net word** | (same arch as BPEByte) | **≈0.51** | — | inferred: identical GPU graph; boundary diff is CPU/async. B200 ratio confirms parity (0.348 vs 0.332) |

### Key correction vs the stale doc
`100M_ablation.md` lists A100 at **~4.1 s/step** (byte). That is stale/old-path. **Current code +
compile on A100 = ~0.51 s/step** — i.e. A100 is only **~1.5× slower than B200** for byte models
(0.508 vs 0.332), not ~12×. Llama factor is ~3.2× (B200 favors its short-seq subword path more).

## Cost & time — 100M ratio-40 (84 B bytes), per the measured A100 rates
Byte = 13,376 opt-steps; Llama iso-byte = 17,600.

| model | pure h | +25% overhead | GPU-h |
|---|---:|---:|---:|
| BPEByte root_greedy | 1.89 | 2.36 | 9.4 |
| AU-Net word | 1.89 | 2.37 | 9.5 |
| Llama | 2.52 | 3.15 | 12.6 |
| **3-family total (1× 4-GPU slice, sequential)** | **6.3** | **~7.9** | **~32** |
| + BLT-entropy v8 (≈byte + 1-time precompute) | +1.9 | +2.4 | +9.4 |

**Takeaway: a full 3–4-family 100M ratio-40 pilot is ~8–10 h wall on ece-agpu11's free 4-GPU slice
(~30–40 GPU-h).** Cheap enough to run the *full* pilot for real performance rather than projecting.
Multilingual run = **identical compute** (multilingual bytes cost the same per step); the only added
cost is data staging.

## Performance
- Training is healthy (BPEByte loss 3.31→2.47 in 50 steps). **No final-quality number** — that needs
  the full ~2 h/model ratio-40 run, which was not run here.
- Multilingual BPB / fertility-slope numbers require staged multilingual+code data (not yet present).
- ⚠️ Recall the ablation lesson: at 100M even ratio-40 separates byte *schemes* by only ~1 pt. **100M
  may be too small to cleanly rank families on multilingual BPB** — the pilot's real job is to
  validate the data pipeline + lock cost; commit to **760M** for publishable separation.

## Blockers to a real multilingual pilot
1. **No multilingual/code data on disk** (B200 or ece) — only `dclm_baseline` (English). Need a small
   CulturaX(~15 lang)+The Stack subset (~10–20 GB) downloaded, tokenized, sharded into the lingua
   format. ece `/home` has 3.6 TB free (96% full but absolute headroom OK for a pilot).
2. **B200 saturated** by the live 1.3B (~90% done, frees in ~12–18 h). **ece-agpu11 GPUs 4–7 are free
   now** and validated working — recommended host for the pilot.

## Next action (recommended)
Stage the multilingual subset on ece-agpu11, write the 4-family 100M configs (reuse the path
overrides above), and run the full ratio-40 pilot (~8–10 h) → first real multilingual BPB +
fertility-slope figure. Then decide 760M.

---

## TEST RUN EXECUTED (2026-06-28, ece-agpu11 GPUs 4–7)
**Multilingual data staged + smoke trained end-to-end — pipeline VALIDATED.**

### Data (built directly on ece; it has internet)
`build_ml_pilot.py` streamed 4 sources into lingua JSONL (`<src>/<src>.chunk.{00..07}.jsonl`,
`{"text":...}`), capped 52 MB/source (~210 MB total, 8 chunks each), tokenized online (no preprocess):

| source | dataset | docs | axis probed |
|---|---|---:|---|
| mlpilot_en | wikipedia 20231101.en | 2,321 | control (low fertility, spaced) |
| mlpilot_fi | wikipedia 20231101.fi | 5,534 | agglutinative, **high BPE fertility** → BPEByte vs Llama |
| mlpilot_zh | wikipedia 20231101.zh | 3,790 | **no whitespace** → BPEByte vs AU-Net word |
| mlpilot_code | codeparrot-clean-valid (Python) | 5,076 | no-space + symbols → both axes |

### Run
`apps/aunet/configs/bpebyte_100M_mlpilot.yaml` (BPEByte root_greedy, 4 sources equal-weight, bs24 ga8,
seq8192, 300-step smoke). Result: **loss 3.82 → 2.46** (steps 10→50, monotone), **iter_time 0.511 s
(identical to English — confirms multilingual = same compute)**, 39.7 GB reserved (fits 80 GB).

### What this proved
✓ multilingual+code text loads & mixes across 4 sources · ✓ byte boundaries (greedy BPE-trie, root)
compute correctly on **no-space zh + code** · ✓ trains (loss falls) · ✓ memory fits · ✓ cost estimate
(0.51 s/step) holds for multilingual.

### Caveat observed
Wall-clock per step ran > the 0.51 s compute (steps lagged real time) — the **online BPE-trie
tokenization is CPU-side** and may be data-bound for zh/code. The full ratio-40 run could be
**data-bound, not compute-bound** → wall-clock possibly above the 1.9 h/model compute estimate. Check
`data_load_time` / raise `prefetch_size` + loader workers in the full run.

### Persisted for the full run
Data at `/home/hwbae/AUNet/data/mlpilot_{en,fi,zh,code}`; config `bpebyte_100M_mlpilot.yaml`; builder
`/home/hwbae/AUNet/build_ml_pilot.py`. To go full: set `steps=13376`, add AU-Net-word + Llama configs
(same 4 sources), add per-language held-out + FLORES BPB eval, produce the fertility-slope figure.

---

## FULL 4-FAMILY RATIO-40 PILOT — LAUNCHED (2026-06-28 23:30, ece-agpu11 GPUs 4–7)
Detached resume-safe orchestrator `runs/ml_pilot_r40/orch_ml_r40.sh` (setsid, pid 610283). Runs the
4 families **sequentially** on GPUs 4–7, each to ratio-40 (84B bytes), stops on failure, `_done_<tag>`
sentinels for resume. Mirrors the existing 1.3B comparison structure (Llama / AU-Net-word / root_greedy
/ online-bt); **BLT-entropy deferred** (needs a trained entropy model on this multilingual data +
~14 h precompute — separate job).

| order | family | module | config | steps | ~compute |
|---|---|---|---|---:|---:|
| 1 | BPEByte root_greedy (leak-free, method) | apps.aunet | `ml_r40_bpebyte_rg.yaml` | 13,376 | 1.9 h |
| 2 | BPEByte online-bt (bt/before_root) | apps.aunet | `ml_r40_bpebyte_bt.yaml` | 13,376 | 1.9 h |
| 3 | AU-Net word (whitespace, `word1`) | apps.aunet | `ml_r40_aunet_word.yaml` | 13,376 | 1.9 h |
| 4 | Llama subword (iso-byte) | apps.main | `ml_r40_llama.yaml` | 17,600 | 2.5 h |

All byte: bs24·ga8 seq8192 (6.29M bytes/opt-step, ratio-40 preserved). Llama: bs32·ga4 seq2048.
**Total ETA ≈ 8–10 h (→ ~07:30–09:30 2026-06-29).** First family compute-bound at 99% GPU util,
loss 3.71 @ step 10. All 4 configs pre-validated (20-step smokes: word loss 3.77, llama 10.48).

**Monitor:** `ssh ece-agpu11 'tail runs/ml_pilot_r40/orch.log'` (per-family DONE/FAIL) and
`.../<tag>/metrics.jsonl` (loss/BPB). **After training:** per-language held-out + FLORES BPB eval →
fertility-slope figure (the headline). For byte models BPB = `loss/out`/ln2; Llama BPB =
`loss/out`/(ln2·bytes_per_token).

### PARALLELIZED across 2 boxes (2026-06-28 23:52) — ETA ~10 h → ~3.8 h
ece-agpu18 is the **same NFS home** (configs/data/run dir all shared) with **all 8 GPUs free**, giving
**3× 4-GPU slices total**. Reassigned via pre-touched `_done` sentinels (no run-dir collisions, each
family writes its own dir):

| slice | host · GPUs | family | driver |
|---|---|---|---|
| A | agpu11 · 4–7 | `bpebyte_rg` → then `aunet_word` | `orch_ml_r40.sh` (skips bt/llama via sentinels) |
| B | agpu18 · 0–3 | `bpebyte_bt` | `orch18.sh` (concurrent) |
| C | agpu18 · 4–7 | `llama` | `orch18.sh` (concurrent) |

All slices confirmed at 99–100% GPU util. Makespan floor = 3.8 h (slice A runs two 1.9 h jobs
sequentially; agpu18 does bt 1.9 h ‖ llama 2.5 h). **ETA ≈ 03:20 2026-06-29.** Monitor agpu18 via
`ssh ece-agpu18 'tail runs/ml_pilot_r40/orch18.log'`.

---

## SIDE ABLATION — down-projection (down_pool): `start` vs `last_byte` (QUEUED 2026-06-29 00:35)
Separate from the multilingual pilot (English DCLM, ratio-5). Isolates the **down-projection**: which
encoder byte-state becomes each patch's trunk token.

| | down_pool | patch_read_delay | trunk token sees |
|---|---|---|---|
| **A0 (existing bpebyte_100M)** | `start` (default) | 0 | encoder state at the patch's **first** byte only |
| **A1 (`bpebyte_100M_A1_lastbyte.yaml`)** | `last_byte` | **1** (required) | encoder state at the patch's **last** byte → summarizes the whole patch interior |

`last_byte` lets the trunk token represent the *content* of the patch (not just its opening byte), but
because the last byte is in the patch's future it **requires `patch_read_delay≥1`** (each position reads
a strictly-earlier patch token) to stay leak-free — that 1-patch read shift is the coupled cost.

**Matched design (confound-free):** watcher `runs/ablation_100M/A_downpool.sh` (pid 213046, agpu18,
fires when GPUs 0–3 free behind bt) runs **both** A0_start and A1_lastbyte at **bs24 ga8** (≡ bs96 ga2),
ratio-5 / 1672 steps, seed 777, English DCLM — only difference is the down-pool + its required delay.
~14 min each → **done ≈ 02:15**. Metric: **ΔBPB = A1 − A0** (BPB = `loss/out`/ln2).
Reference (existing A0 on B200, bs96): `loss/out` **0.9189** → BPB **1.326**.
Monitor: `ssh ece-agpu18 'tail runs/ablation_100M/A_downpool.log'`.
