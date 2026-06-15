# PoC: BPEByte boundary-handling ablation (100M / 2.3B tokens, ratio-5)

Status: code complete + verified, 4 trainings launched 2026-06-10. Results filled on completion.

## 1. Question

BPEByte (the AU-Net hierarchical byte model) consumes raw bytes; "tokenization" only sets the
**patch boundaries** (the pooling schedule), not the input. We have a family of boundary schemes and
don't know their relative quality, especially the **train/gen gap**: the boundaries used at training
and the boundaries the autoregressive decoder can actually reproduce differ. This PoC trains four
schemes at matched scale (100M params, ~2.3B tokens / ratio-5) and measures where they diverge.

The informative metric is the **mc_gen-vs-mc gap**: MC (HellaSwag/ARC-Easy loglikelihood, `acc_norm`)
only *ranks* supplied options; mc_gen (the `*_gen` tasks) makes the model *generate* the answer
letter (`exact_match`). A scheme whose decode-time boundaries match its training boundaries should
degrade little from mc to mc_gen; one that leaks future structure into training (offline) should rank
well but collapse when forced to generate.

## 2. The variants

| # | Name | `bpe_online` | mode | placement | committed_view | what it isolates |
|---|------|----|----|----|----|----|
| **V1** | Full fix (delayed-masked) | true | bt | before_root | **true** (exact) | closing the train/gen availability gap |
| **V2** | Current online BPEByte | true | bt | before_root | false | the online baseline (gap present) |
| **V3** | Offline tokenization | false | — | — | — | causality (offline leaks the future) |
| **V4** | Root + no backtracking | true | greedy | root | false | placement + backtracking |
| **V5** | Distilled predictor (S1) | true | distilled | before_root | false | imitating offline BPE with a zero-lag causal selector |
| **V6** | Prefix-free vocab (S3) | true | prefix_free | before_root | false | instant-decidable cuts (no munch lookahead, no revision) |

All variants share the **same 100M model, data, optimizer, and step count** — only the boundary
scheme changes. V1 = V2 + the exact delayed-mask data view. V3 is the original (non-causal)
AU-Net patching. V4 is a different causal regime (greedy, boundary at next-token-start).
V5/V6 (added 2026-06-10, queued behind v2-v4 via `run_ablation_100M_v56.sh`) test the Phase-3
"BPE as boundary selector" replacements (generation_available_trick_on_offline_BPEByte.md):
V5 = k-gram backoff classifier distilled from offline BPE cuts (held-out imitation acc 0.938,
bytes/patch 4.61 vs teacher 4.67; train mask == streaming mask by construction, zero commitment
lag); V6 = frequency-pruned prefix-free vocab (27,065 tokens, bytes/patch 3.92; cut the instant
a kept token completes, dead-end fallback, zero revision). Artifacts in `tokenizer/boundary_v56/`
(rebuild: `python -m apps.aunet.data.boundary_distill`).

## 3. Model & budget (validated)

- **Model:** dims `[512, 768]`, layers `[3, 10]` (3 enc + 3 dec + 10 trunk), head_dims `[64, 128]`,
  sliding `[512, 4096]`, `multiple_of 256`, `simple_indexed_matmul`. Built param count =
  **98,591,488 (98.6M)** — byte level frozen at dim 512, only the trunk scaled from the 1.3B config.
- **Budget:** `steps 2229`, `batch 72 × grad_acc 2 × 4 GPU × seq 8192 = 4,718,592 bytes/step` →
  **10.5B bytes ≈ 2.3B LLaMA tokens** (data-to-model **ratio 5** — halved from ratio-10 to cut
  wall-clock, since the byte encoder/decoder are frozen at dim 512 and dominate step time regardless
  of trunk size). Microbatch **72** (3× the initial 24) fills the otherwise-idle B200 memory (was
  ~22%) and amortizes per-step overhead → fewer, fuller steps. `lr 2e-3`, `warmup 133`,
  `lr_min_ratio 0.01`, global batch 576. Identical across all four variants, so the cross-variant
  comparison stays fair. (Note: gb 576 with lr 2e-3 is on the large-batch / low-lr side — fine for a
  relative ablation, would want lr scaling for an absolute-quality run.)
- **Eval:** clean only — `hellaswag`, `arc_easy`, `hellaswag_gen`, `arc_easy_gen` (milestone 1.0,
  inline). Plus held-out BPB from the validation stream.

Configs: `apps/aunet/configs/bpebyte_100M_{v1_committed,v2_online,v3_offline,v4_root_greedy}.yaml`.
Run sequentially on 4×B200 via `run_ablation_100M.sh` (ece-agpu18 prefix-2048 left running untouched).

## 4. Code changes

### Change 1 — exact `committed_patch_idx` as a 3rd data view (V1)
The streaming decoder commits each patch boundary ~1 patch late (it holds `commit_margin` token-ends
speculative). Training normally reads each patch the moment it closes → an availability gap. The fix
feeds the model, as a 3rd data row, the **settled committed-patch index** the decoder actually has,
used as `repeat_idx` in `up()`'s teacher-forced branch (replacing `cumsum(mask)`); down-pooling and
the trunk stay on the final mask (causal, correct extents) → train == decode, gap = 0.

- `regex_cutting.py`: `RegexArgs.bpe_online_committed_view`; `RegexPool.online_committed_patch_idx()`
  — computes the per-byte settled frontier in **O(n)** by re-tokenizing only the speculative tail from
  the last committed token boundary (a true boundary, so the trie match is exact), capped at the final
  cumsum so it is strictly causal and ≤ the cumsum index it replaces. (A naive per-byte full-window
  re-tokenization was 4.7 s/doc — 96× too slow; the tail-only version is ~50 ms/doc worst case and is
  fully hidden by the async prefetch.)
- `data.py`: `tokenize()` stacks a 3rd row when enabled; `batch_and_shuffle_prefetched_sequences`
  carries an `n_rows` axis and yields a 4-tuple; `build_dataloader` derives `n_rows`.
- `train.py`: unpacks by arity, passes `committed_patch_idx` to `model(...)` and the probe.
- `hierarchical.py`: `forward(..., committed_patch_idx=None)` → `SimpleTransition.up()` uses it as the
  teacher-forced `repeat_idx` (base-0 normalized, clamped to `#patches-1`); generation branch untouched.

Default off → 2-row data, bit-identical to before.

### Change 2 — "root" placement (V4)
`bpe_online_placement: before_root|root`. before_root puts the boundary on the token's last byte
(`e-1`); root at the next-token start (`e`, clamped to the last byte for the final token).
- `regex_cutting.py`: placement-aware offset in `online_byte_boundaries` and
  `_online_levels_mask_bytes`; threaded through `make_incremental_bt_parser`.
- `byte_trie.py`: `ByteTrieIncrementalParser(..., placement=...)` `_window_mask` uses the same offset.
- `generate_bt.py`: `OFFSET` stays 1 (BOS shift only) — placement lives entirely in the parser, so
  generation reproduces the trained boundaries.

Greedy mode ("no backtracking") was already supported (`bpe_online_mode: greedy` → commit_margin 1).

## 5. Verification

- **CPU (49 checks, all pass):** length parity for the 3rd row; committed index is monotone, causal,
  and ≤ the cumsum repeat_idx for {bt,greedy}×{before_root,root}; streaming `finalize()` == batch mask
  for both placements; root != before_root (boundary shifted by 1); the fast committed index matches
  the slow streaming parser everywhere except a rare bt over-commit, which the causal cap removes.
- **GPU smoke (1 GPU, 15 steps each):** V1 committed_view and V4 root+greedy both train end-to-end
  through the real fmha forward with finite, decreasing loss. (A line-627 `existing_saves[-1]` error in
  the smoke is a config artifact — no checkpoint saved + eval-at-final-step — and does not occur in the
  real configs, which save at milestone 1.0.)
- **Throughput:** V1 (committed_view, seq 8192, 4 GPU) runs at `data: ~0.007s` vs `iter: ~0.6s` even
  after the prefetch buffer drains — not data-bound.
- **Sizing:** built model = 98.6M params.

## 6. Results

_To be filled when the four trainings + clean evals complete (~2.5–3.5 h)._

| variant | HellaSwag mc (acc_norm) | HellaSwag gen (EM) | ARC-E mc | ARC-E gen | mc_gen−mc gap | held-out BPB |
|---|---|---|---|---|---|---|
| V1 full fix | | | | | | |
| V2 online | | | | | | |
| V3 offline | | | | | | |
| V4 root+greedy | | | | | | |

**Expected:** at 100M / 21B bytes the absolute MC numbers are near-random — the **relative
mc_gen-vs-mc divergence** is the result. V3 (offline) should rank well (mc) but collapse on mc_gen
(non-causal boundaries the decoder can't reproduce). V1 should show a **smaller mc_gen-vs-mc gap than
V2** if the exact delayed-mask closes the availability gap. V4 characterizes the alternative
greedy/root regime.

See `Delayed_mask_training.md` for the gap analysis and `run_ablation_100M.sh` for the launch.
