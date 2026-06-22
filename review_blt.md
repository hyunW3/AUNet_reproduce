# Review: Entropy-patching implementation (BLT-style boundaries in AU-Net)

**Date:** 2026-06-22 · **Branch:** `bpebyte-root-greedy-eval-fixes` · **Status of code:** uncommitted
**Reference plan:** `plan_entropy_patching.md`
**Method:** three independent review passes — (A) causal correctness & leak-freeness, (B)
integration / engineering / committability, (C) research validity & BLT fidelity. Findings that were
flagged independently by ≥2 passes are marked **[converged]** (higher confidence).

## Files reviewed
- `lingua/apps/aunet/data/entropy_patch.py` — core module (rules, `EntropyModel`, `EntropyPatcher`,
  `EntropyIncrementalParser`, calibration CLI)
- `lingua/apps/aunet/data/regex_cutting.py` — wiring (`RegexArgs`, seg-cache sig, `online_byte_boundaries`,
  `_online_levels_mask_bytes`, `make_incremental_bt_parser`)
- `lingua/apps/aunet/data/byte_trie.py` — `prefix_vocab` mode + `ByteTrieIncrementalParser` interface
- `lingua/apps/aunet/test_entropy_patch.py` — tests
- configs: `bpebyte_{100M_v7,300M_v7}_entropy.yaml`, `bpebyte_entropy_{760M,1.3B}_b200.yaml`,
  `apps/main/configs/entropy_byte_50M.yaml`

---

## Bottom line

The **per-byte causal core is sound** — root (patch-start) placement + a causal byte-LM gives genuine
leak-freeness; the wiring faithfully mirrors `root_greedy` across all three surfaces (training
`online_byte_boundaries`, eval `_online_levels_mask_bytes`, generation `make_incremental_bt_parser`);
`hierarchical.py` is untouched as designed. **But the code is not committable or runnable as-is**, and
the headline *train==generate* guarantee is currently **broken** under the configured settings.

---

## CRITICAL

### 1. The core module is silently git-ignored — "DONE (committed)" is false **[converged]**
`git check-ignore -v apps/aunet/data/entropy_patch.py` → `.gitignore:14:data/`. The artifact rule
`data/` matches the *source* dir `apps/aunet/data/`. `byte_trie.py`/`regex_cutting.py` survive only
because they were tracked before that rule existed (`bpe_root_stats.py` already fell through the same
trap). A normal `git add -A && commit` ships the wiring + configs **that import `entropy_patch`** but
**not the module itself** → `ModuleNotFoundError` on any fresh checkout / CI, and the tests fail to
import.
- **Fix (durable):** add `.gitignore` negation under the artifact rule:
  ```
  data/
  !apps/aunet/data/
  !apps/aunet/data/**
  ```
  then `git add apps/aunet/data/entropy_patch.py`. A one-off `git add -f` leaves the trap for the next file.

### 2. Streaming generation ignores `newline_reset` → train ≠ generate on structured/MCQ text **[converged]**
`EntropyIncrementalParser._next_entropy`/`feed` (`entropy_patch.py:224-247`) always feed the full
`[BOS]+buf` and never reset context at `\n`, and never reference `patcher.newline_reset`. The
batch/data path (`EntropyModel.entropies(..., newline_reset=True)`, `entropy_patch.py:143-162`) *does*
reset per newline. The plan mandates `entropy_newline_reset: true` at 760M/1.3B, so the model trains on
newline-reset boundaries and generates with non-reset ones — **exactly on the text the drift guard
exists for**. Realizes the plan §7 risk as a live bug; distorts MCQ eval patch sizes → eval FLOPs.
- **Fix:** on feeding `\n`, reset the effective context start and set `_prev_H=None`; add a parity test
  with `newline_reset=True`.

### 3. Chunked batch entropies shift RoPE positions → batch ≠ streaming on long docs
`EntropyModel._entropies_one` (`entropy_patch.py:128-141`) feeds `[BOS]+bytes[ctx:end]` with
`tok_idx=None`, so a byte at absolute position *t* gets RoPE position `t-ctx+1` in chunk ≥2 but `t+1`
in streaming / single-pass. Receptive *content* is correct (overlap ≥ sliding_window; the off-by-one
`H[(start-ctx):(end-ctx)]` was verified correct), but the *positions* differ → H differs → boundaries
diverge past the first `chunk` (8192) bytes. DCLM docs routinely exceed this.
- **Fix:** pass absolute positions via `tok_idx`, or make streaming use the identical chunk/overlap
  windowing.

### 4. The entropy model is built on CUDA inside a forked dataloader worker
`data.py` runs `tokenize()` → `RegexPool` → `entropy_patcher` inside a fork-started producer process.
`EntropyModel.__init__` → `load_consolidated_model_and_tokenizer` does an **unconditional `model.cuda()`**
(`apps/main/generate.py:427`); the `device=` arg threaded through `EntropyPatcher`/`EntropyModel` is
silently ignored. Result: classic "cannot re-initialize CUDA in a forked subprocess," or N× VRAM (one
model per worker). As written, entropy mode **cannot run in the training dataloader** — only
construction was ever verified, never a forward pass.
- **Fix:** force CPU in workers (needs a `device`/`map_location` param into the loader, or `.to('cpu')`
  after load), **or** precompute boundaries offline into `REGEX_SEG_CACHE` (the plan's intended path)
  and never construct the model in a worker.

---

## MAJOR

### 5. Calibration is per-line; the data path is per-document with overlap chunking
The CLI (`entropy_patch.py:291-298`) computes entropies one `line.encode()` at a time, each from a
fresh BOS. The data path segments whole documents with `overlap=512` chunking and (optionally)
per-newline reset. Per-line calibration over-counts boundaries (BOS-primed line starts spike entropy)
and never exercises overlap/reset, so **θ tuned by the CLI will not hit 4.5 bytes/patch on the data the
model actually sees** — threatening the *iso-bytes-per-patch* premise the whole fair comparison rests on
(plan §6.1: "trunk FLOPs match per-step").
- **Fix:** calibrate on document-shaped inputs through the same `entropies(..., newline_reset=...)`
  path; verify achieved bytes/patch on held-out DCLM **and** an MCQ-shaped probe before any FLOP claim.

### 6. Tests do not test the load-bearing claims **[converged]**
`test_streaming_matches_batch` (`test_entropy_patch.py:71-83`) injects a *scripted* `_next_entropy`
returning per-position constants, so batch==streaming holds **by construction** — it cannot catch #2 or
#3, never touches the model, the BOS alignment, or `newline_reset`. There is **no leak probe** (plan §8
step 3) and no real-model batch-vs-stream test. The monotonic-rule calibration is also untested. The
"tests green / parity verified" claim (plan §8) is unsubstantiated for the model path.
- **Fix:** add a gated (`ENTROPY_CKPT`) test asserting (a) causality
  `entropies(seq)[:k] == entropies(seq[:k])`, and (b) streaming `feed` reproduces batch boundaries on
  the model's own entropies, including a newline and a >`chunk` sequence.

### 7. Seg-cache signature omits the entropy-model weights hash
`regex_cutting.py:166-171` keys `_seg_sig` on `entropy_model_path / rule / theta / newline_reset` but
**not** a hash of the checkpoint. Configs hardcode one fixed path; retraining to that path (or
recalibrating without changing θ) serves **stale cached boundaries** → silent train/eval contamination.
Plan §6.7 explicitly wants the entropy-model hash embedded.
- **Fix:** fold `hashlib` of `consolidated.pth` (or mtime/size) into `_seg_sig`.

### 8. `entropy_theta: 0.0` placeholder is unguarded
All four configs ship `entropy_theta: 0.0`. Missing model path fails loudly (good), but θ=0.0 does not:
with `rule: monotonic`, `ΔH>0` fires on nearly every byte → ~1–2 bytes/patch **degenerate run that
looks valid**. If someone fills the model path but forgets to calibrate θ, training proceeds silently
wrong (non-iso-FLOP).
- **Fix:** assert/log achieved bytes/patch on the first batch in a plausible band.

### 9. Monotonic rule is a simplification of BLT — disclose, don't claim exact reproduction
Implemented as adjacent difference `H_t − H_{t-1} > θ` (`entropy_patch.py:50`). BLT's "approximate
monotonicity" is relative to the running in-patch minimum, not the adjacent byte. Direction is
defensible (fires on the entropy rise that breaks the in-patch decay), but the adjacent-difference form
gives different boundaries on slowly-rising entropy. Also: `calibrate_theta` assumes bytes/patch is
monotone in θ — true for both rules, but the function is a **step**, so bisection can land arbitrarily
on a flat plateau and miss the target beyond the test's tolerance (monotonic calibration is untested).
Separately, confirm the consolidated entropy model loads with `sliding_window=512` (else
`overlap=max(512,sw)=512` is wrong and attention is silently full-context).

---

## MINOR / disclosure

- **`prefix_vocab` mode in `byte_trie.py` is unrelated scope creep** — a separate trie-tokenization
  experiment (`prefix_vocab_tokenize_boundaries`, `mode=="prefix_vocab"`), no entropy code, no test,
  and it confusingly shares the "v7" label with the entropy configs. Split into its own commit.
- **Entropy units are nats** (`log_softmax`/`exp`; max ≈ ln 258 ≈ 5.55). `calibrate_theta` `hi=12` is
  wasteful for the global rule; convert when comparing θ to BLT's bit-quoted values.
- **bf16 inference can flip near-threshold boundaries** between short (streaming) and long (batch)
  forwards. Even after #2/#3, exact parity may need fp32 entropy inference, or document a near-threshold
  tolerance.
- **Entropy model trained with EOS between docs but fed BOS-only chunks at inference** — minor tail
  distribution mismatch.
- **`50M` filename vs ~44M actual / BLT's 100M default** — state the saturated-point choice explicitly
  in the paper (window 512 + RoPE θ=500000 + same DCLM shard are correct/fair).
- **`bpe_context_prefix>0` + entropy mode would crash** on the `None` tokenizer (offline branch).
  Latent (all configs use 0); add an assert.
- **`committed_levels()` returns the full list each step** (margin-0 = everything committed) — correct
  but differs from the trie parser (`levels[:_committed]`); add a one-line comment that this is
  intentional.
- **No fairness accounting in code/eval JSON** — nothing emits the entropy model's one-time training
  FLOPs, per-byte O(n²) generation overhead, or the segmenter hash/θ/rule (same gap the audit flagged
  for `root_greedy`).

---

## Verified correct (no action)
- Off-by-one in `_entropies_one`: `out[t]` = entropy of `p(x_t|x_{<t})` — worked through chunk 2.
- Overlap ≥ sliding_window guarantees full receptive *content*.
- Root placement / index-0 forcing: `entropy_patch_starts` returns t≥1; index 0 forced by
  `online_levels_mask`; `_online_levels_mask_bytes` peels BOS.
- Leak-freeness on the placement axis (up-pool reads `repeat_idx[e-1]`, excludes boundary at e).
- `commit_margin=0` soundness: the rule never revises a past decision, so the bt-loop rollback never fires.
- Dispatch parity: entropy handled in all three surfaces `root_greedy` is, with eager root-placement /
  no-committed-view asserts; `generate_bt.py` needs no change.
- Configs are genuine iso-segmentation swaps vs their `v4_root_greedy` siblings (only the `regex` block
  + comments differ; `bpe_tokenizer_path` correctly dropped; `patch_read_delay: 0`).

---

## Recommended order before running beyond the 100M wiring sanity check
1. Fix `.gitignore` and actually commit `entropy_patch.py` (decide `prefix_vocab` split). *(#1)*
2. Decide entropy-model execution model — offline-precompute vs CPU-in-worker; **blocks all runs**. *(#4)*
3. Make streaming replicate batch: newline reset + chunked-RoPE parity. *(#2, #3)*
4. Fix calibration to document-level; add real-model parity + leak-probe tests. *(#5, #6)*
5. Add seg-cache weight hash and a θ / bytes-per-patch guard. *(#7, #8)*
