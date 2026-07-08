# Chinese continue-pretrain plan — BPEByte AU-Net 1.3B

**Goal.** Give the English-trained 1.3B (`bpebyte_br_greedy_root_1.3B`@180k) actual Chinese ability by
warm-starting a new run on a Chinese-heavy corpus — the only path left after every eval-side lever
(tokenizer swap, task choice, letter→cloze format) left it at chance (see `reports/` + memory
`chinese-tokenizer-swap-eval`).

## 0. Why (what the eval sweep established)
- ceval/cmmlu (letter **and** cloze), xcopa_zh, xstorycloze_zh: **all at chance** — model has no Chinese knowledge.
- Chinese BPB (wiki_zh): **2.68** (vs English 0.83) — poor compression of Chinese; and swapping the
  patch tokenizer to Qwen2 at eval *hurt* BPB (+0.21) because the ckpt is **bound to its llama3 training
  boundaries**. Conclusion: a fixed ckpt can't be fixed at eval — must train with Chinese data.

## 1. THE central design decision — which patch tokenizer to TRAIN with
This is the actual experiment, and the proper counterpart to the null eval-swap result:
- **Arm A — llama3 boundaries** (char-level zh patches, ~3 B/patch): continuity baseline, same segmentation the ckpt already knows.
- **Arm B — Qwen2 boundaries** (word-level zh patches, ~4 B/patch): Chinese-aware. Training *and* eval now
  matched, so any benefit of Chinese-aware patching can finally materialize (unlike the eval-only swap).

**Recommendation: run BOTH** — identical data + schedule, differing ONLY in `data.regex.bpe_tokenizer_path`.
That directly answers "does Chinese-aware patching help when trained in?" If compute-limited, do **Arm B
(Qwen2) first** (the hypothesis under test); keep the original llama3 ckpt as the reference.

## 2. Warm-start mechanism (VERIFIED in code)
Weights-only warm-start into a fresh run (train.py:292-295 → `load_from_checkpoint(..., model_key="model")`;
resets RoPE buffers; fresh optimizer + scheduler + step=0):
```yaml
checkpoint:
  path: <NEW_EMPTY_DIR>/checkpoints                     # MUST be empty → blocks auto-resume
  init_ckpt_path: .../bpebyte_br_greedy_root_1.3B/checkpoints/0000180000   # weights only
```
NOTE: the agent-cited `llama_1B_283_extend.yaml` precedent does NOT exist in this tree; rely on the
`init_ckpt_path` hook above (confirmed) — not that config.

## 3. LR schedule (domain adaptation, not from-scratch)
Do **not** use the original 1.65e-3 peak. Converged model + distribution shift → low LR + short warmup:
```yaml
steps: 20000            # pilot (scale for full run)
optim:
  lr: 1.0e-4            # ~6% of original peak; ~6× original lr_min (1.65e-5)
  warmup: 1000          # short re-warmup to absorb the EN→ZH shift (avoid a loss spike)
  lr_min_ratio: 0.1     # gentle cosine decay to 1e-5  (use 1.0 for a flat LR)
  weight_decay: 0.1
  clip: 0.2
```
(If loss plateaus early, bump peak to 2e-4. Flat LR = `lr_min_ratio: 1.0, warmup: 0`, but a small warmup is safer.)

## 4. Data
**Format** (streamed, boundaries computed online — NO offline tokenization):
`<root>/<src>/<src>.chunk.NN.jsonl`, one `{"text": "..."}` UTF-8 object per line. Build with
`scripts/build_ml_pilot.py <root>` (streams a HF dataset → sharded chunks; MAX_DOCS/MAX_MB env caps).

**Corpus.** wiki-zh alone (what the B4 pilot used, ~50 MB) is a *pilot-only* size. For real ability use a
larger zh web corpus — prefer **parquet-native** datasets (datasets 4.x dropped loader scripts, cf. the
cmmlu fix): e.g. `wikimedia/wikipedia 20231101.zh` (clean, ~1.3 GB) for the pilot; a larger web set
(SkyPile / CCI3 / Fineweb-Edu-Chinese) for the full run. Target tens of GB.

**Replay (avoid catastrophic forgetting).** Mix in English DCLM so we don't wipe English:
```yaml
data:
  sources:
    zh_corpus: 0.7        # Chinese
    dclm_baseline_1.0_2shards_shuffled: 0.3   # English replay
```
Measure English BPB before/after to quantify forgetting.

**Throughput caveat.** Qwen2 boundary computation is CPU-side and can be **data-bound on no-whitespace
Chinese** (flagged in `reports/b4_pilot_estimate.md`) — watch tokens/s vs the 0.332 s/step English rate;
`REGEX_SEG_CACHE` will rebuild on the tokenizer change.

## 5. Budget (B200 node, 4 GPUs, ~0.332 s/step, 1.57 M bytes/step)
| phase | steps | bytes | wall-clock / arm |
|-------|------:|------:|-----------------:|
| pilot  | 20k  | ~31 GB  | ~1.8 h |
| full   | 50–100k | ~79–157 GB | ~4.6–9.2 h |
Two arms ≈ double. Gated on GPUs (PoC training currently occupies the node).

## 6. Eval (reuse everything already built) — before vs after, both arms
- **Chinese BPB** (`eval_bpb_zh.yaml`, wiki_zh) — **headline** (sensitive metric; should drop well below 2.68).
- ceval/cmmlu **letter** (`run_zh_ceval_now.sh`/`run_zh_cmmlu.sh`) + **cloze** (`run_zh_cloze*.sh`).
- xcopa_zh + xstorycloze_zh (`run_zh_easy.sh`).
- **English BPB** (DCLM, free via eval_on_val) — forgetting check.
- Key comparison: **Qwen2-trained vs llama3-trained** Chinese metrics → does Chinese-aware patching help when trained-in?

**Success criteria:** Chinese BPB drops materially; ≥1 downstream task clears chance with CI; English BPB
not badly regressed. If Qwen2-arm beats llama3-arm on Chinese, the original tokenizer-swap hypothesis is
vindicated *when properly trained-in*.

## 7. Risks
- Catastrophic forgetting → replay + measure EN BPB.
- Corpus too small/low-quality → wiki-zh is pilot-only; size up for the full run.
- CPU boundary bottleneck on zh → watch throughput.
- B4 pilot's anomalous ~0.025 losses were never verified → sanity-check loss curves in the first 100 steps.
- Warm-start didn't help at 100M (undertrained), but that's from-scratch matching; here it's domain
  adaptation of a *converged* model → warm-start is appropriate.

## 8. Next steps
1. Acquire + shard a real Chinese corpus (`build_ml_pilot.py`); ensure a local DCLM replay source.
2. Write `bpebyte_1.3B_zh_continue_qwen2.yaml` + `…_llama3.yaml` (differ only in `bpe_tokenizer_path`).
3. Pilot 20k steps/arm (GPU-gated behind the PoC training).
4. Eval before/after (§6) → decide full run.
