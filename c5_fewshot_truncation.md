# C5 — Few-shot Capacity Asymmetry (BoolQ): Quantification + Fair-Comparison Plan

**Date:** 2026-06-22
**Scope:** CPU tokenization measurement + plan only. No GPU jobs launched.
**TL;DR:** At **10-shot**, **15.90%** of BoolQ prompts overflow the byte model's
**8192-byte** budget (left-truncated), while **0.00%** overflow Llama's **4096-token**
budget. The asymmetry vanishes at **<=5 shots** (byte truncation 0.03% at 5-shot, 0%
at 0/1/2-shot). **Recommended fair operating point: 0-shot** (clean for both, and the
shot count actually used in the deployed `eval_boolq_piqa_b200.yaml`); **5-shot** is the
highest count that is still effectively clean (1/3270 byte docs truncated).

---

## 1. Where truncation happens (code trace)

Both models share identical left-truncation logic; only the *budget unit* differs.

**Byte model** — `apps/aunet/generate.py:391-400` (`Generator.generate`, the path used by
BoolQ since BoolQ is `output_type: multiple_choice` → `loglikelihood`):
```python
max_prompt_len = self.max_prompt_len or min(
    self.model.encoders[0].max_seqlen - self.max_gen_len,   # 8192 - 1
    self.max_tokens - self.max_gen_len,                     # 16384 - 1
)
prompts = [p[-max_prompt_len:] for p in prompts]            # keep TAIL (left-truncate)
```
- During `loglikelihood` (`eval.py:414-419`) `max_gen_len` is temporarily set to **1**.
- `encoders[0].max_seqlen` comes from the byte config `seq_len: 8192`
  (`bpebyte_root_greedy_1.3B_b200.yaml:49`, `bpebyte_br_bt_online_p2048_1.3B.yaml:49`).
- `eval_boolq_piqa_b200.yaml` raises `generator.max_tokens: 16384`, **but** the `min(...)`
  still binds on `seq_len-1 = 8191`. **Effective byte prompt budget ≈ 8192 bytes** regardless
  of `max_tokens`. (A second, tighter limit — the patch trunk `max_seqlens: [-1, 3200]` — only
  caps *generation* packing in the `generate_until` path, not loglikelihood scoring.)
- Units are **bytes**: `tokenizer.encode(...)` for the byte model emits one id per UTF-8 byte
  (+1 BOS), so `len(p)` ≈ byte length.

**Llama** — `apps/main/generate.py:336-344` (identical structure):
```python
max_seqlen = self.model.max_seqlen                          # = seq_len = 4096
max_prompt_len = self.max_prompt_len or min(
    max_seqlen - self.max_gen_len,                          # 4096 - 1
    self.max_tokens - self.max_gen_len,                     # 16384 - 1
)
prompts = [p[-max_prompt_len:] for p in prompts]
```
- `seq_len: 4096` (`apps/main/configs/llama_1B.yaml:33`). Units are **subword tokens**
  (tiktoken `cl_toplang_128k`). **Effective Llama prompt budget = 4096 tokens.**

So the asymmetry is structural: **8192 BYTES vs 4096 TOKENS**. With BoolQ's English text at
roughly ~4 bytes/token (cl100k), 4096 tokens corresponds to ~16k bytes of headroom for Llama,
versus the byte model's 8k-byte wall — Llama has ~2x the effective context for this corpus.

---

## 2. Measured truncation rates (BoolQ validation, 3270 docs)

- Template (exact, from `lm_eval/tasks/super_glue/boolq/default.yaml`):
  `doc_to_text = "{{passage}}\nQuestion: {{question}}?\nAnswer:"`, choices `["no","yes"]`,
  target_delimiter `" "`, fewshot_delimiter `"\n\n"`, few-shot drawn from the **train** split.
- Sampling reproduced exactly per lm-eval `ContextSampler` (`lm_eval/api/samplers.py:26,57`):
  a single `Random(1234)` (default `fewshot_random_seed=1234`) drawn sequentially per doc.
- Byte length = `len(full.encode("utf-8"))` vs **8192**.
- Llama token length = `len(cl100k_base.encode(full))` vs **4096**.
  (cl100k is a proxy for the deployed 128k `cl_toplang_128k.tiktoken`, which is not present on
  this host; for English prose token counts are within a few % — and Llama's margin to 4096 is
  enormous, so the 0% conclusion is robust to tokenizer choice.)

| shots | byte > 8192 B | **byte trunc %** | tok > 4096 | **Llama trunc %** | byte p50 / p95 / max (B) | Llama p50 / max (tok) |
|------:|-------------:|-----------------:|-----------:|------------------:|--------------------------|-----------------------|
| 0     | 0 / 3270     | **0.00%**        | 0          | **0.00%**         | 590 / 1226 / 4871        | 131 / 1164            |
| 1     | 0 / 3270     | **0.00%**        | 0          | **0.00%**         | 1246 / 2116 / 5048       | 272 / 1208            |
| 2     | 0 / 3270     | **0.00%**        | 0          | **0.00%**         | 1901 / 2953 / 6096       | 417 / 1425            |
| 5     | 1 / 3270     | **0.03%**        | 0          | **0.00%**         | 3864 / 5350 / 9168       | 848 / 1955            |
| 10    | 520 / 3270   | **15.90%**       | 0          | **0.00%**         | 7114 / 9074 / 13470      | 1562 / 2960           |

**On the "17.57%" prior figure:** the structural finding reproduces (10-shot: ~16% byte vs 0%
Llama). The exact percentage is sensitive to the few-shot RNG seed and to whether the scored
continuation length is included; this clean run with the lm-eval default seed (1234) and the
gold continuation yields **15.90%**. Either way the conclusion is identical: **byte 10-shot
BoolQ is materially truncated; Llama 10-shot is not.** The cross-model 10-shot BoolQ numbers
are therefore not a clean comparison — ~16% of the byte model's prompts lost their leading
few-shot exemplars (and, for the longest passages, part of the question/passage itself).

Reproduce: `lingua/.venv/bin/python` on `/tmp/c5_measure2.py` (single shared `Random(1234)`,
BoolQ from `aps/super_glue` cache).

---

## 3. Largest clean shot count for the byte model

- **0, 1, 2 shots → 0.00% byte truncation** (and 0% Llama).
- **5 shots → 0.03% (1/3270)** byte truncation — effectively clean.
- **10 shots → 15.90%** byte truncation — not clean.

The **largest fair operating point is 5-shot** (a single borderline doc); the **safest /
already-deployed point is 0-shot**.

---

## 4. Fair-comparison plan (no GPU executed here)

**Recommendation (in priority order):**

1. **Report BoolQ at 0-shot for cross-model comparison.** Both models truncate 0% → directly
   comparable. This is already the configuration in `eval_boolq_piqa_b200.yaml`
   (no global `num_fewshot`; the standard `boolq` task defaults to 0-shot). If the headline
   table currently cites 10-shot BoolQ, switch it to 0-shot or add the 0-shot row as the
   clean comparator.

2. **If a few-shot number is required, use 5-shot** (highest count clean for both: byte 0.03%,
   Llama 0%). Avoid 10-shot for any byte-vs-Llama BoolQ claim.

3. **If 10-shot must be reported** (e.g. matching an external leaderboard), add the caveat
   explicitly and report it **two ways**: (a) full-set accuracy with the note that ~15.9% of
   byte prompts were left-truncated, and (b) **accuracy on the non-truncated subset only**
   (the 2750 docs ≤ 8192 B) for *both* models, so the comparison is over identical, untruncated
   inputs. Do **not** present raw full-set 10-shot byte-vs-Llama as apples-to-apples.

4. **(Optional, separate effort) Remove the asymmetry at the source** by evaluating the byte
   model at a byte budget that matches Llama's effective context (~16k B ≈ 4096 tok). This
   requires raising the byte `seq_len`/`encoders[0].max_seqlen` to ≥ ~16384 (the `max_tokens`
   override alone does **not** help — the `min(seq_len-1, ...)` binds on seq_len). That changes
   the model's eval-time context window and is a heavier change; the 0-shot report above is the
   cheap, clean fix.

### Ready-to-run commands (launch when a GPU frees)

Fair **0-shot** BoolQ (clean for both; `num_fewshot=0` is explicit for clarity). Standard MC
`boolq` task (loglikelihood, `acc`). Edit `CKPT` only if a different variant is wanted.

**Byte model** (e.g. root_greedy 1.3B @180k):
```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua
source .venv/bin/activate
CKPT=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000
DUMP=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/c5_boolq_fair/byte_root_greedy_0shot
torchrun --nproc-per-node 4 --master-port 29541 -m apps.aunet.eval \
  config=apps/aunet/configs/eval_boolq_piqa_b200.yaml \
  harness.tasks='["boolq"]' harness.num_fewshot=0 \
  validation=null ckpt_dir="$CKPT" dump_dir="$DUMP"
```

**Llama** (1.8B paper @60k):
```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua
source .venv/bin/activate
LCKPT=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/llama_1.8B_paper/checkpoints/0000060000
LDUMP=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/c5_boolq_fair/llama_1.8B_0shot
torchrun --nproc-per-node 4 --master-port 29542 -m apps.main.eval \
  config=apps/main/configs/eval_gen_mc_llama_1.8B_b200.yaml \
  harness.tasks='["boolq"]' harness.num_fewshot=0 \
  validation=null ckpt_dir="$LCKPT" dump_dir="$LDUMP"
```

To instead run the **5-shot** fair point, change `harness.num_fewshot=0` → `harness.num_fewshot=5`
in **both** commands (seed defaults to `fewshot_random_seed=1234` for both harnesses, so the
exemplars match). 5-shot is clean for both (byte 0.03%, Llama 0%); do not use 10-shot for the
cross-model claim.

**Notes**
- Both commands use `--nproc-per-node 4`; bump `--master-port` if a port is busy.
- `eval_boolq_piqa_b200.yaml` sets `generator.max_tokens: 16384`, but the byte budget stays at
  the 8192-B `seq_len` wall (see §1) — fine for 0/5-shot where nothing truncates.
- Verify the byte checkpoint path (`bpebyte_br_greedy_root_1.3B`, or `bpebyte_br_bt_online_1.3B`
  @180k for the online-bt variant) before launching; both exist on disk.
