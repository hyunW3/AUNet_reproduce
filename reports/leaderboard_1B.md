# 1.3B Leaderboard — BPEByte hybrid vs baselines

_Updated 2026-07-09._ Downstream = 5-bench (`acc_norm` for HellaSwag/ARC-Easy/PIQA, `acc`
for BoolQ/WinoGrande), each model evaluated in its **native tokenization regime**. Two BPB
measures are shown:
- **BPB train-log** = per-byte training `loss/out` ÷ ln2 (full-context, seq_len 8192; Llama
  subword → `loss/(ln2·4.5376 bytes/token)`). Read from `metrics.jsonl` — available at **any
  training %** for every model.
- **BPB 512-cap** = held-out `eval_hybrid_bpb` (512-byte cap, exact sdpa). Byte models only.
  The **hybrid shows Full / Decode** over its native uniform (N/3, 2N/3) boundary.

gr & hybrid downstream on A100 (native + `force_bpe_online_mode=greedy`); AU-Net & Llama = canonical B200.

## Main table

| model (1.3B) | train | HellaSwag | ARC-E | BoolQ | PIQA | WinoGrande | **Avg** | **BPB train-log** ⁴ | **BPB 512-cap** ⁵ |
|---|---|---|---|---|---|---|---|---|---|
| **BPEByte root_greedy (gr)** | 30% | 0.453 | 0.471 | 0.435 | 0.669 | 0.530 | **0.512** | 0.965 | 1.065 |
| **BPEByte root_greedy (gr)** | 60% | 0.488 | 0.576 | 0.527 | 0.701 | 0.575 | **0.573** | 0.935 | 1.022 |
| **BPEByte root_greedy (gr)** | 100% | 0.573 | 0.674 | 0.635 | 0.739 | 0.612 | **0.647** | 0.853 | 0.933 |
| **BPEByte hybrid(leaf-offline/root-online)** | 30% | 0.447 | 0.478 | 0.539 | 0.663 | 0.549 | **0.535** | Full 0.938 † | Full **1.026** / Decode **0.997** |
| **AU-Net** | 100% | 0.627 | 0.657 | 0.611 | 0.742 | 0.615 | **0.650** | 0.872 | not measured |
| **Llama** (1.8B) | 100% | 0.622 | 0.654 | 0.635 | 0.753 | 0.616 | **0.656** | 0.872 | N/A |

† Hybrid train-log BPB is **Full only** — the training objective is over the whole sequence, so it
includes the non-causal offline-leaf **prefill** region (leak-contaminated → optimistically low).
The **decode-part BPB is not recorded** in the training log or checkpoint; it is an **eval-only**
quantity (recomputed from the content-hash region by `eval_hybrid_bpb`) → see the 512-cap column.

## 30% training-stage BPB (train-log) — all four families
| model @ 30% | step | train `loss/out` | **BPB** |
|---|---|---|---|
| Llama (1.8B) | 18k | 2.888 /tok | **0.918** |
| BPEByte hybrid(leaf-offline/root-online) | 54k | 0.6499 /byte | Full **0.938** |
| BPEByte root_greedy (gr) | 54k | 0.669 /byte | **0.965** |
| AU-Net | 54k | 0.6876 /byte | **0.992** |

_(AU-Net/Llama kept only their final checkpoints, but their training LOGS pass through 30%, so the
train-log BPB is available for all four even though held-out eval at 30% is not.)_

## Hybrid 512-cap BPB across b/N (native boundary = uniform (N/3, 2N/3), 95% CI)

| b/N | 0.00 | 0.25 | 0.333 | 0.417 | 0.50 | 0.583 | 0.667 | 0.75 |
|---|---|---|---|---|---|---|---|---|
| **Full** | 1.108 | 1.056 | 1.045 | 1.034 | 1.026 | 1.016 | 1.007 | 0.998 |
| **Decode** | 1.108 | 1.011 | 1.003 | 0.997 | 0.996 | 0.994 | 0.997 | 1.002 |

→ Table value = average over (N/3, 2N/3): **Full 1.026 / Decode 0.997**.

## Takeaways
- **Matched 30% (both native):** hybrid **0.535 > gr 0.512** downstream (+0.023); hybrid Decode BPB **0.997** (512-cap) / Full 0.938 (train-log).
- **gr trajectory:** downstream 0.512 → 0.573 → 0.647; train-log BPB 0.965 → 0.935 → 0.853.
- **hybrid@60% / 100%** fill in as training progresses (currently ~45% trained; 108k ≈ 15h, 180k ≈ 2d).
- Train-log BPB at 100% matches held-out full-context (gr 0.853 train ≈ 0.858 scaling), so it's a sound proxy.

## Footnotes
1. **512-cap vs train-log are different measures** (512-cap = held-out, capped context; train-log = training loss, full context) — compare within a column, not across.
2. **AU-Net 512-cap** not run (`aunet2_1.3B` not staged on ece); **Llama 512-cap = N/A** (byte-only scorer, Llama is subword).
3. Held-out **full-context** BPB (scaling, B200; 100% only): gr **0.858**, AU-Net **0.866**, Llama **0.840**.
4. BPB train-log: byte models = `loss/out`/ln2; Llama = `loss/out`/(ln2·4.5376). Full-context; on train data (single-step, mildly noisy).
5. BPB 512-cap: hybrid = Full/Decode; gr = single (pure greedy). Only gr & hybrid measured.
6. AU-Net@30% / Llama@30% held-out evals impossible (only final checkpoints kept); **eval fix:** ece/A100 byte downstream requires `force_bpe_online_mode=greedy`.
