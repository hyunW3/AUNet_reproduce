# Data-to-model ratio of our trained ladders (paper's γ formula)

Computes the **data-to-model ratio γ** for our trained ladders — BPEByte online root_greedy (rg),
AU-Net-word, and Llama (subword) — at 100M / 300M / 760M / 1.3B, using the definition from the
AU-Net paper ([arXiv:2506.14761](https://arxiv.org/pdf/2506.14761), §2.3), **not** the naive
tokens÷params.

## The paper's definition

The paper follows Bi et al. 2024 (DeepSeek): "model size" is **FLOPs per input-unit**, not raw
params. For a decoder transformer:

```
F_model/input-unit = 6·N_non-embed_params + 6·d·L·S      (linear term + attention term)
γ_input-unit       = N_input-unit / F_model/input-unit
```

Since the linear term dominates, **γ ≈ N_data / (6·N_non-embed)**. The paper always expresses γ in
**LLaMa-3 tokens** (γ_token); for byte models the byte budget is converted to tokens via the DCLM /
LLaMa-3 compression factor **k ≈ 4.56** (`N_token = N_byte / k`). The byte-unit and token-unit
ratios are related by `γ_byte = k²·γ_token`.

- **Numerator N** = total training data actually consumed (`optim/total_tokens` from `metrics.jsonl`
  = bytes for a byte model), converted to LLaMa-3 tokens.
- **Denominator** = `6 × N_non-embed` (logged `Model size:`; byte-model embeddings ≈ 256·d are
  negligible, so `Model size ≈ N_non-embed`).

## Numbers — rg (BPEByte root_greedy, byte)

| model | N_non-embed | N_bytes (actual) | N_tokens (÷4.56) | **γ_token = N_tok / (6·N)** | γ if N=bytes | naive tok/param |
|---|---:|---:|---:|---:|---:|---:|
| **100M** | 98,591,488 | 21.01 B | 4.61 B | **7.79** | 35.5 | 46.8 |
| **300M** | 296,659,712 | 62.29 B | 13.66 B | **7.67** | 35.0 | 46.0 |
| **760M** | 714,426,880 | 142.97 B | 31.35 B | **7.32** | 33.4 | 43.9 |
| **1.3B** | 1,324,203,008 | 283.12 B | 62.09 B | **7.81** | 35.6 | 46.9 |

Sources: `runs/small/cmp_g10/{v4_root_greedy,rg_300M,rg_760M}`,
`runs/1.3B/bpebyte_br_greedy_root_1.3B` — final `optim/total_tokens` + logged `Model size`.

## Peers — AU-Net-word and Llama

**AU-Net-word** shares the exact byte trunk with rg (same dims/layers → same `Model size`) and
consumed the **identical byte budget** at every scale (verified: same `optim/total_tokens` to the
byte). So **its γ is identical to rg's** — 7.79 / 7.67 / 7.32 / 7.81.

**Llama (subword)** differs on two axes: (1) it trains on **LLaMa-3 tokens directly**, so
`optim/total_tokens` is already tokens — **no ÷k conversion**; (2) its **128k-vocab embedding is
huge and untied**, so `N_non-embed = Model_size − 2·vocab·dim` must subtract it before the formula.

| scale | config | Model size (total) | − embedding (2·128256·d) | **N_non-embed** | N_tokens (actual) | **γ_token** |
|---|---|---:|---:|---:|---:|---:|
| **100M** | dim768/14L | 296,113,920 | 196,921,344 | 99,192,576 | 4.61 B | **7.75** |
| **300M** | dim1280/15L | ~633 M | 328,335,360 | 304,742,400 | 13.63 B | **7.46** |
| **760M** | dim1536/24L | 1,073,554,944 | 393,842,688 | 679,712,256 | 31.38 B | **7.69** |
| **1.3B** | dim2048/25L¹ | 1,809,946,624 | 525,336,576 | 1,284,610,048 | 62.91 B | **8.16** |

¹ 1.3B-scale Llama peer = `runs/1.3B/llama_1.8B_paper` (1.8B total, 525M embed — matches the paper's
Table 2 "Emb 525M" 1.8B row). Others: `runs/small/cmp_g10/llama_{100M,300M,760M}` (dims from ckpt
`params.json`). Note the **non-embed** counts (99M/305M/680M/1.28B) are trunk-matched to the byte
models (99M/297M/714M/1.32B), and the token budgets match the byte token-equivalents — so Llama
lands at the same γ band despite ~3× larger *total* params.

## Combined γ (LLaMa-3 tokens)

| scale | rg | AU-Net-word | Llama |
|---|---:|---:|---:|
| 100M | 7.79 | 7.79 | 7.75 |
| 300M | 7.67 | 7.67 | 7.46 |
| 760M | 7.32 | 7.32 | 7.69 |
| 1.3B | 7.81 | 7.81 | 8.16 |

All three families, all scales, sit in **γ ≈ 7.3–8.2** — the paper's "ratio ~8" band (just under
their nominal ratio-10). Because the peers are matched on both non-embed params and data budget, no
family is meaningfully over/under-trained relative to the others.

## Result (rg)

- **γ_token ≈ 7.3–7.8 across the whole ladder** — roughly constant (760M is slightly light at 7.32),
  which is the intent of the fixed per-param budget.
- This is **just under the paper's ratio-10 baseline**, not at 10. Equivalently our **naive
  tokens/param ≈ 44–47**, vs the paper's ratio-10 block at **60 tokens/param** (60 B tok ÷ 1 B
  params) → we sit at roughly the paper's **"ratio ~8."**

## Reconciliation with our internal "γ10.4" label

- The **"60"** often quoted is the paper's ratio-10 *naive* tokens/param (60 B ÷ 1 B). Our ladder is
  ~46 tokens/param, i.e. γ_token ≈ 7.7 after the ÷6 FLOP factor.
- Our internal **"γ10.4"** label came from dividing `bytes/param` (≈213) by **k² ≈ 20.5**. The paper
  instead divides by **6·k ≈ 27.4** (the 6N FLOP factor, in token space), giving **≈7.8**. So the
  "10.4" label over-states the paper-convention γ by the factor **6/k ≈ 1.32** (it used k² where the
  paper uses 6k).

**Bottom line:** by the paper's exact formula, our 100M→1.3B rg models are all at **γ ≈ 7.3–7.8
LLaMa-3 tokens** (≈33–36 in raw bytes/6), a bit below the paper's ratio-10.

## Notes / caveats

- k = 4.56 is the DCLM · LLaMa-3 compression rate reported in the paper; our data is `dclm_baseline`
  with the LLaMa-3 tokenizer, so it applies directly.
- `Model size` is treated as N_non-embed (byte-vocab embedding ≈ 0.1–0.7 M, <0.1% — negligible).
  For **Llama / subword peers** this would NOT hold: the 128k-vocab embedding is large, so their γ
  must subtract the embedding params explicitly before applying the formula.
- The attention term `6·d·L·S` is dropped here (linear term only); including it would lower γ by a
  few %, largest for the smaller models with proportionally longer sequences.
