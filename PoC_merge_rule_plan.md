# PoC implementation plan — BPE merge-rule suffix features (from BPEByte_with_merge_rule.md)

## Decisions (after codebase map)
- **Base model:** `apps/main` `LMTransformer` (flat byte transformer) with `tokenizer.name=bytes`
  (vocab 258 = 256 bytes + BOS + EOS). Flat, no hierarchy. `apps/main/eval.py:250` already
  emits `nll_per_byte` → **BPB** = nll_per_byte / ln2. So baseline + metric are free.
- **Merge feature:** new self-contained module `apps/bytepoc/merge_feature.py`:
  - extract ranked merge units from llama3 tiktoken (`load_tiktoken_bpe` → {bytes: rank}),
    filter rank<R and len<=K.
  - reversed ByteTrie (insert reversed unit bytes; node carries unit_id).
  - `compute_merge_ids(byte_ids[seqlen]) -> ids[seqlen, M]` (top-M by priority, -1 pad). Pure
    function of the prefix ending at t → causal, no future leak.
- **Injection:** extend `LMTransformer.forward` to accept `merge_ids` and do
  `h = E_byte(x) + gate * Proj(mean(E_merge(merge_ids)))` right after the byte embedding
  (`apps/main/transformer.py:105`). Embeddings: `E_merge=nn.Embedding(num_units+1, d_merge)`
  (last idx = pad), `Proj: d_merge->dim`, `gate=sigmoid(Linear([E_byte;Proj(m)]->1))`.
- **Threading merge_ids:** compute per-batch from the byte ids (deterministic). For the PoC,
  compute in the train/eval step from the token batch (CPU numpy, overlap via workers) and pass
  into forward. Avoids deep data-pipeline surgery; merge_ids is a pure function of token_values.

## Ablation methods (spec §8 Stage-2)
1 pure byte · 2 random n-gram feat · 3 frequency n-gram feat · 4 BPE vocab suffix · 5 BPE
merge-rule suffix (main) · 6 merge-rule shuffled-rank · 7 merge-rule + boundary feat · 8 BPE
token LM (byte-normalized BPB). Key: 5<1, 5<2, 5<3, 5<6; 7≈8.

## Stage-1 sanity (first): ~30-50M, 1-3B bytes, ctx 2048
A pure byte · B +top10k merges K<=4 M=4 · C +top30k merges K<=8 M=4. Want C BPB < A BPB
(promising >=0.005, strong >=0.01).

## Leakage tests (spec §11) — MUST pass before any training
- T1 suffix-only: every active unit u at t equals byte_ids[t-len(u)+1 : t+1].
- T2 adversarial: "This is" → "_" target; "is" allowed, "is_" forbidden.
- T3 shuffled-future: active_units at t invariant to bytes after t.

## GPU plan
Training waits for a free GPU (B200 v4 done ~Jun19, or ece slot after ratio-20 sweep+evals).
No-GPU foundation (module + leakage tests) built and verified first.
