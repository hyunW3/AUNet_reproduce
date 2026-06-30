# BPEByte `root_greedy`: leak-free, causal online patch boundaries

**TL;DR.** `root_greedy` is the boundary scheme that makes a hierarchical byte model's *training*
patch boundaries **identical to the boundaries its own autoregressive decoder produces at
generation** — with **zero future-byte leakage**. It combines two choices: a **greedy** byte-trie
walk (commit at the trie dead-end, never backtrack → `O(n)`, prefix-only) and **root** placement
(put the patch boundary at the *next* token's start, so predicting byte *e* never reads the boundary
that depends on byte *e*). It is the honest byte-level number; the leaky `bt` variants score lower
train BPB only because they peek.

Companion: `BPEByte_with_merge_rule.md` (offline/merge-rule background), `PoC_merge_rule_plan.md`.
Curves: `runs/bpb_all_1.3B.png` (regenerated). Code: `lingua/apps/aunet/data/byte_trie.py`,
`lingua/apps/aunet/data/regex_cutting.py`. Config: `apps/aunet/configs/bpebyte_300M_v4_root_greedy.yaml`,
`…/bpebyte_root_greedy_760M_b200.yaml`; 1.3B run dir `runs/bpebyte_br_greedy_root_1.3B`.

---

## 1. Setting: AU-Net / BPEByte hierarchy and what a "patch boundary" is

A BPEByte model is a hierarchical AU-Net (`apps/aunet/hierarchical.py`): a **byte encoder**
(`dim 512`, 3 layers) reads the raw byte stream; bytes are **pooled into patches** at a set of
*patch boundaries*; a **trunk** transformer (the "global" model, e.g. `dim 1536 / 24 L` at 760M)
runs on the pooled patch sequence; a **decoder** upsamples patch states back to per-byte states and
predicts the next byte. The tokenizer is raw bytes (vocab 258 = 0–255 + BOS 256 + EOS 257); the
"BPE" in BPEByte refers only to *where the patch boundaries fall* — they are placed at the token
boundaries of a reference BPE vocab (here Llama-3's), applied **over the byte stream**.

So the only thing that distinguishes BPEByte variants is the **segmentation function** `bytes →
boundary positions`. That function is the subject of this document.

Why it matters: the patch boundaries decide (a) how bytes are pooled for the trunk and (b) via the
decoder's `repeat_idx`/`patch_read_delay`, *which patch state each byte is allowed to read when it
is predicted*. If the boundary for byte *e* is computed using bytes at or after *e*, the model is
handed information about the future during teacher forcing that it cannot have at generation — a
**train/generation leak**. Removing that leak is the entire point of `root_greedy`.

---

## 2. The boundary-generation problem (why naïve schemes leak)

There are two independent ways a boundary scheme can leak future information.

### 2a. Mode — how the trie walk decides token ends

`bt` (backtracking longest-match, `byte_trie.py:43`
`longest_match_with_backtracking_tokenize_boundaries`): walk the trie from `root`; remember the
**longest position that is a valid vocab token**; at a dead-end, commit that longest valid token and
**re-feed** the bytes consumed past it.

```python
while pos < n:
    node = root; longest_match = 1
    for i in range(pos, n):
        if byte[i] not in node.children: break
        node = node.children[byte[i]]
        if node.is_token: longest_match = i - pos + 1   # remember last VALID token
    boundaries.append(pos + longest_match); pos += longest_match
```

This is maximal-munch BPE. To decide where token *k* ends it **consumes bytes belonging to token
k+1** (the failed extension) and then backtracks. The end of token *k* is therefore a function of
bytes *after* token *k*. This is the offline/`bt` behaviour and it is **leaky** at the boundary.

`greedy` (`byte_trie.py:66` `greedy_tokenize_boundaries`): walk the trie and **commit ALL
accumulated bytes the instant the walk hits a dead-end** — no "longest valid token", no re-feed.

```python
while pos < n:
    node = root; longest_match = 1
    for i in range(pos, n):
        if byte[i] in node.children: node = node.children[byte[i]]
        else: longest_match = max(1, i - pos); break    # dead-end: commit the walked prefix
    else: longest_match = i - pos + 1; pos += longest_match; continue
    boundaries.append(pos + longest_match); pos += longest_match
```

Properties:
- **Prefix-only / causal.** The cut at the dead-end depends only on the bytes walked so far; no
  byte past the cut is inspected. (The `bt` re-feed *also* only touches already-seen bytes, but the
  *position* it commits is chosen with lookahead; greedy chooses without.)
- **`O(n)`**, single left-to-right pass, **never rolls back** — the same forward-only commitment the
  streaming decoder makes at generation.
- The committed patch may **not be a valid vocab token** (it is "whatever the trie walked before
  dying"). That is fine: patches are pooling spans, not tokens to be decoded.
- **BOS/special-byte guard** (`max(1, i - pos)`): a byte absent from the trie root (e.g. BOS = 256)
  would give `i == pos → 0` advance → infinite loop; it is committed as a 1-byte patch (matching
  `bt`'s minimum advance of 1). This was a real bug once (stale `byte_trie.py` on a remote box
  caused a greedy infinite-loop on BOS).

### 2b. Placement — where the boundary sits relative to a token

`boundaries()` returns **end-exclusive** token ends. The patch-boundary index is then
(`regex_cutting.py:285-290`):

```python
off = 0 if placement == "root" else 1
return [min(e - off, n - 1) for e in ends if 0 < e <= n]
```

- **`before_root`** (`off = 1`): boundary at the token's **last byte**, index `e-1`.
- **`root`** (`off = 0`): boundary at the **next token's first byte**, index `e` (spacebyte's
  "root").

This shift is what closes the leak in the decoder. In the upsampling decoder's teacher-forced
branch, byte *e* gathers the patch state indexed by `repeat_idx[e-1]` — i.e. **the patch state
settled strictly before *e***. Under **root** placement the boundary that opens the patch containing
*e* sits *at* *e*, so it is **excluded** from `repeat_idx[e-1]`: predicting byte *e* cannot see the
boundary decision that *e* itself triggers. Under `before_root` the boundary sits at `e-1` and is
visible when predicting *e* — leak. (See the config comment in
`bpebyte_300M_v4_root_greedy.yaml`: "root placement makes predicting e use repeat_idx[e-1] which
excludes the e-boundary → structurally leak-free".)

### 2c. The combination

| scheme | mode | placement | lookahead to cut? | boundary visible when predicting it? | net |
|--------|------|-----------|-------------------|--------------------------------------|-----|
| offline `bt` (`br_bt`) | longest-match (offline real-BPE) | before_root | yes | yes | **leaky** (×2) |
| online `bt` (`br_bt` online) | longest-match + backtrack | before_root | yes | yes | **leaky** |
| **`root_greedy`** | **greedy** | **root** | **no** | **no** | **leak-free + causal** |

`root_greedy` is the only combination that is leak-free on *both* axes.

---

## 3. `root_greedy` end-to-end

1. **Reference vocab → trie.** Load the BPE vocab (Llama-3 tokenizer at
   `tokenizer/llama3/tokenizer.model`), insert every token's byte sequence into a `ByteTrie`
   (`byte_trie.py:34`).
2. **Segment (training, batched).** `RegexPool.get_levels_mask(byte)` →
   `byte_trie.boundaries(byte_seq, mode="greedy")` → greedy ends → `root` placement (`off=0`) →
   per-byte level mask consumed by the encoder's pooling and the decoder's upsampling.
3. **Segment (generation, streaming).** `ByteTrieIncrementalParser`
   (`byte_trie.py:180`, `regex_cutting.committed_patch_idx`) feeds one byte at a time and emits
   **committed** boundaries with `commit_margin = 1` (greedy forces margin 1 internally). Because
   greedy never backtracks, the streaming frontier is monotone and equals the batched result —
   **train/gen boundary gap = 0**. The decoder's constant `patch_read_delay` is a margin
   approximation of this committed index; `online_committed_patch_idx` (`regex_cutting.py:308`)
   provides the *exact* settled count as a 3rd data view (`repeat_idx`) so training reads precisely
   the patch the streaming decoder will have settled — verified byte-identical to the streaming
   parser.
4. **Reproducibility.** Segmentation is a pure function of the byte prefix, so it is memoised in a
   disk cache (`REGEX_SEG_CACHE`, `regex_cutting.py`); cache hits are byte-identical to recompute.

### Config (the operative block)

```yaml
data:
  tokenizer: { name: bytes }
  regex:
    strategy: { bpe_br: 1@1 }
    bpe_tokenizer_path: .../tokenizer/llama3/tokenizer.model
    bpe_online: true
    bpe_online_mode: greedy        # no backtracking (commit_margin 1) — §2a
    bpe_online_placement: root     # boundary at next-token start, index e — §2b
    bpe_context_prefix: 0          # fully online (whole sequence)
    bpe_online_committed_view: false
model:
  patch_read_delay: 0
```

---

## 4. Results (1.3B scale, same DCLM data, 180k steps ≈ 283 GB bytes)

Train **BPB** (bits/byte, EMA α=0.01, tokenizer-agnostic; byte: `loss/ln2`, Llama:
`loss/(ln2·bytes_per_token)`, `bytes_per_token = 4.55`). See `runs/bpb_all_1.3B.png` (regenerated —
the earlier render showed root-greedy at 0.908 because the run had not finished; final is **0.860**).

| run | scheme | train BPB @283 GB (EMA) |
|-----|--------|------------------------|
| `bpebyte_br_bt_1.3B` | offline `bt` (leaky) | **0.780** |
| `bpebyte_br_bt_online_1.3B` | online `bt` (leaky) | 0.787 |
| `llama_1.8B_paper` | subword (Llama 1.8B, canonical baseline) | **0.839** |
| `bpebyte_br_greedy_root_1.3B` | **root_greedy (leak-free)** | **0.860** |
| `aunet2_1.3B` | AU-Net2 pure byte (whitespace patches) | 0.866 |

**Budget note (iso-byte).** The subword baseline is `llama_1.8B_paper` (dim2048/25L, paper Table
10/11 recipe: seq 4096, lr 3e-3, 60k steps) — **NOT** `llama_1B_dm10`. All five runs reach ≥283 GB,
so a single iso-byte readout at **283 GB** suffices: the byte models *end* exactly at 283 GB (endpoint
= @283 GB), and the 1.8B overshoots to 286 GB but reads 0.839 at 283 GB (vs 0.840 at its 286 GB
endpoint — a 0.001 difference). The x-axis converts Llama tokens→bytes (×4.5483 measured
bytes/token). At 283 GB the 1.8B subword model (**0.839**) is *below* the leak-free byte models
(root_greedy 0.860, AU-Net2 0.866): the leak-free byte BPB is higher than the subword baseline, while
the leaky `bt` variants (0.78–0.79) only beat it via boundary lookahead not realizable at generation.

**Reading the curves.**
- **The leak is worth ~0.07 BPB.** online-`bt` (0.787) vs `root_greedy` (0.860) train on the same
  bytes and the same architecture; the only difference is the boundary leak. That 0.073 BPB gap is
  *not realizable at generation* — it is the model exploiting boundary lookahead it won't have when
  decoding. `root_greedy` is the honest number.
- **Leak-free byte ≈ leak-free byte.** `root_greedy` (0.860) ≈ AU-Net2 (0.866): two different
  leak-free byte segmentations land in the same place. BPE-placed leak-free boundaries buy little
  over plain whitespace patches *in train BPB* at this scale.
- **Subword sits between.** Llama (0.839) is below the leak-free byte models on train BPB but lacks
  their character-level strengths (see below).

**Downstream (0-shot, full sets; `model_results_1.3B.md`).** Despite the higher train BPB,
`root_greedy` is competitive-to-best on reasoning (HellaSwag acc 48.3, ARC-Easy acc_norm **65.9**,
ARC-Challenge acc **35.6**, PIQA acc 73.7) and wins the character/robustness axes that motivate byte
models: CUTE 20.6 (vs Llama 18.4), most typo-robust noisy-downstream (boolq **+0.8**, smallest
arc/piqa drops), and **PBP ≈ 0** (cut-invariant: ΔBPC +0.000, pbp_mc ΔAcc +0.16% vs Llama's −9.85%).
i.e. the leak that inflates `bt`'s train BPB does **not** translate into downstream gains, while
`root_greedy`'s causal boundaries keep all the byte-level robustness.

---

## 5. When to use it

- **Use `root_greedy`** whenever the training boundary scheme must match the generation decoder
  exactly (any honest BPC/BPB claim, any deployment where the model autoregressively produces its
  own boundaries) — it is the only scheme that is leak-free on both the *mode* and *placement* axes
  and is `O(n)` causal at both train and generation.
- **`bt` variants** are useful only as an *upper-bound diagnostic*: they show how much a model could
  gain from boundary lookahead (the leak gap), which is exactly the quantity you do **not** want to
  bank as a real result.
- **Cost.** Greedy patches are slightly coarser/irregular than maximal-munch tokens, costing the
  ~0.07 BPB seen above; in exchange you get train/gen parity, reproducibility, and the full byte
  robustness profile.

---

*Generated alongside the regenerated `runs/bpb_all_1.3B.png`. Numbers from
`runs/*/metrics.jsonl` (180k) and `model_results_1.3B.md`.*
