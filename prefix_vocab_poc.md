# prefix_vocab — prefix-closed causal byte-trie segmentation (V7)

**Idea.** A new online byte-trie patch-boundary rule for AU-Net BPEByte: walking left-to-right
from the trie root, **cut at the first node whose `is_token` is False** (or at a trie dead-end).
Each committed patch is therefore a byte string for which **every prefix is itself a vocab token**
— the *prefix-closed* property. Causal + leak-free (the cut uses only past bytes + the static
trie, never a future byte), with **root** placement and `commit_margin = 1` (never re-merges).

Example (trie with tokens `{a, at, att, atte, attend}`): walking `attend` →
`a`(✓) `at`(✓) `att`(✓) `atte`(✓) `atten`(✗, not a token) → **cut at `n`** → patch `atte`,
then continue: `attend → atte | n | d`.

## How it differs from the other causal modes
| mode | walk rule | cut point |
|------|-----------|-----------|
| `greedy` | follow any byte in the trie | trie **dead-end** (byte not in children) |
| `prefix_free` (V6) | follow until a token completes | **first** `is_token=True` node (needs a prefix-free vocab) |
| **`prefix_vocab` (V7)** | follow while every node `is_token=True` | **first** `is_token=False` node (or dead-end) |
| `bt` | longest valid match + re-feed | dead-end −1 byte (has the 1-byte before_root leak) |

`prefix_free` cuts at the *first* token; `prefix_vocab` cuts at the *first non*-token — they are
opposite-direction rules on opposite vocab structures (prefix-free = antichain / leaves only;
prefix-closed = every prefix kept).

## Effective vocabulary — the central statistic
A token can be emitted only if **all of its prefixes are also tokens**; otherwise the walk cuts
before reaching it. Counting the prefix-closed (reachable) subset of the llama3 BPE vocab:

| | tokens |
|---|---|
| Original BPE vocab | **128,256** |
| Reachable under prefix_vocab (prefix-closed) | **45,293 (35.3%)** |
| Unreachable (some prefix is not a token) | 82,963 (64.7%) |

**~65% of the vocab is inaccessible.** Reachability collapses with token length (single bytes are
always tokens, so all short tokens survive; long tokens almost never have all-token prefixes):

| patch length (bytes) | reachable / total |
|---|---|
| 1 | 256 / 256 (100%) |
| 2 | 4,287 / 4,287 (100%) |
| 3 | 14,989 / 15,369 (98%) |
| 4 | 13,570 / 18,557 (73%) |
| 5 | 7,854 / 17,226 (46%) |
| 6 | 3,042 / 19,356 (16%) |
| 7 | 829 / 15,062 (5.5%) |
| 8 | 187 / 10,710 (1.7%) |

Mean patch length **3.48 bytes/patch** (vs `greedy` 4.70, `bt` 4.69) with **11% 1-byte** patches
(vs 15%) — the prefix-closed constraint shortens patches and shrinks the usable vocab.

## prefix-closed set IS the effective vocab (verified)
Running prefix_vocab on the full 128k trie gives the **identical** segmentation to running it on a
trie built from only the 45,293 prefix-closed tokens (901 == 901 patches on a test text). So:
- the 45k prefix-closed set is not a separate construct — it is exactly what prefix_vocab uses;
- the 82,963 unreachable tokens are **inert**: deleting them changes **zero** boundaries (a long
  unreachable token like `attend` only creates the non-token node `atten` that triggers the cut at
  `atte`; in the 45k trie that node is simply a dead-end at `atte` — same cut);
- the prefix-closed trie is **all-leaf** (every node `is_token=True`), the defining property —
  on it there is no non-token node to cut at except dead-ends.
- prefix_vocab is therefore *almost* "`greedy` restricted to a prefix-closed vocab," differing
  from greedy-on-the-45k-trie by ~1 patch (a final/EOS edge case in the two code paths).

**Invariant check (PASS):** on 1,241 patches over a mixed text/number/punctuation sample, every
patch is an all-`is_token` trie path (0 violations) and every cut lands at the first non-token node
or dead-end (0 violations).

## Why it might help / hurt
- **Leak-free + causal by construction**, like root_greedy (V4) and prefix_free (V6) — no future
  byte ever sets a boundary. Reproducible byte-by-byte at generation (`commit_margin = 1`).
- Unlike V6 it needs **no pre-built pruned vocab** — the prefix-closed restriction is derived
  on-the-fly from the full tokenizer's `is_token` flags.
- Cost: a **smaller effective vocab (35%)** and **shorter patches (3.48 B)** → more patches per
  byte → more level-2 compute and a coarser-to-finer granularity than greedy/bt. Whether the
  cleaner prefix-closed structure offsets the shorter patches is the empirical question.

## Implementation
`apps/aunet/data/byte_trie.py`: `ByteTrie.prefix_vocab_tokenize_boundaries` + dispatch in
`boundaries()`; `ByteTrieIncrementalParser.commit_margin = 1` for the mode (causal, no re-merge).
No new vocab artifact; built from the standard tokenizer trie. Config:
`apps/aunet/configs/r20_v7_prefix_vocab.yaml` (`bpe_online_mode: prefix_vocab`, root placement).

## Experiment (in progress)
100M ratio-40 (13376 steps / 84B bytes), GPUs 0,4,5,7 on ece-agpu18 (`v7_now.sh`). Compare HS /
ARC-E / PIQA / ARC-C acc_norm against the same-scale scratch leak-free+causal baselines:
**v4_root_greedy 31.92** and **v6_prefix_free 32.70** (and word/subword: aunet 32.37, llama 35.60).
Result ~05:30. *To be filled in.*
