# PoC: BPE Merge-Rule Suffix Features for Byte-Level Language Modeling

## 0. One-line Summary

Use BPE merge rules not as output tokens, but as causal auxiliary features for a byte-level language model.

At each byte position, look up BPE-derived merge units that end at the current position using a reversed ByteTrie, inject their embeddings into the byte embedding, and train the model with standard next-byte prediction. The main evaluation metric is validation BPB.

---

## 1. Motivation

Standard BPE token-level language models benefit from several factors:

1. Shorter sequence length.
2. Larger lexical prediction units.
3. Vocabulary-level softmax.
4. Statistical priors learned by the tokenizer.
5. Boundary and merge-rule information from tokenizer training.

The key research question is:

> Is BPE useful mainly because it defines the model's prediction units, or because the tokenizer encodes reusable statistical priors over byte sequences?

This PoC tests the second hypothesis.

Instead of using BPE tokens as prediction units, we keep byte-level autoregressive generation and inject BPE-derived merge-rule information as a causal inductive bias.

---

## 2. Core Idea

Given a byte sequence:

```text
T / h / i / s / _ / i / s / _ / a / _ / t / e / s / t
```

A BPE tokenizer has learned merge-rule units such as:

```text
Th, is, is_, the, ing, tion, ...
```

At position `t`, we only use units that are fully contained in the observed prefix and end at `t`.

Example:

```text
prefix: This is
target: space

Allowed features:
s
is
This
...

Not allowed:
is_
This_
```

The model remains a next-byte LM:

```text
p(x_{t+1} | x_≤t)
```

BPE-derived units are auxiliary features, not prediction targets.

---

## 3. Proposed Model

### 3.1 Baseline byte-level LM

```text
byte_id → byte embedding → causal Transformer → byte softmax
```

### 3.2 Proposed BPE-augmented byte LM

```text
byte_id
  → byte embedding
  + causal BPE merge-rule suffix feature
  → causal Transformer
  → byte softmax
```

At position `t`:

```text
U_t = {u ∈ BPEMergeUnits | u ends at t, len_bytes(u) ≤ K, rank(u) ≤ R}

m_t = Agg({E_merge(u) | u ∈ U_t})

h_t^0 = E_byte(x_t) + gate_t · Proj(m_t)

p(x_{t+1} | x_≤t) = Transformer(h_≤t)
```

Recommended initial fusion:

```text
m_t = mean(E_merge(u) for u in U_t)

gate_t = sigmoid(W [E_byte(x_t); Proj(m_t)])

h_t^0 = E_byte(x_t) + gate_t · Proj(m_t)
```

If `U_t` is empty:

```text
m_t = 0
h_t^0 = E_byte(x_t)
```

---

## 4. Causality Rule

The most important implementation rule:

> A BPE-derived unit can be used only after its final byte has been observed.

Example:

```text
Text: This is_

At prefix "This is":
  target = "_"
  usable feature = "is"
  unusable feature = "is_"

At prefix "This is_":
  target = next byte
  usable feature = "is_"
```

This prevents future leakage.

Implementation rule:

```text
At position t:
  query only suffixes ending at t.
  never query units that extend beyond t.
```

A reversed ByteTrie naturally enforces this.

---

## 5. ByteTrie Construction

### 5.1 Candidate unit source

Use BPE merge-rule results rather than only final BPE vocabulary.

For each merge rule:

```text
(a, b) -> c
```

add `c` as a candidate unit.

Each candidate unit has:

```text
unit bytes
merge rank
byte length
optional corpus frequency
```

### 5.2 Recommended initial filtering

Start with:

```text
R = top 30k high-priority merge-rule results
K = max byte length 8
M = max active units per position 4
d_merge = 128
```

Candidate set:

```text
U = {u | rank(u) ≤ R and len_bytes(u) ≤ K}
```

Optional frequency filtering:

```text
freq_train(u) ≥ f_min
```

### 5.3 Reversed ByteTrie lookup

Build a trie over reversed byte sequences.

Example unit:

```text
is_ → bytes [i, s, _]
reversed key: [_, s, i]
```

At each byte position `t`, scan backward up to `K` bytes and return all matched units ending at `t`.

Then select at most `M` active units.

Possible selection strategies:

1. Highest merge priority.
2. Longest units.
3. Mixed: top 2 by priority + top 2 by length.

Recommended v1:

```text
M = 4
select highest-priority matched units
```

---

## 6. Training Objective

Use standard next-byte prediction only for the first PoC.

```text
L_lm = - Σ_t log p(x_{t+1} | x_≤t)
```

No boundary loss is required for the initial version.

Optional later losses:

```text
L_boundary = BCE(pred_boundary_t, bpe_boundary_t)
L_merge = multi-label BCE(predicted_units_t, active_units_t)
```

But the first experiment should keep the objective simple.

---

## 7. Evaluation Metric

Primary metric:

```text
BPB = total negative log likelihood / total number of bytes / log(2)
```

Report validation BPB for all models.

Important:

For BPE token-level baselines, convert token NLL to byte-normalized BPB:

```text
BPB_token_model = Σ_k -log p(z_k | z_<k) / num_bytes / log(2)
```

---

## 8. Minimal Experiment Plan

### Stage 1: sanity check

```text
Model size: 30M–50M
Training data: 1B–3B bytes
Context length: 1K–2K bytes
Goal: verify implementation, causality, and BPB signal
```

Methods:

```text
A. Pure byte LM
B. Byte + top 10k merge units, K≤4, M=4
C. Byte + top 30k merge units, K≤8, M=4
```

Expected result:

```text
C BPB < A BPB
```

Even a stable improvement of `0.005–0.01 BPB` is promising.

---

### Stage 2: main ablation

```text
Model size: 100M–150M
Training data: 10B–30B bytes
Context length: 2K–4K bytes
Goal: produce main BPB table
```

Methods:

```text
1. Pure byte LM
2. Random n-gram feature
3. Frequency n-gram feature
4. BPE vocab suffix feature
5. BPE merge-rule suffix feature
6. BPE merge-rule suffix feature with shuffled rank
7. BPE merge-rule suffix + BPE boundary feature
8. BPE token-level LM
```

Key comparisons:

```text
5 > 1:
BPE merge-rule prior helps byte-level modeling.

5 > 2:
Improvement is not from arbitrary extra n-gram features.

5 > 3:
BPE merge trajectory is more useful than raw n-gram frequency.

5 > 6:
High-priority merge order matters.

7 ≈ 8:
BPE token-level advantage may be recoverable without using BPE tokens as prediction units.
```

Since lower BPB is better, the actual table should show:

```text
BPB(5) < BPB(1)
BPB(5) < BPB(2)
BPB(5) < BPB(3)
BPB(5) < BPB(6)
```

---

### Stage 3: scaling confirmation

```text
Model size: 300M–500M
Training data: 50B–100B bytes
Context length: 4K bytes
Goal: confirm whether the effect scales
```

Methods:

```text
A. Pure byte LM
B. Byte + BPE merge-rule suffix feature
C. Byte + BPE boundary + merge-rule suffix feature
D. BPE token-level LM
```

---

## 9. Recommended First Config

```yaml
model:
  backbone: causal_byte_transformer
  output_vocab_size: 256

bpe_feature:
  source: merge_rule_results
  rank_cutoff: 30000
  max_byte_length: 8
  max_active_units_per_position: 4
  lookup: reversed_bytetrie
  merge_embedding_dim: 128
  aggregation: mean
  fusion: gated_add

training:
  objective: next_byte_lm
  metric: bpb
  context_length: 2048
  stage1_model_size: 50M
  stage1_data: 1B-3B_bytes
```

---

## 10. Implementation Sketch

### 10.1 Preprocess merge-rule units

```python
units = []
for rank, (left, right, merged) in enumerate(bpe_merge_rules):
    b = bytes_of(merged)
    if len(b) <= K and rank < R:
        units.append({
            "id": len(units),
            "bytes": b,
            "rank": rank,
            "length": len(b),
        })
```

### 10.2 Build reversed ByteTrie

```python
trie = ByteTrie()
for u in units:
    trie.insert(u["bytes"][::-1], value=u["id"])
```

### 10.3 Lookup active units

```python
def lookup_active_units(byte_buffer, t, trie, K, M):
    # byte_buffer contains x_≤t
    node = trie.root
    matches = []

    for offset in range(K):
        pos = t - offset
        if pos < 0:
            break

        b = byte_buffer[pos]
        if b not in node.children:
            break

        node = node.children[b]

        if node.values:
            matches.extend(node.values)

    # choose top-M by merge priority
    matches = sorted(matches, key=lambda uid: unit_rank[uid])
    return matches[:M]
```

### 10.4 Build model input

```python
byte_emb = E_byte[x_t]

active_ids = lookup_active_units(x, t, trie, K, M)

if len(active_ids) > 0:
    merge_emb = E_merge[active_ids].mean(dim=0)
    merge_emb = merge_proj(merge_emb)
    gate = sigmoid(gate_proj(concat(byte_emb, merge_emb)))
    h0_t = byte_emb + gate * merge_emb
else:
    h0_t = byte_emb
```

### 10.5 Next-byte loss

```python
logits = model(h0_sequence)
loss = cross_entropy(logits[:, :-1], byte_ids[:, 1:])
```

---

## 11. Leakage Tests

Before training large models, run these checks.

### Test 1: suffix-only guarantee

For every active unit `u` at position `t`:

```text
u == x[t-len(u)+1 : t+1]
```

No unit may include `x[t+1]`.

### Test 2: adversarial example

For text:

```text
This is a test
```

At prefix:

```text
This is
```

target:

```text
space
```

Allowed:

```text
is
```

Forbidden:

```text
is_
```

### Test 3: shuffled future check

Shuffle bytes after position `t`. Active units at `t` must not change.

```text
active_units(x_≤t + future_a, t)
==
active_units(x_≤t + future_b, t)
```

---

## 12. Main Claims to Test

### Claim 1

BPE merge-rule suffix features improve byte-level BPB.

```text
Byte + BPE merge-rule suffix < Pure byte
```

### Claim 2

The improvement is not merely from adding extra n-gram embeddings.

```text
Byte + BPE merge-rule suffix < Byte + random/frequency n-gram
```

### Claim 3

High-priority BPE merge rules encode useful statistical priors.

```text
Top-ranked merge units < random merge units with same size
```

### Claim 4

A byte-output model can recover part of the BPE token-level advantage.

```text
Byte + BPE priors approaches BPE token LM in BPB
```

---

## 13. Related Work Positioning

This PoC is related to:

1. Byte-level and character-level LMs.
2. Charformer / GBST-style learned subword blocks.
3. CANINE-style subword inductive bias in token-free models.
4. MEGABYTE and BLT-style byte/patch models.
5. Hierarchical autoregressive byte/word models.
6. Incremental BPE Tokenization.
7. Formalizing BPE Tokenization.
8. Decoupling the Benefits of Subword Tokenization via Byte-level Simulation.

Main distinction:

> We do not use BPE tokens as prediction units. Instead, we use BPE merge rules as causal suffix features for byte-level next-byte prediction.

---

## 14. Suggested Paper Framing

### Research question

```text
Is BPE useful because it defines the model's discrete prediction units,
or because it encodes reusable statistical priors over byte sequences?
```

### Method summary

```text
We inject BPE-derived merge-rule units into a byte-level autoregressive model as causal suffix features, while retaining byte-level prediction.
```

### Main hypothesis

```text
A substantial portion of the advantage of BPE tokenization can be recovered by transferring the tokenizer's merge-rule prior to a byte-level model.
```

### Safe wording

```text
Our results suggest that BPE tokenizers are not merely compression mechanisms or output vocabularies, but also encode corpus-level statistical structure that can be reused as an inductive bias for byte-level language modeling.
```

Avoid overly strong wording such as:

```text
BPE token-level models work only because of tokenizer priors.
```

---

## 15. Go / No-Go Criteria

### Promising signal

```text
BPB improvement over pure byte ≥ 0.005
```

### Strong signal

```text
BPB improvement over pure byte ≥ 0.01
```

### Very strong signal

```text
Byte + BPE merge-rule suffix feature approaches BPE token-level LM
under similar parameter or compute budget.
```

### Stop or revise if

```text
BPE merge-rule suffix feature ≈ random n-gram feature
```

This would suggest that the gain may come from generic local n-gram features rather than BPE-specific priors.

---

## 16. Next Steps

1. Extract BPE merge-rule result units.
2. Build reversed ByteTrie.
3. Implement suffix lookup and leakage tests.
4. Add merge-feature embedding path to byte LM.
5. Run 30M–50M sanity check.
6. Run 100M main ablation if BPB improves.
7. Compare against BPE token-level LM using byte-normalized BPB.
