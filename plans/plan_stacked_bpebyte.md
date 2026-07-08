# Stacking BPEByte — a deeper (3–4 level) AU-Net hierarchy

_2026-07-07. Design note: can BPEByte be extended from the current 2-level (bytes→BPE-patches→trunk)
into a 3–4 level hierarchy (bytes→BPE-patches→words→sentences), i.e. "AU-Net 3 / 4" with BPE boundaries?_

**TL;DR.** Yes, and the model is most of the way there — `hierarchical.py` already runs **N levels** from
per-level config lists, and the repo's own `2B_2level.yaml` is a 3-dim / 2-pooling hierarchy. The only
real blocker is that BPEByte's **online (causal) boundary path is hard-locked to a single BPE stage**
(`regex_cutting.py:522`). Stacking = lifting that to emit **N nested, leak-free boundary levels** + adding
the extra dims/layers. Whether it *helps* is a genuine experiment: the upside is compute (a bigger trunk
over fewer super-patches); the risk is that pooling-induced content-starvation **compounds** across levels
(see `../PoC/PoC_down_pooling.md`).

---

## 1. What already exists (multi-level is not new)

- `model.dimensions` / `layers` / `head_dims` / `sliding_windows` / `max_seqlens` are **per-level lists**,
  one entry per level. `2B_2level.yaml` uses `dimensions: [512, 2048, 3456]` — a **byte-enc → mid → trunk**
  hierarchy with **two pooling stages**, widening to dim 3456 at the coarsest (cheapest-per-token) level.
- Boundaries per stage come from `data.regex.strategy`, a dict of stages, e.g. `2B_2level.yaml`:
  `strategy: {word1: 1@1, word2: 2@1}` → bytes→words→word-groups.
- `regex_cutting.py:240` already loops `for i, strategy in enumerate(args.strategy)` building **per-level
  masks** — the **offline / word** multi-stage path works today.

**The blocker.** The online causal BPE path (`online_levels_mask`, `regex_cutting.py:522`) asserts
`len(self.strategy) == 1 and strategy.startswith("bpe")`. BPEByte's leak-free streaming boundaries are
therefore single-level by construction. This is the one piece to extend.

---

## 2. Two ways to stack

### (A) Mixed — BPE bottom, coarser linguistic levels above  *(easiest; reuses existing word stages)*
```
bytes ─[BPE-trie, online]→ subword-patches ─[whitespace]→ words ─[newline/punct]→ lines/sentences
  L0                    L1                        L2                       L3
```
Word boundaries **nest inside** BPE boundaries (in the llama3 tokenizer a `" word"` token *starts* at a BPE
boundary), so L2/L3 are valid poolings of L1. Every source (BPE-trie, whitespace, newline) is a deterministic
function of past bytes → **leak-free preserved**. New code is minimal: combine the online BPE L1 mask with
the already-implemented offline word L2/L3 masks.

### (B) All-BPE multi-resolution  *(cleaner story, more work)*
Use one BPE merge ranking **truncated at decreasing vocab cutoffs** → `bytes → fine-BPE → coarse-BPE`.
Because BPE merges only ever *remove* boundaries, a coarser (larger-vocab) boundary set is a **subset** of a
finer one → nests by construction, no whitespace dependency. Works on **no-space scripts / code** where (A)'s
word level degrades. Cost: need the truncated-merge / merge-tree machinery and an incremental parser per level.

---

## 3. Three hard requirements

1. **Nesting** — coarser boundaries ⊆ finer boundaries at every level, or the pooling gather indices are
   ill-defined. (A) gets this for free; (B) needs the truncated-merge construction (longest-match only
   *approximately* nests — verify or use the merge tree).
2. **Leak-free per level** — holds for all candidate sources, but each new level needs its **incremental /
   streaming parser** (analogous to the existing online BPE parser) for train==generation parity.
3. **Per-level budget** — set a target bytes/patch per level (~4.5 B/BPE-patch → ~6 B/word → ~30 B/sentence)
   and give each level its own `max_seqlen`; calibrate so patch counts don't blow the level buffers.

---

## 4. Will it help? (the real question)

**Upside — compute / trunk scaling.** More levels → the widest trunk runs over *far fewer* super-patches →
cheaper per FLOP → a bigger/deeper trunk fits the same budget. This is the U-Net efficiency argument and
exactly why `2B_2level` widens to dim 3456 at the top. Byte-level robustness (PBP / CUTE / typo) is a
**bottom-level** property, so it survives a deeper stack.

**Downside — content-starvation compounds (our own data warns here).** `../PoC/PoC_down_pooling.md` found the
single pooling already discards patch interior (first-byte gather was near-optimal; the `last_byte` "win" was
a leak, `../reports/` A0/A1). Each extra pooling abstracts further, so a deeper stack risks *worsening* the
BPB gap that 2-level BPEByte already loses to subword — **unless the bottleneck is compute, not information.**

Net: a real experiment, not a sure win. It is the same crossroads **H-Net** explores (learned multi-level
chunking); a **fixed, leak-free, BPE-derived hierarchy** is the cheap principled baseline against it.

---

## 5. First experiment

3-level **BPE(L1) + word(L2)** at 100M, matched budget, against:
- (i) 2-level BPEByte `v4_root_greedy` (the current bottom),
- (ii) 3-level all-word AU-Net (`2B_2level` analog at 100M),
- (iii) Llama subword baseline.

**Reads:** train/held-out **BPB** (primary — does the coarser level + bigger trunk close the gap?), tokens/sec
& trunk-FLOP fraction (the compute-win check), leak probe (must stay 0% at every level), HS/ARC-E/PIQA/ARC-C.
If BPB improves and the leak stays 0, scale the level count (4-level: +sentence) and the trunk width.

---

## 6. Implementation touch-points

- `apps/aunet/data/regex_cutting.py`
  - `online_levels_mask` (~:515–531): drop the `len(strategy)==1` assert; emit an **N-level** mask by
    composing the online BPE L0→L1 boundaries with additional nested stages (whitespace/newline for (A);
    truncated-merge for (B)). Reuse the offline `for i, strategy in enumerate(args.strategy)` loop (:240).
  - Add per-level incremental parsers (mirror the existing online BPE parser: feed / committed_levels /
    snapshot / restore) so streaming generation matches training.
- `apps/aunet/hierarchical.py` — already N-level; just receives the longer per-level config lists.
- Config: copy `bpebyte_100M_v4_root_greedy.yaml`, extend `dimensions`/`layers`/`sliding_windows`/`max_seqlens`
  to 3 entries, set `regex.strategy: {bpe_br: 1@1, word1: 2@1}` (BPE L1 + word L2), target bytes/patch per level.
- **Nesting assertion** in the mask builder (coarser ⊆ finer) as a correctness guard.
- Reuse `probe_root_causal.py` to confirm 0% leak per level.

---

## 7. Open choices (need a decision before building)

1. **Design (A) mixed vs (B) all-BPE** — (A) is a fast first result and reuses word stages; (B) is the
   stronger multilingual/code story (no whitespace dependency) but more machinery.
2. **Levels** — 3 (bytes→BPE→word) first; 4 (+sentence) only if 3 shows a BPB or compute win.
3. **Trunk-scaling axis** — hold trunk size (isolate the pooling effect) vs grow the trunk with the saved
   compute (test the efficiency thesis). Run both if the first is ambiguous.

_Companions: `../PoC/PoC_down_pooling.md` (the pooling-representative / content-starvation finding this must
beat), `../methods/BPEByte_root_greedy_method.md` (the L0→L1 boundary scheme being stacked),
`plan_entropy_patching.md` (another N-of-a-family boundary source), `../PROJECT_STATUS.md`._
