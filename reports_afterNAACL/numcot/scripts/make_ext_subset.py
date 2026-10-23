#!/usr/bin/env python3
"""Reduced NUMCoT item set for the external byte models (BLT-1B, H-Net 1-stage XL).

Their harnesses generate without a KV cache (BLT also at batch 1), so the full protocol (18,828 generations, most of
the bytes in the CoT conditions) does not fit. Subset, a strict subset of items.jsonl (same prompts and budgets):
  few-shot                          every item (4,557)
  few-shot CoT, few-shot CoT + knowledge   N questions per split (seeded random; rows are ordered by magnitude)
  zero-shot, zero-shot CoT          left out (base models do not follow the bare instruction; the trio's 0-shot
                                    outputs mostly ramble to the budget)
The trio is run on every item, so trio-vs-external comparisons use these keys.

  python reports_afterNAACL/numcot/scripts/make_ext_subset.py [N=50]
"""
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parents[1]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
items = [json.loads(l) for l in open(D / "items.jsonl")]
idx = defaultdict(set)
for it in items:
    idx[(it["task"], it["split"])].add(it["idx"])
pick = {k: set(random.Random(f"numcot_ext/{k[0]}/{k[1]}").sample(sorted(v), min(N, len(v)))) for k, v in idx.items()}
out = [it for it in items if it["cond"] == "fewshot"
       or (it["cond"] in ("cot_fewshot", "numcot") and it["idx"] in pick[(it["task"], it["split"])])]
with open(D / "items_ext.jsonl", "w") as f:
    for it in out:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(Counter(it["cond"] for it in out), len(out), "->", D / "items_ext.jsonl")
