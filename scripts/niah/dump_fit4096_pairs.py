#!/usr/bin/env python3
"""Dump S-NIAH pairs for the largest input length that fits BLT's 4096-byte
window, so the top length cell is scored without left-truncation for any model.

The existing sniah_full_pairs.jsonl labels cells by `target_bytes`, which sizes
the haystack BODY only; the instruction, the needle and the query add ~310+ bytes
on top, so its "4096" cell is really a 4405-5466-byte input and BLT (window 4096)
silently drops the leading 318-1379 bytes -- including the needle itself in a few
items. target_bytes=3200 is the largest round value whose worst-case scored
sequence (prompt + " " + value + BOS) stays under 4096 for every item.

Generation is byte-identical to the stored dump apart from target_bytes: depths
(i+0.5)/50 for S-NIAH-1/2/word, and [.1,.3,.5,.7,.9] x 10 for S-NIAH-3, seed 0.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import niah_data                                              # noqa: E402
from niah_data import make_sample, make_dataset, TASKS        # noqa: E402

TARGET_BYTES = int(sys.argv[1]) if len(sys.argv) > 1 else 3200
OUT = Path(__file__).resolve().parents[2] / f"reports/niah/sniah_fit4096_{TARGET_BYTES}_pairs.jsonl"

CELLS = [("1", "sniah1"), ("2", "sniah2"), ("3", "sniah3"), ("4", "sniahword")]

recs = []
for task, name in CELLS:
    ht, vt = TASKS[task]
    if task == "3":
        samples = make_dataset(TARGET_BYTES, [0.1, 0.3, 0.5, 0.7, 0.9], 10, 7, 0, ht, vt)
    else:
        rng = random.Random(0)
        samples = [make_sample(TARGET_BYTES, (i + 0.5) / 50, 7, rng, ht, vt) for i in range(50)]
    for s in samples:
        recs.append({"probe": name, "cell": f"{name}/{TARGET_BYTES}",
                     "target_bytes": TARGET_BYTES, "prompt": s["prompt"],
                     "values": [s["value"]], "prompt_bytes": s["prompt_bytes"],
                     "scored_bytes": len((s["prompt"] + " " + s["value"]).encode())})

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

worst = max(r["scored_bytes"] for r in recs) + 1        # +1 BOS
print(f"wrote {len(recs)} -> {OUT}")
print(f"worst-case BLT sequence (bytes + BOS): {worst}  (limit 4096, headroom {4096 - worst})")
for name in ("sniah1", "sniah2", "sniah3", "sniahword"):
    sb = [r["scored_bytes"] for r in recs if r["probe"] == name]
    print(f"  {name:10s} n={len(sb)}  scored bytes min={min(sb)} max={max(sb)}")
