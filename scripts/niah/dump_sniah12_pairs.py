#!/usr/bin/env python3
"""Dump S-NIAH-1/2 teacher-forcing pairs in the sniah3_pairs.jsonl format, so the
external references (BLT-1B, H-Net) can be scored on byte-identical prompts via
eval_suite's _score_cells. Mirrors the sniah3 dump config: lengths 512/1024/2048/4096,
n=50 per cell, seed 0, colon-free query; depths spread evenly over (0,1)."""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from niah_data import make_sample, TASKS  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "reports/niah/sniah12_pairs.jsonl"

recs = []
for task in ("1", "2"):
    ht, vt = TASKS[task]
    for tb in (512, 1024, 2048, 4096):
        rng = random.Random(0)
        for i in range(50):
            depth = (i + 0.5) / 50.0
            s = make_sample(tb, depth, 7, rng, ht, vt)
            recs.append({"probe": f"sniah{task}", "cell": f"sniah{task}/{tb}",
                         "target_bytes": tb, "prompt": s["prompt"], "values": [s["value"]]})
with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print(f"wrote {len(recs)} -> {OUT}")
for k, v in sorted(Counter(r["cell"] for r in recs).items()):
    print(f"  {k:14s} {v}")
