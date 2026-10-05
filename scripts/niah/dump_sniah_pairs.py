#!/usr/bin/env python3
"""Dump S-NIAH teacher-forcing pairs (sniah3_pairs.jsonl format) with configurable
n per cell, so all five models can be scored on byte-identical prompts:
matched trio via scripts/niah/score_probe_pairs.py, externals via
eval_suite/score_pairs_file.py.

Generalizes dump_sniah12_pairs.py / dump_fit4096_pairs.py (n was a hardcoded 50):
depths spread evenly over (0,1) as (i+0.5)/n, seed 0 per cell, colon-free query.

  python scripts/niah/dump_sniah_pairs.py --tasks 1 2 3 --n 250 \
      --lengths 512 1024 2048 4096 6144 --out reports/niah/sniah123_n250_pairs.jsonl
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from niah_data import make_sample, TASKS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--lengths", nargs="+", type=int, default=[512, 1024, 2048, 4096, 6144])
    ap.add_argument("--value_digits", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = []
    for task in args.tasks:
        ht, vt = TASKS[task]
        for tb in args.lengths:
            rng = random.Random(args.seed)
            for i in range(args.n):
                depth = (i + 0.5) / args.n
                s = make_sample(tb, depth, args.value_digits, rng, ht, vt)
                recs.append({"probe": f"sniah{task}", "cell": f"sniah{task}/{tb}",
                             "target_bytes": tb, "prompt": s["prompt"],
                             "values": [s["value"]]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} -> {out}")
    for k, v in sorted(Counter(r["cell"] for r in recs).items()):
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
