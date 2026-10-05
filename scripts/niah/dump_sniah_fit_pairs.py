#!/usr/bin/env python3
"""Dump the S-NIAH "4k" cell with the WHOLE scored sequence capped at 4096 bytes.

In sniah123_n250_le4096_pairs.jsonl the cell label (`target_bytes`) sizes only the haystack body;
the instruction, needle and query add ~310+ bytes, and the essay haystack is sliced by characters
(multi-byte UTF-8 inflates it further). So every "4096" item was actually 4.4-5.9 KB, and BLT
(4096-byte context) scored left-truncated prompts: the instruction was lost in all items and the needle
in 8/10/12 of 250.

Here each item keeps the same task, depth (i+0.5)/n and seed-0 RNG stream as dump_sniah_pairs.py, but
its haystack size is the largest that keeps BOS + prompt + " " + value <= --budget bytes (binary
search per item from a saved RNG state; make_sample draws the same number of random values for any
target size, so the stream stays aligned). Cell names keep the "/4096" label.

  python scripts/niah/dump_sniah_fit_pairs.py --out reports/niah/sniah123_n250_fit4k_pairs.jsonl
"""
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from niah_data import make_sample, TASKS  # noqa: E402


def scored_bytes(s):
    return 1 + len((s["prompt"] + " " + s["value"]).encode())      # +1 BOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--value_digits", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs, stats = [], defaultdict(list)
    for task in a.tasks:
        ht, vt = TASKS[task]
        rng = random.Random(a.seed)
        for i in range(a.n):
            depth = (i + 0.5) / a.n
            state = rng.getstate()

            def build(tb):
                rng.setstate(state)
                return make_sample(tb, depth, a.value_digits, rng, ht, vt)
            lo, hi = 64, a.budget                      # largest haystack target that fits the budget
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if scored_bytes(build(mid)) <= a.budget:
                    lo = mid
                else:
                    hi = mid - 1
            s = build(lo)                              # leaves rng after this item's draws
            sb = scored_bytes(s)
            assert sb <= a.budget, (task, i, sb)
            recs.append({"probe": f"sniah{task}", "cell": f"sniah{task}/{a.budget}", "target_bytes": lo,
                         "budget_bytes": a.budget, "scored_bytes_with_bos": sb, "depth": depth,
                         "prompt": s["prompt"], "values": [s["value"]]})
            stats[task].append((lo, sb))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} -> {out}")
    for t, v in stats.items():
        print(f"  sniah{t}: haystack target {min(x[0] for x in v)}-{max(x[0] for x in v)} | "
              f"scored+BOS {min(x[1] for x in v)}-{max(x[1] for x in v)} (budget {a.budget})")


if __name__ == "__main__":
    main()
