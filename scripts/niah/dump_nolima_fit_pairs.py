#!/usr/bin/env python3
"""NoLiMa items (tab:ext_retrieval) with the 4k cell capped at 4096 scored bytes.

The 1k/2k cells are those of dump_probe_pairs.py (make_dataset("nolima", tb, ..., seed 0); max 2,611 bytes). The
old 4k cell sized only the haystack, so its items were 4,681-4,685 bytes and BLT (4096-byte context) scored them
left-truncated. Here the 4k cell keeps the same seed-0 RNG stream per mode, but each item's haystack is the largest
for which BOS + prompt + " " + value <= 4096 bytes (binary search from a saved RNG state, as
dump_sniah_fit_pairs.py). Output: score_items.py "gen" items (greedy teacher-forced exact match).

  AUNET_ROOT=/mnt/ssd2/hyun2/AUNet python scripts/niah/dump_nolima_fit_pairs.py --out reports/niah/nolima_fit4k/items.jsonl
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import niah_ext_data as D  # noqa: E402


def scored(s):
    return 1 + len((s["prompt"] + " " + s["values"][0]).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = []
    for tb in (1024, 2048, a.budget):
        for mode, lit in (("literal", True), ("paraphrase", False)):
            rng = random.Random(a.seed)
            for i in range(a.n):
                if tb == a.budget:
                    state = rng.getstate()

                    def build(t):
                        rng.setstate(state)
                        return D.make_nolima(t, rng, lit, 3, "essay")
                    lo, hi = 400, a.budget
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        lo, hi = (mid, hi) if scored(build(mid)) <= a.budget else (lo, mid - 1)
                    s, hay = build(lo), lo
                else:
                    s, hay = D.make_nolima(tb, rng, lit, 3, "essay"), tb
                n = scored(s)
                assert n <= 4096 and s["values"][0] in s["prompt"]
                cell = f"nolima/{mode}/{tb}"
                recs.append({"type": "gen", "probe": "nolima", "task": f"nolima_{mode}", "cond": cell, "cell": cell,
                             "length": tb, "depth": None, "target_bytes": hay, "scored_bytes_with_bos": n,
                             "key": f"{cell}/{i}", "prompt": s["prompt"], "value": s["values"][0]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} -> {a.out}")
    for c in sorted(Counter(r["cell"] for r in recs)):
        v = [r["scored_bytes_with_bos"] for r in recs if r["cell"] == c]
        print(f"  {c:24s} n={len(v)} bytes {min(v)}-{max(v)}")


if __name__ == "__main__":
    main()
