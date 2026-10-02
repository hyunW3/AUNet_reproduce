#!/usr/bin/env python3
"""NoLiMa-lite (scripts/niah/niah_ext_data.make_nolima) as run_longctx.py gen rows, sized so that
official BLT (<= 4096 B incl. generation) can score every row. literal and paraphrase rows share
the seed, so item i of both conditions has the identical haystack / needles / values and differs
only in the query noun (needle noun vs. its zero-overlap synonym) -> paired NoLiMa effect.

  python scripts/probes/blt_weak/build_nolima.py --n 200 --out reports/blt_weak/data/nolima200.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "niah"))
from niah_ext_data import make_nolima  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 3400])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for L in a.lengths:
        for mode in ("literal", "paraphrase"):
            rng = random.Random(a.seed * 1000 + L)
            for i in range(a.n):
                s = make_nolima(L, rng, literal=(mode == "literal"))
                v = s["values"][0]
                rows.append({"id": f"nolima-{mode}-{L}-{i}", "task": "nolima", "cond": mode, "length": L,
                             "pos": i, "mode": "gen", "score": "substr", "prompt": s["prompt"],
                             "answers": [v], "window": len(v.encode()) + 2,
                             "prompt_bytes": len(s["prompt"].encode())})
    # the haystack is cut in characters, so non-ASCII text can push a row past 4096 B: drop the
    # whole (literal, paraphrase) pair so the paired comparison stays balanced
    big = {(r["length"], r["pos"]) for r in rows if 1 + r["prompt_bytes"] + r["window"] > 4096}
    rows = [r for r in rows if (r["length"], r["pos"]) not in big]
    print(f"dropped {len(big)} oversize item pairs: {sorted(big)}")
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows; max prompt", max(r["prompt_bytes"] for r in rows))


if __name__ == "__main__":
    main()
