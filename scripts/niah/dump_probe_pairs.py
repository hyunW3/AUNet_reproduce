#!/usr/bin/env python3
"""Dump model-free teacher-forcing pairs for NoLiMa and MK-NIAH so the SAME
prompts can be scored by every model (Llama/AU-Net/BPEByte via the local harness,
BLT-1B in its own repo). Each record is one item:

  {"probe","cell","target_bytes","prompt","values"}

Scoring convention (byte + subword identical here): teacher-force
`prompt + " " + value` and check greedy exact-match over the value bytes.

Configs match the paper runs:
  NoLiMa : lengths 1024/2048/4096, n=50, 3 distractors, literal+paraphrase (seed 0)
  MK-NIAH: K in {1,2,4,8} at 2k bytes, n=200 per K, target needle at a spread of
           depths, K-1 distractors at random depths (seed 0)
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import niah_ext_data as D                                   # noqa: E402  (nolima)
from niah_data import KEYS, _essay_pool, INSTRUCTION        # noqa: E402  (mk primitives)


def mk_sample(target_bytes, target_depth, K, rng):
    """MK-NIAH item: 1 queried needle at target_depth + K-1 distractors (mk_grid)."""
    keys = rng.sample(KEYS, K)
    vals = ["".join(rng.choice("0123456789") for _ in range(7)) for _ in keys]
    needles = [f"One of the special magic numbers for {k} is {v}. "
               for k, v in zip(keys, vals)]
    pool = _essay_pool()
    start = rng.randint(0, max(0, len(pool) - target_bytes - 1))
    offs = [target_depth] + [rng.random() for _ in range(K - 1)]
    order = sorted(range(K), key=lambda i: offs[i])
    qk, ans = keys[0], vals[0]
    query = (f"What is the special magic number for {qk} mentioned in the "
             f"provided text? The special magic number for {qk} is")
    body = pool[start:start + target_bytes]
    parts, prev = [], 0
    for i in order:
        c = min(len(body), max(prev, int(offs[i] * len(body))))
        while 0 < c < len(body) and body[c] != " ":
            c += 1
        parts.append(body[prev:c]); parts.append(needles[i]); prev = c
    parts.append(body[prev:])
    return {"prompt": f"{INSTRUCTION}\n\n{''.join(parts)}\n\n{query}", "values": [ans]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    recs = []

    # --- NoLiMa (literal + paraphrase) ---
    for tb in (1024, 2048, 4096):
        for mode, lit in (("literal", True), ("paraphrase", False)):
            sm = D.make_dataset("nolima", tb, 1, 50, args.seed, "essay", lit, 3)
            for s in sm:
                recs.append({"probe": "nolima", "cell": f"nolima/{mode}/{tb}",
                             "target_bytes": tb, "prompt": s["prompt"], "values": s["values"]})

    # --- MK-NIAH (K sweep at 2k) ---
    for K in (1, 2, 4, 8):
        rng = random.Random(args.seed * 977 + K)
        for i in range(200):
            depth = (i + 0.5) / 200.0                        # spread target depth over [0,1]
            s = mk_sample(2048, depth, K, rng)
            recs.append({"probe": "mkniah", "cell": f"mkniah/K{K}",
                         "K": K, "target_bytes": 2048, "prompt": s["prompt"], "values": s["values"]})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r["cell"] for r in recs)
    print(f"wrote {len(recs)} pairs -> {args.out}")
    for k in sorted(c):
        print(f"  {k:22} {c[k]}")


if __name__ == "__main__":
    main()
