#!/usr/bin/env python3
"""Multi-Key NIAH (RULER): K needles with distinct keys, retrieve ONE key's value.

Adds disambiguation (the FFLM failure mode) on top of the copy-fidelity we
already measured with single-NIAH: the model must find the RIGHT needle among K.
Exact-match retrieval vs. number of needles K (at fixed length).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L
sys.path.insert(0, f"{L}/scripts/niah")
from niah_data import _essay_pool, KEYS, INSTRUCTION  # noqa: E402


def make_mk(target_bytes, num_needles, rng):
    keys = rng.sample(KEYS, num_needles)
    vals = ["".join(rng.choice("0123456789") for _ in range(7)) for _ in keys]
    needles = [f"One of the special magic numbers for {k} is: {v}. "
               for k, v in zip(keys, vals)]
    pool = _essay_pool()
    start = rng.randint(0, max(0, len(pool) - target_bytes - 1))
    body = pool[start:start + target_bytes]
    seg = max(1, len(body) // (num_needles + 1))
    parts = []
    for i in range(num_needles):
        parts.append(body[i * seg:(i + 1) * seg])
        parts.append(needles[i])
    parts.append(body[num_needles * seg:])
    haystack = "".join(parts)
    ti = rng.randrange(num_needles)
    qk, ans = keys[ti], vals[ti]
    query = (f"What is the special magic number for {qk} mentioned in the "
             f"provided text? The special magic number for {qk} is")
    return f"{INSTRUCTION}\n\n{haystack}\n\n{query}", ans


def run_model(tag, n, target_bytes, needle_counts, seed, batch, pbp=False, query_sep=""):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rng = random.Random(seed)
    rows = []
    for K in needle_counts:
        items = [make_mk(target_bytes, K, rng) for _ in range(n)]  # (prompt, ans) each
        if query_sep and not pbp:  # align query end to the needle "...is: <value>" format
            items = [(p.rstrip(" ") + query_sep, a) for p, a in items]
        if pbp:
            # Same items, both boundary cuts -> delta = space - canonical.
            stats = {}
            for var in ("canonical", "space"):
                res = score(gen, tok, [fmt_pair(p, a, fam, var) for p, a in items], batch)
                stats[var] = sum(r[0] for r in res) / len(res)
            d = stats["space"] - stats["canonical"]
            rows.append({"tag": tag, "num_needles": K, "target_bytes": target_bytes, "n": n,
                         "exact_canonical": stats["canonical"], "exact_space": stats["space"],
                         "pbp_delta_exact": round(d, 4)})
            print(f"  {tag:16s} K={K:2d} PBP exact canon={stats['canonical']:.3f} "
                  f"space={stats['space']:.3f} Δ={d:+.3f}")
        else:
            res = score(gen, tok, [fmt_pair(p, a, fam) for p, a in items], batch)
            em = sum(r[0] for r in res) / len(res)
            rows.append({"tag": tag, "num_needles": K, "target_bytes": target_bytes,
                         "n": len(res), "exact_match": em})
            print(f"  {tag:16s} K={K:2d} exact={em:.3f}")
    del gen
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--target_bytes", type=int, default=2048)
    ap.add_argument("--needle_counts", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--pbp", action="store_true",
                    help="prompt-boundary problem: score canonical vs trailing-space cut per K, "
                         "report delta (byte models ~0, subword moves)")
    ap.add_argument("--query_sep", default="",
                    help="append to the query (e.g. ':') to match the needle format and remove "
                         "the boundary-format confound in the standard eval")
    ap.add_argument("--out", default=f"{L}/reports/statetrack/mkniah_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"MK-NIAH | len={args.target_bytes} K={args.needle_counts} n={args.n}"
          + (" | PBP" if args.pbp else ""))
    for tag in args.models:
        for r in run_model(tag, args.n, args.target_bytes, args.needle_counts,
                           args.seed, args.batch_size, args.pbp, args.query_sep):
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
