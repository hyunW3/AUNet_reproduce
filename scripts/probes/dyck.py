#!/usr/bin/env python3
"""Dyck-k bracket matching (nested-stack state tracking, in-context probe).

Random balanced Dyck-k words over k bracket types; at every closing position the
model must predict the closer that matches the most recent unmatched opener (top
of stack). Tests hierarchical/stack state, complementary to S5's group state.
Chance ~ 1/k (if the model knows a close is due). We score close-prediction
accuracy overall and bucketed by nesting depth.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L

OPEN, CLOSE = "([{<", ")]}>"


def gen_dyck(length, k, maxdepth, rng):
    opens, closes = OPEN[:k], CLOSE[:k]
    match = dict(zip(opens, closes))
    toks, stack, reads, steps = [], [], [], 0
    while steps < length or stack:
        can_open = len(stack) < maxdepth and steps < length
        can_close = len(stack) > 0
        if can_open and (not can_close or rng.random() < 0.5):
            o = rng.choice(opens)
            toks.append(o); stack.append(o); steps += 1
        else:
            depth = len(stack)                       # nesting level being closed
            o = stack.pop(); c = match[o]
            reads.append((" ".join(toks), c, depth))  # predict c from prefix
            toks.append(c)
    return " ".join(toks), reads


def fewshot(shots, length, k, maxdepth, seed):
    if shots <= 0:
        return ""
    rng = random.Random(seed + 777)
    return "".join(gen_dyck(length, k, maxdepth, rng)[0] + "\n" for _ in range(shots))


def run_model(tag, n, length, k, maxdepth, shots, seed, batch):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rng = random.Random(seed)
    prefix = fewshot(shots, length, k, maxdepth, seed)
    pairs, meta = [], []
    for _ in range(n):
        _txt, reads = gen_dyck(length, k, maxdepth, rng)
        for ctx, ans, depth in reads:
            pairs.append(fmt_pair(prefix + ctx, ans, fam))
            meta.append(depth)
    res = score(gen, tok, pairs, batch)
    ok = [r[0] for r in res]
    by = defaultdict(lambda: [0, 0])
    for m, r in zip(meta, res):
        by[min(m, 8)][0] += r[0]; by[min(m, 8)][1] += 1
    del gen
    return {"tag": tag, "n_reads": len(ok), "acc": sum(ok) / len(ok), "k": k,
            "by_depth": {kk: v[0] / v[1] for kk, v in sorted(by.items())},
            "by_depth_n": {kk: v[1] for kk, v in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--maxdepth", type=int, default=6)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/dyck_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"Dyck-{args.k} | length={args.length} maxdepth={args.maxdepth} "
          f"shots={args.shots} n={args.n} (chance~{1/args.k:.2f})")
    for tag in args.models:
        r = run_model(tag, args.n, args.length, args.k, args.maxdepth,
                      args.shots, args.seed, args.batch_size)
        print(f"  {tag:16s} acc={r['acc']:.3f}  n={r['n_reads']}", flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
