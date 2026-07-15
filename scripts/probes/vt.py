#!/usr/bin/env python3
"""RULER Variable Tracking (VT): chained bindings, retrieve the final value.

A target chain X0=V, X1=X0, X2=X1, ... X_{R-1}=X_{R-2} (V a 7-digit number) is
mixed (shuffled) with distractor chains and light noise; the query asks for the
value of the LAST variable in the target chain. Solving it needs multi-hop
reference following == recall + state tracking. Exact-match vs #hops R.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L
sys.path.insert(0, f"{L}/scripts/niah")
from niah_data import NOISE  # noqa: E402


def _var(rng):
    return "X" + "".join(rng.choice("0123456789") for _ in range(4))


def make_vt(num_hops, num_chains, rng, noise_sents=6):
    stmts = []
    target_last, target_val = None, None
    for c in range(num_chains):
        V = "".join(rng.choice("0123456789") for _ in range(7))
        vs = [_var(rng) for _ in range(num_hops)]
        stmts.append(f"VAR {vs[0]} = {V}.")
        for i in range(1, num_hops):
            stmts.append(f"VAR {vs[i]} = {vs[i-1]}.")
        if c == 0:                                # chain 0 is the target
            target_last, target_val = vs[-1], V
    rng.shuffle(stmts)
    # sprinkle light noise between statements
    body = []
    for s in stmts:
        body.append(s)
        if rng.random() < 0.5:
            body.append(NOISE.strip())
    text = " ".join(body)
    prompt = (f"Memorize the variable assignments below, then answer.\n\n{text}\n\n"
              f"Trace the references: the numeric value of VAR {target_last} is")
    return prompt, target_val


def run_model(tag, n, hop_counts, num_chains, seed, batch):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rng = random.Random(seed)
    rows = []
    for R in hop_counts:
        pairs = []
        for _ in range(n):
            prompt, ans = make_vt(R, num_chains, rng)
            pairs.append(fmt_pair(prompt, ans, fam))
        res = score(gen, tok, pairs, batch)
        em = sum(r[0] for r in res) / len(res)
        rows.append({"tag": tag, "num_hops": R, "num_chains": num_chains,
                     "n": len(res), "exact_match": em})
        print(f"  {tag:16s} hops={R} exact={em:.3f}")
    del gen
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--hop_counts", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--num_chains", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/vt_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"VT | hops={args.hop_counts} chains={args.num_chains} n={args.n}")
    for tag in args.models:
        for r in run_model(tag, args.n, args.hop_counts, args.num_chains,
                           args.seed, args.batch_size):
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
