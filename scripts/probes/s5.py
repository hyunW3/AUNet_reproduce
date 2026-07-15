#!/usr/bin/env python3
"""S5 permutation-composition state tracking (in-context probe).

5 slots hold distinct symbols A..E; "swap i j" transposes two slots; "get k ="
asks for the symbol currently in slot k (== the composed permutation applied to
the initial arrangement). This is the canonical HARD state-tracking task
(Merrill et al., "The Illusion of State"; "(How) Do LMs Track State?", ICML'25);
FFLM's 1-bit register is its easiest special case. Chance = 1/5 = 0.20.

We score read accuracy (greedy argmax == correct symbol) overall and bucketed by
the number of swaps applied so far (a difficulty/distance axis).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L

SYMS = "ABCDE"


def gen_sequence(num_events, p_read, rng):
    arr = list(SYMS)
    text = "start " + " ".join(SYMS) + " ."
    reads = []                     # (prefix_ending_at '=', answer, nswaps)
    nsw = 0
    for _ in range(num_events):
        if rng.random() < p_read:
            k = rng.randint(0, 4)
            ans = arr[k]
            ctx = text + f" get {k} ="
            reads.append((ctx, ans, nsw))
            text = ctx + f" {ans} ."
        else:
            i, j = rng.sample(range(5), 2)
            arr[i], arr[j] = arr[j], arr[i]
            nsw += 1
            text += f" swap {i} {j} ."
    return text, reads


def fewshot(shots, num_events, p_read, seed):
    if shots <= 0:
        return ""
    rng = random.Random(seed + 555)
    return "".join(gen_sequence(num_events, p_read, rng)[0] + "\n" for _ in range(shots))


def run_model(tag, n, num_events, p_read, shots, seed, batch):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rng = random.Random(seed)
    prefix = fewshot(shots, num_events, p_read, seed)
    pairs, meta = [], []
    for _ in range(n):
        _txt, reads = gen_sequence(num_events, p_read, rng)
        for ctx, ans, nsw in reads:
            pairs.append(fmt_pair(prefix + ctx, ans, fam))
            meta.append(nsw)
    res = score(gen, tok, pairs, batch)
    ok = [r[0] for r in res]
    by = defaultdict(lambda: [0, 0])
    for m, r in zip(meta, res):
        b = min(m, 20)
        by[b][0] += r[0]; by[b][1] += 1
    del gen
    return {"tag": tag, "n_reads": len(ok), "acc": sum(ok) / len(ok),
            "by_nswaps": {k: v[0] / v[1] for k, v in sorted(by.items())},
            "by_nswaps_n": {k: v[1] for k, v in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--num_events", type=int, default=40)
    ap.add_argument("--p_read", type=float, default=0.3)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/s5_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"S5 | events={args.num_events} p_read={args.p_read} shots={args.shots} "
          f"n={args.n} (chance=0.20)")
    for tag in args.models:
        r = run_model(tag, args.n, args.num_events, args.p_read, args.shots,
                      args.seed, args.batch_size)
        print(f"  {tag:16s} acc={r['acc']:.3f}  n={r['n_reads']}")
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
