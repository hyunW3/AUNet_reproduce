#!/usr/bin/env python3
"""Paired comparison of two downstream runs from eval_downstream_updater.py.

Accuracy alone cannot resolve the effects under test here (BPB moves ~0.4%; a 1.3B model's
accuracy on these sets carries ~1pt of noise).  Same items, same order, so this reports what
accuracy hides: how many items FLIPPED in each direction (McNemar), and how the gold-vs-best-rival
logprob margin moved -- a continuous signal that registers changes far below one flipped item.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

METRIC = {"hellaswag": "pred_norm", "arc_challenge": "pred_norm", "arc_easy": "pred_norm",
          "boolq": "pred"}
SCORE = {"hellaswag": "lp_norm", "arc_challenge": "lp_norm", "arc_easy": "lp_norm",
         "boolq": "lp"}


def load(d, tag, task):
    return [json.loads(l) for l in open(Path(d) / f"{tag}_{task}.jsonl")]


def margin(row, key):
    """gold logprob minus the best rival's -- positive means the item is answered correctly."""
    s = np.asarray(row[key], dtype=float)
    g = row["gold"]
    rival = np.max(np.delete(s, g))
    return s[g] - rival


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/patch_lag/downstream")
    ap.add_argument("--base", default="base_frozen")
    ap.add_argument("--runs", required=True, help="comma-separated tags to compare against base")
    ap.add_argument("--tasks", default="hellaswag,arc_challenge,boolq")
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    print(f"\npaired vs `{args.base}` — same items, same order\n")
    print("| run | task | acc | Δ acc (pt) | 좋아짐 / 나빠짐 | McNemar p | Δ margin (nats) |")
    print("|---|---|---|---|---|---|---|")
    for tag in args.runs.split(","):
        accs = []
        for task in tasks:
            b = load(args.dir, args.base, task)
            r = load(args.dir, tag, task)
            assert len(b) == len(r) and all(x["gold"] == y["gold"] for x, y in zip(b, r))
            key, skey = METRIC.get(task, "pred"), SCORE.get(task, "lp")
            cb = np.array([x[key] == x["gold"] for x in b])
            cr = np.array([x[key] == x["gold"] for x in r])
            n01 = int((~cb & cr).sum())          # base wrong -> run right
            n10 = int((cb & ~cr).sum())          # base right -> run wrong
            # exact binomial (McNemar) on the discordant pairs
            n = n01 + n10
            if n == 0:
                p = 1.0
            else:
                k = min(n01, n10)
                p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
            mb = np.array([margin(x, skey) for x in b])
            mr = np.array([margin(x, skey) for x in r])
            accs.append(cr.mean())
            print(f"| {tag} | {task} | {cr.mean() * 100:.2f} | {(cr.mean() - cb.mean()) * 100:+.2f} "
                  f"| {n01} / {n10} | {p:.3f} | {(mr - mb).mean():+.4f} |")
        print(f"| **{tag}** | **avg3** | **{np.mean(accs) * 100:.2f}** | | | | |")


if __name__ == "__main__":
    main()
