#!/usr/bin/env python3
"""NumerSense (Lin et al., EMNLP 2020) on frozen 1.3B LMs, as a cloze.

Each item is a commonsense sentence with a masked number word ("Latin verbs
have <mask> principal parts." -> "four"). We split at <mask> and score each of
the 12 candidate words (zero..ten, no) substituted into the sentence, i.e. the
continuation "<candidate><suffix>" after the prefix; argmax over per-character
mean loglikelihood. Uses the labeled train split (test labels are
leaderboard-hidden). Uniform chance = 1/12; majority-class ("two") = 0.214.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L

URL = "https://raw.githubusercontent.com/INK-USC/NumerSense/main/data/train.masked.tsv"
CANDS = ["no", "zero", "one", "two", "three", "four", "five", "six",
         "seven", "eight", "nine", "ten"]


def load(n):
    cache = Path.home() / ".cache" / "numersense" / "train.masked.tsv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, cache)
    items = []
    for line in open(cache, encoding="utf-8"):
        if "\t" not in line or "<mask>" not in line:
            continue
        sent, gold = line.rstrip("\n").split("\t")
        prefix, suffix = sent.split("<mask>", 1)
        # need a non-empty prefix for causal scoring and a non-empty suffix so
        # every candidate shares the same continuation tail
        if not prefix.strip() or not suffix.strip() or gold not in CANDS:
            continue
        items.append((prefix, suffix, gold))
        if len(items) >= n:
            break
    return items


def run_model(tag, n, batch, out):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 2048)
    items = load(n)
    pairs = []
    for prefix, suffix, _gold in items:
        for c in CANDS:
            pairs.append(fmt_pair(prefix.rstrip(" "), c + suffix, fam))
    res = score(gen, tok, pairs, batch)
    correct = 0
    for j, (_p, suffix, gold) in enumerate(items):
        lls = [res[j * len(CANDS) + i][3] / max(1, len(CANDS[i] + suffix))
               for i in range(len(CANDS))]
        pred = CANDS[max(range(len(CANDS)), key=lambda i: lls[i])]
        correct += int(pred == gold)
    acc = correct / max(1, len(items))
    row = {"tag": tag, "task": "numersense", "n": len(items), "acc": acc,
           "chance": 1 / len(CANDS)}
    print(f"  {tag:16s} numersense             acc={acc:.3f} (chance {1/len(CANDS):.3f}) n={len(items)}",
          flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/numersense_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"NumerSense cloze | n={args.n} candidates={len(CANDS)}")
    for tag in args.models:
        run_model(tag, args.n, args.batch_size, args.out)


if __name__ == "__main__":
    main()
