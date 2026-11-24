#!/usr/bin/env python3
"""Dyck-k MECHANISM probe: is the close-prediction a stack, or recency copying?

`dyck.py` reports close-prediction accuracy only, which cannot distinguish
"maintains a stack" from "copies the closer of the most recent opener it saw".
Those two policies AGREE whenever the token right before the read is an opener
(the top of stack IS the most recent opener) and DISAGREE otherwise.

So every read is labelled:
  agree    : recency closer == gold closer -> both policies predict the SAME thing
  disagree : recency closer != gold closer -> recency predicts a WRONG closer

Note the split is on (recency == gold), NOT on "is the previous token an opener".
Those differ: after a close, the last opener in the prefix may coincidentally be
the same bracket TYPE as the top of stack (prob ~1/k), in which case recency is
accidentally right and the read is not diagnostic.

We report
  acc_agree / acc_disagree         : a stack keeps these equal; recency copying
                                     collapses acc_disagree toward 0
  lure_disagree = P(pred == recency): on diagnostic reads, how often the model
                                     emits exactly the recency answer. A stack ->
                                     ~1/(k-1) (no preference among wrong closers).
                                     Pure recency copying -> ~1.

Prediction is FORCED-CHOICE: each of the k closers is scored as the continuation
and we take the argmax of the answer log-likelihood, so `pred` is always defined
(unlike dyck.py's free-argmax exact-match, which only yields a boolean).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L

OPEN, CLOSE = "([{<ABCD", ")]}>abcd"
LETTERS_OPEN, LETTERS_CLOSE = "ABCDEFGH", "abcdefgh"


def gen_dyck_mech(length, k, maxdepth, rng, opens, closes):
    """Balanced Dyck word; yield one record per close position.

    Mirrors dyck.py:gen_dyck exactly (same rng call sequence, same 0.5 open bias,
    same ' '.join context) so accuracies stay comparable, and adds the labels the
    mechanism split needs.
    """
    match = dict(zip(opens, closes))
    toks, stack, reads, steps = [], [], [], 0
    open_pos = []                       # stack of positions, parallel to `stack`
    while steps < length or stack:
        can_open = len(stack) < maxdepth and steps < length
        can_close = len(stack) > 0
        if can_open and (not can_close or rng.random() < 0.5):
            o = rng.choice(opens)
            toks.append(o); stack.append(o); open_pos.append(len(toks) - 1); steps += 1
        else:
            depth = len(stack)
            o = stack.pop(); c = match[o]
            opos = open_pos.pop()
            # recency heuristic: closer of the LAST opener anywhere in the prefix,
            # matched or not. Equals gold iff the previous token is that opener.
            last_open = next((t for t in reversed(toks) if t in match), None)
            recency = match[last_open] if last_open is not None else None
            prev_is_open = bool(toks) and toks[-1] in match
            reads.append({
                "ctx": " ".join(toks),
                "gold": c,
                "recency": recency,
                "aligned": prev_is_open,
                "depth": depth,
                "dist": len(toks) - opos,        # tokens from matching opener to here
            })
            toks.append(c)
    return " ".join(toks), reads


def fewshot(shots, length, k, maxdepth, seed, opens, closes):
    if shots <= 0:
        return ""
    rng = random.Random(seed + 777)
    return "".join(gen_dyck_mech(length, k, maxdepth, rng, opens, closes)[0] + "\n"
                   for _ in range(shots))


def run_model(tag, n, length, k, maxdepth, shots, seed, batch, letters, dump):
    opens = (LETTERS_OPEN if letters else OPEN)[:k]
    closes = (LETTERS_CLOSE if letters else CLOSE)[:k]
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rng = random.Random(seed)
    prefix = fewshot(shots, length, k, maxdepth, seed, opens, closes)

    reads = []
    for _ in range(n):
        _txt, rs = gen_dyck_mech(length, k, maxdepth, rng, opens, closes)
        reads.extend(rs)

    # forced choice: score every candidate closer for every read
    pairs, ntok = [], None
    for r in reads:
        for c in closes:
            pairs.append(fmt_pair(prefix + r["ctx"], c, fam))
    res = score(gen, tok, pairs, batch)
    del gen

    ntoks = {t[2] for t in res}
    for i, r in enumerate(reads):
        lls = [res[i * k + j][3] for j in range(k)]
        r["pred"] = closes[max(range(k), key=lambda j: lls[j])]
        r["correct"] = int(r["pred"] == r["gold"])
        r["is_lure"] = int(r["pred"] == r["recency"])

    def agg(sel):
        s = [r for r in reads if sel(r)]
        if not s:
            return None
        return {"n": len(s), "acc": sum(r["correct"] for r in s) / len(s),
                "lure": sum(r["is_lure"] for r in s) / len(s)}

    agree = agg(lambda r: r["recency"] == r["gold"])
    dis = agg(lambda r: r["recency"] != r["gold"])
    # secondary split, kept for reference: prev-token-is-an-opener
    al, mis = agg(lambda r: r["aligned"]), agg(lambda r: not r["aligned"])
    by_depth = {d: agg(lambda r, d=d: min(r["depth"], 6) == d) for d in range(1, 7)}
    out = {
        "tag": tag, "k": k, "letters": bool(letters), "n_reads": len(reads),
        "shots": shots, "answer_ntoks": sorted(ntoks),
        "acc_all": sum(r["correct"] for r in reads) / len(reads),
        "agree": agree, "disagree": dis,
        "aligned": al, "misaligned": mis,
        "lure_baseline_if_stack": 1.0 / (k - 1) if k > 1 else None,
        "by_depth": {str(d): v for d, v in by_depth.items() if v},
    }
    if dump:
        Path(dump).write_text("\n".join(json.dumps(r) for r in reads) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--maxdepth", type=int, default=6)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--letters", action="store_true")
    ap.add_argument("--dump", default=None, help="per-read jsonl (one model only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Dyck-mech k={args.k} letters={args.letters} n={args.n} shots={args.shots}",
          flush=True)
    with open(args.out, "a") as fh:
        for tag in args.models:
            r = run_model(tag, args.n, args.length, args.k, args.maxdepth,
                          args.shots, args.seed, args.batch_size, args.letters,
                          args.dump)
            a, m = r["agree"], r["disagree"]
            print(f"  {tag:16s} acc={r['acc_all']:.3f} | agree {a['acc']:.3f} "
                  f"(n={a['n']}) | disagree {m['acc']:.3f} (n={m['n']}) "
                  f"lure={m['lure']:.3f} vs {r['lure_baseline_if_stack']:.3f} if-stack",
                  flush=True)
            fh.write(json.dumps(r) + "\n"); fh.flush()


if __name__ == "__main__":
    main()
