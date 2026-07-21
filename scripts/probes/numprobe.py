#!/usr/bin/env python3
"""Number probes: how tokenization affects number handling. No free generation --
teacher-forced greedy EXACT-MATCH (as in NIAH) for generation-style answers, and
option-loglikelihood CLOZE for the comparison task.

Subtasks (prompts end at a clean separator, so the boundary-format confound that
inflated NIAH does not apply here):
  arith    a {+,-,x} b =            -> exact-match of the numeric result
  reverse  digits of N reversed     -> exact-match
  index    k-th digit (from left)   -> exact-match
  compare  which of A,B is larger   -> cloze: argmax length-normalized LL over {A,B}
  copy_c   strip commas from N      -> exact-match  (1,234,567 -> 1234567)
  copy_s   strip inter-digit spaces -> exact-match  (1 2 3 4 -> 1234)

Byte / digit-level access should help where subword splits numbers inconsistently.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L


def _num(rng, d):
    return rng.randint(0, 9) if d <= 1 else rng.randint(10 ** (d - 1), 10 ** d - 1)


def _ord(k):
    return {1: "st", 2: "nd", 3: "rd"}.get(k if k < 20 else k % 10, "th")


# ---- exact-match subtasks -> [(prompt, answer)] ----
_OPS = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("x", lambda a, b: a * b)]


def gen_arith(rng, d, n):
    out = []
    for _ in range(n):
        sym, f = rng.choice(_OPS)
        a, b = _num(rng, d), _num(rng, d)
        if sym == "-" and b > a:
            a, b = b, a
        out.append((f"{a} {sym} {b} =", str(f(a, b))))
    return out


def gen_reverse(rng, d, n):
    out = []
    for _ in range(n):
        s = str(_num(rng, max(d, 2)))
        out.append((f"The digits of {s} written in reverse order are:", s[::-1]))
    return out


def gen_index(rng, d, n):
    out = []
    for _ in range(n):
        s = str(_num(rng, max(d, 2)))
        k = rng.randint(1, len(s))
        out.append((f"The {k}{_ord(k)} digit from the left of {s} is:", s[k - 1]))
    return out


def gen_copy(rng, d, n, sep):
    out = []
    for _ in range(n):
        s = str(_num(rng, max(d, 4)))
        if sep == "comma":
            out.append((f"Write {int(s):,} without commas:", s))
        else:
            out.append((f"Write {' '.join(s)} without spaces:", s))
    return out


# ---- cloze subtasks -> [(prompt, [correct, distractor], gold_idx=0)] ----
def _flip(rng, s):  # change one digit to a different one (same length distractor)
    i = rng.randrange(len(s))
    return s[:i] + rng.choice([c for c in "0123456789" if c != s[i]]) + s[i + 1:]


def gen_compare(rng, d, n):
    out = []
    for _ in range(n):
        a, b = _num(rng, d), _num(rng, d)
        while b == a:
            b = _num(rng, d)
        out.append((f"Which number is larger, {a} or {b}? Answer:", [str(a), str(b)],
                    0 if a > b else 1))
    return out


def gen_arith_cloze(rng, d, n):
    out = []
    for _ in range(n):
        sym, f = rng.choice(_OPS)
        a, b = _num(rng, d), _num(rng, d)
        if sym == "-" and b > a:
            a, b = b, a
        c = str(f(a, b))
        dist = _flip(rng, c)
        while dist == c:
            dist = _flip(rng, c)
        out.append((f"{a} {sym} {b} =", [c, dist], 0))
    return out


def gen_reverse_cloze(rng, d, n):
    out = []
    for _ in range(n):
        s = str(_num(rng, max(d, 2)))
        r = s[::-1]
        dist = _flip(rng, s) if r == s else s  # natural distractor: the un-reversed original
        out.append((f"The digits of {s} written in reverse order are:", [r, dist], 0))
    return out


def gen_copy_cloze(rng, d, n):
    out = []
    for _ in range(n):
        s = str(_num(rng, max(d, 4)))
        dist = _flip(rng, s)
        while dist == s:
            dist = _flip(rng, s)
        out.append((f"Write {int(s):,} without commas:", [s, dist], 0))
    return out


def run_exact(gen, tok, fam, items, batch):
    res = score(gen, tok, [fmt_pair(p, a, fam) for p, a in items], batch)
    return sum(r[0] for r in res) / len(res)


def run_cloze(gen, tok, fam, items, batch):
    pairs, spans = [], []
    for p, opts, gold in items:
        spans.append((len(pairs), len(opts), gold))
        pairs += [fmt_pair(p, o, fam) for o in opts]
    res = score(gen, tok, pairs, batch)  # r = (exact, n_correct, n_tok, ll_sum)
    correct = 0
    for start, k, gold in spans:
        ll = [res[start + j][3] / max(1, res[start + j][2]) for j in range(k)]  # length-norm
        correct += (max(range(k), key=lambda j: ll[j]) == gold)
    return correct / len(items)


SUBTASKS = {
    "arith":         ("exact", gen_arith),
    "reverse":       ("exact", gen_reverse),
    "index":         ("exact", gen_index),
    "copy_comma":    ("exact", lambda r, d, n: gen_copy(r, d, n, "comma")),
    "copy_space":    ("exact", lambda r, d, n: gen_copy(r, d, n, "space")),
    "compare":       ("cloze", gen_compare),
    "arith_cloze":   ("cloze", gen_arith_cloze),
    "reverse_cloze": ("cloze", gen_reverse_cloze),
    "copy_cloze":    ("cloze", gen_copy_cloze),
}
DEFAULT_SUBTASKS = ["arith", "reverse", "index", "compare", "copy_comma", "copy_space"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--digits", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--subtasks", nargs="+", default=DEFAULT_SUBTASKS, choices=list(SUBTASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/numprobe.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"NUMPROBE | models={args.models} digits={args.digits} subtasks={args.subtasks} n={args.n}")
    for tag in args.models:
        fam, ckpt = ckpt_for(tag)
        gen, tok = build_generator(fam, ckpt, os.environ.get("AUNET_TOK"), 8192)
        for d in args.digits:
            rng = random.Random(args.seed + d)  # same items per model at each d
            r = {"tag": tag, "digits": d, "n": args.n}
            for st in args.subtasks:
                kind, genf = SUBTASKS[st]
                items = genf(rng, d, args.n)
                r[st] = (run_exact if kind == "exact" else run_cloze)(gen, tok, fam, items, args.batch_size)
            print("  " + f"{tag:16s} d={d:2d}: "
                  + " ".join(f"{st}={r[st]:.2f}" for st in args.subtasks), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
        del gen


if __name__ == "__main__":
    main()
