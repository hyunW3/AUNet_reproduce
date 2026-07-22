#!/usr/bin/env python3
"""Evaluate the OFFICIAL FFLM dataset (synthseq/flipflop, Liu et al. 2023) on the
frozen 1.3B LMs, in-context -- instead of our home-grown generator.

Splits: val (FFL(0.8), in-dist), val_dense, val_sparse. Each row is a dense
2-char-token string 'w0i1r1...' (w=write, i=ignore, r=read; no spaces). At every
read 'r' the model must predict the bit == the most recent write. Metrics mirror
fflm_probe: greedy exact-match and binary-rank (LL(correct) > LL(wrong)).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fflm_probe import build_generator, score_pairs, DEFAULT_TOK  # noqa: E402


def parse_reads(text: str):
    """'w0i1r1...' -> [(context_ending_at_'r', correct_bit, wrong_bit)] per read."""
    reads, state = [], None
    for k in range(len(text) // 2):
        instr, bit = text[2 * k], text[2 * k + 1]
        if instr == "w":
            state = bit
        elif instr == "r" and state is not None:
            reads.append((text[:2 * k + 1], state, "1" if state == "0" else "0"))
    return reads


def run(family, ckpt, tag, splits, n_seq, max_reads, batch, tok_path, out):
    from datasets import load_dataset
    generator, tokenizer = build_generator(family, ckpt, tok_path, 4096)
    rows = []
    for split in splits:
        ds = load_dataset("synthseq/flipflop", split=split, streaming=True)
        cpairs, wpairs = [], []
        seen = 0
        for ex in ds:
            reads = parse_reads(ex["text"])
            if max_reads and len(reads) > max_reads:
                reads = reads[:max_reads]
            for ctx, corr, wrong in reads:
                cpairs.append((ctx, corr))     # dense format: no space between 'r' and bit
                wpairs.append((ctx, wrong))
            seen += 1
            if seen >= n_seq:
                break
        corr = score_pairs(generator, tokenizer, cpairs, batch)
        wrong = score_pairs(generator, tokenizer, wpairs, batch)
        greedy = [c[1] for c in corr]
        binary = [c[0] > w[0] for c, w in zip(corr, wrong)]
        N = len(greedy)
        row = {"tag": tag, "split": split, "n_seq": seen, "n_reads": N,
               "greedy_acc": sum(greedy) / N, "binary_acc": sum(binary) / N}
        rows.append(row)
        print(f"  {tag:16s} {split:10s} greedy={row['greedy_acc']:.3f} "
              f"binary={row['binary_acc']:.3f}  (n_reads={N})", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["subword", "aunet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "val_dense", "val_sparse"])
    ap.add_argument("--n_seq", type=int, default=200, help="sequences per split")
    ap.add_argument("--max_reads", type=int, default=8, help="reads scored per sequence")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.family, args.ckpt, args.tag, args.splits, args.n_seq, args.max_reads,
        args.batch_size, args.tok_path, args.out)


if __name__ == "__main__":
    main()
