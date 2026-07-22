#!/usr/bin/env python3
"""AcademicEval (ulab-ai/AcademicEval) on frozen 1.3B LMs, converted to a top-1
token-accuracy (cloze-style, no generation) task.

Configs: title_10K, abs_9K, intro_8K. Each row gives the paper body
(main_content) and the gold section to write (gt = title / abstract / intro). We
truncate the body to ~8k bytes to fit our models' window, append a section cue,
teacher-force the gold section, and report the fraction of gold tokens that are
the model's top-1 prediction (plus whole-section exact-match, which floors).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import build_generator, ckpt_for, score, CKPT, L

CUES = {"title_10K": "Title:", "abs_9K": "Abstract:", "intro_8K": "Introduction:"}


def _trunc_bytes(s: str, nbytes: int) -> str:
    return s.encode("utf-8")[:nbytes].decode("utf-8", "ignore")


def build(ex, cfg, ctx_bytes, gold_bytes):
    ctx = _trunc_bytes(ex["main_content"], ctx_bytes).rstrip() + f"\n\n{CUES[cfg]}"
    gold = _trunc_bytes(" " + ex["gt"].strip(), gold_bytes)
    return ctx, gold


def run_model(tag, cfgs, n, ctx_bytes, gold_bytes, batch, max_tokens):
    from datasets import load_dataset
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, max_tokens)
    rows = []
    for cfg in cfgs:
        ds = load_dataset("ulab-ai/AcademicEval", cfg, split="test", streaming=True)
        pairs, seen = [], 0
        for ex in ds:
            pairs.append(build(ex, cfg, ctx_bytes, gold_bytes))
            seen += 1
            if seen >= n:
                break
        res = score(gen, tok, pairs, batch)               # raw (ctx, gold): score handles the cut
        tf = sum(r[1] / r[2] if r[2] else 0.0 for r in res) / len(res)
        ex_acc = sum(r[0] for r in res) / len(res)
        rows.append({"tag": tag, "config": cfg, "n": len(res), "ctx_bytes": ctx_bytes,
                     "top1_tok_acc": tf, "exact": ex_acc})
        print(f"  {tag:16s} {cfg:10s} top1_tok={tf:.3f} exact={ex_acc:.3f} n={len(res)}",
              flush=True)
    del gen
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--configs", nargs="+", default=["title_10K", "abs_9K", "intro_8K"])
    ap.add_argument("--n", type=int, default=60, help="papers per config")
    ap.add_argument("--ctx_bytes", type=int, default=6500)
    ap.add_argument("--gold_bytes", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/academiceval_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"AcademicEval | configs={args.configs} n={args.n} ctx_bytes={args.ctx_bytes} "
          f"gold_bytes={args.gold_bytes}")
    for tag in args.models:
        for r in run_model(tag, args.configs, args.n, args.ctx_bytes, args.gold_bytes,
                           args.batch_size, args.max_tokens):
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
