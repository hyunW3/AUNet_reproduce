#!/usr/bin/env python3
"""GPT-3 arithmetic suite (EleutherAI/arithmetic, Brown et al. 2020) on frozen
1.3B LMs, teacher-forced exact-match (our protocol). The HF dataset ships a
loader script that datasets>=4 rejects, so we download the raw JSONL directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import build_generator, ckpt_for, score, CKPT, L

CONFIGS = {
    "2da": "two_digit_addition", "2ds": "two_digit_subtraction",
    "2dm": "two_digit_multiplication",
    "3da": "three_digit_addition", "3ds": "three_digit_subtraction",
    "4da": "four_digit_addition", "4ds": "four_digit_subtraction",
    "5da": "five_digit_addition", "5ds": "five_digit_subtraction",
    "1dc": "single_digit_three_ops",
}


def load_cfg(cfg, n):
    f = hf_hub_download("EleutherAI/arithmetic", f"data/{CONFIGS[cfg]}.jsonl",
                        repo_type="dataset")
    items = []
    for line in open(f):
        d = json.loads(line)
        items.append((d["context"], d["completion"]))   # ctx="...A:", completion=" 143"
        if len(items) >= n:
            break
    return items


def run_model(tag, configs, n, batch, out):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 2048)
    rows = []
    for cfg in configs:
        items = load_cfg(cfg, n)
        res = score(gen, tok, items, batch)              # teacher-forced exact-match
        em = sum(r[0] for r in res) / len(res)
        tf = sum(r[1] / r[2] if r[2] else 0.0 for r in res) / len(res)
        rows.append({"tag": tag, "config": cfg, "n": len(res),
                     "exact": em, "digit_acc": tf})
        print(f"  {tag:16s} arithmetic_{cfg:4s} exact={em:.3f} digit_acc={tf:.3f} n={len(res)}",
              flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--configs", nargs="+",
                    default=["2da", "2ds", "2dm", "3da", "3ds", "4da", "4ds", "5da", "5ds", "1dc"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/arithmetic_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"GPT-3 arithmetic | configs={args.configs} n={args.n}")
    for tag in args.models:
        run_model(tag, args.configs, args.n, args.batch_size, args.out)


if __name__ == "__main__":
    main()
