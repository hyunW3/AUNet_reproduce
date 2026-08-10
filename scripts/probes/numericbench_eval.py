#!/usr/bin/env python3
"""NumericBench (TreeAILab/NumericBench, Yang et al. 2025) num_list config on
frozen 1.3B LMs, as a value cloze. Abilities: comparison / summary /
contextual retrieval (500 each).

Each item: a number list (struct_data, a JSON-encoded string -- parse it!),
a question with lettered options ("Options: A: 25, B: 62, ..."), and a letter
answer. We strip the option letters from the question and score each option's
*value* as the continuation of "... The answer is"; argmax over per-character
mean loglikelihood (values differ in digit count, and per-char normalization
is tokenizer-agnostic across the subword/byte families) -- no generation.
NB: NumericBench targets frontier models, so 1.3B is expected near the
per-item chance (1/#options).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L

FILES = {"100": "num_list/num_list_500_per_sample_100_length.json",
         "1000": "num_list/num_list_500_per_sample_1000_length.json"}


def load(length):
    f = hf_hub_download("TreeAILab/NumericBench", FILES[length], repo_type="dataset")
    D = json.load(open(f))
    return D["system_prompt"], D["data"]


def split_cloze(q):
    """(question stem without the option list, [(letter, value_str), ...])."""
    if "Options:" not in q:
        return q.strip(), []
    stem, tail = q.split("Options:", 1)
    opts = re.findall(r"([A-Z]):\s*(.*?)\s*(?=,\s*[A-Z]:|$)", tail.strip(), re.S)
    return stem.strip(), opts


def run_model(tag, abilities, n_per, length, batch, out):
    _sysp, Q = load(length)
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 4096)
    rows = []
    for ab in abilities:
        items = [q for q in Q if q.get("ability") == ab][:n_per]
        pairs, layout = [], []
        for q in items:
            stem, opts = split_cloze(q["question"])
            if len(opts) < 2:
                continue
            vals = [s.strip() for s in q["struct_data"].strip()[1:-1].split(",")]
            lst = ", ".join(f"{i}:{v}" for i, v in enumerate(vals))
            ctx = f"List (index:value): {lst}\n\n{stem}\nThe answer is"
            base = len(pairs)
            for _letter, val in opts:
                pairs.append(fmt_pair(ctx, val, fam))
            layout.append((base, opts, q["answer"]))
        res = score(gen, tok, pairs, batch)
        correct = 0
        for base, opts, gold in layout:
            lls = [res[base + i][3] / max(1, len(v)) for i, (_l, v) in enumerate(opts)]
            pred = opts[max(range(len(opts)), key=lambda i: lls[i])][0]
            correct += int(pred == gold)
        acc = correct / max(1, len(layout))
        chance = sum(1 / len(o) for _, o, _ in layout) / max(1, len(layout))
        rows.append({"tag": tag, "ability": ab, "n": len(layout),
                     "acc": acc, "chance": chance})
        print(f"  {tag:16s} {ab:22s} acc={acc:.3f} (chance {chance:.3f}) n={len(layout)}",
              flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--abilities", nargs="+",
                    default=["comparison", "summary", "contextual retrieval"])
    ap.add_argument("--n_per", type=int, default=150)
    ap.add_argument("--length", choices=["100", "1000"], default="100")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/numericbench_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"NumericBench num_list ({args.length}) | abilities={args.abilities} n_per={args.n_per}")
    for tag in args.models:
        run_model(tag, args.abilities, args.n_per, args.length, args.batch_size, args.out)


if __name__ == "__main__":
    main()
