#!/usr/bin/env python3
"""NumericBench (TreeAILab/NumericBench, Yang et al. 2025) num_list config on
frozen 1.3B LMs, as a multiple-choice cloze (loglikelihood over the letter
options). Abilities: comparison / summary / contextual retrieval (500 each).

Each item: a number list (struct_data), a question with lettered options
("Options: A: 25, B: 62, ..."), and a letter answer. We present the list + the
question, score P(" <letter>") for each option, and take the argmax -- no
generation. NB: NumericBench targets frontier models, so 1.3B is expected near
the per-item chance (1/#options).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from common import build_generator, ckpt_for, score, CKPT, L

FILES = {"100": "num_list/num_list_500_per_sample_100_length.json",
         "1000": "num_list/num_list_500_per_sample_1000_length.json"}


def load(length):
    f = hf_hub_download("TreeAILab/NumericBench", FILES[length], repo_type="dataset")
    D = json.load(open(f))
    return D["system_prompt"], D["data"]


def options(q):
    return re.findall(r"([A-Z]):", q.split("Options:")[-1]) if "Options:" in q else []


def run_model(tag, abilities, n_per, length, batch, out):
    sysp, Q = load(length)
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 4096)
    rows = []
    for ab in abilities:
        items = [q for q in Q if q.get("ability") == ab][:n_per]
        pairs, layout = [], []
        for q in items:
            opts = options(q["question"])
            if len(opts) < 2:
                continue
            lst = ", ".join(f"{i}:{v}" for i, v in enumerate(q["struct_data"]))
            ctx = f"{sysp}\n\nList (index:value): {lst}\n\n{q['question']}\nThe answer is"
            base = len(pairs)
            for opt in opts:
                pairs.append((ctx, " " + opt))
            layout.append((base, opts, q["answer"]))
        res = score(gen, tok, pairs, batch)
        correct = 0
        for base, opts, gold in layout:
            lls = [res[base + i][3] for i in range(len(opts))]
            pred = opts[max(range(len(opts)), key=lambda i: lls[i])]
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
