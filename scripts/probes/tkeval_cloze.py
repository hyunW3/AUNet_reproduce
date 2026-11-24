#!/usr/bin/env python3
"""TKEval single-char probes re-scored as CLOZE (constrained top-1 over a-z).

The generation scoring showed byte models continuing the question pattern instead of
answering ("\\n\\n##..."), so their ~0 measured format compliance, not character skill.
Here the SAME official 2-shot prompts are scored by ranking the 26 lowercase letters as
the continuation and taking argmax total loglikelihood -- format-free: correct iff the
top-1 letter is in the official target list.
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

from common import build_generator, ckpt_for, score, L

BASE = f"{L}/benchmarks/TKEval/token_structure_probing/test"
TASKS = {
    "char_count":   "token_struct_prob.character_count.all_data_0123_shots.json",
    "nth_char":     "token_struct_prob.nth_character.all_data_0123_shots.json",
    "nth_char_end": "token_struct_prob.nth_character_from_end.all_data_0123_shots.json",
}
CANDS = list(string.ascii_lowercase)


def run_model(tag, shots, batch):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rows = []
    for name, fn in TASKS.items():
        data = json.load(open(f"{BASE}/{fn}"))[str(shots)]
        pairs = [(x["input"], c) for x in data for c in CANDS]
        res = score(gen, tok, pairs, batch)
        ncand = len(CANDS)
        correct = 0
        for i, x in enumerate(data):
            lls = [res[i * ncand + j][3] for j in range(ncand)]
            pred = CANDS[max(range(ncand), key=lambda j: lls[j])]
            correct += int(pred in x["target"])
        acc = correct / len(data)
        print(f"[{tag}] {name:14s} n={len(data)} cloze_acc={acc:.3f}", flush=True)
        rows.append({"tag": tag, "task": name, "shots": shots, "n": len(data),
                     "cloze_acc": acc})
    del gen
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=f"{L}/reports/tkeval/results_cloze.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for tag in args.models:
            for row in run_model(tag, args.shots, args.batch_size):
                f.write(json.dumps(row) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
