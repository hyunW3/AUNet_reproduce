#!/usr/bin/env python3
"""TKEval (EMNLP'24 Findings "Tokenization Falling Short", github.com/FloatAI/TKEval).

Runs the OFFICIAL benchmark data (HF floatai/TKEval, cloned under benchmarks/TKEval)
on the 1.3B paper checkpoints with the shared teacher-forced greedy scorer:
an item is correct if the model greedily reproduces ANY of the official acceptable
targets verbatim (the data ships a list of valid answers per item).

Two task groups, official prompts used verbatim (prebuilt k-shot in the JSON):
  complex_problem_solving/  : "...\\nQ: ...\\nA: " prompts -> answer follows " "
  token_structure_probing/  : "##...?\\n" prompts        -> answer follows directly
Big tasks (cycled/unscramble: 9k-21k items) are subsampled with a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import build_generator, ckpt_for, score, L

BASE = f"{L}/benchmarks/TKEval"
# task -> (relative json, answer-cut style)
TASKS = {
    "cycled_letters":   ("complex_problem_solving/cycled_letters_all_data_0123_shots.json", "qa"),
    "unscramble":       ("complex_problem_solving/word_unscrambling_all_data_0123_shots.json", "qa"),
    "math_theorems":    ("complex_problem_solving/identify_math_theorems_all_data_0123_shots.json", "qa"),
    "common_substr":    ("token_structure_probing/test/multi_token_prob.common_substrings.all_data_0123_shots.json", "raw"),
    "lcsubseq":         ("token_structure_probing/test/multi_token_prob.longest_common_subsequences.all_data_0123_shots.json", "raw"),
    "lcsubstr":         ("token_structure_probing/test/multi_token_prob.longest_common_substrings.all_data_0123_shots.json", "raw"),
    "case_conversion":  ("token_structure_probing/test/token_struct_prob.char_case_conversion.all_data_0123_shots.json", "raw"),
    "char_count":       ("token_structure_probing/test/token_struct_prob.character_count.all_data_0123_shots.json", "raw"),
    "nth_char":         ("token_structure_probing/test/token_struct_prob.nth_character.all_data_0123_shots.json", "raw"),
    "nth_char_end":     ("token_structure_probing/test/token_struct_prob.nth_character_from_end.all_data_0123_shots.json", "raw"),
}


def build_pairs(items, cut, family):
    """[(ctx, cont, item_idx)] — one pair per acceptable target; OR-scored per item."""
    pairs = []
    for i, x in enumerate(items):
        inp = x["input"]
        for tgt in x["target"]:
            if cut == "qa":     # prompt ends "A: " -> continuation carries the space
                pairs.append((inp.rstrip(" "), " " + tgt, i))
            else:               # struct probes: answer directly after the trailing \n
                pairs.append((inp, tgt, i))
    return pairs


def run_model(tag, tasks, shots, limit, seed, batch):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rows = []
    for name in tasks:
        rel, cut = TASKS[name]
        data = json.load(open(f"{BASE}/{rel}"))[str(shots)]
        if limit and len(data) > limit:
            data = random.Random(seed).sample(data, limit)
        pairs = build_pairs(data, cut, fam)
        res = score(gen, tok, [(c, k) for c, k, _ in pairs], batch)
        best_exact = {}
        best_tf = {}
        for (c, k, ii), r in zip(pairs, res):
            best_exact[ii] = max(best_exact.get(ii, 0), int(r[0]))
            best_tf[ii] = max(best_tf.get(ii, 0.0), r[1] / r[2] if r[2] else 0.0)
        n = len(data)
        row = {"tag": tag, "task": name, "shots": shots, "n": n,
               "acc": sum(best_exact.values()) / n,
               "tok_frac": sum(best_tf.values()) / n}
        print(f"[{tag}] {name:16s} n={n:4d} acc={row['acc']:.3f} tok_frac={row['tok_frac']:.3f}",
              flush=True)
        rows.append(row)
    del gen
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--limit", type=int, default=500)   # caps the 9k/21k tasks
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/tkeval/results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for tag in args.models:
            for row in run_model(tag, args.tasks, args.shots, args.limit, args.seed,
                                 args.batch_size):
                f.write(json.dumps(row) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
