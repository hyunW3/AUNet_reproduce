#!/usr/bin/env python3
"""Dump lm-eval MC items ({task, idx, context, choices, gold}) with the repo's item builder
(apps.aunet.eval_pbp_mc._items_from_task: doc_to_text / doc_to_choice / doc_to_target, first N docs),
so build_tasks2.py can perturb byte-identical prompts offline. Run in the lingua venv with
PYTHONPATH=<lingua> and the HF datasets cache available.

  python export_mc.py --tasks arc_easy arc_challenge piqa hellaswag --n 500 --out mc_items.jsonl
"""
import argparse
import json

from apps.aunet.eval_pbp_mc import _items_from_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["arc_easy", "arc_challenge", "piqa", "hellaswag"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for t in a.tasks:
            items = _items_from_task(t, a.n)
            for i, it in enumerate(items):
                f.write(json.dumps({"task": t, "idx": i, **it}, ensure_ascii=False) + "\n")
            print(t, len(items))


if __name__ == "__main__":
    main()
