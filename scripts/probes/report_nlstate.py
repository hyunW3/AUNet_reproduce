#!/usr/bin/env python3
"""Aggregate the natural-language state-tracking (BIG-bench MC) results.

Reads reports/statetrack/nlstate/<tag>/results.json and emits the markdown block
for reports/statetrack/REPORT.md.

Every task here has a fixed, known chance floor, and at 1.3B base scale the
absolute accuracies are expected to sit near it -- so the table reports accuracy,
its stderr, and how many stderr the model is ABOVE CHANCE. That z is the number
to read; the raw accuracy on its own says little.
"""
import json
import math
import os
import sys

ORDER = ["subword_llama", "aunet_static", "byte_greedyroot"]
NAME = {"subword_llama": "Llama", "aunet_static": "AUNet",
        "byte_greedyroot": "BPEByte rg"}
# task -> (label, chance, n)
TASKS = [
    ("bb_tso3_mc", "Tracking shuffled objects, 3", 1 / 3, 750),
    ("bb_tso5_mc", "Tracking shuffled objects, 5", 1 / 5, 1250),
    ("bb_tso7_mc", "Tracking shuffled objects, 7", 1 / 7, 1750),
    ("bigbench_temporal_sequences_multiple_choice", "Temporal sequences", 1 / 4, 1000),
    ("bigbench_navigate_multiple_choice", "Navigate", 1 / 2, 1000),
]
TSO = ["bb_tso3_mc", "bb_tso5_mc", "bb_tso7_mc"]


def load(d, tag):
    p = os.path.join(d, tag, "results.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("results", {})


def cell(acc, se, chance):
    if acc is None:
        return "—"
    z = (acc - chance) / se if se else float("nan")
    return f"{acc:.3f} ±{se:.3f} ({z:+.1f}σ)"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "reports/statetrack/nlstate"
    res = {t: load(d, t) for t in ORDER}
    have = [t for t in ORDER if res[t]]
    if not have:
        print("<!-- nlstate: not run -->"); return

    print("Accuracy ±stderr, and (in parentheses) how many stderr **above the "
          "chance floor**. Multiple choice scored by option log-likelihood; the "
          "generate_until / CoT variants of these tasks are not meaningful for "
          "1.3B base checkpoints.\n")
    print("| task | chance | n | " + " | ".join(NAME[t] for t in have) + " |")
    print("|---|---|---|" + "---|" * len(have))
    for key, label, chance, n in TASKS:
        row = []
        for t in have:
            r = (res[t] or {}).get(key)
            if not r:
                row.append("—"); continue
            row.append(cell(r.get("acc,none"), r.get("acc_stderr,none"), chance))
        print(f"| {label} | {chance:.3f} | {n} | " + " | ".join(row) + " |")

    # pooled tracking_shuffled_objects: chance is exactly 0.2000 for the 750/1250/1750
    # mix, which makes it directly comparable to the synthetic S5 probe (chance 1/5).
    print()
    pooled = {}
    for t in have:
        num = den = 0.0
        ok = True
        for key in TSO:
            r = (res[t] or {}).get(key)
            if not r:
                ok = False; break
            n = dict((k, nn) for k, _l, _c, nn in TASKS)[key]
            num += r["acc,none"] * n; den += n
        pooled[t] = (num / den if ok and den else None, den)
    print("**Pooled tracking_shuffled_objects** (750+1250+1750 = 3750 items). The "
          "3/5/7 mix has a mean chance floor of exactly **0.2000**, the same floor "
          "as the synthetic S5 permutation probe — so these two numbers measure "
          "the same abstract task (composition of transpositions) in symbols vs "
          "in English.\n")
    print("| model | NL tracking_shuffled_objects | synthetic S5 (depth-controlled) | chance |")
    print("|---|---|---|---|")
    S5 = {"subword_llama": 0.166, "aunet_static": 0.192, "byte_greedyroot": 0.212}
    for t in have:
        p = pooled[t][0]
        print(f"| {NAME[t]} | {p:.3f} | {S5[t]:.3f} | 0.200 |" if p is not None
              else f"| {NAME[t]} | — | {S5[t]:.3f} | 0.200 |")
    print("\n(S5 column = mean of the D=1/2/3 exact-match rows in §\"S5 permutation "
          "at controlled depth\", n=900/model.)")


if __name__ == "__main__":
    main()
