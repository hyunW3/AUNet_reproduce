#!/usr/bin/env python3
"""Is AUNet's below-chance Temporal-sequences score a DISTRACTOR bias?

BIG-bench temporal_sequences has a perfect structural asymmetry (all 1000 items):
the GOLD interval never appears verbatim in the passage (0/1000), while every
DISTRACTOR does (3000/3000). The distractors ARE the observed intervals; the
answer is the unstated gap. So "prefer a span you already read" is an
anti-correct strategy that scores 0%, well under the 25% floor.

Reads the harness per-doc samples (needs harness.log_samples=true; the stock
eval config sets it false) and reports, per decision rule:

  acc        accuracy
  seen_rate  P(pick appears verbatim in the passage)
             0.75 = picking uniformly at random
             1.00 = pure already-seen preference
             0.00 = perfect solver
  pos_dist   distribution over choice index, to separate a positional bias

seen_rate is the statistic accuracy cannot give: every wrong option is a seen
span and the right one is unseen, so accuracy alone cannot say WHICH wrong
answer, only that it was wrong.
"""
import json
import math
import sys
from collections import Counter

NAME = {"subword_llama": "Llama", "aunet_static": "AUNet",
        "byte_greedyroot": "BPEByte rg"}
TASK = "bb_temporal_mc"


def wilson(p, n, z=1.96):
    if not n:
        return (float("nan"),) * 2
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def analyse(path):
    recs = json.load(open(path))["samples"][TASK]
    rules = {"acc": [], "acc_norm": [], "pmi": []}
    seen_flags, golds = [], []
    for r in recs:
        doc = r["doc"]
        ch = list(doc["multiple_choice_targets"])
        k = len(ch)
        body = doc["inputs"].split("  choice:")[0]
        passage = body.split("We know that:")[1] if "We know that:" in body else body
        seen = [c in passage for c in ch]
        # filtered_resps is a flat [[ll, is_greedy], ...]; resps nests one level deeper
        lls = [x[0] for x in r["filtered_resps"]]
        cond, uncond = lls[:k], lls[k:2 * k]
        clen = [float(len(c)) for c in ch]
        rules["acc"].append(max(range(k), key=lambda j: cond[j]))
        rules["acc_norm"].append(max(range(k), key=lambda j: cond[j] / clen[j]))
        if len(uncond) == k:
            rules["pmi"].append(max(range(k), key=lambda j: cond[j] - uncond[j]))
        seen_flags.append(seen)
        golds.append(int(r["target"]))
    n = len(recs)
    out = {"n": n,
           "gold_seen": sum(s[g] for s, g in zip(seen_flags, golds)) / n,
           "seen_if_uniform": sum(sum(s) / len(s) for s in seen_flags) / n}
    for rule, picks in rules.items():
        if not picks:
            continue
        acc = sum(p == g for p, g in zip(picks, golds)) / n
        sr = sum(s[p] for s, p in zip(seen_flags, picks)) / n
        pos = Counter(picks)
        out[rule] = {"acc": acc, "seen": sr, "seen_ci": wilson(sr, n),
                     "pos": {j: pos.get(j, 0) / n for j in range(4)}}
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "reports/statetrack/temporal_samples"
    res = {t: analyse(f"{d}/{t}/results.json") for t in NAME}
    a = next(iter(res.values()))
    print(f"n = {a['n']} items. GOLD appears verbatim in the passage: "
          f"{a['gold_seen']:.1%}. Mean options seen: "
          f"{a['seen_if_uniform']*4:.2f}/4, so picking uniformly gives "
          f"seen_rate = {a['seen_if_uniform']:.3f}.\n")
    for rule, label in [("acc", "acc (raw sum)"), ("acc_norm", "acc_norm (length)"),
                        ("pmi", "PMI")]:
        print(f"**{label}**\n")
        print("| model | acc (chance .250) | seen_rate (uniform .750, copy-bias 1.0) | choice-index dist |")
        print("|---|---|---|---|")
        for t, nm in NAME.items():
            r = res[t].get(rule)
            if not r:
                print(f"| {nm} | — | — | — |"); continue
            lo, hi = r["seen_ci"]
            pos = "/".join(f"{r['pos'][j]:.2f}" for j in range(4))
            print(f"| {nm} | {r['acc']:.3f} | **{r['seen']:.3f}** [{lo:.3f},{hi:.3f}] | {pos} |")
        print()


if __name__ == "__main__":
    main()
