#!/usr/bin/env python3
"""Scoring-rule ablation for the natural-language state-tracking battery.

Compares the three multiple-choice decision rules the harness can apply to the
SAME forward passes:

  acc              argmax of the raw summed option log-likelihood. Biased toward
                   short options; 99.5% of TSO items have unequal-length choices.
  acc_norm         argmax of lls / len(choice) -- length-normalised.
  acc_mutual_info  argmax of log P(choice|ctx) - log P(choice) -- removes each
                   option's unconditional prior (PMI).

The point is to decide whether the first run's uniformly BELOW-chance TSO numbers
were a scoring artifact or a real absence of signal. Also cross-checks that `acc`
reproduces the original nlstate/ run exactly.
"""
import json
import math
import os
import sys

ORDER = ["subword_llama", "aunet_static", "byte_greedyroot"]
NAME = {"subword_llama": "Llama", "aunet_static": "AUNet",
        "byte_greedyroot": "BPEByte rg"}
TASKS = [
    ("bb_tso3_mc", "TSO, 3 objects", 1 / 3, 750),
    ("bb_tso5_mc", "TSO, 5 objects", 1 / 5, 1250),
    ("bb_tso7_mc", "TSO, 7 objects", 1 / 7, 1750),
    ("bb_temporal_mc", "Temporal sequences", 1 / 4, 1000),
    ("bb_navigate_mc", "Navigate", 1 / 2, 1000),
]
METRICS = [("acc,none", "acc_stderr,none", "acc (raw sum)"),
           ("acc_norm,none", "acc_norm_stderr,none", "acc_norm (length)"),
           ("acc_mutual_info,none", "acc_mutual_info_stderr,none", "PMI")]
# old-run task names -> new
OLD = {"bb_tso3_mc": "bb_tso3_mc", "bb_tso5_mc": "bb_tso5_mc",
       "bb_tso7_mc": "bb_tso7_mc",
       "bb_temporal_mc": "bigbench_temporal_sequences_multiple_choice",
       "bb_navigate_mc": "bigbench_navigate_multiple_choice"}


def load(d, tag):
    p = os.path.join(d, tag, "results.json")
    return json.load(open(p)).get("results", {}) if os.path.exists(p) else None


def z(acc, se, ch):
    return (acc - ch) / se if se else float("nan")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "reports/statetrack/nlstate_pmi"
    old_d = sys.argv[2] if len(sys.argv) > 2 else "reports/statetrack/nlstate"
    res = {t: load(d, t) for t in ORDER}
    have = [t for t in ORDER if res[t]]
    if not have:
        print("<!-- nlstate_pmi: not run -->"); return

    for mkey, skey, mlabel in METRICS:
        print(f"**{mlabel}** — accuracy ±stderr (σ above chance)\n")
        print("| task | chance | " + " | ".join(NAME[t] for t in have) + " |")
        print("|---|---|" + "---|" * len(have))
        for key, label, ch, n in TASKS:
            row = []
            for t in have:
                r = (res[t] or {}).get(key)
                if not r or mkey not in r:
                    row.append("—"); continue
                a, se = r[mkey], r.get(skey) or 0.0
                row.append(f"{a:.3f} ±{se:.3f} ({z(a,se,ch):+.1f}σ)")
            print(f"| {label} | {ch:.3f} | " + " | ".join(row) + " |")
        print()

    # pooled TSO per metric
    print("**Pooled tracking_shuffled_objects** (n=3750, mean chance exactly 0.2000 "
          "— the same floor as the synthetic S5 probe)\n")
    print("| model | " + " | ".join(m[2] for m in METRICS) + " | synthetic S5 | chance |")
    print("|---|" + "---|" * (len(METRICS) + 2))
    S5 = {"subword_llama": 0.166, "aunet_static": 0.192, "byte_greedyroot": 0.212}
    for t in have:
        cells = []
        for mkey, _s, _l in METRICS:
            num = den = 0.0; ok = True
            for key, _lab, _ch, n in TASKS[:3]:
                r = (res[t] or {}).get(key)
                if not r or mkey not in r:
                    ok = False; break
                num += r[mkey] * n; den += n
            cells.append(f"{num/den:.3f}" if ok and den else "—")
        print(f"| {NAME[t]} | " + " | ".join(cells) + f" | {S5[t]:.3f} | 0.200 |")

    # consistency: acc must match the original run exactly
    print("\n**Reproducibility check** — `acc` vs the original `nlstate/` run "
          "(identical forward passes, so these must be equal):\n")
    bad = 0
    for t in have:
        o = load(old_d, t)
        if not o:
            print(f"- {NAME[t]}: original run not found"); continue
        diffs = []
        for key, _l, _c, _n in TASKS:
            new = (res[t] or {}).get(key, {}).get("acc,none")
            oldv = (o or {}).get(OLD[key], {}).get("acc,none")
            if new is None or oldv is None:
                continue
            if abs(new - oldv) > 1e-9:
                diffs.append(f"{key} {oldv:.4f}->{new:.4f}"); bad += 1
        print(f"- {NAME[t]}: " + ("all 5 tasks identical" if not diffs else "MISMATCH " + "; ".join(diffs)))
    if bad:
        print(f"\n**{bad} mismatched cell(s)** — investigate before trusting the ablation.")


if __name__ == "__main__":
    main()
