#!/usr/bin/env python3
"""Turn dyck_mech jsonl into the markdown block for reports/statetrack/REPORT.md.

Reads reports/statetrack/dyck_mech/*.jsonl and prints, per config, the
stack-vs-recency split with Wilson 95% CIs.
"""
import glob
import json
import math
import os
import sys

ORDER = ["subword_llama", "aunet_static", "byte_greedyroot"]
NAME = {"subword_llama": "Llama", "aunet_static": "AUNet",
        "byte_greedyroot": "BPEByte rg"}
CFGS = [("dyck3_brackets", "Dyck-3, ASCII brackets"),
        ("dyck4_brackets", "Dyck-4, ASCII brackets"),
        ("dyck4_letters",  "Dyck-4, letter delimiters (control)")]


def wilson(p, n, z=1.96):
    if not n:
        return (float("nan"),) * 2
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt(p, n):
    lo, hi = wilson(p, n)
    return f"{p:.3f} [{lo:.3f},{hi:.3f}]"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "reports/statetrack/dyck_mech")
    for stem, title in CFGS:
        paths = sorted(glob.glob(os.path.join(d, stem + "__*.jsonl")))
        if not paths:
            p1 = os.path.join(d, stem + ".jsonl")
            paths = [p1] if os.path.exists(p1) else []
        if not paths:
            print(f"<!-- {stem}: not run -->\n"); continue
        rows = {}
        for path in paths:
            for l in open(path):
                if l.strip():
                    r = json.loads(l); rows[r["tag"]] = r
        if not rows:
            continue
        any_r = next(iter(rows.values()))
        k = any_r["k"]
        base = any_r["lure_baseline_if_stack"]
        print(f"**{title}** — k={k}, {any_r['n_reads']} scored reads/model, "
              f"**forced choice** over the {k} closers (argmax of answer "
              f"log-likelihood), so `acc` here is not comparable to `dyck.py`'s "
              f"free-argmax exact-match — see the note below the table.\n")
        print(f"`recency-share` = of the ERRORS on diagnostic reads, the fraction "
              f"that are exactly the recency answer. Unbiased errors give "
              f"1/(k-1) = {base:.3f}; pure recency copying gives 1.0. (The raw "
              f"lure rate P(pred==recency) is in the jsonl; it is not the right "
              f"comparison because it is bounded by the error rate.)\n")
        print("| model | acc all | acc (recency agrees) | acc (recency **disagrees**) | Δ | recency-share of errors |")
        print("|---|---|---|---|---|---|")
        for t in ORDER:
            r = rows.get(t)
            if not r:
                continue
            a, m = r["agree"], r["disagree"]
            err = 1.0 - m["acc"]
            n_err = int(round(err * m["n"]))
            share = (m["lure"] / err) if err > 1e-9 else float("nan")
            print(f"| {NAME[t]} | {r['acc_all']:.3f} | {fmt(a['acc'], a['n'])} "
                  f"| {fmt(m['acc'], m['n'])} | {a['acc'] - m['acc']:+.3f} "
                  f"| {fmt(share, n_err)} |")
        print(f"\nn: agree={any_r['agree']['n']}, disagree={any_r['disagree']['n']} "
              f"({any_r['disagree']['n']/any_r['n_reads']:.1%} of reads). "
              f"Answer token counts seen: {any_r['answer_ntoks']} "
              f"(uniform => the log-likelihood argmax is not length-biased).\n")

        # --- distance-matched version -------------------------------------
        # The agree/disagree split above is CONFOUNDED by dependency distance:
        # whenever the previous token is an opener the answer is at distance 1
        # and recency necessarily agrees, so `agree` is dominated by short
        # dependencies (mean dist 3.1 vs 11.3 for `disagree`).
        # Restricting BOTH subsets to reads whose previous token is a CLOSER
        # removes it: "agree by luck" (the last opener coincidentally has the
        # same bracket type as the top of stack) and `disagree` are then matched
        # on distance (11.3 vs 11.3) and depth (2.78 vs 2.77).
        # In that regime the recency answer is literally the PREVIOUS TOKEN ~71%
        # of the time, so this measures a copy-the-previous-token bias:
        #   Δ > 0  -> helped when the answer repeats the previous token (copy bias)
        #   Δ < 0  -> hurt by it (repetition avoidance)
        print("Distance-matched (both subsets have a **closer** as the previous "
              "token, so dependency distance and depth are matched):\n")
        print("| model | acc, recency right by luck | acc, recency wrong | copy-bias Δ |")
        print("|---|---|---|---|")
        for t in ORDER:
            r = rows.get(t)
            if not r:
                continue
            mis, dis = r["misaligned"], r["disagree"]
            n_ma = mis["n"] - dis["n"]
            if n_ma <= 0:
                continue
            acc_ma = (mis["acc"] * mis["n"] - dis["acc"] * dis["n"]) / n_ma
            print(f"| {NAME[t]} | {fmt(acc_ma, n_ma)} (n={n_ma}) "
                  f"| {fmt(dis['acc'], dis['n'])} (n={dis['n']}) "
                  f"| {acc_ma - dis['acc']:+.3f} |")
        print(f"\nForced-choice chance on both subsets = 1/{k} = {1.0/k:.3f}.\n")


if __name__ == "__main__":
    main()
