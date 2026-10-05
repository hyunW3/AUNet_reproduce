#!/usr/bin/env python3
"""Base-vs-hard overview for the state-tracking / recall battery (3 models)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/reports/statetrack"
ORDER = ["subword_llama", "aunet_static", "byte_greedyroot"]
LBL = {"subword_llama": "Llama", "aunet_static": "AUNet", "byte_greedyroot": "BPEByte rg"}
COL = {"subword_llama": "#0072B2", "aunet_static": "#E69F00", "byte_greedyroot": "#009E73"}


def load(f):
    p = Path(R) / f
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    return rows


def acc_by_tag(f):  # dedup, keep last
    d = {}
    for r in load(f):
        d[r["tag"]] = r["acc"]
    return d


def main():
    s5b, s5h = acc_by_tag("s5_results.jsonl"), acc_by_tag("s5_hard_results.jsonl")
    dyb, dyh = acc_by_tag("dyck_results.jsonl"), acc_by_tag("dyck_hard_results.jsonl")
    mkb, mkh = load("mkniah_results.jsonl"), load("mkniah_hard_results.jsonl")
    vtb, vth = load("vt_results.jsonl"), load("vt_hard_results.jsonl")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    x = range(len(ORDER)); w = 0.36

    def bars(a, base, hard, chance_b, chance_h, title, ylab):
        a.bar([i - w/2 for i in x], [base.get(t, 0) for t in ORDER], w,
              color=[COL[t] for t in ORDER], label="base")
        a.bar([i + w/2 for i in x], [hard.get(t, 0) for t in ORDER], w,
              color=[COL[t] for t in ORDER], alpha=0.55, hatch="///", label="hard")
        a.axhline(chance_b, ls="--", lw=1, color="#888")
        if abs(chance_h - chance_b) > 1e-3:
            a.axhline(chance_h, ls=":", lw=1, color="#888")
        a.set_xticks(list(x)); a.set_xticklabels([LBL[t] for t in ORDER], fontsize=9)
        a.set_ylim(0, 1); a.set_title(title, fontsize=11); a.set_ylabel(ylab)
        a.legend(fontsize=8, loc="upper left")

    bars(ax[0][0], s5b, s5h, 0.20, 0.20,
         "S5 permutation (base: 40ev/2-shot · hard: 60ev/4-shot)", "read acc")
    bars(ax[0][1], dyb, dyh, 0.33, 0.25,
         "Dyck (base: k3/depth6 · hard: k4/depth10)", "close-pred acc")
    ax[0][0].text(0.02, 0.205, "chance", fontsize=7, color="#888")
    ax[0][1].text(0.02, 0.34, "chance k3", fontsize=7, color="#888")
    ax[0][1].text(0.02, 0.26, "chance k4", fontsize=7, color="#888")

    def lines(a, base, hard, xkey, title, xlab):
        for t in ORDER:
            rb = sorted([r for r in base if r["tag"] == t], key=lambda r: r[xkey])
            rh = sorted([r for r in hard if r["tag"] == t], key=lambda r: r[xkey])
            if rb:
                a.plot([r[xkey] for r in rb], [r["exact_match"] for r in rb], "o-",
                       color=COL[t], label=f"{LBL[t]} base")
            if rh:
                a.plot([r[xkey] for r in rh], [r["exact_match"] for r in rh], "s--",
                       color=COL[t], alpha=0.7, label=f"{LBL[t]} hard")
        a.set_title(title, fontsize=11); a.set_xlabel(xlab); a.set_ylabel("exact-match")
        a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3); a.legend(fontsize=7, ncol=3)

    lines(ax[1][0], mkb, mkh, "num_needles",
          "MK-NIAH (base: 2KB · hard: 4KB)", "# needles (K)")
    lines(ax[1][1], vtb, vth, "num_hops",
          "Variable Tracking (base: 3 chains · hard: 6 chains)", "# hops")
    fig.suptitle("State-tracking & recall — base vs hard (frozen 1.3B, in-context)",
                 fontweight="bold")
    fig.tight_layout()
    out = Path(R) / "statetrack_base_vs_hard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
