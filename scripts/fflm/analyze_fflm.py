#!/usr/bin/env python3
"""Diagnostics that explain *why* a family wins on FFLM.

Reads reports/fflm/per_read.jsonl (emitted by fflm_probe.py --per_read_out) and
produces two artifacts:

  fflm_acc_vs_distance.png  — greedy read accuracy vs dependency distance
                              (#instructions back to the governing write). A flat
                              line = robust long-range state tracking; a line that
                              decays = the model loses the bit over distance.

  recency_glitch.md         — accuracy split by whether the NEAREST distractor bit
                              disagrees with the true state. A model that "copies
                              the nearest bit" (attention glitch) scores far worse
                              on the disagree subset; a true state-tracker is
                              indifferent. Reports the gap per family.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_LABEL = {"subword_llama": "subword (Llama)",
                "aunet_static": "AUNet",
                "byte_greedyroot": "BPEByte rg"}
FAMILY_COLOR = {"subword_llama": "#4C78A8", "aunet_static": "#F58518",
                "byte_greedyroot": "#54A24B"}
# distance buckets (instructions back to governing write)
BUCKETS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 1 << 30)]
BLABEL = ["1-2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]


def bucket(d):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= d <= hi:
            return i
    return len(BUCKETS) - 1


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def acc_vs_distance(rows, outdir):
    # agg[tag][bucket] = [n_ok, n_total]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        b = bucket(r["dist"])
        agg[r["tag"]][b][0] += r["greedy_ok"]
        agg[r["tag"]][b][1] += 1
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    tags = [t for t in FAMILY_LABEL if t in agg]
    for tag in tags:
        xs, ys, ns = [], [], []
        for b in range(len(BUCKETS)):
            ok, tot = agg[tag][b]
            if tot >= 20:            # skip tiny buckets
                xs.append(b); ys.append(ok / tot); ns.append(tot)
        ax.plot(xs, ys, "o-", label=FAMILY_LABEL.get(tag, tag),
                color=FAMILY_COLOR.get(tag))
    ax.axhline(0.5, ls="--", lw=1, color="grey", label="chance")
    ax.set_xticks(range(len(BLABEL)))
    ax.set_xticklabels(BLABEL)
    ax.set_xlabel("dependency distance (instructions back to governing write)")
    ax.set_ylabel("greedy read accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("FFLM read accuracy vs. dependency distance")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(outdir) / "fflm_acc_vs_distance.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


def recency_glitch(rows, outdir):
    # agg[tag][prev_disagrees] = [n_ok, n_total]
    agg = defaultdict(lambda: {0: [0, 0], 1: [0, 0]})
    for r in rows:
        cell = agg[r["tag"]][r["prev_disagrees"]]
        cell[0] += r["greedy_ok"]; cell[1] += 1
    lines = ["# FFLM recency-glitch test", "",
             "Accuracy split by whether the **nearest distractor bit disagrees** "
             "with the true state. A model that copies the nearest bit (attention "
             "glitch) collapses on the *disagree* subset; a true state-tracker is "
             "flat. `gap = acc(agree) − acc(disagree)` — bigger = more glitch-prone.",
             "",
             "| family | acc (prev agrees) | acc (prev disagrees) | gap | n |",
             "|---|---|---|---|---|"]
    for tag in [t for t in FAMILY_LABEL if t in agg]:
        a_ok, a_n = agg[tag][0]
        d_ok, d_n = agg[tag][1]
        a = a_ok / a_n if a_n else float("nan")
        d = d_ok / d_n if d_n else float("nan")
        lines.append(f"| {FAMILY_LABEL.get(tag, tag)} | {a:.3f} ({a_n}) | "
                     f"{d:.3f} ({d_n}) | {a - d:+.3f} | {a_n + d_n} |")
    out = Path(outdir) / "recency_glitch.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_read", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows = load(args.per_read)
    if not rows:
        print("no per-read rows"); return
    acc_vs_distance(rows, args.outdir)
    recency_glitch(rows, args.outdir)


if __name__ == "__main__":
    main()
