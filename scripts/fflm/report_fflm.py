#!/usr/bin/env python3
"""Render the FFLM probe results: markdown table + grouped bar chart.

Reads reports/fflm/results.jsonl (rows written by fflm_probe.py) and writes
  reports/fflm/summary.md
  reports/fflm/fflm_read_accuracy.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIME_ORDER = ["dense", "indist", "sparse"]
REGIME_LABEL = {"dense": "dense\nFFL(0.1)", "indist": "in-dist\nFFL(0.8)",
                "sparse": "sparse-OOD\nFFL(0.98)"}
# colorblind-safe, distinct in light/dark
FAMILY_COLOR = {"subword_llama": "#4C78A8", "aunet_static": "#F58518",
                "byte_greedyroot": "#54A24B"}
FAMILY_LABEL = {"subword_llama": "subword (Llama)",
                "aunet_static": "AUNet",
                "byte_greedyroot": "BPEByte rg"}


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    # index[tag][regime] = row
    idx = defaultdict(dict)
    for r in rows:
        idx[r["tag"]][r["regime"]] = r
    return rows, idx


def write_md(rows, idx, outdir):
    lines = ["# FFLM in-context state-tracking probe", "",
             "Read-token accuracy of frozen 1.3B checkpoints on flip-flop sequences "
             "(Liu et al. 2023, arXiv:2306.00946). No training — in-context only.", "",
             "- **greedy_acc**: model's argmax next token == correct bit (the paper's "
             "strict metric; <100% = a *reasoning error*). Chance = 0.50.",
             "- **binary_acc**: P(correct bit) > P(wrong bit) — pure 0/1 discrimination.",
             "- **margin**: mean logprob(correct) − logprob(wrong).", ""]
    meta = rows[0]
    lines.append(f"Config: T={2*meta['num_instr']} symbols, shots={meta['shots']}, "
                 f"max_reads/seq={meta.get('max_reads','?')}.")
    lines.append("")
    lines.append("| model | step | regime | greedy_acc | binary_acc | margin | n_reads |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag in [t for t in FAMILY_LABEL if t in idx]:
        for reg in REGIME_ORDER:
            r = idx[tag].get(reg)
            if not r:
                continue
            step = Path(r["ckpt"]).parts[-2]
            lines.append(f"| {FAMILY_LABEL.get(tag, tag)} | {step} | {reg} | "
                         f"{r['greedy_acc']:.3f} | {r['binary_acc']:.3f} | "
                         f"{r['mean_margin']:+.2f} | {r['n_reads']} |")
    (Path(outdir) / "summary.md").write_text("\n".join(lines) + "\n")
    print("wrote", Path(outdir) / "summary.md")


def plot(idx, outdir):
    tags = [t for t in FAMILY_LABEL if t in idx]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in [(axes[0], "greedy_acc", "Greedy read accuracy (paper metric)"),
                              (axes[1], "binary_acc", "Binary discrimination (0 vs 1)")]:
        x = range(len(REGIME_ORDER))
        w = 0.8 / max(len(tags), 1)
        for i, tag in enumerate(tags):
            ys = [idx[tag].get(r, {}).get(metric, float("nan")) for r in REGIME_ORDER]
            ax.bar([xi + i * w for xi in x], ys, w,
                   label=FAMILY_LABEL.get(tag, tag),
                   color=FAMILY_COLOR.get(tag, None))
        ax.axhline(0.5, ls="--", lw=1, color="grey", label="chance")
        ax.set_xticks([xi + w * (len(tags) - 1) / 2 for xi in x])
        ax.set_xticklabels([REGIME_LABEL[r] for r in REGIME_ORDER])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("FFLM state tracking — frozen 1.3B checkpoints, in-context", fontweight="bold")
    fig.tight_layout()
    out = Path(outdir) / "fflm_read_accuracy.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows, idx = load(args.results)
    if not rows:
        print("no results to report")
        return
    write_md(rows, idx, args.outdir)
    plot(idx, args.outdir)


if __name__ == "__main__":
    main()
