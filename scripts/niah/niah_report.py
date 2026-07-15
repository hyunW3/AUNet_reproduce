#!/usr/bin/env python3
"""Render S-NIAH-1 results: table + exact-match-vs-length + depth heatmaps."""
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


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


TASK_DESC = {"1": "repeated-noise haystack + 7-digit number",
             "2": "natural-essay (DCLM) haystack + 7-digit number",
             "3": "natural-essay (DCLM) haystack + UUID"}


def table(rows, outdir, suffix, task):
    lens = sorted({r["target_bytes"] for r in rows})
    idx = {(r["tag"], r["target_bytes"]): r for r in rows}
    lines = [f"# S-NIAH-{task} single-needle retrieval (RULER)", "",
             f"Haystack/value: {TASK_DESC.get(task, '')}. Exact-match retrieval "
             "(teacher-forced greedy == model would emit the value verbatim), "
             "iso-byte context length.", "",
             "| model | " + " | ".join(f"{l}B" for l in lens) + " |",
             "|---|" + "---|" * len(lens)]
    for tag in [t for t in FAMILY_LABEL if any(r["tag"] == t for r in rows)]:
        cells = []
        for l in lens:
            r = idx.get((tag, l))
            cells.append(f"{r['exact_match']:.2f}" if r else "-")
        lines.append(f"| {FAMILY_LABEL.get(tag, tag)} | " + " | ".join(cells) + " |")
    (Path(outdir) / f"summary{suffix}.md").write_text("\n".join(lines) + "\n")
    print("wrote", Path(outdir) / f"summary{suffix}.md")


def line_plot(rows, outdir, suffix, task):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    by = defaultdict(list)
    for r in rows:
        if r["tag"] in FAMILY_LABEL:
            by[r["tag"]].append(r)
    for tag in [t for t in FAMILY_LABEL if t in by]:
        rs = sorted(by[tag], key=lambda r: r["target_bytes"])
        xs = [r["avg_prompt_bytes"] for r in rs]
        ys = [r["exact_match"] for r in rs]
        ax.plot(xs, ys, "o-", label=FAMILY_LABEL.get(tag, tag),
                color=FAMILY_COLOR.get(tag))
    ax.set_xlabel("prompt length (bytes, iso-byte)")
    ax.set_ylabel("exact-match retrieval")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"S-NIAH-{task}: needle retrieval vs. context length\n{TASK_DESC.get(task,'')}",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(outdir) / f"niah_exact_vs_length{suffix}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


def heatmap(per, outdir, suffix, task):
    tags = [t for t in FAMILY_LABEL if any(r["tag"] == t for r in per)]
    lens = sorted({r["target_bytes"] for r in per})
    depths = sorted({r["depth"] for r in per})
    fig, axes = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 3.6), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        agg = defaultdict(lambda: [0, 0])
        for r in per:
            if r["tag"] != tag:
                continue
            c = agg[(r["target_bytes"], r["depth"])]
            c[0] += r["exact"]; c[1] += 1
        grid = [[(agg[(l, d)][0] / agg[(l, d)][1] if agg[(l, d)][1] else float("nan"))
                 for l in lens] for d in depths]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(lens))); ax.set_xticklabels([f"{l}" for l in lens], fontsize=7)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d:.1f}" for d in depths], fontsize=7)
        ax.set_xlabel("length (B)"); ax.set_title(FAMILY_LABEL.get(tag, tag), fontsize=9)
        if ax is axes[0][0]:
            ax.set_ylabel("needle depth")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, label="exact-match")
    fig.suptitle(f"S-NIAH-{task} exact-match by needle depth x context length", fontweight="bold")
    out = Path(outdir) / f"niah_depth_heatmap{suffix}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--per", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--task", default="1")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows = load(args.results)
    if not rows:
        print("no results"); return
    table(rows, args.outdir, args.suffix, args.task)
    line_plot(rows, args.outdir, args.suffix, args.task)
    if args.per and Path(args.per).exists():
        heatmap(load(args.per), args.outdir, args.suffix, args.task)


if __name__ == "__main__":
    main()
