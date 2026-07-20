#!/usr/bin/env python3
"""Re-plot the MK-NIAH pressure-test grid from mk_grid.jsonl (no eval / no GPU).

Renders the depth x context exact-match heatmap per model. By default the 8192
byte column is excluded (out-of-window for the byte models -> all n/a), leaving
the fully in-window 512..6144 grid used in the paper figure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = [("subword_llama", "subword (Llama)"), ("aunet_static", "AUNet"),
          ("byte_greedyroot", "BPEByte rg")]


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="reports/niah/mk_grid.jsonl")
    ap.add_argument("--out", default="reports/niah/mk_grid_no8k.png")
    ap.add_argument("--drop-context", type=int, nargs="*", default=[8192],
                    help="context sizes (bytes) to omit from the x-axis")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    drop = set(args.drop_context)
    contexts = sorted({r["context"] for r in rows} - drop)
    depths = sorted({r["depth"] for r in rows})
    lbl = dict(MODELS)
    tags = [t for t, _ in MODELS]

    # (tag, depth, context) -> exact_match (None -> NaN)
    cell = {(r["tag"], r["depth"], r["context"]): r.get("exact_match") for r in rows}
    grids = {}
    for t in tags:
        g = [[cell.get((t, d, c)) for c in contexts] for d in depths]
        grids[t] = np.array([[np.nan if v is None else v for v in row] for row in g],
                            dtype=float)

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d0d0d0")
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 4.6), squeeze=False)
    im = None
    for ax, tag in zip(axes[0], tags):
        g = grids[tag]
        im = ax.imshow(g, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(contexts)))
        ax.set_xticklabels([f"{c // 1024}k" if c >= 1024 else str(c) for c in contexts],
                           fontsize=8)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{int(d * 100)}%" for d in depths], fontsize=7)
        ax.set_xlabel("context size (bytes)")
        if ax is axes[0][0]:
            ax.set_ylabel("needle depth (position in haystack)")
        ax.set_title(lbl.get(tag, tag), fontsize=10)
        for di in range(len(depths)):
            for ci in range(len(contexts)):
                v = g[di][ci]
                txt = "n/a" if math.isnan(v) else f"{v:.2f}"
                ax.text(ci, di, txt, ha="center", va="center", fontsize=6,
                        color="#777" if math.isnan(v) else "black")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, label="exact-match")
    fig.text(0.5, -0.02, "Exact-match, n=8 samples/cell, K=4 (1 queried + 3 distractor "
             "needles). 8k body omitted: out-of-window for byte models (8192 B) and "
             "mostly for Llama (2048 tok); largest fully testable body is ~6k.",
             ha="center", fontsize=8, color="#777")
    fig.suptitle("MK-NIAH pressure test (K=4 needles) — depth x context, frozen 1.3B",
                 fontweight="bold")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print("wrote", args.out, "contexts:", contexts)


if __name__ == "__main__":
    run()
