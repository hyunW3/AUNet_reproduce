#!/usr/bin/env python3
"""Rebuild the combined MK-NIAH pressure-test heatmap (fig:niah_matrix) from the
per-model grid_*.jsonl files produced by mk_grid.py, so the three panels share
one figure. Mirrors mk_grid.py's plotting block; no model eval here."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/niah/highn_1p3b")
FILES = [("subword_llama", "Llama"), ("aunet_static", "AU-Net (static)"),
         ("byte_greedyroot", "BPEByte-rg")]
CONTEXTS = [512, 1024, 2048, 4096, 6144]
DEPTHS = [round(i / 9, 3) for i in range(10)]

cmap = plt.get_cmap("RdYlGn").copy()
cmap.set_bad("#d0d0d0")
fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 4.6), squeeze=False)
means = {}
for ax, (tag, lbl) in zip(axes[0], FILES):
    grid = [[float("nan")] * len(CONTEXTS) for _ in range(len(DEPTHS))]
    di_of = {d: i for i, d in enumerate(DEPTHS)}
    ci_of = {c: i for i, c in enumerate(CONTEXTS)}
    for r in map(json.loads, open(OUT / f"grid_{tag}.jsonl")):
        if r["exact_match"] is None:
            continue
        grid[di_of[round(r["depth"], 3)]][ci_of[r["context"]]] = r["exact_match"]
    g = np.array(grid, dtype=float)
    vals = g[~np.isnan(g)]
    means[lbl] = vals.mean()
    im = ax.imshow(g, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(CONTEXTS)))
    ax.set_xticklabels([f"{c//1024}k" if c >= 1024 else str(c) for c in CONTEXTS], fontsize=8)
    ax.set_yticks(range(len(DEPTHS)))
    ax.set_yticklabels([f"{int(d*100)}%" for d in DEPTHS], fontsize=7)
    ax.set_xlabel("context size (bytes)")
    if ax is axes[0][0]:
        ax.set_ylabel("needle depth (position in haystack)")
    ax.set_title(f"{lbl}   (grid mean {means[lbl]:.2f})", fontsize=10)
    for di in range(len(DEPTHS)):
        for ci in range(len(CONTEXTS)):
            v = g[di][ci]
            txt = "n/a" if np.isnan(v) else f"{v:.2f}"
            ax.text(ci, di, txt, ha="center", va="center", fontsize=6,
                    color="#777" if np.isnan(v) else "black")
fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, label="exact-match")
fig.text(0.5, -0.02, "K=4 needles, n=32/cell. At the longest context (6k) samples "
         "whose prompt exceeds the model window (byte: 8192 B; Llama: 2048 tok) are "
         "measured at their largest fitting length (filler trimmed, needle depth kept).",
         ha="center", fontsize=8, color="#777")
fig.suptitle("MK-NIAH pressure test (K=4 needles) — depth × context, frozen 1.3B "
             "(no-colon needle, n=32/cell)", fontweight="bold")
out = OUT / "mk_grid.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out, "| means:", {k: round(v, 3) for k, v in means.items()})

# per-model depth/length summaries for the figure text
for tag, lbl in FILES:
    g = {}
    for r in map(json.loads, open(OUT / f"grid_{tag}.jsonl")):
        if r["exact_match"] is not None:
            g[(round(r["depth"], 3), r["context"])] = r["exact_match"]
    def band(pred):
        vs = [v for (d, c), v in g.items() if pred(d, c)]
        return sum(vs) / len(vs) if vs else float("nan")
    ends = band(lambda d, c: d <= 0.15 or d >= 0.85)
    mid = band(lambda d, c: 0.3 <= d <= 0.7)
    short = band(lambda d, c: c <= 1024)
    longc = band(lambda d, c: c >= 4096)
    print(f"  {lbl:18} ends={ends:.2f} mid={mid:.2f} short(<=1k)={short:.2f} long(>=4k)={longc:.2f}")
