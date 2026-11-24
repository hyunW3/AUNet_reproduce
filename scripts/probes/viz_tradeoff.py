#!/usr/bin/env python3
"""Two-axis robustness trade-off at 1.3B.

  X = tokenization-brittleness  = mean Delta-acc over {PBP, Noise, Typo}
  Y = irregular-input brittle.  = mean Delta-acc over {despace(all100-clean), S-NIAH-3(UUID-easy)}

Both are accuracy drops in points (nearer 0 = more robust), so the top-right
corner is best. Each point is annotated with its zero-shot downstream average.
Output: reports/parse/tradeoff.{png,pdf}
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/ssd2/hyun2/AUNet"
OUT = f"{ROOT}/reports/parse"
COL = {"Llama": "#0072B2", "AU-Net": "#E69F00", "BPEByte": "#009E73"}
MARK = {"Llama": "o", "AU-Net": "s", "BPEByte": "^"}

# --- component numbers (1.3B; Delta-acc in points, negative = worse) ----------
PBP = {"Llama": -8.52, "AU-Net": +0.01, "BPEByte": -0.05}
NOISE = {"Llama": (-18.2, -27.6), "AU-Net": (-16.3, -22.3), "BPEByte": (-14.5, -21.9)}
TYPO = {"Llama": (-6.1, -4.4), "AU-Net": (-5.7, -4.2), "BPEByte": (-4.9, -3.8)}
SNIAH3 = {"Llama": 0.94, "AU-Net": 0.57, "BPEByte": 0.95}
SNIAH_EASY = 1.00
DOWN = {"Llama": 56.5, "AU-Net": 56.3, "BPEByte": 56.6}


def despace_drops():
    out = {}
    for m, lbl in [("llama", "Llama"), ("aunet", "AU-Net"), ("bpebyte", "BPEByte")]:
        r = json.load(open(f"{ROOT}/reports/statetrack/despace_grid/{m}/results.json"))["results"]
        out[lbl] = 100 * (r["despace_mc_avg_despaceall"]["acc"] - r["despace_mc_avg_clean"]["acc"])
    return out


def points():
    desp = despace_drops()
    P = {}
    for lbl in COL:
        X = st.mean([PBP[lbl], st.mean(NOISE[lbl]), st.mean(TYPO[lbl])])
        Y = st.mean([desp[lbl], 100 * (SNIAH3[lbl] - SNIAH_EASY)])
        P[lbl] = (X, Y)
    return P


def draw(path, dark=False):
    ink = "#e7eaf0" if dark else "#111"
    muted = "#9aa4b2" if dark else "#666"
    surf = "#171a21" if dark else "white"
    grid = "#2a2f3a" if dark else "#e6e6e6"
    P = points()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.patch.set_facecolor(surf); ax.set_facecolor(surf)

    xs = [P[m][0] for m in P]; ys = [P[m][1] for m in P]
    xpad = (max(xs) - min(xs)) * 0.42 + 0.6
    ypad = (max(ys) - min(ys)) * 0.18 + 2.0
    xlo, xhi = min(xs) - xpad, max(xs) + xpad
    ylo, yhi = min(ys) - ypad, max(ys) + ypad
    # shade the best (top-right) corner
    ax.axhspan(sum(ys) / 3, yhi, xmin=0, xmax=1, color="#2ca02c", alpha=0.04, zorder=0)
    ax.axvspan(sum(xs) / 3, xhi, color="#2ca02c", alpha=0.04, zorder=0)
    ax.grid(True, color=grid, lw=0.7, zorder=1)

    # label offsets per model to avoid overlap
    off = {"Llama": (0, 2.4, "center", "bottom"),
           "AU-Net": (0.35, -2.6, "center", "top"),
           "BPEByte": (-0.28, 2.0, "right", "bottom")}
    for m in P:
        x, y = P[m]
        ax.scatter([x], [y], s=210, marker=MARK[m], color=COL[m],
                   edgecolor="white" if not dark else "#171a21", lw=1.4, zorder=5)
        dx, dy, ha, va = off[m]
        ax.annotate(f"{m}\n(avg {DOWN[m]})", (x, y), (x + dx, y + dy),
                    ha=ha, va=va, fontsize=10.5, color=COL[m], fontweight="bold",
                    zorder=6)

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Tokenization brittleness  —  mean $\\Delta$acc, PBP + Noise + Typo (pts)",
                  fontsize=10.5, color=ink)
    ax.set_ylabel("Irregular-input brittleness  —  mean $\\Delta$acc,\ndespace + S-NIAH-3 (pts)",
                  fontsize=10.5, color=ink)
    ax.tick_params(colors=muted, labelsize=9)
    for s in ax.spines.values():
        s.set_color(grid)
    # "more robust" cue toward top-right
    ax.annotate("more\nrobust", (xhi - 0.1, ylo + 0.06 * (yhi - ylo)), ha="right",
                va="bottom", fontsize=9.5, color="#2ca02c", fontweight="bold")
    ax.annotate("", xy=(xhi - 0.2, ylo + 0.22 * (yhi - ylo)),
                xytext=(xhi - 1.8, ylo + 0.02 * (yhi - ylo)),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.6))
    ax.set_title("Robustness trade-off at 1.3B (nearer 0 on both axes = better; "
                 "downstream in parentheses)", fontsize=10.5, color=ink, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=surf)
    print("wrote", path, "| points:", {m: (round(P[m][0], 1), round(P[m][1], 1)) for m in P})


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    draw(f"{OUT}/tradeoff.png")
    draw(f"{OUT}/tradeoff.pdf")
    draw(f"{OUT}/tradeoff_dark.png", dark=True)


if __name__ == "__main__":
    main()
