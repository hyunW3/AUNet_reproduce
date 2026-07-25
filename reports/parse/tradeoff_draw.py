#!/usr/bin/env python3
"""Self-contained two-axis robustness trade-off at 1.3B (matplotlib only).

  X = tokenization brittleness  = mean Delta-acc over {PBP, Noise, Typo}
  Y = irregular-input brittle.  = mean Delta-acc over {despace(all100-clean), S-NIAH-3(UUID-easy)}

Both are accuracy drops in points (nearer 0 = more robust) -> top-right corner is best.
Each point is annotated with its zero-shot downstream average.
Run:  python reports/parse/tradeoff_draw.py    (writes tradeoff.png/.pdf beside this file)
"""
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COL = {"Transformer": "#0072B2", "AU-Net": "#E69F00", "BPEByte": "#009E73", "BLT-1B": "#CC79A7",
       "H-Net-1s": "#56B4E9", "H-Net-2s": "#D55E00"}
MARK = {"Transformer": "o", "AU-Net": "s", "BPEByte": "^", "BLT-1B": "D",
        "H-Net-1s": "v", "H-Net-2s": "P"}

# --- component numbers (1.3B; Delta-acc in points, negative = worse) ----------
# BLT-1B and H-Net (1-/2-stage XL): external byte-level references, scored on identical perturbations.
PBP = {"Transformer": -8.52, "AU-Net": +0.01, "BPEByte": -0.05, "BLT-1B": 0.00,
       "H-Net-1s": 0.00, "H-Net-2s": 0.00}
NOISE = {"Transformer": (-18.2, -27.6), "AU-Net": (-16.3, -22.3), "BPEByte": (-14.5, -21.9),
         "BLT-1B": (-16.0, -16.1), "H-Net-1s": (-16.7, -21.8), "H-Net-2s": (-16.9, -23.7)}  # HS, ARC-E
TYPO = {"Transformer": (-6.1, -4.4), "AU-Net": (-5.7, -4.2), "BPEByte": (-4.9, -3.8),
        "BLT-1B": (-5.2, -5.2), "H-Net-1s": (-7.7, -7.7), "H-Net-2s": (-6.1, -6.1)}  # HS, ARC-C  (byte refs: HS-typo only)
DESPACE = {"Transformer": -14.4, "AU-Net": -22.4, "BPEByte": -12.5, "BLT-1B": -6.8,
           "H-Net-1s": -14.6, "H-Net-2s": -14.3}   # all100 - clean
SNIAH3 = {"Transformer": 0.94, "AU-Net": 0.57, "BPEByte": 0.95, "BLT-1B": 0.11,
          "H-Net-1s": 0.84, "H-Net-2s": 0.81}       # UUID recall
# 0-shot / 5-shot downstream averages (5-benchmark set of tab:main_13b: HS/ARC-E/ARC-C/PIQA/WG)
DOWN0 = {"Transformer": 60.0, "AU-Net": 60.1, "BPEByte": 60.4, "BLT-1B": 59.1,
         "H-Net-1s": 59.2, "H-Net-2s": 61.7}
DOWN5 = {"Transformer": 62.0, "AU-Net": 62.7, "BPEByte": 62.8, "BLT-1B": 57.2,
         "H-Net-1s": 62.6, "H-Net-2s": 64.1}


def axes():
    P = {}
    for m in COL:
        X = st.mean([PBP[m], st.mean(NOISE[m]), st.mean(TYPO[m])])       # tokenization
        Y = st.mean([DESPACE[m], 100 * (SNIAH3[m] - 1.00)])             # irregular-input
        P[m] = (X, Y)
    return P


GREEN = "#2ca02c"


def draw(path):
    P = axes()
    xs = [P[m][0] for m in P]
    ys = [P[m][1] for m in P]
    xpad = (max(xs) - min(xs)) * 0.42 + 0.8
    ypad = (max(ys) - min(ys)) * 0.18 + 2.5
    xlo, xhi = min(xs) - xpad, max(xs) + xpad
    ylo, yhi = min(ys) - ypad, max(ys) + ypad
    w, hgt = xhi - xlo, yhi - ylo
    mx, my = st.mean(xs), st.mean(ys)   # centroid = quadrant divider

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    # "robust on BOTH axes" quadrant (top-right) + faint centroid dividers
    ax.add_patch(Rectangle((mx, my), xhi - mx, yhi - my, facecolor=GREEN,
                           alpha=0.07, edgecolor="none", zorder=0))
    ax.axvline(mx, color="#c9c9c9", lw=0.8, ls=(0, (4, 4)), zorder=1)
    ax.axhline(my, color="#c9c9c9", lw=0.8, ls=(0, (4, 4)), zorder=1)
    ax.grid(True, color="#eeeeee", lw=0.6, zorder=0)

    off = {"Transformer": (0, 2.0, "center", "bottom"),        # per-point label offset
           "AU-Net": (0.0, -2.0, "center", "top"),
           "BPEByte": (0, 2.0, "center", "bottom"),
           "BLT-1B": (0.0, -2.2, "center", "top"),
           "H-Net-1s": (0.35, 2.1, "left", "bottom"),
           "H-Net-2s": (0.35, -2.3, "left", "top")}
    for m in P:
        x, y = P[m]
        ax.scatter([x], [y], s=230, marker=MARK[m], color=COL[m],
                   edgecolor="white", lw=1.5, zorder=5)
        dx, dy, ha, va = off[m]
        ax.annotate(f"{m}\n({DOWN0[m]} / {DOWN5[m]})", (x, y), (x + dx, y + dy),
                    ha=ha, va=va, fontsize=10.5, color=COL[m], fontweight="bold", zorder=6)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Mean $\Delta$acc across PBP, Noise, and Typo (pts)",
                  fontsize=10.5)
    ax.set_ylabel("Mean of despace $\\Delta$acc aand S-NIAH-3 shortfall (acc$-$100)",
                  fontsize=10.5)
    ax.tick_params(labelsize=9)
    for s in ax.spines.values():
        s.set_color("#cccccc")

    # single directional cue in the empty bottom-left, pointing to the good corner
    # ax.annotate("more robust",
    #             xy=(xlo + 0.36 * w, ylo + 0.55 * hgt),
    #             xytext=(xlo + 0.05 * w, ylo + 0.28 * hgt),
    #             ha="left", va="center", fontsize=10, color=GREEN, fontweight="bold",
    #             arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8))
    # label the best quadrant
    ax.text(xhi - 0.15, my + 0.03 * hgt, "robust on both axes", ha="right", va="bottom",
            fontsize=8.5, color=GREEN, style="italic", alpha=0.9)

    ax.set_title("Robustness trade-off at 1B param scale "
                 "(top-right = robust on both axes)",
                 fontsize=10, pad=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    print("wrote", path, "|", {m: (round(P[m][0], 1), round(P[m][1], 1)) for m in P})


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    draw(os.path.join(here, "tradeoff.png"))
    draw(os.path.join(here, "tradeoff.pdf"))
