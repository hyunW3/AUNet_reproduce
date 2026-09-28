#!/usr/bin/env python3
"""Star (radar) plot of the 1.3B trio + BLT over 7 axes, all oriented higher = better.

Axes (raw values from tab:main_13b): 0-shot downstream avg, forward latency,
PBP / Noise / Typo / Despace |Delta-acc|, S-NIAH-3 (formatted-string) recall.
Each axis is min-max normalized across the plotted models (rim = best model,
inner ring = worst); lower-is-better metrics are inverted before normalizing.
Axis labels carry the raw worst -> best range so directions stay readable.
Axes are grouped into three families (Quality / Retrieval / Robustness),
marked by colored arcs outside the rim.

Run:  python reports/parse/star_draw.py   (writes star.png/.pdf beside this file)
"""
import os
from math import pi

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

COL = { "BPEByte (ours)": "#009E73","Llama": "#0072B2", "AU-Net": "#E69F00", "BLT": "#CC79A7",
       "H-Net": "#56B4E9"}
STYLE = {"Llama": "-", "AU-Net": "-", "BPEByte (ours)": "-", "BLT": "--",
         "H-Net": "--"}   # dashed = external (H-Net = 1-stage XL)

# (label, higher_is_better, {model: raw}) — raw numbers of tab:main_13b,
# ordered clockwise from the top so the three families are contiguous.
AXES = [
    # -- Quality (right sector) --
    ("0-shot", True,
     {"Llama": 60.0, "AU-Net": 60.1, "BPEByte (ours)": 60.4, "BLT": 59.1,
      "H-Net": 59.2}),
    ("3-shot", True,
     {"Llama": 61.8, "AU-Net": 62.7, "BPEByte (ours)": 62.4, "BLT": 59.4,
      "H-Net": 61.9}),
    ("5-shot", True,
     {"Llama": 61.9, "AU-Net": 63.0, "BPEByte (ours)": 62.8, "BLT": 57.2,
      "H-Net": 62.6}),
    ("    Latency", False,   # tab:main_13b byte-matched forward (ms)
     {"Llama": 406.3, "AU-Net": 652.0, "BPEByte (ours)": 650.3, "BLT": 1315.3,
      "H-Net": 578.7}),
    # -- Retrieval (bottom sector) --
    ("  S-NIAH-3", True,   # tab:sniah_full <=4k means, n=250/cell
     {"Llama": 0.943, "AU-Net": 0.551, "BPEByte (ours)": 0.952, "BLT": 0.147,
      "H-Net": 0.853}),
    ("S-NIAH-1", True,   # <=4k means; externals scored on identical dumped pairs
     {"Llama": 1.000, "AU-Net": 0.996, "BPEByte (ours)": 1.000, "BLT": 0.428,
      "H-Net": 0.999}),
    ("S-NIAH-2", True,
     {"Llama": 0.998, "AU-Net": 0.991, "BPEByte (ours)": 0.999, "BLT": 0.781,
      "H-Net": 0.961}),
    # -- Robustness (upper-left sector) --
    # unified 5-task protocol (HS/ARC-E/ARC-C/PIQA/BoolQ, limit 2000):
    # scripts/probes/paper_robustness_tables.py (axis means printed there).
    ("Despace", False,
     {"Llama": 12.16, "AU-Net": 21.83, "BPEByte (ours)": 13.18, "BLT": 5.82,
      "H-Net": 12.37}),
    ("PBP", False,   # BLT: table shows "--" (HS/ARC-E unmeasured); mean of the 3 measured tasks
     {"Llama": 10.65, "AU-Net": 0.05, "BPEByte (ours)": 0.14, "BLT": 0.03,
      "H-Net": 0.00}),
    ("Noise", False,
     {"Llama": 14.26, "AU-Net": 11.21, "BPEByte (ours)": 11.69, "BLT": 14.01,
      "H-Net": 12.07}),
    ("Typo", False,
     {"Llama": 5.70, "AU-Net": 3.98, "BPEByte (ours)": 3.86, "BLT": 4.64,
      "H-Net": 5.00}),
]
LO = 0.15   # worst model sits on this ring, not at the center pole

# Downstream-accuracy axes: absolute differences are small, so min-max
# stretching the ~1-pp spread over the full radius overstates them. These axes
# instead use a fixed absolute scale: LO ring = 55.0, rim = 63.0 (values
# outside are clamped), so equal radial distances mean equal accuracy gaps.
PERF_RANGE = {"0-shot": (55.0, 63.0), "3-shot": (55.0, 63.0),
              "5-shot": (55.0, 63.0)}

# (name, first axis idx, last axis idx, arc color, label color)
GROUPS = [
    ("Performance",    0, 3,  "#F2DE7C", "#9C7A00"),
    ("Retrieval",  4, 6,  "#8FD694", "#1F7A33"),
    ("Robustness", 7, 10, "#F5B26B", "#B35A00"),
]
ARC_R, LAB_R = 1.45, 1.56   # radius of the arc band / of the family label


def norm(raw, hib, vrange=None):
    vals = list(raw.values())
    best, worst = (max(vals), min(vals)) if hib else (min(vals), max(vals))
    if vrange is not None:
        lo, hi = vrange
        frac = {m: min(1.0, max(0.0, (raw[m] - lo) / (hi - lo))) for m in raw}
        return {m: LO + (1 - LO) * frac[m] for m in raw}
    if best == worst:
        return {m: 1.0 for m in raw}
    return {m: LO + (1 - LO) * (raw[m] - worst) / (best - worst) for m in raw}


def curved_text(ax, theta_c, r, text, color, fontsize=14):
    """Draw `text` centered on data angle theta_c, each glyph following the
    circle of radius r (letters flip upright when the label sits below the
    horizontal midline). Call only after the figure layout is final."""
    fig = ax.figure
    fig.canvas.draw()   # finalize transforms before measuring
    tr = ax.transData.transform
    x0, y0 = tr((0.0, 0.0))
    px_per_unit = np.hypot(*(tr((0.0, 1.0)) - (x0, y0)))
    fp = FontProperties(size=fontsize, weight="bold")
    px = fig.dpi / 72.0
    widths = []
    for ch in text:
        ext = TextPath((0, 0), ch, prop=fp).get_extents()
        widths.append((ext.width if ext.width > 0 else 0.5 * fontsize) * px)
    kern = 0.10 * fontsize * px
    span = (sum(widths) + kern * (len(text) - 1)) / (r * px_per_unit)
    xc, yc = tr((theta_c, r))
    flip = yc < y0          # lower half: mirror so the label reads upright
    if flip:
        text, widths = text[::-1], widths[::-1]
    t = theta_c - span / 2
    for ch, w in zip(text, widths):
        aw = w / (r * px_per_unit)
        tc = t + aw / 2
        x, y = tr((tc, r))
        rot = np.degrees(np.arctan2(y - y0, x - x0)) - 90 + (180 if flip else 0)
        ax.text(tc, r, ch, rotation=rot, rotation_mode="anchor",
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, clip_on=False)
        t += aw + kern / (r * px_per_unit)


def draw(path, paper=False):
    scores = {m: [] for m in COL}
    for label, hib, raw in AXES:
        n = norm(raw, hib, vrange=PERF_RANGE.get(label.strip()))
        for m in COL:
            scores[m].append(n[m])
    K = len(AXES)
    ang = [2 * pi * i / K for i in range(K)]
    ang_c = ang + ang[:1]
    step = 2 * pi / K

    fig, ax = plt.subplots(figsize=(6.4, 6.0), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    for m in COL:
        v = scores[m] + scores[m][:1]
        ax.plot(ang_c, v, STYLE[m], color=COL[m], lw=2, zorder=3, label=m)
        ax.fill(ang_c, v, color=COL[m], alpha=0.06, zorder=2)
    ax.set_xticks(ang)
    ax.set_xticklabels([a[0] for a in AXES], fontsize=12)
    ax.set_yticks([LO, 0.5, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.06)
    ax.grid(color="#e1e0d9", lw=0.7)
    ax.spines["polar"].set_visible(False)
    ax.tick_params(pad=2)

    # family arcs outside the rim (sketch: Quality right, Retrieval bottom,
    # Robustness upper left)
    for name, i0, i1, c_arc, c_lab in GROUPS:
        t0, t1 = ang[i0] - 0.38 * step, ang[i1] + 0.38 * step
        ts = np.linspace(t0, t1, 80)
        ax.plot(ts, np.full_like(ts, ARC_R), color=c_arc, lw=7.5, alpha=0.9,
                solid_capstyle="round", clip_on=False, zorder=1)

    if paper:
        leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.24, 1.16), fontsize=12,
                        frameon=False, handlelength=1.6)
    else:
        leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.46, 1.18), fontsize=12,
                        frameon=False)
        # ax.set_title("1B-scale performance & robustness — higher is better\n"
        #              "(rim = best model per axis; dashed = external reference)", fontsize=9.5, pad=42)
    fig.tight_layout()
    # curved family labels, drawn after tight_layout so transforms are final
    for name, i0, i1, _, c_lab in GROUPS:
        t0, t1 = ang[i0] - 0.38 * step, ang[i1] + 0.38 * step
        curved_text(ax, 0.5 * (t0 + t1), LAB_R, name, c_lab)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    draw(os.path.join(here, "radar.png"))
    draw(os.path.join(here, "radar.pdf"))
    # draw(os.path.join(here, "radar_paper.pdf"), paper=True)
