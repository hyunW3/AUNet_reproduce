#!/usr/bin/env python3
"""Paper figure: training-BPB curves for the three matched 1.3B models.

Same conversion as runs/_scripts/plot_bpb_all.py (byte models: loss/ln2 nats/byte;
Llama: loss/(ln2*4.5483 bytes/token)), EMA alpha=0.01, x = training bytes seen (GB),
all series truncated at the common 283.1 GB iso-byte endpoint. Colors/markers follow
the paper's model convention (plot_despace_grid.py, CVD-validated).

Output: reports/despace/../bpb_curves_1col.pdf -> copy to paper_overleaf/figures.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/ssd2/hyun2/AUNet")
LN2 = math.log(2.0)
BPT_LLAMA = 4.5483            # measured on 5000 DCLM docs
END_GB = 283.1                # common iso-byte endpoint

SERIES = [  # (tag, label, run_dir, bytes_per_token)
    ("llama", "Llama", "llama_1.8B_paper", BPT_LLAMA),
    ("aunet", "AU-Net", "aunet2_1.3B", 1.0),
    ("bpebyte", "BPEByte", "bpebyte_br_greedy_root_1.3B", 1.0),
]
COL = {"llama": "#0072B2", "aunet": "#E69F00", "bpebyte": "#009E73"}


def load(path, bpt):
    by_step = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        loss, tok, step = d.get("loss/out"), d.get("optim/total_tokens"), d.get("global_step")
        if loss is None or tok is None or step is None:
            continue
        by_step[step] = (tok * bpt / 1e9, loss / (LN2 * bpt))
    steps = sorted(by_step)
    return [by_step[s][0] for s in steps], [by_step[s][1] for s in steps]


def ema(xs, a=0.01):
    out, m = [], xs[0]
    for x in xs:
        m = a * x + (1 - a) * m
        out.append(m)
    return out


def main():
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ends = []
    for tag, label, run, bpt in SERIES:
        gb, bpb = load(ROOT / "main/main/1.3B" / run / "metrics.jsonl", bpt)
        sm = ema(bpb)
        pts = [(g, v) for g, v in zip(gb, sm) if g <= END_GB]
        gb, sm = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(gb, sm, color=COL[tag], lw=1.8, label=label, zorder=3)
        ends.append((sm[-1], tag, label))
        print(f"{label:8s} end={gb[-1]:.1f} GB  BPB(EMA)={sm[-1]:.4f}")
    # dodged end labels with values
    ends.sort(reverse=True)
    ypos = []
    for y, _, _ in ends:
        if ypos and ypos[-1] - y < 0.016:
            y = ypos[-1] - 0.016
        ypos.append(y)
    for (yv, tag, label), y in zip(ends, ypos):
        ax.annotate(label, (END_GB, yv), xytext=(END_GB + 4, y),
                    fontsize=8.5, color=COL[tag], va="center", annotation_clip=False)
    ax.set_xlim(0, END_GB + 2)
    ax.set_ylim(0.82, 1.25)
    ax.set_xlabel("training bytes seen (GB)", fontsize=9, color="#898781")
    ax.set_ylabel("training BPB (EMA)", fontsize=9, color="#898781")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.tick_params(colors="#898781", labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#898781")
    leg = ax.legend(fontsize=8.5, loc="upper right", frameon=False)
    for t in leg.get_texts():
        t.set_color("#0b0b0b")
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    out = ROOT / "reports" / "bpb_curves_1col.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
