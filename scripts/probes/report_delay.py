#!/usr/bin/env python3
"""Patch-read-delay tables: does the FRESH patch vector delivered at the boundary position carry
the global information that makes the pos-in-patch-1 byte predictable?"""
import argparse, math
from pathlib import Path
import numpy as np

LN2 = math.log(2.0)
ROOT = Path(__file__).resolve().parents[2]
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 10 ** 6)]
NAMES = {"full": "full (fresh patch vector at the boundary position)",
         "delay_bnd": "delay 1 patch at BOUNDARY positions only",
         "delay1": "delay 1 patch everywhere", "delay2": "delay 2 patches everywhere",
         "local_only": "global path cut entirely"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "reports/global_ablation"))
    ap.add_argument("--tags", default="aunet_static,byte_greedyroot")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = []
    for tag in a.tags.split(","):
        d = np.load(Path(a.dir) / f"{tag}.npz")
        pos = d["pos_in_patch"]
        modes = [m for m in NAMES if f"nll_{m}" in d]
        hdr = "| mode | all | pos 0 | pos 1 | pos 2 | pos 3–4 | pos 5–9 | pos 10+ |"
        out += [f"\n### {tag} — patch-read delay, BPB by patch position\n", hdr, "|---" * 8 + "|"]
        for m in modes:
            n = d[f"nll_{m}"]
            c = [f"{n.mean() / LN2:.3f}"] + [f"{n[(pos >= x) & (pos <= y)].mean() / LN2:.3f}" for x, y in BINS]
            out.append(f"| {NAMES[m]} | " + " | ".join(c) + " |")
        f = d["nll_full"]
        out += ["\n**Δ BPB vs full**\n", hdr, "|---" * 8 + "|"]
        for m in modes:
            if m == "full":
                continue
            n = d[f"nll_{m}"] - f
            c = [f"{n.mean() / LN2:+.3f}"] + [f"{n[(pos >= x) & (pos <= y)].mean() / LN2:+.3f}" for x, y in BINS]
            out.append(f"| {NAMES[m]} | " + " | ".join(c) + " |")
    txt = "\n".join(out)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
