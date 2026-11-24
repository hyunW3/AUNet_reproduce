#!/usr/bin/env python3
"""Decompose each arm's BPB gain into the offset-0 part and the patch-lag (offset>=1) part.

The point of the split: offset 0 is the byte that predicts the FIRST byte of a new patch, which no
patch-lag mechanism can reach by construction -- the read vector is fresh there.  Any gain an arm
shows at offset 0 is therefore generic capability, and only the offset>=1 column is patch lag.
Contributions are count-weighted so the columns sum to the arm's total BPB delta.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LN2 = math.log(2.0)
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 1 << 30)]


def lab(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}+" if hi > 1 << 20 else f"{lo}-{hi}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "reports/patch_lag/offset"))
    ap.add_argument("--blocks", nargs="+", required=True,
                    help="base_tag:arm1,arm2,...  (one per block; deltas are within-block)")
    args = ap.parse_args()
    d = Path(args.dir)
    load = lambda t: np.load(d / f"{t}.npz")

    for spec in args.blocks:
        btag, arms = spec.split(":")
        b = load(btag)
        nll_b, off = b["nll"], b["offset"]
        N = nll_b.size
        masks = [(lab(lo, hi), (off >= lo) & (off <= hi)) for lo, hi in BINS]

        print(f"\n=== block base = {btag}   BPB {nll_b.mean() / LN2:.4f} ===")
        print(f"{'offset':>7s} {'bytes':>7s} {'loss%':>6s} {'BPB':>8s}   "
              + "".join(f"{a:>21s}" for a in arms.split(",")))
        for name, m in masks:
            row = (f"{name:>7s} {m.mean():>6.1%} "
                   f"{nll_b[m].sum() / nll_b.sum():>5.1%} {nll_b[m].mean() / LN2:>8.4f}   ")
            for a in arms.split(","):
                x = load(a)["nll"]
                dd = (x[m].mean() - nll_b[m].mean()) / LN2
                row += f"{dd:>+11.4f}({dd * m.mean():>+8.5f})"
            print(row)

        m0 = off == 0
        print(f"{'-' * 7} {'':>6s} {'':>5s} {'':>8s}   "
              + "".join(f"{'-' * 21}" for _ in arms.split(",")))
        for label, m in [("offset 0", m0), ("offset>=1", ~m0), ("TOTAL", np.ones(N, bool))]:
            row = f"{label:>7s} {m.mean():>6.1%} {nll_b[m].sum() / nll_b.sum():>5.1%} " \
                  f"{nll_b[m].mean() / LN2:>8.4f}   "
            for a in arms.split(","):
                x = load(a)["nll"]
                contrib = (x[m].sum() - nll_b[m].sum()) / (LN2 * N)
                dd = (x[m].mean() - nll_b[m].mean()) / LN2
                row += f"{dd:>+11.4f}({contrib:>+8.5f})"
            print(row)
    print("\n  cell = BPB delta within the bin (count-weighted contribution to the total).")


if __name__ == "__main__":
    main()
