#!/usr/bin/env python3
"""Tables + paired bootstrap CIs for the patch-lag probes (reports/patch_lag/*.npz).

Bootstrap resamples WINDOWS (not bytes), which is the unit the eval set is built from, and pairs
every mode against `full` inside each resample, so the CI is on the difference.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (8, 15), (16, 10 ** 6)]


def paired_ci(nll, base, win, n_boot=1000, seed=0):
    """95% CI of mean(nll - base) in BPB, resampling windows."""
    rng = np.random.default_rng(seed)
    ws = np.unique(win)
    idx = {w: np.flatnonzero(win == w) for w in ws}
    d = (nll - base) / LN2
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ws, size=ws.size, replace=True)
        sel = np.concatenate([idx[w] for w in pick])
        out[b] = d[sel].mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--md", action="store_true", help="emit markdown tables")
    args = ap.parse_args()

    d = np.load(args.npz)
    # files written before the rename carry the key as "stale"
    st = d["patch_offset"] if "patch_offset" in d.files else d["stale"]
    win = d["window"]
    modes = [k[4:] for k in d.files if k.startswith("nll_")]
    modes = ["full"] + [m for m in modes if m != "full"]
    base = d["nll_full"]

    P = print
    P(f"\n{Path(args.npz).name}: {base.size} bytes, {np.unique(win).size} windows, "
      f"baseline BPB {base.mean() / LN2:.4f}\n")
    P("| mode | BPB | Δ BPB vs full | 95% CI (Δ) | Δ at patch_offset≥1 |")
    P("|---|---|---|---|---|")
    m1 = st >= 1
    for m in modes:
        v = d[f"nll_{m}"]
        lo, hi = paired_ci(v, base, win, args.n_boot)
        P(f"| {m} | {v.mean() / LN2:.4f} | {(v.mean() - base.mean()) / LN2:+.4f} "
          f"| [{lo:+.4f}, {hi:+.4f}] | {(v[m1].mean() - base[m1].mean()) / LN2:+.4f} |")

    P("\n**By read staleness** (BPB; staleness 0 = the vector was pooled at this very byte)\n")
    hdr = "| staleness | bytes | share |" + "".join(f" {m} |" for m in modes)
    P(hdr); P("|" + "---|" * (3 + len(modes)))
    for lo_, hi_ in BINS:
        sel = (st >= lo_) & (st <= hi_)
        if not sel.any():
            continue
        name = f"{lo_}" if lo_ == hi_ else (f"{lo_}–{hi_}" if hi_ < 10 ** 5 else f"{lo_}+")
        row = f"| {name} | {sel.sum()} | {100 * sel.mean():.1f}% |"
        for m in modes:
            row += f" {d[f'nll_{m}'][sel].mean() / LN2:.4f} |"
        P(row)

    P(f"\nLoss mass: staleness 0 = {100 * base[~m1].sum() / base.sum():.1f}% of total, "
      f"staleness≥1 = {100 * base[m1].sum() / base.sum():.1f}% "
      f"({base[m1].sum() / base.size / LN2:.4f} of {base.mean() / LN2:.4f} bits/byte)")


if __name__ == "__main__":
    main()
