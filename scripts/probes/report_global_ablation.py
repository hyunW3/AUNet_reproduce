#!/usr/bin/env python3
"""Aggregate reports/global_ablation/*.npz into the loss tables (overall + boundary split)."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
ROOT = Path(__file__).resolve().parents[2]
MODES = ("full", "no_trunk", "local_only", "mean_patch")
LABEL = {"full": "full (local+global)", "no_trunk": "trunk=identity", "local_only": "local only (global cut)",
         "mean_patch": "mean-patch control"}


def boot_ci(nll, win, idx, iters=2000, seed=0):
    """95% CI on the mean NLL over the selected bytes, resampling WINDOWS."""
    n, w = nll[idx], win[idx]
    if n.size == 0:
        return float("nan"), float("nan")
    order = np.argsort(w, kind="stable")
    n, w = n[order], w[order]
    ids, starts = np.unique(w, return_index=True)
    sums = np.add.reduceat(n, starts)
    cnts = np.diff(np.append(starts, n.size))
    rng = np.random.default_rng(seed)
    b = rng.integers(0, ids.size, size=(iters, ids.size))
    m = sums[b].sum(1) / cnts[b].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stats(nll, win, idx):
    x = nll[idx]
    lo, hi = boot_ci(nll, win, idx)
    return dict(n=int(x.size), mean=float(x.mean()), bpb=float(x.mean() / LN2),
                bpb_lo=lo / LN2, bpb_hi=hi / LN2, med=float(np.median(x)),
                std=float(x.std()), p90=float(np.percentile(x, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "reports/global_ablation"))
    ap.add_argument("--tags", default="aunet_static,byte_greedyroot")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = []
    P = lines.append
    for tag in args.tags.split(","):
        d = np.load(Path(args.dir) / f"{tag}.npz")
        win, bnd, pos = d["window"], d["is_boundary"], d["pos_in_patch"]
        modes = [m for m in MODES if f"nll_{m}" in d]
        groups = {"all": np.ones_like(bnd, bool), "boundary": bnd, "interior": ~bnd}
        P(f"\n### {tag}   ({win.max() + 1} windows, {bnd.size} scored bytes, "
          f"{100 * bnd.mean():.1f}% boundary bytes, {1 / bnd.mean():.2f} bytes/patch)\n")
        P("| mode | group | bytes | share | mean NLL (nats) | BPB | 95% CI | median | p90 | std | ΔNLL vs full | ΔBPB vs full |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        base = {g: d["nll_full"][m].mean() for g, m in groups.items()} if "nll_full" in d else {}
        for m in modes:
            nll = d[f"nll_{m}"]
            for g, mask in groups.items():
                s = stats(nll, win, mask)
                dn = s["mean"] - base[g] if base else float("nan")
                P(f"| {LABEL[m]} | {g} | {s['n']} | {100 * mask.mean():.1f}% | {s['mean']:.4f} | "
                  f"{s['bpb']:.4f} | [{s['bpb_lo']:.4f}, {s['bpb_hi']:.4f}] | {s['med']:.4f} | "
                  f"{s['p90']:.4f} | {s['std']:.4f} | "
                  f"{dn:+.4f} | {dn / LN2:+.4f} |")
        # loss by index inside the patch (0 = the boundary byte itself)
        P("\n| mode | pos-in-patch 0 (boundary) | 1 | 2 | 3 | 4 | 5+ |")
        P("|---|---|---|---|---|---|---|")
        buckets = [(pos == k) for k in range(5)] + [(pos >= 5)]
        for m in modes:
            nll = d[f"nll_{m}"]
            cells = " | ".join(f"{nll[b].mean() / LN2:.3f}" if b.any() else "-" for b in buckets)
            P(f"| {LABEL[m]} | {cells} |")
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
