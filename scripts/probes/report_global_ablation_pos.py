#!/usr/bin/env python3
"""Positional full-vs-local_only comparison for the global-trunk ablation.

Two notions of "position", both from reports/global_ablation/<tag>.npz:
  * pos-in-patch  : bytes since the patch start (0 = the boundary byte itself)
  * byte offset   : absolute offset inside the 3840-byte held-out window (= how much left
                    context the model has; the LOCAL path is capped at a 512-byte window)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
ROOT = Path(__file__).resolve().parents[2]
POS_BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 9), (10, 14), (15, 10**6)]
OFF_EDGES = [0, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3840]


def offsets_in_window(win):
    """Byte offset of every scored byte inside its window (rows are stored in byte order)."""
    idx = np.arange(win.size)
    starts = np.zeros(win.max() + 1, dtype=np.int64)
    first = np.searchsorted(win, np.arange(win.max() + 1))
    starts[:] = first
    return idx - starts[win]


def boot_delta_ci(full, loc, win, mask, iters=2000, seed=0):
    """95% CI on Δ(mean NLL) for the selected bytes, resampling WINDOWS (paired)."""
    f, l, w = full[mask], loc[mask], win[mask]
    if f.size == 0:
        return float("nan"), float("nan")
    o = np.argsort(w, kind="stable")
    f, l, w = f[o], l[o], w[o]
    ids, starts = np.unique(w, return_index=True)
    fs, ls = np.add.reduceat(f, starts), np.add.reduceat(l, starts)
    cnt = np.diff(np.append(starts, f.size))
    rng = np.random.default_rng(seed)
    b = rng.integers(0, ids.size, size=(iters, ids.size))
    n = cnt[b].sum(1)
    d = (ls[b].sum(1) - fs[b].sum(1)) / n
    return float(np.percentile(d, 2.5)) / LN2, float(np.percentile(d, 97.5)) / LN2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "reports/global_ablation"))
    ap.add_argument("--tags", default="aunet_static,byte_greedyroot")
    ap.add_argument("--ablation", default="local_only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = []
    P = lines.append
    for tag in args.tags.split(","):
        d = np.load(Path(args.dir) / f"{tag}.npz")
        win, bnd, pos = d["window"], d["is_boundary"], d["pos_in_patch"]
        full, loc = d["nll_full"], d[f"nll_{args.ablation}"]
        off = offsets_in_window(win)

        P(f"\n### {tag} — full vs {args.ablation}\n")
        P("**By position inside the patch** (0 = boundary byte)\n")
        P("| pos-in-patch | bytes | share | full BPB | local-only BPB | Δ BPB | 95% CI (Δ) | ×  |")
        P("|---|---|---|---|---|---|---|---|")
        for lo, hi in POS_BINS:
            m = (pos >= lo) & (pos <= hi)
            if not m.any():
                continue
            f, l = full[m].mean() / LN2, loc[m].mean() / LN2
            clo, chi = boot_delta_ci(full, loc, win, m)
            name = f"{lo}" if lo == hi else (f"{lo}+" if hi > 1000 else f"{lo}–{hi}")
            P(f"| {name} | {int(m.sum())} | {100 * m.mean():.1f}% | {f:.3f} | {l:.3f} | "
              f"{l - f:+.3f} | [{clo:+.3f}, {chi:+.3f}] | {l / f:.2f} |")

        P("\n**By byte offset in the window** (left-context length; the local path's byte encoder "
          "has a 512-byte sliding window)\n")
        P("| byte offset | bytes | full BPB | local-only BPB | Δ BPB | 95% CI (Δ) | × |")
        P("|---|---|---|---|---|---|---|")
        for a, b in zip(OFF_EDGES[:-1], OFF_EDGES[1:]):
            m = (off >= a) & (off < b)
            if not m.any():
                continue
            f, l = full[m].mean() / LN2, loc[m].mean() / LN2
            clo, chi = boot_delta_ci(full, loc, win, m)
            P(f"| {a}–{b - 1} | {int(m.sum())} | {f:.3f} | {l:.3f} | {l - f:+.3f} | "
              f"[{clo:+.3f}, {chi:+.3f}] | {l / f:.2f} |")

        # staleness = age (in bytes) of the patch vector the predicting position reads.
        # Verified against the model's own repeat_idx: staleness[t] == pos_in_patch[t-1].
        first = np.r_[True, win[1:] != win[:-1]]
        stale = np.r_[0, pos[:-1]].astype(int)
        P("\n**Boundary vs interior at MATCHED patch-vector staleness** (removes the confound that "
          "a pos-0 byte reads the PREVIOUS patch's vector, i.e. an older one)\n")
        P("| staleness (B) | boundary n | boundary full | boundary Δ | interior n | interior full | interior Δ |")
        P("|---|---|---|---|---|---|---|")
        for sv in range(6):
            m = ~first & (stale == sv)
            b, i = m & bnd, m & ~bnd
            if b.sum() < 50 or i.sum() < 50:
                continue
            P(f"| {sv} | {int(b.sum())} | {full[b].mean() / LN2:.3f} | {(loc[b] - full[b]).mean() / LN2:+.3f} "
              f"| {int(i.sum())} | {full[i].mean() / LN2:.3f} | {(loc[i] - full[i]).mean() / LN2:+.3f} |")

        P("\n**Patch position × boundary/interior split of the offset trend** (Δ BPB)\n")
        P("| byte offset | Δ boundary | Δ interior |")
        P("|---|---|---|")
        for a, b in zip(OFF_EDGES[:-1], OFF_EDGES[1:]):
            m = (off >= a) & (off < b)
            db = (loc[m & bnd].mean() - full[m & bnd].mean()) / LN2
            di = (loc[m & ~bnd].mean() - full[m & ~bnd].mean()) / LN2
            P(f"| {a}–{b - 1} | {db:+.3f} | {di:+.3f} |")

    txt = "\n".join(lines)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
