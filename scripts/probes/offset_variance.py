#!/usr/bin/env python3
"""Dispersion and uncertainty for the patch_offset-binned BPB.

Two different questions get confused if only the bin mean is reported:

  1. How SPREAD OUT is the loss inside a bin?  offset 0 has BPB 2.85 -- is every boundary byte
     about that expensive, or is it a few catastrophic bytes dragging a cheap majority up?  The
     answer changes what an intervention there could even do.
  2. How CERTAIN is a bin's mean, and a difference between two arms in that bin?  Bytes inside a
     window are not independent, so a per-byte standard error is far too small.  Resampling the
     320 WINDOWS is the honest unit, and arms are compared paired (same bytes, same order).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "reports/patch_lag/offset"
LN2 = math.log(2.0)
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 1 << 30)]


def lab(a, b):
    return f"{a}" if a == b else (f"{a}+" if b > 1 << 20 else f"{a}-{b}")


def per_window(vals, win_id, mask, n_win):
    """-> (per-window sum over the masked bytes, per-window count)."""
    s = np.bincount(win_id[mask], weights=vals[mask], minlength=n_win)
    c = np.bincount(win_id[mask], minlength=n_win).astype(float)
    return s, c


def boot(s, c, n_boot, rng):
    """95% CI of sum(s)/sum(c) resampling windows with replacement."""
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    r = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1e-9)
    return np.percentile(r, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ece_base")
    ap.add_argument("--arms", default="")
    ap.add_argument("--win_lens", required=True)
    ap.add_argument("--n_boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lens = np.load(args.win_lens)
    win_id = np.repeat(np.arange(len(lens)), lens)
    n_win = len(lens)
    rng = np.random.default_rng(args.seed)

    b = np.load(OFF / f"{args.base}.npz")
    nll, off = b["nll"] / LN2, b["offset"]          # per-byte BPB
    assert nll.size == win_id.size, (nll.size, win_id.size)

    print(f"\n=== {args.base}: 구간 내 분산 (단위 bit/byte) ===")
    print(f"{'offset':>7s}{'n':>9s}{'평균':>8s}{'표준편차':>9s}{'중앙값':>8s}"
          f"{'p90':>7s}{'p99':>7s}{'>4비트 비율':>12s}{'   평균의 95% CI (윈도우 부트스트랩)':>36s}")
    for lo, hi in BINS:
        m = (off >= lo) & (off <= hi)
        v = nll[m]
        s, c = per_window(nll, win_id, m, n_win)
        lo_, hi_ = boot(s, c, args.n_boot, rng)
        print(f"{lab(lo,hi):>7s}{m.sum():>9d}{v.mean():>8.4f}{v.std():>9.4f}"
              f"{np.median(v):>8.4f}{np.percentile(v,90):>7.3f}{np.percentile(v,99):>7.3f}"
              f"{(v>4).mean():>11.1%}   [{lo_:.4f}, {hi_:.4f}]  ±{(hi_-lo_)/2:.4f}")

    if not args.arms:
        return
    print(f"\n=== 짝지은 차이 vs {args.base} (같은 바이트, 윈도우 부트스트랩 95% CI) ===")
    print(f"{'arm':>22s}{'offset':>9s}{'Δ BPB':>10s}{'   95% CI':>22s}{'  0 포함?':>10s}")
    for a in args.arms.split(","):
        x = np.load(OFF / f"{a}.npz")["nll"] / LN2
        d = x - nll
        for lo, hi in [(0, 0), (1, 1 << 30)]:
            m = (off >= lo) & (off <= hi)
            s, c = per_window(d, win_id, m, n_win)
            l_, h_ = boot(s, c, args.n_boot, rng)
            name = "offset 0" if lo == 0 else "offset>=1"
            print(f"{a:>22s}{name:>9s}{d[m].mean():>+10.4f}   [{l_:>+8.4f}, {h_:>+8.4f}]"
                  f"{('예 (유의하지 않음)' if l_ < 0 < h_ else '아니오'):>12s}")
        m = np.ones(off.size, bool)
        s, c = per_window(d, win_id, m, n_win)
        l_, h_ = boot(s, c, args.n_boot, rng)
        print(f"{a:>22s}{'전체':>9s}{d.mean():>+10.4f}   [{l_:>+8.4f}, {h_:>+8.4f}]"
              f"{('예 (유의하지 않음)' if l_ < 0 < h_ else '아니오'):>12s}\n")


if __name__ == "__main__":
    main()
