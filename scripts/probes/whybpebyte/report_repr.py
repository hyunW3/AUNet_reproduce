#!/usr/bin/env python3
"""Experiment E tables: representation locality at the global transformer's input.

    python report_repr.py            # both regimes, whatever rows exist
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"
ORDER = ["llama", "aunet", "bpebyte"]
NAME = {"llama": "Llama (subword)", "aunet": "AU-Net", "bpebyte": "BPEByte"}
BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 10 ** 9)]
MAXD = 256


def boot_ci(x, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    idx = {v: np.flatnonzero(groups == v) for v in g}
    out = [x[np.concatenate([idx[v] for v in rng.choice(g, len(g))])].mean() for _ in range(n)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def rows_for(tag, arm):
    p = RES / f"repr_{tag}_{arm}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def single():
    have = {a: rows_for("single", a) for a in ORDER}
    if not any(have.values()):
        return
    print("### Experiment E1 — one character edit, representation at the trunk input\n")
    print("Centred cosine between the clean and perturbed vector covering each raw byte "
          "(1 = identical, 0 = as unrelated as two random units of the same model).\n")
    hdr = "| Model | null cos (raw) | " + " | ".join(
        f"d={lo}" if lo == hi else (f"d≥{lo}" if hi > 10 ** 8 else f"d={lo}-{hi}")
        for lo, hi in BINS) + " | R₀.₉₅ p90 |"
    print(hdr)
    print("|" + "---|" * (len(BINS) + 3))
    def emit(label, curve_path, r):
        cur = np.load(curve_path)
        sim, cnt = cur[0], cur[1]
        cells = []
        for lo, hi in BINS:
            hi = min(hi, MAXD)
            c = cnt[lo:hi + 1].sum()
            cells.append(f"{(sim[lo:hi+1]*cnt[lo:hi+1]).sum()/c:.3f}" if c else "--")
        rad = np.array([x.get("repr_radius_095", 0) for x in r], float)
        nullraw = np.mean([x["null_cos"] for x in r])
        print(f"| {label} | {nullraw:.3f} | " + " | ".join(cells) +
              f" | {np.percentile(rad, 90):.0f} |")

    any_depth = False
    for arm in ORDER:
        r = have[arm]
        if not r:
            continue
        emit(NAME[arm], RES / f"repr_single_{arm}.curve.npy", r)
        dp = RES / f"repr_single_{arm}.curve_depth.npy"
        if dp.exists():
            emit(NAME[arm] + ", after 3 layers", dp, r)
            any_depth = True
    print()
    if any_depth:
        print("The depth-matched row exists so the comparison is not \"a lookup table against a "
              "3-layer encoder\": it is the subword model's hidden state after as many layers as "
              "the byte arms' local encoder.\n")
    print("Read the byte-distance axis with one caveat: a model with longer units spreads a single "
          "edit over more bytes, so part of AU-Net's advantage at d=1 is that its patches are "
          "longer, not that its vectors move less. E2 is the cleaner comparison, because there "
          "every unit is corrupted.\n")


def suite():
    have = {a: rows_for("suite", a) for a in ORDER}
    if not any(have.values()):
        return
    fams = sorted({x["family"] for r in have.values() for x in r})
    print("### Experiment E2 — the paper's corruption families, whole span\n")
    print("Unit-sequence Levenshtein per CLEAN unit (↓ better; it can exceed 1 when the "
          "corruption lengthens the sequence), mean centred cosine of the trunk input (↑ better), "
          "and how many units the corrupted span is re-segmented into relative to the clean one. "
          "95% CI bootstrapped over documents.\n")
    print("| Family | " + " | ".join(f"{NAME[a]} Lev/unit" for a in ORDER) + " | " +
          " | ".join(f"{NAME[a]} cos" for a in ORDER) + " | " +
          " | ".join(f"{NAME[a]} units×" for a in ORDER) + " |")
    print("|" + "---|" * (1 + 3 * len(ORDER)))
    agg = defaultdict(dict)
    for fam in fams:
        lev, cos, ratio = [], [], []
        for a in ORDER:
            r = [x for x in have[a] if x["family"] == fam]
            if not r:
                lev.append("--"); cos.append("--"); ratio.append("--"); continue
            L = np.array([x["unit_levenshtein_norm"] for x in r])
            C = np.array([x.get("cos_centered_mean", np.nan) for x in r])
            w = np.array([x["window"] for x in r])
            lo, hi = boot_ci(L, w)
            clo, chi = boot_ci(C, w)
            uc = np.array([x["n_units_clean"] for x in r], float)
            up = np.array([x["n_units_pert"] for x in r], float)
            lev.append(f"{L.mean():.3f} [{lo:.3f},{hi:.3f}]")
            cos.append(f"{np.nanmean(C):.3f} [{clo:.3f},{chi:.3f}]")
            ratio.append(f"{up.sum()/max(uc.sum(), 1):.2f}x")
            agg[fam][a] = (float(L.mean()), float(np.nanmean(C)))
        print(f"| {fam} | " + " | ".join(lev) + " | " + " | ".join(cos) + " | " +
              " | ".join(ratio) + " |")
    print()
    print("| Model | macro Lev/unit | macro centred cos | bytes below cos 0.95 |")
    print("|" + "---|" * 4)
    for a in ORDER:
        r = have[a]
        if not r:
            continue
        L = np.array([x["unit_levenshtein_norm"] for x in r])
        C = np.array([x.get("cos_centered_mean", np.nan) for x in r])
        F = np.array([x.get("frac_bytes_below_095", np.nan) for x in r])
        print(f"| {NAME[a]} | {L.mean():.3f} | {np.nanmean(C):.3f} | {np.nanmean(F)*100:.1f}% |")
        D = np.array([x.get("cos_depth_centered_mean", np.nan) for x in r])
        if not np.isnan(D).all():
            FD = np.array([x.get("frac_bytes_depth_below_095", np.nan) for x in r])
            print(f"| {NAME[a]}, after 3 layers | -- | {np.nanmean(D):.3f} | "
                  f"{np.nanmean(FD)*100:.1f}% |")


if __name__ == "__main__":
    single()
    suite()
