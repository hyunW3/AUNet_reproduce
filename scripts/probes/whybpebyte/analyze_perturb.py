#!/usr/bin/env python3
"""Experiment D: how far does ONE local byte edit propagate through each segmentation?

Design section 7. For every (window, family, site) triple the same perturbed raw string is fed to
every arm -- never a model-specific perturbation -- and boundary agreement is measured in CLEAN
byte coordinates through the exact alignment from perturb.apply_edit.

    python analyze_perturb.py --domain en --sites 16

Outputs
    results/perturb_<domain>.json    aggregate curves, radii, changed-boundary rates
    results/perturb_<domain>.jsonl   one row per (arm, window, family, site) for the
                                     downstream-robustness join (design section 9 / stage 5)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"

import perturb as P                                                    # noqa: E402
from segmenters import build, starts_to_mask, fixed_k_starts           # noqa: E402

MAXD = 256                       # per-byte distance resolution kept for the recovery distance
BINS = [(0, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 9)]
PROTECT_KS = (4, 8, 16, 32)
ARMS = ["llama", "aunet", "bpebyte", "whitespace"]


def distances(n: int, touched: np.ndarray) -> np.ndarray:
    """Distance in clean byte coordinates from the nearest edited byte."""
    lo, hi = int(touched[0]), int(touched[-1])
    d = np.zeros(n, dtype=np.int64)
    d[:lo] = lo - np.arange(lo)
    d[hi + 1:] = np.arange(hi + 1, n) - hi
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="en")
    ap.add_argument("--sites", type=int, default=16, help="edit sites per window per family")
    ap.add_argument("--families", nargs="*", default=P.FAMILIES)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fixed-control", action="store_true", default=True)
    a = ap.parse_args()

    from dump import load_windows
    wins = load_windows(a.domain, a.limit)
    segs = {arm: build(arm) for arm in a.arms}

    # Fixed-width control at llama's mean bytes/unit: an offset-indexed scheme, so a length-
    # changing edit re-phases every later chunk. It is the "maximum cascade" reference point.
    K_FIXED = 5

    acc = {}          # (arm, family) -> dict of accumulators
    rows = []
    rng = random.Random(a.seed)

    def blank():
        return {"agree": np.zeros(MAXD + 1), "tot": np.zeros(MAXD + 1),
                "radii": [], "nchg": [], "nchg_k": defaultdict(list), "n_samples": 0,
                "nbytes": [], "nunit_chg": [], "nunit": [], "unit_radii": []}

    for wi, w in enumerate(wins):
        text = w["text"]
        n = len(text.encode())
        clean_mask = {}
        for arm in a.arms:
            clean_mask[arm] = starts_to_mask(segs[arm](text), n)
        if a.fixed_control:
            clean_mask["fixed5"] = starts_to_mask(fixed_k_starts(n, K_FIXED), n)

        for fam in a.families:
            for ci in P.sample_edits(text, fam, a.sites, rng):
                r = P.apply_edit(text, fam, ci, rng)
                if r is None:
                    continue
                new, align, touched = r
                nn = len(new.encode())
                d = distances(n, touched)
                keep = (align >= 0) & (d >= 1)
                for arm in list(a.arms) + (["fixed5"] if a.fixed_control else []):
                    if arm == "fixed5":
                        pm = starts_to_mask(fixed_k_starts(nn, K_FIXED), nn)
                    else:
                        pm = starts_to_mask(segs[arm](new), nn)
                    A = clean_mask[arm][keep] == pm[align[keep]]
                    dk = np.minimum(d[keep], MAXD)
                    key = (arm, fam)
                    st = acc.setdefault(key, blank())
                    np.add.at(st["agree"], dk, A.astype(float))
                    np.add.at(st["tot"], dk, 1.0)
                    bad = ~A
                    # --- unit-level view (design 10.1, architecture-specific supplement).
                    # A clean unit SURVIVES iff every aligned byte it covers, and its closing
                    # boundary, agree. For the subword arm a broken unit is a different input
                    # SYMBOL (a different embedding row) fed to the model; for the byte arms the
                    # bytes are unchanged and only the pooling of them moves. Same count, very
                    # different consequence -- which is the point of separating the two views.
                    agree_full = np.zeros(n, bool)
                    agree_full[keep] = A
                    agree_full[~keep] = False              # edited bytes count as broken
                    cs_ag = np.concatenate(([0], np.cumsum(agree_full)))
                    st_c = np.flatnonzero(clean_mask[arm])
                    en_c = np.append(st_c[1:], n)
                    intact = (cs_ag[en_c] - cs_ag[st_c]) == (en_c - st_c)
                    ud = np.minimum(d[st_c], MAXD)
                    st["nunit_chg"].append(int((~intact).sum()))
                    st["nunit"].append(int(len(st_c)))
                    st["unit_radii"].append(int(ud[~intact].max()) if (~intact).any() else 0)
                    R = int(dk[bad].max()) if bad.any() else 0
                    st["radii"].append(R)
                    st["nchg"].append(int(bad.sum()))
                    st["nbytes"].append(int(keep.sum()))
                    for k in PROTECT_KS:
                        st["nchg_k"][k].append(int((bad & (dk > k)).sum()))
                    st["n_samples"] += 1
                    rows.append({"arm": arm, "window": wi, "family": fam, "site_char": int(ci),
                                 "n_bytes": n, "radius": R, "n_changed": int(bad.sum()),
                                 "n_units_changed": st["nunit_chg"][-1],
                                 "n_units": st["nunit"][-1],
                                 "unit_radius": st["unit_radii"][-1],
                                 "changed_frac": float(bad.sum()) / max(int(keep.sum()), 1),
                                 **{f"n_changed_gt{k}": int((bad & (dk > k)).sum())
                                    for k in PROTECT_KS}})
        if (wi + 1) % 20 == 0:
            print(f"  {wi+1}/{len(wins)} windows", flush=True)

    out = {"domain": a.domain, "sites_per_window": a.sites, "families": a.families,
           "n_windows": len(wins), "arms": {}}
    for (arm, fam), st in sorted(acc.items()):
        S = st["agree"] / np.maximum(st["tot"], 1)
        # recovery distance: smallest d beyond which agreement never dips below tau again
        rec = {}
        for tau in (0.95, 0.99):
            ok = (S >= tau) | (st["tot"] == 0)
            bad = np.flatnonzero(~ok)
            rec[str(tau)] = int(bad[-1] + 1) if bad.size else 1
        binned = []
        for lo, hi in BINS:
            hi = min(hi, MAXD)
            t = st["tot"][lo:hi + 1].sum()
            binned.append(float(st["agree"][lo:hi + 1].sum() / t) if t else float("nan"))
        radii = np.array(st["radii"], float)
        nchg = np.array(st["nchg"], float)
        nb = np.array(st["nbytes"], float)
        out["arms"].setdefault(arm, {})[fam] = {
            "n_samples": st["n_samples"],
            "agreement_by_bin": binned,
            "bins": [f"{lo}-{hi if hi < 10**9 else ''}" for lo, hi in BINS],
            "agreement_by_distance": S[:65].tolist(),
            "radius": {"mean": float(radii.mean()), "p50": float(np.percentile(radii, 50)),
                       "p90": float(np.percentile(radii, 90)), "p95": float(np.percentile(radii, 95)),
                       "p99": float(np.percentile(radii, 99)), "max": float(radii.max())},
            "changed_boundaries": {"mean": float(nchg.mean()),
                                   "per_kb": float((nchg / nb).mean() * 1000),
                                   **{f"gt{k}_mean": float(np.mean(st['nchg_k'][k]))
                                      for k in PROTECT_KS}},
            "recovery_distance": rec,
            "units_changed": {
                "mean": float(np.mean(st["nunit_chg"])),
                "p90": float(np.percentile(st["nunit_chg"], 90)),
                "frac": float(np.sum(st["nunit_chg"]) / max(np.sum(st["nunit"]), 1)),
                "radius_p50": float(np.percentile(st["unit_radii"], 50)),
                "radius_p90": float(np.percentile(st["unit_radii"], 90)),
                "radius_p99": float(np.percentile(st["unit_radii"], 99)),
            },
        }
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / f"perturb_{a.domain}.json", "w"), indent=1)
    with open(RES / f"perturb_{a.domain}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"-> {RES}/perturb_{a.domain}.json  ({len(rows)} rows)")

    # console summary, pooled over families
    print(f"\n{'arm':11s} {'p50 R':>6s} {'p90 R':>6s} {'p99 R':>6s} {'chg/kB':>7s} "
          f"{'chg>8':>6s} {'D95':>5s} {'D99':>5s} {'units':>6s} {'uR p90':>7s}")
    for arm in out["arms"]:
        fams = out["arms"][arm]
        rr = np.concatenate([np.array(acc[(arm, f)]["radii"], float) for f in fams])
        cc = np.concatenate([np.array(acc[(arm, f)]["nchg"], float) /
                             np.array(acc[(arm, f)]["nbytes"], float) for f in fams]) * 1000
        g8 = np.concatenate([np.array(acc[(arm, f)]["nchg_k"][8], float) for f in fams])
        d95 = max(v["recovery_distance"]["0.95"] for v in fams.values())
        d99 = max(v["recovery_distance"]["0.99"] for v in fams.values())
        uu = np.concatenate([np.array(acc[(arm, f)]["nunit_chg"], float) for f in fams])
        ur = np.concatenate([np.array(acc[(arm, f)]["unit_radii"], float) for f in fams])
        print(f"{arm:11s} {np.percentile(rr,50):6.0f} {np.percentile(rr,90):6.0f} "
              f"{np.percentile(rr,99):6.0f} {cc.mean():7.2f} {g8.mean():6.2f} {d95:5d} {d99:5d} "
              f"{uu.mean():6.2f} {np.percentile(ur,90):7.0f}")


if __name__ == "__main__":
    main()
