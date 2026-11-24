#!/usr/bin/env python3
"""M5 / Experiment F statistics: does segmentation disruption predict loss degradation?

Reads results/downstream_<arm>.jsonl (exp_downstream.py) and reports, per arm:
  * mean Delta BPB on the unedited tail, with a paired document-level bootstrap CI;
  * Spearman rho between Delta BPB and each disruption measure (design section 9);
  * an OLS fit Delta y ~ b0 + b1*R + b2*Nhat + b3*L, the design's example-level regression.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"
ARMS = ["llama", "aunet", "bpebyte"]
NAME = {"llama": "Llama (subword)", "aunet": "AU-Net", "bpebyte": "BPEByte"}


def boot_mean(x, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    idx = {v: np.flatnonzero(groups == v) for v in g}
    out = []
    for _ in range(n):
        pick = rng.choice(g, len(g))
        sel = np.concatenate([idx[v] for v in pick])
        out.append(x[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, p, r2


def main():
    out = {"arms": {}}
    print(f"{'arm':16s} {'n':>5s} {'dBPB':>9s} {'95% CI':>18s} {'rho(Nchg)':>10s} "
          f"{'rho(R)':>8s} {'mismatch':>9s}")
    for arm in ARMS:
        p = RES / f"downstream_{arm}.jsonl"
        if not p.exists():
            print(f"{arm:16s}  (missing)")
            continue
        rows = [json.loads(l) for l in open(p)]
        y = np.array([r["delta_bpb"] for r in rows])
        win = np.array([r["window"] for r in rows])
        nchg = np.array([r["n_changed_downstream"] for r in rows], float)
        rad = np.array([r["radius"] for r in rows], float)
        L = np.array([r["region_bytes_clean"] for r in rows], float)
        lo, hi = boot_mean(y, win)
        rho_n = stats.spearmanr(nchg, y)
        rho_r = stats.spearmanr(rad, y)
        beta, se, pv, r2 = ols(y, [rad, nchg / np.maximum(L, 1) * 100, L])
        mism = float(np.mean([r["region_token_mismatch"] for r in rows]))
        out["arms"][arm] = {
            "n": len(rows), "mean_delta_bpb": float(y.mean()), "ci": [lo, hi],
            "median_delta_bpb": float(np.median(y)),
            "spearman_nchg": [float(rho_n.statistic), float(rho_n.pvalue)],
            "spearman_radius": [float(rho_r.statistic), float(rho_r.pvalue)],
            "ols": {"names": ["const", "radius", "changed_per_100B", "region_bytes"],
                    "beta": beta.tolist(), "se": se.tolist(), "p": pv.tolist(),
                    "r2": float(r2)},
            "region_token_mismatch": mism,
            "mean_changed_downstream": float(nchg.mean()),
        }
        print(f"{NAME[arm]:16s} {len(rows):5d} {y.mean():+9.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {rho_n.statistic:10.3f} {rho_r.statistic:8.3f} "
              f"{mism:9.3f}")
    if out["arms"]:
        json.dump(out, open(RES / "downstream_stats.json", "w"), indent=1)
        print(f"\n-> {RES}/downstream_stats.json")
        for arm, d in out["arms"].items():
            b, se, pv = d["ols"]["beta"], d["ols"]["se"], d["ols"]["p"]
            print(f"\n{NAME[arm]}  OLS  R2={d['ols']['r2']:.4f}")
            for nm, bi, si, pi in zip(d["ols"]["names"], b, se, pv):
                print(f"   {nm:18s} {bi:+.6f}  (se {si:.6f}, p={pi:.3g})")


if __name__ == "__main__":
    main()
