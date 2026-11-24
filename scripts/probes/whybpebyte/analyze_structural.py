#!/usr/bin/env python3
"""Experiments A + B: within-unit predictability and boundaries as information transitions.

Reads the Stage-1/2 cache (dump.py) and emits per-arm structural metrics with document-level
bootstrap CIs, plus the two controls the design insists on (section 4.2 / 11): a length-matched
random segmentation and fixed-width chunks at the arm's own mean bytes-per-unit.

    python analyze_structural.py --domain en --probe blt_entropy

Output: reports_afterAAAIsub/whybpebyte/results/structural_<probe>_<domain>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
CACHE = ROOT / "reports_afterAAAIsub/whybpebyte/cache"
RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"

import metrics as M                                                    # noqa: E402
from segmenters import fixed_k_starts, random_matched_starts, starts_to_mask, starts_to_lengths  # noqa: E402

REAL = ["llama", "aunet", "bpebyte"]
N_BOOT = 1000
N_BOOT_RANK = 200
N_RAND = 200          # randomisations for the length-matched control / z-score


# ----------------------------------------------------------------------- aggregation
def agg(rows: list[dict]) -> dict:
    """Corpus aggregate from per-window sufficient statistics (ratio of sums)."""
    T = {k: float(np.sum([r.get(k, 0.0) for r in rows])) for k in rows[0]}
    d = {}
    d["bytes_per_unit"] = T["n_bytes"] / T["n_units"]
    d["boundary_rate"] = T["n_units"] / T["n_bytes"]
    d["S_boundary"] = T["sb"] / max(T["nb"], 1)
    d["S_internal"] = T["si"] / max(T["ni"], 1)
    d["dS"] = d["S_boundary"] - d["S_internal"]
    d["C_unit"] = T["sum_c_unit"] / max(T["n_units_c"], 1)
    d["C_byte"] = T["sum_c_byte"] / max(T["sum_len_c"], 1)
    for k in M.PEAK_KS:
        d[f"peak_k{k}"] = T[f"sum_peak{k}"] / max(T[f"n_peak{k}"], 1)
    for k in (4, 8):
        d[f"D_boundary_k{k}"] = T[f"sDb{k}"] / max(T[f"nDb{k}"], 1)
        d[f"D_internal_k{k}"] = T[f"sDi{k}"] / max(T[f"nDi{k}"], 1)
        d[f"dD_k{k}"] = d[f"D_boundary_k{k}"] - d[f"D_internal_k{k}"]
    # offset-swept gap: Delta S(delta) = E[s_{i+delta} | boundary] - E[s_{j+delta} | interior]
    d["dS_shift"] = {str(k): (T[f"sb{k:+d}"] / max(T[f"nb{k:+d}"], 1)) -
                             (T[f"si{k:+d}"] / max(T[f"ni{k:+d}"], 1)) for k in M.SHIFTS}
    if "nb_nws" in T:
        d["S_boundary_nws"] = T["sb_nws"] / max(T["nb_nws"], 1)
        d["S_internal_nws"] = T["si_nws"] / max(T["ni_nws"], 1)
        d["dS_nws"] = d["S_boundary_nws"] - d["S_internal_nws"]
        d["frac_boundary_ws"] = T["nb_ws"] / max(T["nb"], 1)
    for tag in ("L2", "L34", "L58", "L916", "L17p"):
        n = T.get(f"n_c_{tag}", 0.0)
        d[f"C_{tag}"] = (T.get(f"sum_c_{tag}", 0.0) / n) if n else float("nan")
        d[f"n_{tag}"] = n
    return d


def boot_ci(rows: list[dict], keys, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    W = len(rows)
    cols = {k: np.array([r.get(k, 0.0) for r in rows]) for k in rows[0]}
    out = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, W, W)
        T = {k: v[idx].sum() for k, v in cols.items()}
        a = agg([T])
        for k in keys:
            out[k].append(a[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


# ----------------------------------------------------------------------- arms
def arm_masks(seg, arm, n_bytes):
    from dump import unpack
    starts = unpack(seg[f"starts_{arm}"], seg[f"offs_{arm}"])
    return [starts_to_mask(s, int(n)) for s, n in zip(starts, n_bytes)]


def control_masks(kind, real_starts, n_bytes, rng):
    out = []
    for s, n in zip(real_starts, n_bytes):
        n = int(n)
        if kind == "random_matched":
            out.append(starts_to_mask(random_matched_starts(starts_to_lengths(s, n), rng), n))
        elif kind == "fixed":
            k = max(1, int(round(n / len(s))))
            out.append(starts_to_mask(fixed_k_starts(n, k), n))
    return out


def measure(S, masks, RAW=None):
    rows = [M.window_stats(s, m, r) for s, m, r in zip(S, masks, RAW or [None] * len(S))]
    K = M.PROFILE_K
    psum = np.zeros(2 * K + 1); pcnt = np.zeros(2 * K + 1)
    rs = np.zeros(M.REL_BINS); rn = np.zeros(M.REL_BINS)
    asum = np.zeros(M.ABS_OFFSETS); acnt = np.zeros(M.ABS_OFFSETS)
    for s, m in zip(S, masks):
        a, b = M.boundary_profile(s, m, K)
        psum += a; pcnt += b
        c, d, e, f = M.within_unit_profile(s, m)
        rs += c; rn += d; asum += e; acnt += f
    return rows, {
        "boundary_profile": (psum / np.maximum(pcnt, 1)).tolist(),
        "boundary_profile_offsets": list(range(-K, K + 1)),
        "within_unit_rel": (rs / np.maximum(rn, 1)).tolist(),
        "within_unit_abs": (asum / np.maximum(acnt, 1)).tolist(),
    }


def retrieval(S, masks, seed=0, n_boot=N_BOOT_RANK, shift=0):
    """Boundary retrieval from probe surprisal alone (design 5.3), bootstrapped over windows.

    `shift` scores boundary i with s_{i+shift}. shift=0 is the design's literal metric; shift=1 is
    the placement-corrected one, because every production scheme puts the separating space at
    offset 0 of the unit and the information transition at offset +1.
    """
    d = int(shift)
    sc = [s[1 + d:len(s) - max(0, -d)] if d else s[1:] for s in S]
    lb = [m[1:len(m) - d] if d > 0 else m[1:] for m in masks]
    assert all(len(x) == len(y) for x, y in zip(sc, lb))
    flat_s, flat_l = np.concatenate(sc), np.concatenate(lb)
    point = {"shift": d, "auroc": M.auroc(flat_s, flat_l), "auprc": M.auprc(flat_s, flat_l),
             "prec_at_rate": M.precision_at_rate(flat_s, flat_l),
             "base_rate": float(flat_l.mean())}
    rng = np.random.default_rng(seed)
    W = len(sc)
    acc = {k: [] for k in ("auroc", "auprc", "prec_at_rate")}
    for _ in range(n_boot):
        idx = rng.integers(0, W, W)
        fs = np.concatenate([sc[i] for i in idx])
        fl = np.concatenate([lb[i] for i in idx])
        acc["auroc"].append(M.auroc(fs, fl))
        acc["auprc"].append(M.auprc(fs, fl))
        acc["prec_at_rate"].append(M.precision_at_rate(fs, fl))
    point["ci"] = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                   for k, v in acc.items()}
    return point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="en")
    ap.add_argument("--probe", default="blt_entropy")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--n_rand", type=int, default=N_RAND)
    ap.add_argument("--no-retrieval", action="store_true")
    a = ap.parse_args()

    from dump import unpack
    seg = np.load(CACHE / f"seg_{a.domain}.npz")
    pr = np.load(CACHE / f"probe_{a.probe}_{a.domain}.npz", allow_pickle=True)
    probe_info = json.loads(str(pr["info"]))
    offs = pr["offs"]
    S = [pr["s"][offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
    from dump import load_windows
    RAW = [np.frombuffer(w["text"].encode(), dtype=np.uint8) for w in load_windows(a.domain)]
    n_bytes = seg["n_bytes"]
    assert all(len(s) == n for s, n in zip(S, n_bytes)), "probe/segmentation length mismatch"
    arms = a.arms or [k[len("starts_"):] for k in seg.files if k.startswith("starts_")]

    out = {"domain": a.domain, "probe": probe_info, "n_windows": len(S),
           "total_bytes": int(sum(n_bytes)), "arms": {}}
    CI_KEYS = ["dS", "S_boundary", "S_internal", "C_unit", "C_byte",
               "peak_k2", "peak_k4", "peak_k8", "dD_k4", "dD_k8", "bytes_per_unit"]

    for arm in arms:
        print(f"[{arm}]", flush=True)
        masks = arm_masks(seg, arm, n_bytes)
        rows, prof = measure(S, masks, RAW)
        d = agg(rows)
        d["ci"] = boot_ci(rows, CI_KEYS)
        d.update(prof)
        if not a.no_retrieval:
            d["retrieval"] = retrieval(S, masks, shift=0)
            d["retrieval_shift1"] = retrieval(S, masks, shift=1)
        # ---- controls, defined relative to this arm
        real_starts = unpack(seg[f"starts_{arm}"], seg[f"offs_{arm}"])
        rng = np.random.default_rng(12345)
        rand_dS, rand_peak4, rand_C, rand_shift, rand_prof = [], [], [], [], []
        for r in range(a.n_rand):
            rm = control_masks("random_matched", real_starts, n_bytes, rng)
            rrows, rprof = measure(S, rm, RAW)
            ra = agg(rrows)
            rand_dS.append(ra["dS"]); rand_peak4.append(ra["peak_k4"]); rand_C.append(ra["C_unit"])
            rand_shift.append([ra["dS_shift"][str(k)] for k in M.SHIFTS])
            rand_prof.append(rprof["boundary_profile"])
        mu, sd = float(np.mean(rand_dS)), float(np.std(rand_dS, ddof=1))
        d["control_random"] = {
            "n_rand": a.n_rand,
            "dS_mean": mu, "dS_std": sd,
            "dS_z": (d["dS"] - mu) / sd if sd > 0 else float("nan"),
            "peak_k4_mean": float(np.mean(rand_peak4)),
            "peak_k4_z": (d["peak_k4"] - float(np.mean(rand_peak4))) /
                         max(float(np.std(rand_peak4, ddof=1)), 1e-12),
            "C_unit_mean": float(np.mean(rand_C)),
            "dS_shift_mean": {str(k): float(v) for k, v in
                              zip(M.SHIFTS, np.mean(rand_shift, axis=0))},
            "dS_shift_std": {str(k): float(v) for k, v in
                             zip(M.SHIFTS, np.std(rand_shift, axis=0, ddof=1))},
            "boundary_profile_mean": np.mean(rand_prof, axis=0).tolist(),
        }
        # placement-corrected headline: the offset where the arm's boundaries carry the most
        # information ABOVE a length-matched random cut at the same offset.
        gap = {k: d["dS_shift"][k] - d["control_random"]["dS_shift_mean"][k] for k in d["dS_shift"]}
        best = max(gap, key=lambda k: gap[k])
        d["dS_gap_vs_random"] = gap
        d["dS_gap_best_offset"] = int(best)
        d["dS_gap_best"] = gap[best]
        fm = control_masks("fixed", real_starts, n_bytes, rng)
        fr, fprof = measure(S, fm)
        fa = agg(fr)
        d["control_fixed"] = {k: fa[k] for k in ("bytes_per_unit", "dS", "peak_k4", "C_unit", "C_byte")}
        d["control_fixed"]["boundary_profile"] = fprof["boundary_profile"]
        out["arms"][arm] = d
        print(f"   B/unit={d['bytes_per_unit']:.3f}  dS={d['dS']:+.4f} "
              f"(rand {mu:+.4f}, z={d['control_random']['dS_z']:+.1f}; fixed {fa['dS']:+.4f})  "
              f"C={d['C_unit']:+.4f}  peak4={d['peak_k4']:+.4f}\n"
              f"   dS_nws={d.get('dS_nws', float('nan')):+.4f} "
              f"(ws-initial boundaries {100*d.get('frac_boundary_ws', 0):.0f}%)  "
              f"gap_vs_random: best {d['dS_gap_best']:+.4f} @ offset {d['dS_gap_best_offset']:+d}",
              flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    p = RES / f"structural_{a.probe}_{a.domain}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
