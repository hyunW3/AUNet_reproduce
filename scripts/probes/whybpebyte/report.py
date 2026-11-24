#!/usr/bin/env python3
"""Assemble the structured-locality report: the design's section-14 main table plus the
per-experiment tables, from the JSON emitted by the analysis scripts.

    python report.py --domain en --probe blt_entropy > ../../../reports_afterAAAIsub/whybpebyte/results.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"
ORDER = ["llama", "aunet", "bpebyte"]
NAME = {"llama": "Llama (subword)", "aunet": "AU-Net", "bpebyte": "BPEByte",
        "whitespace": "Whitespace (ref.)", "fixed5": "Fixed-width k=5 (control)"}


def f(x, n=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def radius_bootstrap(path, n=1000, seed=0):
    """Document-level bootstrap CI for the radius quantiles and the changed-boundary rate
    (design section 12: resample documents, not byte positions)."""
    if not Path(path).exists():
        return {}
    import collections
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        per[r["arm"]][r["window"]].append((r["radius"], r["changed_frac"]))
    rng = np.random.default_rng(seed)
    out = {}
    for arm, byw in per.items():
        wins = list(byw)
        acc = {"p50": [], "p90": [], "p99": [], "chg_per_kb": []}
        for _ in range(n):
            pick = rng.choice(len(wins), len(wins))
            v = np.concatenate([np.array(byw[wins[i]]) for i in pick])
            acc["p50"].append(np.percentile(v[:, 0], 50))
            acc["p90"].append(np.percentile(v[:, 0], 90))
            acc["p99"].append(np.percentile(v[:, 0], 99))
            acc["chg_per_kb"].append(v[:, 1].mean() * 1000)
        allv = np.concatenate([np.array(v) for v in byw.values()])
        out[arm] = {k: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
                    for k, x in acc.items()}
        out[arm]["point"] = {"p50": float(np.percentile(allv[:, 0], 50)),
                             "p90": float(np.percentile(allv[:, 0], 90)),
                             "p99": float(np.percentile(allv[:, 0], 99)),
                             "chg_per_kb": float(allv[:, 1].mean() * 1000)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="en")
    ap.add_argument("--probe", default="blt_entropy")
    a = ap.parse_args()
    S = json.load(open(RES / f"structural_{a.probe}_{a.domain}.json"))
    boot = radius_bootstrap(RES / f"perturb_{a.domain}.jsonl")
    P = json.load(open(RES / f"perturb_{a.domain}.json")) if (RES / f"perturb_{a.domain}.json").exists() else None
    arms = [x for x in ORDER + ["whitespace"] if x in S["arms"]]

    pr = S["probe"]
    print(f"### Setup ({a.domain})\n")
    print(f"- Corpus: {S['n_windows']} held-out windows, {S['total_bytes']:,} raw bytes.")
    parts = Path(pr["ckpt"]).parts
    pname = next((x for x in reversed(parts) if x not in ("consolidated",) and
                  not x.isdigit() and x != "checkpoints"), parts[-1])
    print(f"- Frozen byte-LM probe: `{pname}`, {pr['n_params']/1e6:.0f}M "
          f"params, local window {pr['window'] or 'full'}, validation **{pr['bpb']:.4f} bits/byte** "
          f"on this corpus.\n")

    # ---------------------------------------------------------------- main table
    print("### Main table\n")
    hdr = ("| Segmentation | Bytes/unit | S_boundary | S_internal | Gap ΔS | ΔS vs random "
           "| ΔS(+1) vs random | ΔS non-ws | R p50 | R p90 | R p99 | Chg/kB | D95 |")
    print(hdr)
    print("|" + "---|" * 13)
    for arm in arms:
        d = S["arms"][arm]
        g0 = d["dS_gap_vs_random"]["0"]
        g1 = d["dS_gap_vs_random"]["1"]
        row = [NAME[arm], f(d["bytes_per_unit"]), f(d["S_boundary"]), f(d["S_internal"]),
               f(d["dS"], sign=True), f(g0, sign=True), f(g1, sign=True),
               f(d.get("dS_nws"), sign=True)]
        if P and arm in P["arms"]:
            fams = P["arms"][arm]
            d95 = max(v["recovery_distance"]["0.95"] for v in fams.values())
            b = boot.get(arm)
            r, chg = b["point"], b["point"]["chg_per_kb"]
            row += [f(r["p50"], 1), f(r["p90"], 1),
                    f"{r['p99']:.1f} [{b['p99'][0]:.0f}, {b['p99'][1]:.0f}]",
                    f"{chg:.3f} [{b['chg_per_kb'][0]:.3f}, {b['chg_per_kb'][1]:.3f}]",
                    str(d95)]
        else:
            row += ["--"] * 5
        print("| " + " | ".join(row) + " |")
    if P and "fixed5" in P["arms"]:
        fams = P["arms"]["fixed5"]
        d95 = max(v["recovery_distance"]["0.95"] for v in fams.values())
        r = boot["fixed5"]["point"]
        print("| " + " | ".join([NAME["fixed5"], "5.000"] + ["--"] * 6 +
                                [f(r["p50"], 1), f(r["p90"], 1), f(r["p99"], 1),
                                 f(r["chg_per_kb"], 3), str(d95)]) + " |")

    # ---------------------------------------------------------------- experiment A
    print("\n### Experiment A -- within-unit predictability\n")
    print("| Segmentation | ΔS [95% CI] | z vs random | ΔS fixed-width | C(u) unit | C(u) byte "
          "| C L=2 | C 3-4 | C 5-8 | C 9-16 | C >16 |")
    print("|" + "---|" * 11)
    for arm in arms:
        d = S["arms"][arm]
        ci = d["ci"]["dS"]
        print("| " + " | ".join([
            NAME[arm], f"{d['dS']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]",
            f"{d['control_random']['dS_z']:+.1f}", f(d["control_fixed"]["dS"], sign=True),
            f(d["C_unit"], sign=True), f(d["C_byte"], sign=True),
            f(d["C_L2"], 2, True), f(d["C_L34"], 2, True), f(d["C_L58"], 2, True),
            f(d["C_L916"], 2, True), f(d["C_L17p"], 2, True)]) + " |")

    # ---------------------------------------------------------------- experiment B
    print("\n### Experiment B -- boundaries as information transitions\n")
    print("| Segmentation | peak k=2 | k=4 | k=8 | ΔD k=4 | ΔD k=8 | AUROC | AUPRC | P@rate "
          "| AUROC(+1) | AUPRC(+1) | P@rate(+1) | base rate |")
    print("|" + "---|" * 13)
    for arm in arms:
        d = S["arms"][arm]
        r0, r1 = d.get("retrieval"), d.get("retrieval_shift1")
        cells = [NAME[arm], f(d["peak_k2"], sign=True), f(d["peak_k4"], sign=True),
                 f(d["peak_k8"], sign=True), f(d["dD_k4"], sign=True), f(d["dD_k8"], sign=True)]
        for r in (r0, r1):
            cells += ([f(r["auroc"]), f(r["auprc"]), f(r["prec_at_rate"])] if r else ["--"] * 3)
        cells.append(f(r0["base_rate"]) if r0 else "--")
        print("| " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- experiment D
    if P:
        print("\n### Experiment D -- perturbation locality, by family\n")
        fams = sorted(next(iter(P["arms"].values())).keys())
        print("| Segmentation | " + " | ".join(fams) + " |")
        print("|" + "---|" * (len(fams) + 1))
        for arm in [x for x in ORDER + ["whitespace", "fixed5"] if x in P["arms"]]:
            cells = [f"{P['arms'][arm][fa]['radius']['p90']:.0f} / "
                     f"{P['arms'][arm][fa]['radius']['p99']:.0f}" for fa in fams]
            print(f"| {NAME[arm]} (R p90/p99) | " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- experiment F
    ds = RES / "downstream_stats.json"
    if ds.exists():
        D = json.load(open(ds))
        print("\n### Experiment F -- structural disruption vs loss degradation (100M trio)\n")
        print("| Model | n | ΔBPB tail [95% CI] | mean Δboundaries | ρ(Δboundaries, ΔBPB) "
              "| ρ(radius, ΔBPB) | region re-tok. |")
        print("|" + "---|" * 7)
        for arm in ORDER:
            if arm not in D["arms"]:
                continue
            d = D["arms"][arm]
            print("| " + " | ".join([
                NAME[arm], str(d["n"]),
                f"{d['mean_delta_bpb']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]",
                f(d["mean_changed_downstream"], 2),
                f"{d['spearman_nchg'][0]:+.3f} (p={d['spearman_nchg'][1]:.2g})",
                f"{d['spearman_radius'][0]:+.3f} (p={d['spearman_radius'][1]:.2g})",
                f"{d['region_token_mismatch']*100:.1f}%"]) + " |")

    # ---------------------------------------------------------------- experiment C
    cp = RES / f"classes_{a.domain}.json"
    if cp.exists():
        C = json.load(open(cp))
        print("\n### Experiment C -- what becomes a boundary (P(class|B) / enrichment)\n")
        ca = [x for x in ORDER + ["whitespace"] if x in C["arms"]]
        print("| Class | corpus | " + " | ".join(NAME[x] for x in ca) + " |")
        print("|" + "---|" * (len(ca) + 2))
        for i, cls in enumerate(C["classes"]):
            if C["corpus_freq"][i] < 1e-6:
                continue
            cells = [f"{C['arms'][x]['p_given_boundary'][i]*100:.1f}% "
                     f"({C['arms'][x]['enrichment'][i]:.2f}x)" for x in ca]
            print(f"| {cls} | {C['corpus_freq'][i]*100:.1f}% | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
