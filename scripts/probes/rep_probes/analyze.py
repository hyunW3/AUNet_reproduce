#!/usr/bin/env python3
"""Summaries + paired-bootstrap CIs for the repetition probes A-D (all five models).

Inputs: reports/rep_probes/<P>/scored/<family>.jsonl (score_items.py output, merged from both hosts).
Metric: A, C (multiple choice) use acc_norm for HellaSwag/ARC/PIQA and acc for BoolQ (= the robustness
suite); B, D use greedy exact match. Repetition effects are difference-in-differences against a
same-length non-repetitive control on the SAME items, CI = 95% percentile of a paired item bootstrap.

  python analyze.py A|B|C|D [--B 2000]   -> reports/rep_probes/<P>/summary.md, summary.json
"""
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

L = Path("/mnt/ssd2/hyun2/AUNet/reports/rep_probes")
FAM = ("llama", "aunet", "bpebyte", "blt", "hnet")
NAME = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte", "blt": "BLT", "hnet": "H-Net"}


def load(probe):
    D = {}
    for f in FAM:
        p = L / probe / "scored" / f"{f}.jsonl"
        if p.exists():
            D[f] = [json.loads(l) for l in open(p)]
    return D


def score(r):
    if "exact" in r:
        return r["exact"]
    return r["acc"] if r["task"] == "boolq" else r["acc_norm"]


def boot_did(a, b, c, d, B, rng):
    """mean((a-b) - (c-d)) over paired items with CI; a,b,c,d aligned arrays."""
    x = (np.array(a) - np.array(b)) - (np.array(c) - np.array(d))
    n = len(x)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    return 100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def probe_A(D, B, rng):
    lines = ["# Probe A — repeated vs natural filler before the question", "",
             "Δ = accuracy change vs clean (pp), mean over 5 tasks; DiD = Δrep − Δnat [95% CI].", ""]
    out = {}
    for n in (256, 1024, 2048):
        lines += [f"## filler {n} B", "", "| model | clean | Δ rep | Δ nat | DiD (rep−nat) | BoolQ DiD |", "|---|---|---|---|---|---|"]
        for f, rows in D.items():
            by = defaultdict(dict)
            for r in rows:
                by[r["key"].rsplit("/", 1)[0]][r["cond"]] = (score(r), r["task"])
            keys = [k for k in by if all(c in by[k] for c in ("clean", f"rep{n}", f"nat{n}"))]
            g = lambda c, ks: [by[k][c][0] for k in ks]
            clean = 100 * st.mean(g("clean", keys))
            drep = 100 * (st.mean(g(f"rep{n}", keys)) - st.mean(g("clean", keys)))
            dnat = 100 * (st.mean(g(f"nat{n}", keys)) - st.mean(g("clean", keys)))
            did = boot_did(g(f"rep{n}", keys), g("clean", keys), g(f"nat{n}", keys), g("clean", keys), B, rng)
            bk = [k for k in keys if by[k]["clean"][1] == "boolq"]
            bdid = boot_did(g(f"rep{n}", bk), g("clean", bk), g(f"nat{n}", bk), g("clean", bk), B, rng)
            out[(f, n)] = {"clean": clean, "d_rep": drep, "d_nat": dnat, "did": did, "boolq_did": bdid}
            lines.append(f"| {NAME[f]} | {clean:.1f} | {drep:+.1f} | {dnat:+.1f} | {did[0]:+.1f} [{did[1]:+.1f}, {did[2]:+.1f}] | "
                         f"{bdid[0]:+.1f} [{bdid[1]:+.1f}, {bdid[2]:+.1f}] |")
        lines.append("")
    return lines, {f"{k[0]}|{k[1]}": v for k, v in out.items()}


def probe_B(D, B, rng):
    periods = ("P64", "P128", "P256", "P512", "P1024", "Pnat")
    lines = ["# Probe B — needle retrieval vs haystack repetition period", "",
             "Exact recall; DiD = recall(P) − recall(nat) on the same key/value/window/depth [95% CI].", ""]
    out = {}
    for length in (2048, 4096):
        lines += [f"## length {length}", "", "| model | " + " | ".join(periods) + " | P64 − nat |", "|---" * 8 + "|"]
        for f, rows in D.items():
            by = defaultdict(dict)
            for r in rows:
                if r["length"] == length:
                    by[r["key"].split("/", 1)[1]][r["task"]] = r["exact"]
            ks = [k for k in by if all(p in by[k] for p in periods)]
            acc = {p: st.mean(by[k][p] for k in ks) for p in periods}
            x = np.array([by[k]["P64"] - by[k]["Pnat"] for k in ks], float)
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
            d = (100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5))
            out[(f, length)] = {"acc": acc, "p64_minus_nat": d}
            lines.append(f"| {NAME[f]} | " + " | ".join(f"{acc[p]:.2f}" for p in periods) +
                         f" | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] |")
        lines.append("")
    return lines, {f"{k[0]}|{k[1]}": v for k, v in out.items()}


def probe_C(D, B, rng):
    lines = ["# Probe C — duplicated passage vs distinct passages before the evidence (BoolQ acc)", "",
             "DiD = (dup_k − clean) − (ctrl_k − clean) [95% CI].", ""]
    out = {}
    lines += ["| model | clean | dup2 | ctrl2 | dup4 | ctrl4 | DiD k=2 | DiD k=3 | DiD k=4 |", "|---" * 9 + "|"]
    for f, rows in D.items():
        by = defaultdict(dict)
        for r in rows:
            by[r["key"].split("/")[0]][r["cond"]] = score(r)
        ks = list(by)
        m = lambda c: 100 * st.mean(by[k][c] for k in ks)
        dids = {}
        for k_ in (2, 3, 4):
            dids[k_] = boot_did([by[k][f"dup{k_}"] for k in ks], [by[k]["clean"] for k in ks],
                                [by[k][f"ctrl{k_}"] for k in ks], [by[k]["clean"] for k in ks], B, rng)
        out[f] = {"acc": {c: m(c) for c in ("clean", "dup2", "ctrl2", "dup3", "ctrl3", "dup4", "ctrl4")}, "did": dids}
        lines.append(f"| {NAME[f]} | {m('clean'):.1f} | {m('dup2'):.1f} | {m('ctrl2'):.1f} | {m('dup4'):.1f} | {m('ctrl4'):.1f} | " +
                     " | ".join(f"{dids[k_][0]:+.1f} [{dids[k_][1]:+.1f}, {dids[k_][2]:+.1f}]" for k_ in (2, 3, 4)) + " |")
    return lines, out


def probe_D(D, B, rng):
    fmts = ("json", "csv", "log", "prose")
    lines = ["# Probe D — key-value retrieval from structured (template-repetitive) text vs prose", "",
             "Exact recall of the 7-digit code; diff = structured mean − prose on the same records [95% CI].", ""]
    out = {}
    for length in (1024, 2048, 4096):
        lines += [f"## length {length}", "", "| model | json | csv | log | prose | structured − prose |", "|---" * 6 + "|"]
        for f, rows in D.items():
            by = defaultdict(dict)
            for r in rows:
                if r["length"] == length:
                    by[r["key"].split("/", 1)[1]][r["fmt"]] = r["exact"]
            ks = [k for k in by if all(x in by[k] for x in fmts)]
            acc = {x: st.mean(by[k][x] for k in ks) for x in fmts}
            x = np.array([st.mean(by[k][y] for y in fmts[:3]) - by[k]["prose"] for k in ks])
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
            d = (100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5))
            out[(f, length)] = {"acc": acc, "struct_minus_prose": d}
            lines.append(f"| {NAME[f]} | " + " | ".join(f"{acc[y]:.2f}" for y in fmts) + f" | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] |")
        lines.append("")
    return lines, {f"{k[0]}|{k[1]}": v for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=list("ABCD"))
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    D = load(a.probe)
    lines, out = {"A": probe_A, "B": probe_B, "C": probe_C, "D": probe_D}[a.probe](D, a.B, rng)
    (L / a.probe / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(out, open(L / a.probe / "summary.json", "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
