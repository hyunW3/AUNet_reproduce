#!/usr/bin/env python3
"""Figures A-E of the structured-locality suite (design section 13).

One figure per file, as the design asks. Colours follow the repo's Okabe-Ito convention
(viz_tradeoff.py, plot_compression_stability.py); the three-arm palette passes the dataviz
validator on every check (worst adjacent CVD dE 11.4 protan, normal-vision dE 24.2). Controls are
drawn as recessive grey references rather than as a fourth hue: they are baselines, not
competitors, and keeping them grey also keeps the categorical palette at three.

    python plots.py --domain en --probe blt_entropy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"
FIG = ROOT / "reports_afterAAAIsub/whybpebyte/figures"

COL = {"llama": "#0072B2", "aunet": "#E69F00", "bpebyte": "#009E73"}
MARK = {"llama": "o", "aunet": "s", "bpebyte": "^"}
NAME = {"llama": "Llama (subword)", "aunet": "AU-Net", "bpebyte": "BPEByte"}
ORDER = ["llama", "aunet", "bpebyte"]
GREY = "#7a7a7a"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG}/{name}.pdf")


# --------------------------------------------------------------------------- A
def fig_boundary_profile(S, tag):
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    off = S["arms"][ORDER[0]]["boundary_profile_offsets"]
    for arm in ORDER:
        d = S["arms"][arm]
        ax.plot(off, d["boundary_profile"], color=COL[arm], marker=MARK[arm], ms=3.5,
                lw=1.6, label=NAME[arm])
        ax.plot(off, d["control_random"]["boundary_profile_mean"], color=COL[arm],
                lw=1.0, ls=(0, (3, 2)), alpha=0.45)
    ax.axvline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("byte offset from unit start  (0 = first byte of the unit)")
    ax.set_ylabel("probe surprisal (bits/byte)")
    ax.plot([], [], color=GREY, lw=1.0, ls=(0, (3, 2)), label="length-matched random")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    pk = int(np.argmax(S["arms"]["bpebyte"]["boundary_profile"])) + off[0]
    ax.axvline(pk, color=GREY, lw=0.8, ls="--", alpha=0.7)
    ax.text(pk + 0.3, ax.get_ylim()[0] + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"peak {pk:+d}", fontsize=7.5, color="#333333")
    save(fig, f"figA_boundary_profile_{tag}")


# --------------------------------------------------------------------------- B
def fig_gap(S, tag):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    ks = sorted((int(k) for k in S["arms"][ORDER[0]]["dS_gap_vs_random"]))
    for arm in ORDER:
        g = S["arms"][arm]["dS_gap_vs_random"]
        a1.plot(ks, [g[str(k)] for k in ks], color=COL[arm], marker=MARK[arm], ms=3.5,
                lw=1.6, label=NAME[arm])
    a1.axhline(0, color=GREY, lw=0.8)
    a1.axvline(0, color=GREY, lw=0.8, ls=":")
    a1.set_xlabel("offset $\\delta$ from unit start")
    a1.set_ylabel("$\\Delta S(\\delta)$ over length-matched random (bits)")
    a1.legend(frameon=False, fontsize=7.5)

    x = np.arange(len(ORDER))
    for j, (key, lbl, hatch) in enumerate([("0", "$\\delta=0$ (as specified)", None),
                                           ("1", "$\\delta=+1$ (placement-corrected)", "//")]):
        vals = [S["arms"][arm]["dS_gap_vs_random"][key] for arm in ORDER]
        a2.bar(x + (j - 0.5) * 0.38, vals, 0.34, label=lbl, hatch=hatch,
               color=[COL[arm] for arm in ORDER], edgecolor="white", linewidth=1.0,
               alpha=1.0 if j == 0 else 0.55)
        for xi, v in zip(x + (j - 0.5) * 0.38, vals):
            a2.text(xi, v + (0.04 if v >= 0 else -0.12), f"{v:+.2f}", ha="center",
                    fontsize=6.8, color="#333333")
    a2.axhline(0, color=GREY, lw=0.8)
    a2.set_xticks(x, [NAME[a].split(" ")[0] for a in ORDER], fontsize=8)
    a2.set_ylabel("gap vs random (bits)")
    a2.legend(frameon=False, fontsize=7, loc="lower center", ncol=1,
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    save(fig, f"figB_gap_vs_random_{tag}")


# --------------------------------------------------------------------------- within-unit
def fig_within_unit(S, tag):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for arm in ORDER:
        y = S["arms"][arm]["within_unit_abs"]
        ax.plot(range(len(y)), y, color=COL[arm], marker=MARK[arm], ms=3.5, lw=1.6,
                label=NAME[arm])
    ax.set_xlabel("byte offset inside the unit")
    ax.set_ylabel("probe surprisal (bits/byte)")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, f"figA2_within_unit_{tag}")


# --------------------------------------------------------------------------- C
def fig_agreement(Pj, tag):
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for arm in ORDER:
        fams = Pj["arms"][arm]
        agr = np.zeros(65); tot = np.zeros(65)
        for f, v in fams.items():
            a = np.array(v["agreement_by_distance"])
            agr += a * v["n_samples"]; tot += v["n_samples"]
        ax.plot(np.arange(1, 65), (agr / tot)[1:], color=COL[arm], lw=1.7, label=NAME[arm])
    if "fixed5" in Pj["arms"]:
        fams = Pj["arms"]["fixed5"]
        agr = np.zeros(65); tot = np.zeros(65)
        for f, v in fams.items():
            agr += np.array(v["agreement_by_distance"]) * v["n_samples"]; tot += v["n_samples"]
        ax.plot(np.arange(1, 65), (agr / tot)[1:], color=GREY, lw=1.3, ls=(0, (4, 1.5, 1, 1.5)),
                marker="D", ms=3.5, markevery=[0, 3, 7, 15, 31, 62], label="fixed-width control")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("distance from the edited byte")
    ax.set_ylabel("boundary agreement with clean input")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.95, color="#c8c8c8", lw=0.6)
    ax.text(64, 0.962, "0.95", fontsize=6.8, color="#9a9a9a", ha="right")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, f"figC_agreement_{tag}")


# --------------------------------------------------------------------------- D
def fig_radius_cdf(Pj, tag, jsonl):
    import collections
    per = collections.defaultdict(list)
    for line in open(jsonl):
        r = json.loads(line)
        per[r["arm"]].append(r["radius"])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for arm in ORDER:
        v = np.sort(np.array(per[arm]))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COL[arm],
                lw=1.7, label=NAME[arm])
    if "fixed5" in per:
        v = np.sort(np.array(per["fixed5"]))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=GREY, lw=1.2,
                ls=(0, (1, 1.5)), label="fixed-width control")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("disturbance radius $R$ (bytes)")
    ax.set_ylabel("fraction of single edits $\\leq R$")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, f"figD_radius_cdf_{tag}")


# --------------------------------------------------------------------------- E
def fig_downstream(tag):
    # Experiment F ran on the English span set only, so this figure exists once, not per domain.
    if tag != "en":
        return
    files = {arm: RES / f"downstream_{arm}.jsonl" for arm in ORDER}
    if not all(p.exists() for p in files.values()):
        print("  (skipping figure E: downstream rows not ready)")
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for arm in ORDER:
        rows = [json.loads(l) for l in open(files[arm])]
        x = np.array([r["n_changed_downstream"] for r in rows], float)
        y = np.array([r["delta_bpb"] for r in rows], float)
        xs = np.unique(x)
        xs = xs[xs <= 6]
        mu = [y[x == v].mean() for v in xs]
        se = [y[x == v].std(ddof=1) / max(np.sqrt((x == v).sum()), 1) for v in xs]
        ax.errorbar(xs, mu, yerr=1.96 * np.array(se), color=COL[arm], marker=MARK[arm],
                    ms=4, lw=1.6, capsize=2, label=NAME[arm])
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("boundaries changed downstream of the edit")
    ax.set_ylabel("$\\Delta$ BPB on the unedited tail (bits/byte)")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, f"figE_downstream_{tag}")


# --------------------------------------------------------------------------- F, G (exp E)
def fig_repr_single(tag):
    if tag != "en":                 # Experiment E ran on the English span set only
        return
    curves = {a: RES / f"repr_single_{a}.curve.npy" for a in ORDER}
    if not all(p.exists() for p in curves.values()):
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    d = np.arange(1, 65)
    for arm in ORDER:
        sim, cnt = np.load(curves[arm])
        ax.plot(d, sim[1:65], color=COL[arm], lw=1.7, label=NAME[arm])
    dp = RES / "repr_single_llama.curve_depth.npy"
    if dp.exists():
        sim, _ = np.load(dp)
        ax.plot(d, sim[1:65], color=COL["llama"], lw=1.3, ls=(0, (3, 2)),
                label="Llama, after 3 layers")
    ax.set_xscale("log", base=2)
    ax.axhline(1.0, color="#c8c8c8", lw=0.6)
    ax.set_xlabel("distance from the edited byte")
    ax.set_ylabel("centred cosine, clean vs perturbed\ntrunk-input vector")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, f"figF_repr_single_{tag}")


def fig_repr_suite(tag):
    if tag != "en":
        return
    import json as _json
    files = {a: RES / f"repr_suite_{a}.jsonl" for a in ORDER}
    if not all(p.exists() for p in files.values()):
        return
    rows = {a: [_json.loads(l) for l in open(files[a])] for a in ORDER}
    fams = sorted({r["family"] for rs in rows.values() for r in rs})
    order = sorted(fams, key=lambda f: -np.mean(
        [r["unit_levenshtein_norm"] for r in rows["llama"] if r["family"] == f] or [0]))
    y = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 0.34 * len(order) + 1.5), sharey=True)
    h = 0.26
    for j, arm in enumerate(ORDER):
        lev = [np.mean([r["unit_levenshtein_norm"] for r in rows[arm] if r["family"] == f] or [np.nan])
               for f in order]
        cos = [np.nanmean([r.get("cos_centered_mean", np.nan) for r in rows[arm] if r["family"] == f]
                          or [np.nan]) for f in order]
        a1.barh(y + (1 - j) * h, lev, h * 0.86, color=COL[arm], label=NAME[arm],
                edgecolor="white", linewidth=0.8)
        a2.barh(y + (1 - j) * h, cos, h * 0.86, color=COL[arm], edgecolor="white", linewidth=0.8)
    a1.set_yticks(y, order, fontsize=7.5)
    a1.invert_yaxis()
    a1.set_xlabel("unit-sequence Levenshtein / clean unit")
    a2.set_xlabel("centred cosine of the trunk input")
    a2.set_xlim(0, 1.02)
    a1.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, f"figG_repr_suite_{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="en")
    ap.add_argument("--probe", default="blt_entropy")
    a = ap.parse_args()
    tag = a.domain
    sp = RES / f"structural_{a.probe}_{a.domain}.json"
    if sp.exists():
        S = json.load(open(sp))
        fig_boundary_profile(S, tag)
        fig_gap(S, tag)
        fig_within_unit(S, tag)
    pp = RES / f"perturb_{a.domain}.json"
    if pp.exists():
        Pj = json.load(open(pp))
        fig_agreement(Pj, tag)
        fig_radius_cdf(Pj, tag, RES / f"perturb_{a.domain}.jsonl")
    fig_downstream(tag)
    fig_repr_single(tag)
    fig_repr_suite(tag)


if __name__ == "__main__":
    main()
