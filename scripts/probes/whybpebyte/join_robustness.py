#!/usr/bin/env python3
"""Does the trunk-input representation similarity explain the measured robustness drops?

The boundary-level version of Experiment F came out null (results/downstream_stats.json:
rho ~ 0.02-0.04). This is the same question asked at the representation level and at the level
of a whole corruption FAMILY rather than a single edit: for each (model, family) pair, does a
lower centred cosine of what the trunk receives go with a larger downstream accuracy drop?

Scale caveat, stated in the output: the representation numbers are the 100M trio
(runs/others/superseded/cmp_100M_ratio20), the accuracy drops are the 1.3B trio from
reports/robust_aligned/ -- and that BPEByte row is the `bt` variant, not root_greedy. The join is
therefore about the ORDERING of families and models, not about matched checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"
ROB = ROOT / "reports/robust_aligned"

ARMS = {"llama": "llama_1.8B.json", "aunet": "aunet2_1.3B.json",
        "bpebyte": "bpebyte_br_bt_1.3B.json"}
NAME = {"llama": "Llama (subword)", "aunet": "AU-Net", "bpebyte": "BPEByte"}
# E2 family -> the aligned-robustness variant measuring the same corruption
VARIANT = {
    "despace": "despace",
    "noise_antspeak": "noise_antspeak_prompt",
    "noise_drop": "noise_drop_prompt",
    "noise_randomcase": "noise_randomcase_prompt",
    "noise_repeat": "noise_repeat_prompt",
    "noise_uppercase": "noise_uppercase_prompt",
    "typo_delete": "typo_delete_char",
    "typo_insert": "typo_insert_char",
    "typo_key": "typo_key_char",
    "typo_swap": "typo_swap_char",
}


def main():
    rows = []
    for arm, f in ARMS.items():
        p = RES / f"repr_suite_{arm}.jsonl"
        rp = ROB / f
        if not (p.exists() and rp.exists()):
            print(f"  (skipping {arm}: missing inputs)")
            continue
        rr = [json.loads(l) for l in open(p)]
        macro = json.load(open(rp))["macro_avg"]
        for fam, var in VARIANT.items():
            sel = [x for x in rr if x["family"] == fam]
            if not sel or var not in macro:
                continue
            rows.append({
                "arm": arm, "family": fam,
                "cos": float(np.nanmean([x.get("cos_centered_mean", np.nan) for x in sel])),
                "lev": float(np.mean([x["unit_levenshtein_norm"] for x in sel])),
                "units_x": float(np.sum([x["n_units_pert"] for x in sel]) /
                                 max(np.sum([x["n_units_clean"] for x in sel]), 1)),
                "delta_acc": 100 * float(macro[var]["delta_acc_norm"]),
            })
    if not rows:
        return
    x = np.array([r["cos"] for r in rows])
    lev = np.array([r["lev"] for r in rows])
    ux = np.array([r["units_x"] for r in rows])
    y = np.array([r["delta_acc"] for r in rows])

    print("### Experiment F2 — representation similarity vs measured accuracy drop\n")
    print("Representations: 100M trio. Accuracy drops: 1.3B trio, `reports/robust_aligned/` "
          "(macro over 5 MC tasks, acc_norm, points). Cross-scale, so this is about ordering.\n")
    print("| Model | Family | centred cos | Lev/unit | units× | Δacc (pp) |")
    print("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["arm"], r["family"])):
        print(f"| {NAME[r['arm']]} | {r['family']} | {r['cos']:.3f} | {r['lev']:.3f} | "
              f"{r['units_x']:.2f}x | {r['delta_acc']:+.2f} |")
    print()
    print("| Predictor | Spearman ρ with Δacc | p | n |")
    print("|---|---|---|---|")
    for nm, v in (("centred cosine", x), ("Levenshtein/unit", -lev), ("units× (re-segmentation)", -ux)):
        s = stats.spearmanr(v, y)
        print(f"| {nm} | {s.statistic:+.3f} | {s.pvalue:.2g} | {len(y)} |")
    print()
    # antspeak is an outlier for every model (cos ~0.1, ~-16 pp), so check the rank correlation
    # survives without it; and remove the model-level offsets, so what is left is purely the
    # family-to-family ordering WITHIN each model.
    keep = np.array([r["family"] != "noise_antspeak" for r in rows])
    s = stats.spearmanr(x[keep], y[keep])
    print(f"Without the `noise_antspeak` outlier: rho = {s.statistic:+.3f} (p={s.pvalue:.2g}, "
          f"n={keep.sum()}).")
    xc, yc = x.copy(), y.copy()
    for arm in ARMS:
        sel = np.array([r["arm"] == arm for r in rows])
        if sel.any():
            xc[sel] -= xc[sel].mean()
            yc[sel] -= yc[sel].mean()
    s2 = stats.spearmanr(xc, yc)
    print(f"Model-demeaned (family ordering only): rho = {s2.statistic:+.3f} "
          f"(p={s2.pvalue:.2g}, n={len(yc)}).\n")
    print("| Model | ρ(cos, Δacc) across families | n |")
    print("|---|---|---|")
    for arm in ARMS:
        sel = [i for i, r in enumerate(rows) if r["arm"] == arm]
        if len(sel) > 3:
            s = stats.spearmanr(x[sel], y[sel])
            print(f"| {NAME[arm]} | {s.statistic:+.3f} (p={s.pvalue:.2g}) | {len(sel)} |")
    json.dump(rows, open(RES / "robustness_join.json", "w"), indent=1)


if __name__ == "__main__":
    main()
