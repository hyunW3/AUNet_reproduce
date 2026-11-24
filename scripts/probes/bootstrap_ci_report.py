#!/usr/bin/env python3
"""Bootstrap 95% CI report for every results table in the paper.

For each accuracy cell we run a percentile bootstrap (B=10,000) of the item-level
Bernoulli process: acc_b ~ Binomial(n, p)/n (equivalent to an item-level resample
when only (p, n) survives aggregation). Aggregate cells (avg3/avgall, CUTE mean,
S-NIAH means, MK mean, macro-averages) bootstrap each component task and average
per replicate. Pairwise model differences use independent replicates (no per-item
pairing data is persisted for most suites), which is CONSERVATIVE: paired designs
on shared items would give narrower CIs, so "significant" here is trustworthy and
"tie" may understate real gaps. BPB is excluded (not item-resamplable; the
moving-block bootstrap in reports/bpb_ci_1.3B.md gives +/-0.001 within-run, with
a +/-0.005 run-to-run seed floor). Latency (tab:throughput) and the qualitative
design table are excluded.

Output: reports/paper_tables_bootstrap_ci.md
"""
import random
from pathlib import Path

B = 10_000
rng = random.Random(0)
MODELS = ["Llama", "AU-Net", "BPEByte"]


def _binom(p, n):
    # inverse-CDF-free binomial draw via normal approx for big n, exact for small
    if n >= 400:
        import math
        z = rng.gauss(0, 1)
        return min(max((p + z * math.sqrt(p * (1 - p) / n)), 0.0), 1.0)
    return sum(rng.random() < p for _ in range(n)) / n


def ci(reps):
    s = sorted(reps)
    return s[int(0.025 * len(s))], s[int(0.975 * len(s))]


def cell(p, n):
    return [_binom(p, n) for _ in range(B)]


def avg_cells(cells):
    return [sum(c[b] for c in cells) / len(cells) for b in range(B)]


def fmt(p, lo, hi, pct=True):
    if pct:
        return f"{p*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def verdicts(reps3):
    out = []
    for (i, j, lbl) in [(0, 1, "L-A"), (0, 2, "L-B"), (1, 2, "A-B")]:
        d = sorted(reps3[i][b] - reps3[j][b] for b in range(B))
        lo, hi = d[int(0.025 * B)], d[int(0.975 * B)]
        if lo > 0:
            out.append(f"{lbl}: {MODELS[i][0] if False else MODELS[i]}>{MODELS[j]}")
        elif hi < 0:
            out.append(f"{lbl}: {MODELS[j]}>{MODELS[i]}")
        else:
            out.append(f"{lbl}: tie")
    return "; ".join(v.split(": ")[1] for v in out)


def table(title, header_note, rows, pct=True):
    """rows: list of (metric, [(p,n) x3]) -> markdown block."""
    lines = [f"### {title}", "", header_note, "",
             "| Metric | Llama | AU-Net | BPEByte | Pairwise (95%) |", "|---|---|---|---|---|"]
    for metric, cells3 in rows:
        reps3 = []
        cols = []
        for spec in cells3:
            if isinstance(spec, list):                # aggregate: list of (p, n) components
                comp = [cell(p, n) for p, n in spec]
                reps = avg_cells(comp)
                pt = sum(p for p, _ in spec) / len(spec)
            else:
                p, n = spec
                reps = cell(p, n)
                pt = p
            reps3.append(reps)
            lo, hi = ci(reps)
            cols.append(fmt(pt, lo, hi, pct))
        lines.append(f"| {metric} | {cols[0]} | {cols[1]} | {cols[2]} | {verdicts(reps3)} |")
    lines.append("")
    return "\n".join(lines)


N = dict(hs=10042, arce=2376, arcc=1172, piqa=1838, boolq=3270, wg=1267, mmlu=14042)
out = ["# Bootstrap 95% CIs for all paper tables",
       "",
       "Percentile bootstrap, B=10,000, of the item-level Bernoulli process per cell; "
       "aggregates bootstrap components jointly. Pairwise differences use independent "
       "replicates (conservative -- no per-item pairing is persisted). "
       "'X>Y' means the 95% CI of the difference excludes 0.",
       ""]

# ---- tab:main_13b ----------------------------------------------------------
zs = {  # zero-shot: task -> (llama, aunet, bpebyte)
    "HellaSwag": (.622, .626, .625), "ARC-Easy": (.655, .657, .668),
    "ARC-Challenge": (.353, .365, .375), "PIQA": (.753, .742, .743),
    "BoolQ": (.635, .611, .620), "WinoGrande": (.616, .615, .611),
    "MMLU-text": (.319, .322, .323)}
ns = [N["hs"], N["arce"], N["arcc"], N["piqa"], N["boolq"], N["wg"], N["mmlu"]]
rows = [(t, [(v[m], n) for m in range(3)]) for (t, v), n in zip(zs.items(), ns)]
tasks = list(zs.values())
rows.append(("avg3 (0-shot)", [[(tasks[i][m], ns[i]) for i in (0, 1, 3)] for m in range(3)]))
rows.append(("avgall (0-shot)", [[(tasks[i][m], ns[i]) for i in range(7)] for m in range(3)]))
fs = {"HellaSwag": (.629, .637, .637), "ARC-Easy": (.702, .724, .715), "PIQA": (.757, .741, .743)}
rows.append(("avg3 (5-shot)", [[(fs[t][m], n) for t, n in
                                [("HellaSwag", N["hs"]), ("ARC-Easy", N["arce"]), ("PIQA", N["piqa"])]]
                               for m in range(3)]))
out.append(table("tab:main_13b -- consolidated 1.3B (downstream)",
                 "Zero-shot per task; aggregates bootstrap all component tasks. Five-shot avgall "
                 "omitted (per-task 5-shot accs for BoolQ/WG/MMLU not persisted); BPB excluded "
                 "(seed floor +/-0.005, within-run +/-0.001); robustness deltas covered under "
                 "tab:pbp / tab:noise_typo below.", rows))

# ---- tab:bpb_scale ---------------------------------------------------------
scale_data = {
 "100M": {"HellaSwag": (.332, .318, .307), "ARC-Easy": (.451, .360, .346),
          "ARC-Challenge": (.238, .244, .235), "PIQA": (.640, .591, .594),
          "BoolQ": (.549, .548, .391), "WinoGrande": (.506, .496, .502)},
 "300M": {"HellaSwag": (.393, .323, .315), "ARC-Easy": (.480, .371, .347),
          "ARC-Challenge": (.247, .238, .239), "PIQA": (.674, .610, .617),
          "BoolQ": (.592, .497, .387), "WinoGrande": (.522, .523, .516)},
 "760M": {"HellaSwag": (.558, .524, .522), "ARC-Easy": (.620, .559, .556),
          "ARC-Challenge": (.315, .317, .315), "PIQA": (.737, .711, .712),
          "BoolQ": (.584, .499, .455), "WinoGrande": (.540, .556, .546)},
}
ns6 = [N["hs"], N["arce"], N["arcc"], N["piqa"], N["boolq"], N["wg"]]
for sc, dat in scale_data.items():
    rows = [(t, [(v[m], n) for m in range(3)]) for (t, v), n in zip(dat.items(), ns6)]
    tasks = list(dat.values())
    rows.append(("avg3", [[(tasks[i][m], ns6[i]) for i in (0, 1, 3)] for m in range(3)]))
    rows.append(("avgall (6 tasks)", [[(tasks[i][m], ns6[i]) for i in range(6)] for m in range(3)]))
    out.append(table(f"tab:bpb_scale -- {sc}",
                     "Zero-shot accuracy per task (1.3B block identical to tab:main_13b above).", rows))

# ---- tab:pbp ---------------------------------------------------------------
out.append("\n".join([
    "### tab:pbp -- prompt-boundary perturbation (Llama canonical vs perturbed)",
    "",
    "Byte models are invariant by construction (identical bytes; Delta==0 exactly), so CIs apply to "
    "Llama only. Unpaired bootstrap of the accuracy difference (conservative; the true paired CI is narrower):",
    "",
    "| Task | canonical | perturbed | Delta [95% CI] | verdict |", "|---|---|---|---|---|"] + [
    (lambda t, p, d, n: (lambda a, b: f"| {t} | {p*100:.1f} | {(p+d)*100:.1f} | {d*100:+.1f} "
     f"[{a*100:+.1f}, {b*100:+.1f}] | {'significant' if (a > 0 or b < 0) else 'tie'} |")(
        *ci([_binom(p + d, n) - _binom(p, n) for _ in range(B)])))(t, p, d, n)
    for t, p, d, n in [("HellaSwag", .622, -.0093, 2000), ("ARC-Easy", .655, -.2146, 2000),
                       ("ARC-Challenge", .353, -.0316, 1172),
                       ("PBP-mc macro", (.622 + .655 + .353) / 3, -.0852, 1724)]] + [""]))

# ---- tab:noise_typo --------------------------------------------------------
rows = [("HellaSwag-Noise", [(.376, 2000), (.415, 2000), (.423, 2000)]),
        ("ARC-Easy-Noise", [(.377, 2376), (.435, 2376), (.447, 2376)]),
        ("HellaSwag-Typo", [(.497, 2000), (.521, 2000), (.519, 2000)]),
        ("ARC-C-Typo", [(.308, 1172), (.318, 1172), (.335, 1172)])]
out.append(table("tab:noise_typo -- noise / typo robustness (perturbed accuracy)",
                 "n = underlying item set (HS subsets 2000, ARC-E 2376, ARC-C 1172). Each row "
                 "averages 5 noise strategies x prompt/completion/both on the SAME items, so "
                 "item-sampling n does not multiply; treating the mean as one Bernoulli(n) is "
                 "the defensible width.", rows))

# ---- tab:state_tracking ----------------------------------------------------
rows = [("FFLM Dense (binary)", [(.811, 3200), (.850, 3200), (.859, 3200)]),
        ("FFLM In-Dist (binary)", [(.768, 3195), (.715, 3195), (.765, 3195)]),
        ("FFLM Sparse (binary)", [(.802, 7126), (.666, 7126), (.866, 7126)]),
        ("Dyck-3 (exact)", [(.520, 6000), (.540, 6000), (.673, 6000)])]
out.append(table("tab:state_tracking -- FFLM + Dyck-3",
                 "n = scored reads (FFLM dense/indist 3200/3195, sparse 7126; Dyck 6000). Reads "
                 "cluster within sequences, so true CIs are somewhat wider than these iid widths.", rows))

# ---- tab:fflm_binary -------------------------------------------------------
rows = [("Dense exact", [(.823, 3200), (.853, 3200), (.865, 3200)]),
        ("Dense binary", [(.811, 3200), (.850, 3200), (.859, 3200)]),
        ("In-Dist exact", [(.783, 3195), (.724, 3195), (.771, 3195)]),
        ("In-Dist binary", [(.768, 3195), (.715, 3195), (.765, 3195)]),
        ("Sparse exact", [(.812, 7126), (.672, 7126), (.872, 7126)]),
        ("Sparse binary", [(.802, 7126), (.666, 7126), (.866, 7126)])]
out.append(table("tab:fflm_binary -- exact-match vs binary-rank (appendix)",
                 "Same reads as tab:state_tracking; metric agreement within 0.016 everywhere.", rows))

# ---- tab:recall_query ------------------------------------------------------
rows = [("S-NIAH-1 mean", [(1.00, 250), (.996, 250), (1.00, 250)]),
        ("S-NIAH-2 mean", [(1.00, 250), (.984, 250), (.996, 250)]),
        ("S-NIAH-3 mean", [(.940, 250), (.572, 250), (.948, 250)]),
        ("S-NIAH-word mean", [(.990, 250), (.990, 250), (.980, 250)]),
        ("MK-NIAH K=1", [(.990, 200), (.985, 200), (1.00, 200)]),
        ("MK-NIAH K=2", [(.905, 200), (.660, 200), (.845, 200)]),
        ("MK-NIAH K=4", [(.800, 200), (.515, 200), (.765, 200)]),
        ("MK-NIAH K=8", [(.700, 200), (.440, 200), (.605, 200)]),
        ("MK-NIAH mean", [[(p, 200) for p in (.990, .905, .800, .700)],
                          [(p, 200) for p in (.985, .660, .515, .440)],
                          [(p, 200) for p in (1.00, .845, .765, .605)]])]
out.append(table("tab:recall_query -- exact recall (boundary-consistent)",
                 "S-NIAH means over 5 context lengths (n=250); MK-NIAH n=200 per K.", rows))

# ---- tab:numbers -----------------------------------------------------------
rows = [("Magnitude comparison", [(.52, 2000), (.56, 2000), (.69, 2000)]),
        ("Copy fidelity", [(.74, 1900), (1.00, 1900), (1.00, 1900)])]
out.append(table("tab:numbers -- numerical-value probes",
                 "n = 100 items per digit length x 20 lengths (comparison) / 19 (copy).", rows))

# ---- tab:cute --------------------------------------------------------------
rows = [("Spelling", [(.186, 2000), (.884, 2000), (.793, 2000)]),
        ("Spelling Inverse", [(.112, 2000), (.485, 2000), (.456, 2000)]),
        ("Contains Char", [(.534, 2000), (.508, 2000), (.494, 2000)]),
        ("Contains Word", [(.470, 2000), (.496, 2000), (.503, 2000)]),
        ("Orthographic sim.", [(.473, 2000), (.370, 2000), (.388, 2000)]),
        ("Semantic sim.", [(.469, 2000), (.450, 2000), (.422, 2000)]),
        ("Manipulation (avg 4)", [(.041, 8000), (.019, 8000), (.021, 8000)]),
        ("CUTE average (14)", [[(.184, 2000)] * 14, [(.239, 2000)] * 14, [(.230, 2000)] * 14])]
out.append(table("tab:cute -- character-level understanding",
                 "n=2000/subtask (manipulation pools 4 subtasks; average approximated as 14 "
                 "equal-n tasks at the mean accuracy -- CI width is what matters).", rows))

# ---- tab:ext_retrieval -----------------------------------------------------
rows = [("MV recall", [(.588, 150), (.538, 150), (.530, 150)]),
        ("MQ recall", [(.307, 150), (.295, 150), (.243, 150)]),
        ("Sequential-NIAH recall", [(.310, 150), (.355, 150), (.198, 150)]),
        ("NoLiMa literal", [(.807, 150), (.680, 150), (.953, 150)]),
        ("NoLiMa paraphrase", [(.507, 150), (.420, 150), (.573, 150)])]
out.append(table("tab:ext_retrieval -- extended retrieval",
                 "n=150 items (3 context lengths x 50). Recall rows score 4 values/item; "
                 "clustering by item makes n=150 the conservative unit.", rows))

Path("/mnt/ssd2/hyun2/AUNet/reports/paper_tables_bootstrap_ci.md").write_text("\n".join(out))
print("\n".join(out))
