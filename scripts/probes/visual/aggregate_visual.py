#!/usr/bin/env python3
"""Aggregate the visual-attack runs (eval_visual.py) into reports/visual_attacks/summary.{md,json}.

  python aggregate_visual.py <run_out_dir> [<report_dir>]

<run_out_dir> holds lingua dumps (<arm>/results.json) and run_visual.py jsons (blt_*.json merged
into one BLT row, hnet.json). Δ = perturbed - clean (pp) per item on the same doc ids, macro over
the 5 tasks; 95% CI by paired item bootstrap within each task (B=2000).
"""
import glob, json, os, sys

import numpy as np

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq"]
METRIC = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa": "acc_norm",
          "boolq": "acc"}
ATTACKS = ["viper", "badchars", "anthro"]
LABEL = {"viper": "VIPER-ICES", "badchars": "BadChars homoglyph", "anthro": "ANTHRO"}
PROBS = ["p10", "p50"]
TARGETS = ["prompt", "both"]
BUDGETS = [0, 1, 2, 3, 4, 5, 10]
MODELS = [("llama", "Transformer (BPE)"), ("aunet", "AU-Net"), ("bpebyte", "BPEByte"),
          ("blt", "BLT-1B"), ("hnet", "H-Net 1-stage XL")]
B = int(os.environ.get("BOOT", 2000))


def load(out):
    M = {}
    for arm in ("llama", "aunet", "bpebyte"):
        f = f"{out}/{arm}/results.json"
        if os.path.exists(f):
            d = json.load(open(f))
            S = {}
            for t, rows in d["samples"].items():
                rows = sorted(rows, key=lambda r: int(r["doc_id"]))
                S[t] = {"doc_id": [int(r["doc_id"]) for r in rows]}
                for m in ("acc", "acc_norm"):
                    if m in rows[0]:
                        S[t][m] = [float(r[m]) for r in rows]
            M[arm] = S
    for name, glb in (("blt", "blt_*.json"), ("hnet", "hnet.json")):
        S = {}
        for f in sorted(glob.glob(f"{out}/{glb}")):
            S.update(json.load(open(f))["samples"])
        if S:
            M[name] = S
    return M


def bits(S, task, metric):
    s = S.get(task)
    return None if s is None or metric not in s else dict(zip(s["doc_id"], s[metric]))


def macro_delta(S, variant_of, rng):
    """variant_of(task) -> perturbed task name (or None). Returns (mean, lo, hi, per_task, n_tasks)."""
    pairs = []
    per = {}
    for t in TASKS:
        v = variant_of(t)
        if v is None:
            continue
        c, p = bits(S, t, METRIC[t]), bits(S, v, METRIC[t])
        if c is None or p is None:
            continue
        ids = sorted(set(c) & set(p))
        d = [p[i] - c[i] for i in ids]
        pairs.append(d)
        per[t] = 100 * sum(d) / len(d)
    if not pairs:
        return None
    mean = sum(per.values()) / len(per)
    tot = 0.0
    for d in pairs:
        a = np.asarray(d, dtype=np.float32)
        tot = tot + a[rng.integers(0, len(a), size=(B, len(a)))].mean(1)
    boots = np.sort(100 * tot / len(pairs))
    return mean, float(boots[int(.025 * B)]), float(boots[int(.975 * B) - 1]), per, len(pairs)


def main():
    out = sys.argv[1]
    rep = sys.argv[2] if len(sys.argv) > 2 else "/mnt/ssd2/hyun2/AUNet/reports/visual_attacks"
    os.makedirs(rep, exist_ok=True)
    M = load(out)
    rng = np.random.default_rng(0)
    J = {"clean": {}, "visual": {}, "mnli": {}}
    L = ["# Visual-attack robustness (1.3B-class byte/BPE models)", "",
         "VIPER-ICES / Bad Characters homoglyphs / ANTHRO, random positions at rate p (char-level for VIPER and",
         "BadChars, word-level for ANTHRO), 0-shot, limit 1000 per task (ARC-C 1000/1172), perturbation seed 1234.",
         "Δ = perturbed − clean on the same items (pp), macro over HellaSwag / ARC-E / ARC-C / PIQA (acc_norm) and",
         "BoolQ (acc); [95% CI] = paired item bootstrap. `prompt` = context only; `both` = context + answer options",
         "(BoolQ is always prompt-only).", ""]
    present = [(k, n) for k, n in MODELS if k in M]
    # clean
    L += ["## Clean accuracy", "", "| model | " + " | ".join(TASKS) + " | mean |", "|---|" + "---|" * (len(TASKS) + 1)]
    for k, n in present:
        vals = {}
        for t in TASKS:
            b = bits(M[k], t, METRIC[t])
            if b:
                vals[t] = 100 * sum(b.values()) / len(b)
        J["clean"][k] = vals
        L.append(f"| {n} | " + " | ".join(f"{vals[t]:.1f}" if t in vals else "–" for t in TASKS) +
                 f" | {sum(vals.values()) / max(1, len(vals)):.1f} |")
    # macro Δ table
    for tg in TARGETS:
        L += ["", f"## Macro Δ (pp), target = `{tg}`", "",
              "| model | " + " | ".join(f"{LABEL[a]} p={p[1:]}%" for a in ATTACKS for p in PROBS) + " |",
              "|---|" + "---|" * (len(ATTACKS) * len(PROBS))]
        for k, n in present:
            cells = []
            for a in ATTACKS:
                for p in PROBS:
                    r = macro_delta(M[k], lambda t: f"{t}_visual_{a}_{p}_{'prompt' if t == 'boolq' else tg}", rng)
                    J["visual"].setdefault(k, {})[f"{a}_{p}_{tg}"] = None if r is None else {
                        "delta": r[0], "ci": [r[1], r[2]], "per_task": r[3], "n_tasks": r[4]}
                    cells.append("–" if r is None else f"{r[0]:+.1f} [{r[1]:+.1f}, {r[2]:+.1f}]" +
                                 ("" if r[4] == len(TASKS) else f" ({r[4]}t)"))
            L.append(f"| {n} | " + " | ".join(cells) + " |")
    # per-task detail at p=50, both
    L += ["", "## Per-task Δ (pp), p = 50%, target = `both` (BoolQ: prompt)", "",
          "| model | attack | " + " | ".join(TASKS) + " |", "|---|---|" + "---|" * len(TASKS)]
    for k, n in present:
        for a in ATTACKS:
            r = J["visual"][k].get(f"{a}_p50_both")
            if r:
                L.append(f"| {n} | {LABEL[a]} | " + " | ".join(
                    f"{r['per_task'][t]:+.1f}" if t in r["per_task"] else "–" for t in TASKS) + " |")
    # MNLI official
    L += ["", "## Bad Characters official MNLI homoglyph set (untargeted, 500 dev-matched items)", "",
          "Adversarial premises released with Boucher et al. (optimised against fairseq RoBERTa-large-MNLI by",
          "differential evolution; budget = max #homoglyphs). Scored 0-shot with lm-eval's `mnli` prompt (acc;",
          "chance = 33.3). Budget 0 = clean.", "",
          "| model | " + " | ".join(f"b={b}" for b in BUDGETS) + " | Δ(b10−b0) |", "|---|" + "---|" * (len(BUDGETS) + 1)]
    for k, n in present:
        acc = {}
        for b in BUDGETS:
            x = bits(M[k], f"mnli_badchars_homoglyphs_b{b}", "acc")
            if x:
                acc[b] = 100 * sum(x.values()) / len(x)
        J["mnli"][k] = acc
        if acc:
            L.append(f"| {n} | " + " | ".join(f"{acc[b]:.1f}" if b in acc else "–" for b in BUDGETS) +
                     (f" | {acc[10] - acc[0]:+.1f} |" if 0 in acc and 10 in acc else " | – |"))
    open(f"{rep}/summary.md", "w").write("\n".join(L) + "\n")
    json.dump(J, open(f"{rep}/summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
