#!/usr/bin/env python3
"""One integrated table for the MCQ listing study (POSITION_SUMMARY.md), five models, all scored on snu55.

Prompt layouts: clean (no options shown; lm-eval cloze), options listed BEFORE the question
("Options: (A) .. (B) ..\\nQuestion: ..\\nAnswer:"), options listed AFTER the question (standard MCQ
"Question: ..\\nA. ..\\nB. ..\\nAnswer:"). Gold positions balanced (A/B/C/D equally often) for the listed
layouts. Scoring: text (" warm"), letter (" A"), full (" (A) warm" / " A. warm"), OR (text right or
letter right; two guesses), logsum (argmax of log(P(text)+P(letter))).

  python scripts/probes/blt_weak/report_pos_summary.py --root reports/blt_weak > reports/blt_weak/POSITION_SUMMARY.md
"""
import argparse
import glob
import json
import math
from collections import defaultdict

MODELS = [("subword_llama", "Llama"), ("aunet_static", "AU-Net"), ("byte_greedyroot", "BPEByte"),
          ("hnet_1stage_XL", "H-Net"), ("blt_1b", "BLT-1B")]
L = "ABCD"


def lse(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/blt_weak")
    a = ap.parse_args()
    R = defaultdict(dict)
    for f in glob.glob(f"{a.root}/results/snu55/pos4*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("score") is not None:
                R[r["tag"]][(r["cond"], r["length"], r["pos"])] = r
    meta = {}
    for l in open(f"{a.root}/data/pos4.jsonl"):
        r = json.loads(l)
        meta[(r["length"], r["pos"])] = (r["gold_pos"], r["n_opt"])
    clean_gold = {}
    for l in open(f"{a.root}/data/pos4_ref.jsonl"):
        r = json.loads(l)
        clean_gold[(r["length"], r["pos"])] = (r["gold"], len(r["options"]))

    # per model: list of item dicts {k, g, correct, pred} per condition
    conds = [("clean", "no options shown", "text", 0.25)]
    for fmt, lab in (("before", "options BEFORE question"), ("after", "options AFTER question (standard MCQ)")):
        for sc, ch in (("text", 0.25), ("letter", 0.25), ("full", 0.25), ("or", 0.4375), ("logsum", 0.25)):
            conds.append((f"{fmt}_{sc}", lab, sc, ch))

    def items(t, cond):
        out = []
        if cond == "clean":
            for (c, m, i), r in R[t].items():
                if c == "ref_clean":
                    g, k = clean_gold[(m, i)]
                    out.append({"k": k, "g": g, "ok": r["score"], "pred": r["pred"]})
            return out
        fmt, sc = cond.split("_")
        for (c, m, i), rt in R[t].items():
            if c != f"{fmt}_text":
                continue
            g, k = meta[(m, i)]
            rl, rf = R[t].get((f"{fmt}_letter", m, i)), R[t].get((f"{fmt}_full", m, i))
            if sc == "text":
                out.append({"k": k, "g": g, "ok": rt["score"], "pred": rt["pred"]})
            elif sc == "letter" and rl:
                out.append({"k": k, "g": g, "ok": rl["score"], "pred": rl["pred"]})
            elif sc == "full" and rf:
                out.append({"k": k, "g": g, "ok": rf["score"], "pred": rf["pred"]})
            elif sc == "or" and rl:
                out.append({"k": k, "g": g, "ok": float(rt["score"] or rl["score"]), "pred": None})
            elif sc == "logsum" and rl:
                comb = [lse(x, y) for x, y in zip(rt["lls"], rl["lls"])]
                p = max(range(k), key=comb.__getitem__)
                out.append({"k": k, "g": g, "ok": float(p == g), "pred": p})
        return out

    D = {(t, c[0]): items(t, c[0]) for t, _ in MODELS for c in conds}
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    print("# MCQ listing study — integrated summary (five models, all scored on snu55)\n")
    print("ARC-E / ARC-C / PIQA / HellaSwag x 500. Layouts: **clean** = lm-eval cloze, no options shown "
          "(`Question: ..\\nAnswer:` + option text); **before** = `Options: (A) .. (B) ..` then the question; "
          "**after** = standard MCQ (`Question: ..\\nA. ..\\nB. ..\\nAnswer:`). Listed layouts use gold positions "
          "balanced over A–D. Scoring: **text** = option text, **letter** = \" A\", **full** = label + text, "
          "**OR** = text right or letter right (two guesses), **logsum** = argmax log(P(text)+P(letter)).\n")
    print("## 1. Accuracy, 4-option items (ARC-E / ARC-C / HellaSwag, n≈1,495)\n")
    print("| layout | scoring | chance | " + " | ".join(n for _, n in MODELS) + " | BLT − BPEByte |")
    print("|---|---|---:|" + "---:|" * (len(MODELS) + 1))
    for c, lab, sc, ch in conds:
        accs = {t: mean([x["ok"] for x in D[(t, c)] if x["k"] == 4]) for t, _ in MODELS}
        print(f"| {c.split('_')[0]} | {sc} | {ch} | " + " | ".join(f"{accs[t]:.3f}" for t, _ in MODELS)
              + f" | {accs['blt_1b'] - accs['byte_greedyroot']:+.3f} |")
    print("\n## 2. Accuracy, all 2,000 items (PIQA included; chance ≈0.31, OR ≈0.49)\n")
    print("| layout | scoring | " + " | ".join(n for _, n in MODELS) + " |")
    print("|---|---|" + "---:|" * len(MODELS))
    for c, lab, sc, ch in conds:
        print(f"| {c.split('_')[0]} | {sc} | " + " | ".join(f"{mean([x['ok'] for x in D[(t, c)]]):.3f}" for t, _ in MODELS) + " |")
    print("\n## 3. By gold position, 4-option items (A / B / C / D, and max − min)\n")
    print("clean shows the same items' original option index (never shown to the model) as a control.\n")
    print("| layout | scoring | " + " | ".join(n for _, n in MODELS) + " |")
    print("|---|---|" + "---|" * len(MODELS))
    for c, lab, sc, ch in conds:
        cells = []
        for t, _ in MODELS:
            acc = [mean([x["ok"] for x in D[(t, c)] if x["k"] == 4 and x["g"] == p]) for p in range(4)]
            cells.append(" / ".join(f"{v:.2f}" for v in acc) + f" (Δ{max(acc) - min(acc):.2f})")
        print(f"| {c.split('_')[0]} | {sc} | " + " | ".join(cells) + " |")
    print("\n## 4. Share of predictions on position A (4-option; 0.25 = no bias)\n")
    print("| layout | scoring | " + " | ".join(n for _, n in MODELS) + " |")
    print("|---|---|" + "---:|" * len(MODELS))
    for c, lab, sc, ch in conds:
        if sc == "or" or c == "clean":
            continue
        print(f"| {c.split('_')[0]} | {sc} | " + " | ".join(
            f"{mean([x['pred'] == 0 for x in D[(t, c)] if x['k'] == 4]):.3f}" for t, _ in MODELS) + " |")


if __name__ == "__main__":
    main()
