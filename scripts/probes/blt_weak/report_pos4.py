#!/usr/bin/env python3
"""POSITION_BALANCED.md: the position-balanced listing probe (build_tasks4.py), five models, all scored on
snu55 (same hardware for every model, incl. the clean / original-order references).

  python scripts/probes/blt_weak/report_pos4.py --root reports/blt_weak
"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

MODELS = [("subword_llama", "Llama"), ("aunet_static", "AU-Net"), ("byte_greedyroot", "BPEByte"),
          ("hnet_1stage_XL", "H-Net"), ("blt_1b", "BLT-1B")]
L = "ABCD"
FMT = {"before": "options listed BEFORE the question", "after": "standard MCQ: options AFTER the question"}
SC = {"text": "option text (\" warm\")", "letter": "letter (\" A\")", "full": "label + text (\" (A) warm\" / \" A. warm\")"}


def ci(v):
    if not v:
        return "—"
    p = sum(v) / len(v)
    return f"{p:.3f} ±{1.96 * math.sqrt(max(p * (1 - p), 1e-9) / len(v)):.2f}"


def table(head, body):
    return ("| " + " | ".join(head) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(head)))
            + "|\n" + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/blt_weak")
    a = ap.parse_args()
    root = Path(a.root)
    S = defaultdict(list)
    for f in glob.glob(f"{root}/results/snu55/pos4*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("score") is not None:
                S[(r["tag"], r["cond"])].append(r)
    meta = {}
    for f in ("pos4.jsonl", "pos4_ref.jsonl"):
        for l in open(root / "data" / f):
            r = json.loads(l)
            meta[r["id"]] = (r["gold_pos"], r["n_opt"])
    gp = lambda r: meta[r["id"]][0]
    k4 = lambda r: meta[r["id"]][1] == 4
    md = ["# Position-balanced listing probe — five models (all scored on snu55)\n",
          "Options are permuted per item so the gold sits at position idx mod k (A/B/C/D equally often). Each listing",
          "format is scored three ways, separately (never OR-ed). `ref_*` = the original-order prompts of the 2nd batch,",
          "re-scored on the same machine. ARC-E / ARC-C / PIQA / HellaSwag x 500. Cells: accuracy ±95% half-width.\n",
          "## 1. Overall accuracy (all 2,000 items)\n"]
    body = [["clean (no listing; option text)"] + [ci([r["score"] for r in S[(t, "ref_clean")]]) for t, _ in MODELS],
            ["original order, options before (echo_all)"] + [ci([r["score"] for r in S[(t, "ref_echo_all")]]) for t, _ in MODELS]]
    for fm in ("before", "after"):
        for sc in ("text", "letter", "full"):
            body.append([f"balanced, {fm} / {sc}"] + [ci([r["score"] for r in S[(t, f'{fm}_{sc}')]]) for t, _ in MODELS])
    md.append(table(["condition"] + [n for _, n in MODELS], body))
    md.append("Chance: ARC 4-way 0.25 (a few 3/5-way), PIQA 0.5, HellaSwag 0.25 → pooled ≈ 0.31.\n")
    for fm in ("before", "after"):
        for sc in ("text", "letter", "full"):
            c = f"{fm}_{sc}"
            md.append(f"## {FMT[fm]} — scored by {SC[sc]}\n")
            md.append("Accuracy by gold position (4-option items: ARC-E / ARC-C / HellaSwag):\n")
            body = []
            for p in range(4):
                body.append([f"gold at {L[p]}"] + [ci([r["score"] for r in S[(t, c)] if k4(r) and gp(r) == p]) for t, _ in MODELS])
            body.append(["all 4-option"] + [ci([r["score"] for r in S[(t, c)] if k4(r)]) for t, _ in MODELS])
            spread = []
            for t, _ in MODELS:
                acc = [sum(r["score"] for r in S[(t, c)] if k4(r) and gp(r) == p) /
                       max(1, sum(1 for r in S[(t, c)] if k4(r) and gp(r) == p)) for p in range(4)]
                spread.append(f"{max(acc) - min(acc):.3f}")
            body.append(["max − min"] + spread)
            md.append(table(["gold position"] + [n for _, n in MODELS], body))
            md.append("\nPrediction distribution (4-option items):\n")
            body = [[f"predicts {L[p]}"] + [(lambda v: f"{sum(r['pred'] == p for r in v) / len(v):.3f}" if v else "—")(
                [r for r in S[(t, c)] if k4(r)]) for t, _ in MODELS] for p in range(4)]
            md.append(table(["", *[n for _, n in MODELS]], body))
            md.append("\nPIQA (2 options) by gold position: " + "; ".join(
                f"{n} " + " / ".join(ci([r["score"] for r in S[(t, c)] if r["length"] == "piqa" and gp(r) == p]).split(" ")[0]
                                     for p in range(2)) for t, n in MODELS) + " (A / B)\n")
    (root / "POSITION_BALANCED.md").write_text("\n".join(md) + "\n")
    print("wrote", root / "POSITION_BALANCED.md")


if __name__ == "__main__":
    main()
