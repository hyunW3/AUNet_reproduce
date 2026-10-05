#!/usr/bin/env python3
"""Graded despace-MC robustness report: acc vs space-removal probability, per model.

Reads each model's evals_despace_graded/results.json (despace_mc_avg_<variant> +
per-task despace_mc_<task>_<variant>) and renders:
  - a robustness curve (acc vs p, context-only solid / context+answer dashed), and
  - a table (variant x model) of macro-avg acc.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
MODELS = [  # (tag, label, results.json)
    ("llama", "Llama", f"{L}/runs/main/1.3B/llama_1.8B_paper/evals_despace_graded/results.json"),
    ("aunet", "AUNet", f"{L}/runs/main/1.3B/aunet2_1.3B/evals_despace_graded/results.json"),
    ("rg", "BPEByte rg", f"{L}/runs/main/1.3B/bpebyte_br_greedy_root_1.3B/evals_despace_graded/results.json"),
]
COL = {"llama": "#0072B2", "aunet": "#E69F00", "rg": "#009E73"}
PROBS = [0, 10, 40, 70]          # x-axis
TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]
OUT = f"{L}/reports/despace"


def load(path):
    if not Path(path).exists():
        return None
    return json.load(open(path)).get("results", {})


def acc(res, variant):
    r = res.get(f"despace_mc_avg_{variant}")
    return r["acc"] if r else float("nan")


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    data = {tag: load(p) for tag, _, p in MODELS}
    have = [(t, lb) for t, lb, _ in MODELS if data[t]]
    if not have:
        print("no results yet"); return

    # --- curve ---
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for tag, lb in have:
        res = data[tag]
        ctx = [acc(res, "clean")] + [acc(res, f"ctx{p:02d}") for p in (10, 40, 70)]
        allv = [acc(res, "clean")] + [acc(res, f"all{p:02d}") for p in (10, 40, 70)]
        ax.plot(PROBS, ctx, "o-", color=COL[tag], label=f"{lb} · context")
        ax.plot(PROBS, allv, "s--", color=COL[tag], alpha=0.7, label=f"{lb} · context+answer")
    ax.set_xlabel("space-removal probability (%)")
    ax.set_ylabel("MCQ accuracy (macro-avg over 5 tasks)")
    ax.set_title("Graded despace robustness — HS · ARC-E · ARC-C · PIQA · WinoGrande")
    ax.set_xticks(PROBS); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/despace_curve.png", dpi=140, bbox_inches="tight")
    print("wrote", f"{OUT}/despace_curve.png")

    # --- table (markdown) ---
    variants = ["clean", "ctx10", "ctx40", "ctx70", "all10", "all40", "all70"]
    lines = ["# Graded despace-MC robustness", "",
             "Each intra-line space removed independently with prob p. `ctxNN` = context only; "
             "`allNN` = context **and** answer despaced. Macro-avg acc over 5 tasks (500/task).", "",
             "| variant | " + " | ".join(lb for _, lb in have) + " |",
             "|---|" + "---|" * len(have)]
    for v in variants:
        cells = []
        for tag, _ in have:
            a = acc(data[tag], v)
            cells.append(f"{a:.3f}" if a == a else "—")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    # per-task clean vs all70 delta
    lines += ["", "## Per-task acc drop at all70 (context+answer, p=70%)", "",
              "| task | " + " | ".join(lb for _, lb in have) + " |", "|---|" + "---|" * len(have)]
    for task in TASKS:
        cells = []
        for tag, _ in have:
            r = data[tag]
            c = r.get(f"despace_mc_{task}_clean", {}).get("acc")
            a = r.get(f"despace_mc_{task}_all70", {}).get("acc")
            cells.append(f"{c:.2f}→{a:.2f}" if c is not None and a is not None else "—")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    Path(f"{OUT}/summary.md").write_text("\n".join(lines) + "\n")
    print("wrote", f"{OUT}/summary.md")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
