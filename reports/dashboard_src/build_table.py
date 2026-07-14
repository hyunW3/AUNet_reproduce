#!/usr/bin/env python3
import json
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
D=json.load(open(f"{ROOT}/reports/downstream_data.json"))
data=D["data"]; L=D["meta"]["labels"]
order=["llama","aunet(word)","bpebyte_rg","hyb_leafQ","hyb_greedyQ"]
cols=["hellaswag","arc_easy","arc_challenge","piqa","boolq","winogrande","mmlu_text"]
def c(v): return f"{v:.1f}" if isinstance(v,(int,float)) else "—"
lines=[]
lines.append("# Downstream accuracy vs scale — 5-way table (0-shot)\n")
lines.append("Families: **Llama** (BPE), **AU-Net** (word-patch), **BPEByte rg** (online root_greedy),")
lines.append("**BPEByte hybrid** (offline-leaf prefill + B3) in two eval regimes: **leafQ** (offline-leaf")
lines.append("question) and **greedyQ** (native online-root-greedy question). Ladder = **constant-ratio")
lines.append("~210 bytes/param**: 100M/300M/760M = `cmp_g10`, 1.3B = main run.\n")
lines.append("| Scale | Family | HS | ARC-E | ARC-C | PIQA | BoolQ | WinoG | MMLU-txt | **Avg3** | **Avg-all** |")
lines.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for s in D["meta"]["scales"]:
    first=True
    for ser in order:
        cells=data[ser].get(s)
        lab=L[ser]
        if not cells:  # e.g. hybrid at 760M
            if ser.startswith("hyb") and s=="760M":
                row=[c(None)]*9
                lines.append(f"| {s if first else ''} | {lab} | "+" | ".join(row)+" |"); first=False
            continue
        vals=[c(cells.get(b)) for b in cols]
        row=" | ".join(vals)+f" | **{c(cells.get('_avg3'))}** | **{c(cells.get('_avgall'))}**"
        lines.append(f"| {'**'+s+'**' if first else ''} | {lab} | {row} |"); first=False
lines.append("")
notes="""**Notes**
- **Metric/shots:** HS/ARC-E/ARC-C/PIQA = `acc_norm`; BoolQ/WinoG/MMLU-text = `acc`; all **0-shot**.
- **Avg3** = mean(HS, ARC-E, PIQA) — the three most reliable tasks (ARC-C dropped: near-chance/noisy at small scale).
  **Avg-all** = mean of every benchmark column present in that row.
- **Hybrid = canonical leaf/B3 checkpoints, FULL eval (2026-07-13, GPU3):** 100M `lb_hybrid_100M`@53504
  (rsynced from ece), 300M `hybrid_300M`@120752, 1.3B `hybrid_1p3B_leaf_B3`@180000. **leafQ** = question
  offline-leaf; **greedyQ** = question online-root-greedy (native, ≈ the BPEByte-rg column at 1.3B).
  Earlier leafQ/greedyQ points had used the under-trained **p1b1 pilots** (300M @6.7k vs canonical @120k)
  — fixed; see `runs/main/HYBRID_CANONICAL.md`. No hybrid **760M** run exists.
- **MMLU-text**: 0-shot acc, macro-avg over 57 subjects; only evaluated at **1.3B**.
- **† 760M BoolQ/WinoG are main-lineage (101 bpp):** cmp_g10 760M was never evaluated on them, so those
  two cells use the *main* 760M run (the other four 760M tasks are cmp_g10).
- **Limit:** all HS/ARC-E/ARC-C/PIQA and the hybrid rows are **full**; base **BoolQ/WinoG at 100M/300M**
  are still `limit=1000` (`evals_fill`) — re-run at full if strict parity is needed.
- **760M lineage:** ratio-matched `cmp_g10` (~210 bpp); the committed `scaling_*.png` use the *main*
  760M (101 bpp), ~5-6 pts lower — this table differs at 760M by design.
"""
lines.append(notes)
open(f"{ROOT}/reports/scaling_downstream_table.md","w").write("\n".join(lines))
print("wrote reports/scaling_downstream_table.md")
print("\n".join(lines[:24]))
