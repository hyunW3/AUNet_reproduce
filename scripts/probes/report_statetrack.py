#!/usr/bin/env python3
"""Combined report for the four state-tracking / recall probes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/reports/statetrack"
ORDER = ["subword_llama", "aunet_static", "byte_greedyroot"]
LBL = {"subword_llama": "subword (Llama)", "aunet_static": "AUNet",
       "byte_greedyroot": "BPEByte rg"}
COL = {"subword_llama": "#4C78A8", "aunet_static": "#F58518",
       "byte_greedyroot": "#54A24B"}


def load(f):
    p = Path(R) / f
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main():
    s5 = {r["tag"]: r["acc"] for r in load("s5_results.jsonl")}
    dyck = {r["tag"]: r["acc"] for r in load("dyck_results.jsonl")}
    mk = load("mkniah_results.jsonl")
    vt = load("vt_results.jsonl")
    tags = [t for t in ORDER if t in s5 or t in dyck]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # S5 bar
    a = ax[0][0]
    a.bar([LBL[t] for t in tags], [s5.get(t, 0) for t in tags], color=[COL[t] for t in tags])
    a.axhline(0.20, ls="--", color="grey", label="chance 0.20")
    a.set_title("S5 permutation composition (HARD state tracking)"); a.set_ylabel("read acc")
    a.set_ylim(0, 1); a.tick_params(axis="x", labelrotation=20, labelsize=8); a.legend(fontsize=8)
    # Dyck bar
    a = ax[0][1]
    a.bar([LBL[t] for t in tags], [dyck.get(t, 0) for t in tags], color=[COL[t] for t in tags])
    a.axhline(0.33, ls="--", color="grey", label="chance 0.33")
    a.set_title("Dyck-3 bracket matching (nested-stack state)"); a.set_ylabel("close-pred acc")
    a.set_ylim(0, 1); a.tick_params(axis="x", labelrotation=20, labelsize=8); a.legend(fontsize=8)
    # MK-NIAH lines
    a = ax[1][0]
    for t in ORDER:
        rows = sorted([r for r in mk if r["tag"] == t], key=lambda r: r["num_needles"])
        if rows:
            a.plot([r["num_needles"] for r in rows], [r["exact_match"] for r in rows],
                   "o-", label=LBL[t], color=COL[t])
    a.set_title("Multi-key NIAH (disambiguation recall)"); a.set_xlabel("# needles (K)")
    a.set_ylabel("exact-match"); a.set_ylim(0, 1.02); a.legend(fontsize=8); a.grid(alpha=0.3)
    # VT lines
    a = ax[1][1]
    for t in ORDER:
        rows = sorted([r for r in vt if r["tag"] == t], key=lambda r: r["num_hops"])
        if rows:
            a.plot([r["num_hops"] for r in rows], [r["exact_match"] for r in rows],
                   "o-", label=LBL[t], color=COL[t])
    a.set_title("Variable Tracking (multi-hop recall)"); a.set_xlabel("# hops")
    a.set_ylabel("exact-match"); a.set_ylim(0, 1.02); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle("State-tracking & recall probes — frozen 1.3B, in-context", fontweight="bold")
    fig.tight_layout()
    out = Path(R) / "statetrack_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)

    # markdown
    lines = ["# State-tracking & recall probes (frozen 1.3B, in-context)", "",
             "| task | subword (Llama) | AUNet | BPEByte rg | chance |",
             "|---|---|---|---|---|",
             f"| S5 permutation (hard) | {s5.get('subword_llama',0):.2f} | {s5.get('aunet_static',0):.2f} | {s5.get('byte_greedyroot',0):.2f} | 0.20 |",
             f"| Dyck-3 brackets | {dyck.get('subword_llama',0):.2f} | {dyck.get('aunet_static',0):.2f} | {dyck.get('byte_greedyroot',0):.2f} | 0.33 |"]
    Path(R, "summary.md").write_text("\n".join(lines) + "\n")
    print("wrote", Path(R, "summary.md"))


if __name__ == "__main__":
    main()
