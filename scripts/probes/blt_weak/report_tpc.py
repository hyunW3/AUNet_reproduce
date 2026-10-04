#!/usr/bin/env python3
"""TRANSPLANT_CTX.md: context-side segmentation transplant (blt_transplant_ctx.py), BLT-1B on snu55.
Clean reference = BLT's snu55 score on the clean prompt (pos4_ref, same machine).

  python scripts/probes/blt_weak/report_tpc.py --root reports/blt_weak
"""
import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

CONDS = [("nat", "natural entropy patching everywhere (echo_all as evaluated)"),
         ("sfx_clean", "question + option boundaries copied from the clean prompt; listing natural"),
         ("sfx_list_byte", "sfx_clean + listing as 1-byte patches"),
         ("sfx_list_word", "sfx_clean + listing split at spaces (one patch per word)"),
         ("sfx_list_opt", "sfx_clean + one patch per listed option")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/blt_weak")
    a = ap.parse_args()
    root = Path(a.root)
    T = defaultdict(dict)
    for f in glob.glob(f"{root}/results/snu55_tpc/transplant_ctx_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            T[r["cond"]][(r["mc"], r["idx"])] = r
    clean = {}
    for l in open(root / "results/snu55/pos4_ref_blt_1b_s0of2.jsonl"):
        r = json.loads(l)
        if r["cond"] == "ref_clean":
            clean[(r["length"], r["pos"])] = r["score"]
    for l in open(root / "results/snu55/pos4_ref_blt_1b_s1of2.jsonl"):
        r = json.loads(l)
        if r["cond"] == "ref_clean":
            clean[(r["length"], r["pos"])] = r["score"]
    rng = random.Random(0)
    mean = lambda v: sum(v) / len(v) if v else float("nan")

    def boot(vals):
        b = sorted(mean([rng.choice(vals) for _ in vals]) for _ in range(2000))
        return b[50], b[1949]

    md = ["# Context-side segmentation transplant — BLT-1B (snu55)\n",
          "Same bytes (echo_all prompt: options listed before the question); only the patch boundaries change.",
          "Everything after the listing is byte-identical to the clean prompt, so `sfx_*` copy the clean prompt's",
          "boundaries there exactly. Clean reference = BLT on the clean prompt, same machine.\n"]
    nat = T["nat"]
    md.append("| cond | what changes | n | acc | Δ vs nat [95% CI] | Δ vs clean [95% CI] | picks 1st listed (non-HS) |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    ks_c = [k for k in nat if k in clean]
    v = [clean[k] for k in ks_c]
    md.append(f"| clean | no listing (reference) | {len(v)} | {mean(v):.3f} | | | |")
    for c, desc in CONDS:
        ks = [k for k in T[c] if k in nat]
        if not ks:
            continue
        acc = mean([T[c][k]["score"] for k in ks])
        dn = [T[c][k]["score"] - nat[k]["score"] for k in ks]
        dc = [T[c][k]["score"] - clean[k] for k in ks if k in clean]
        first = mean([T[c][k]["pred"] == 0 for k in ks if k[0] != "hellaswag"])
        lo, hi = boot(dn)
        lo2, hi2 = boot(dc)
        md.append(f"| {c} | {desc} | {len(ks)} | {acc:.3f} | {mean(dn):+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                  f"{mean(dc):+.3f} [{lo2:+.3f}, {hi2:+.3f}] | {first:.3f} |")
    seg = [s for r in nat.values() for s in r.get("seg", [])]
    if seg:
        md.append("\nSegmentation diagnostics (mean over all scored options):\n")
        md.append(f"- question+option patches after the listing: natural {mean([s['sfx_nat'] for s in seg]):.1f} vs "
                  f"clean-prompt {mean([s['sfx_clean'] for s in seg]):.1f}; boundaries changed by `sfx_clean`: "
                  f"{mean([s['changed'] for s in seg]):.1f} per option")
        md.append(f"- listing patches (natural): {mean([s['list_nat'] for s in seg]):.1f}")
    (root / "TRANSPLANT_CTX.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
