#!/usr/bin/env python3
"""Collect the BPEByte3 level-2 sweep + boundary-dropout arms into one comparison table.

BPB follows the sweep's own convention (`reports_afterAAAIsub/r10_lr_sweep_plan.md`, and the
rg/lr34 row the user reported): tail-mean of `loss/out` over the last N logged points, divided by
ln 2. Compression is read from `nb_toks/enc_*` over the same window, so the compute each arm
actually consumed is reported next to its loss rather than assumed.

  python3 scripts/bpebyte3/collect_results.py [--tail 10] [--md]
"""
import argparse
import json
import math
import os
import statistics

RUNS = "/mnt/ssd2/hyun2/aunet_r10lr/runs/r10lr"
STEPS = 3344

# name -> (label, group)
ARMS = [
    ("r10lr_rg_100M_lr34",          "rg  2-stage (no level 2)",              "reference"),
    ("r10lr_rgl3_100M_lr34",        "rgl3  level-2 = static2",               "level-2 control"),
    ("r10lr_rgbb7_100M_lr34",       "rgbb7  level-2 = bytebudget B=7",       "level-2"),
    ("r10lr_rgend_100M_lr34",       "rgend  level-2 = endtype@deadend",      "level-2"),
    ("r10lr_rgstats_100M_lr34",     "rgstats  level-2 = close-stats θ=0.5",  "level-2"),
    ("r10lr_rgdrop010_100M_lr34",   "rgdrop010  boundary dropout p=.1 free", "dropout"),
    ("r10lr_rgdropps010_100M_lr34", "rgdropps010  dropout p=.1 prefix_safe", "dropout"),
]


def read(name, tail):
    d = os.path.join(RUNS, name)
    mfile = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(mfile):
        return None
    rows = []
    with open(mfile) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows = [r for r in rows if "loss/out" in r]
    if not rows:
        return None
    last = rows[-1].get("global_step", 0)
    w = rows[-tail:]
    loss = statistics.mean(r["loss/out"] for r in w)
    out = {
        "step": last,
        "done": os.path.exists(os.path.join(d, ".DONE")) or last >= STEPS - 10,
        "loss": loss,
        "bpb": loss / math.log(2),
        "n": len(rows),
    }
    e0 = [r["nb_toks/enc_0"] for r in w if "nb_toks/enc_0" in r]
    e1 = [r["nb_toks/enc_1"] for r in w if "nb_toks/enc_1" in r]
    out["l1"] = statistics.mean(e0) if e0 else None
    out["l2"] = statistics.mean(e1) if e1 else None
    out["cr"] = (out["l1"] / out["l2"]) if (out["l1"] and out["l2"]) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=10, help="logged points averaged for the tail mean")
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    args = ap.parse_args()

    got = [(n, lab, grp, read(n, args.tail)) for n, lab, grp in ARMS]
    ctrl = next((r for n, _, _, r in got if n == "r10lr_rgl3_100M_lr34" and r), None)

    if args.md:
        print(f"| arm | 상태 | step | BPB ↓ | Δ vs static2 | L1/L2 | trunk 위치 |")
        print("|---|---|---:|---:|---:|---:|---:|")
    else:
        print(f"{'arm':<42}{'status':<10}{'step':>6}{'BPB':>9}{'Δctrl':>8}{'L1/L2':>8}{'trunk':>9}")
    for name, label, grp, r in got:
        if r is None:
            row = (label, "not started", "", "", "", "", "")
        else:
            st = "DONE" if r["done"] else "running"
            d = f"{r['bpb'] - ctrl['bpb']:+.4f}" if (ctrl and r is not ctrl) else "—"
            cr = f"{r['cr']:.3f}" if r["cr"] else "—"
            tr = f"{r['l2']:,.0f}" if r["l2"] else f"{r['l1']:,.0f}" if r["l1"] else "—"
            row = (label, st, f"{r['step']}", f"{r['bpb']:.4f}", d, cr, tr)
        if args.md:
            print("| " + " | ".join(str(x) for x in row) + " |")
        else:
            print(f"{row[0]:<42}{row[1]:<10}{row[2]:>6}{row[3]:>9}{row[4]:>8}{row[5]:>8}{row[6]:>9}")
    if not args.md:
        print(f"\nBPB = tail-mean(loss/out, last {args.tail} logs) / ln2.  "
              f"trunk = mean level-2 positions/step (level-1 for 2-stage arms).")


if __name__ == "__main__":
    main()
