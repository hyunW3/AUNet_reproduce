#!/usr/bin/env python
"""Summarize Prompt Boundary Problem (PBP) eval results across runs.

  python runs/pbp_table.py <run_dir|results.json> [<run_dir|results.json> ...]

Each arg is either a run dir (reads <run_dir>/eval_pbp/results.json — what run_eval_pbp.sh
writes) or a results.json directly. Prints two tables side-by-side across runs:

  1. ΔBPC per stratum (bits/byte). delta_bpc = BPC(misaligned cut) - BPC(aligned cut).
     A byte model (AU-Net 2 / BPEByte) is cut-invariant -> delta_bpc ~= 0; a BPE model
     (Llama) is distorted by the mid-token cut -> delta_bpc > 0 (largest on zh_char/code).
  2. MCQ boundary (Exp B): acc canonical vs trailing-space, and delta_acc.

So a clean result reads: byte rows ~0.00 across strata, Llama rows positive and growing
on code/Chinese. Run run_eval_pbp.sh on BOTH the byte ckpt and the Llama ckpt, then pass
both run dirs here.
"""
import json
import os
import sys

STRATA = ["pbp_overall", "pbp_en_space", "pbp_code_space", "pbp_zh_char"]
MC_KEYS = ["pbp_mc_canonical", "pbp_mc_space"]


def load(arg):
    """Resolve a run dir or a results.json path to its 'results' dict (or None)."""
    p = arg if arg.endswith(".json") else os.path.join(arg, "eval_pbp", "results.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get("results", {})
    except Exception:
        return None


def label(arg):
    if arg.endswith(".json"):
        # .../runs/<model>/eval_pbp/results.json -> <model>
        parts = os.path.normpath(arg).split(os.sep)
        return parts[-3] if len(parts) >= 3 else parts[0]
    return os.path.basename(arg.rstrip("/"))


def fmt(x, sign=False, pct=False):
    if x is None:
        return "-"
    if pct:
        return f"{x * 100:.1f}"
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def main(args):
    cols = []  # (label, results)
    for a in args:
        r = load(a)
        if r is None:
            print(f"# skip {a!r}: no results.json (has run_eval_pbp.sh run?)", file=sys.stderr)
            continue
        cols.append((label(a), r))
    if not cols:
        print("no PBP results found — run: bash lingua/run_eval_pbp.sh <ckpt>")
        return

    w = max(14, *(len(c[0]) for c in cols)) + 2
    lw = 16

    # --- Table 1: ΔBPC per stratum ---
    print("=== PBP — ΔBPC (bits/byte; byte model ~0.00, BPE > 0) ===")
    strata = [s for s in STRATA if any(s in r for _, r in cols)]
    print("stratum".ljust(lw) + "".join(c[0].rjust(w) for c in cols))
    print("-" * (lw + w * len(cols)))
    for s in strata:
        row = s.replace("pbp_", "").ljust(lw)
        for _, r in cols:
            d = r.get(s, {}).get("delta_bpc")
            row += fmt(d, sign=True).rjust(w)
        print(row)
    # n + absolute aligned->misaligned detail (overall only, compact)
    print()
    print("detail (overall):  aligned -> misaligned BPC  (n items)")
    for name, r in cols:
        o = r.get("pbp_overall", {})
        if o:
            print(f"  {name.ljust(lw)} {fmt(o.get('bpc_aligned'))} -> {fmt(o.get('bpc_misaligned'))}"
                  f"   (n={o.get('n', '?')})")

    # --- Table 2: MCQ boundary (Exp B) ---
    if any("pbp_mc_canonical" in r for _, r in cols):
        print()
        print("=== PBP-MC — accuracy under prompt-boundary perturbation (Exp B) ===")
        print("metric".ljust(lw) + "".join(c[0].rjust(w) for c in cols))
        print("-" * (lw + w * len(cols)))
        for key, lbl in [("pbp_mc_canonical", "acc canonical"), ("pbp_mc_space", "acc +space")]:
            row = lbl.ljust(lw)
            for _, r in cols:
                row += fmt(r.get(key, {}).get("acc"), pct=True).rjust(w)
            print(row)
        row = "Δacc (space)".ljust(lw)
        for _, r in cols:
            d = r.get("pbp_mc_space", {}).get("delta_acc")
            row += (f"{d * 100:+.1f}" if d is not None else "-").rjust(w)
        print(row)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    main(sys.argv[1:])
