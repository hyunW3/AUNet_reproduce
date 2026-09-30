#!/usr/bin/env python3
"""Patch lengths for the matched trio's parsers (CPU): Transformer (Llama-3 tiktoken tokens),
AUNet (RegexPool word1 — the aunet2_1.3B config), BPEByte (OnlineBPE greedy/root).

  lingua/.venv/bin/python scripts/probes/patch_stats/measure_trio.py --model bpebyte [--gate_only]
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # scripts/probes
from patch_common import Hist, check_gate, lengths_from_starts, load_texts  # noqa: E402
from compression_stability import build_parser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("llama", "aunet", "bpebyte"), required=True)
    ap.add_argument("--gate_only", action="store_true")
    a = ap.parse_args()
    parse = build_parser(a.model)
    rows = load_texts(["dclm"] if a.gate_only else None)
    rows.sort(key=lambda r: r["group"] != "dclm")          # gate first
    h, t0 = Hist(), time.time()
    for i, r in enumerate(rows):
        h.add(r, lengths_from_starts(parse(r["text"]), len(r["text"].encode("utf-8"))))
        if r["group"] == "dclm" and (i + 1 == len(rows) or rows[i + 1]["group"] != "dclm"):
            if not check_gate(a.model, h):
                raise SystemExit("gate failed — parser does not reproduce the paper-time measurement")
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    if not a.gate_only:
        print("wrote", h.save(a.model, {"parser": a.model}))


if __name__ == "__main__":
    main()
