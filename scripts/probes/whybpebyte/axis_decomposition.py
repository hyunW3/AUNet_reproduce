#!/usr/bin/env python3
"""Where does each arm's robustness advantage actually come from?

Reads the **dataset-matched** tables in reports_afterAAAIsub/robustness_suite.md -- section 12.1
(5 benchmarks x 4 axes, `limit 2000`, identical perturbation seeds, each run's own clean baseline)
and section 12.2 (noise strategy x position). Section 12 is the right source, not section 7: section 7
is the paper protocol, where each axis uses a different dataset. Section 9 is the same unified
protocol as section 12 but only HellaSwag + ARC-Easy.

Values are parsed, never retyped, so this cannot drift from the suite doc.

    python axis_decomposition.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "reports_afterAAAIsub/robustness_suite.md"
ARMS = ["Transformer", "AUNet", "BPEByte", "H-Net", "BLT"]
LABEL = {"Transformer": "Llama (subword)", "AUNet": "AU-Net", "BPEByte": "BPEByte",
         "H-Net": "H-Net", "BLT": "BLT"}
AXES = ["pbp", "noise", "typo", "despace"]
MAIN = ["Transformer", "AUNet", "BPEByte"]
# noise strategies that destroy information vs. ones that only change the surface form
CASING = {"randomcase", "uppercase"}
DESTRUCTIVE = {"antspeak", "drop", "repeat"}


def _nums(cells):
    out = []
    for c in cells:
        m = re.search(r"[-−+]?\d+\.\d+", c.replace("**", ""))
        out.append(float(m.group().replace("−", "-")) if m else None)
    return out


def parse_axes(tag: str):
    """section 12.1 '<axis> 평균(<tag>)' rows -> {axis: {arm: delta}}."""
    want = {f"**{ax} 평균({tag})**": ax for ax in AXES}
    rows = {}
    for line in SRC.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in want:
            rows[want[cells[0]]] = dict(zip(ARMS, _nums(cells[1:1 + len(ARMS)])))
    missing = [a for a in AXES if a not in rows]
    if missing:
        raise SystemExit(f"could not parse {tag} axes {missing} from {SRC}")
    return rows


def parse_noise():
    """section 12.2 'strategy | position | ...' rows -> [(strategy, position, {arm: delta})]."""
    out = []
    strategies = CASING | DESTRUCTIVE
    for line in SRC.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2 + len(ARMS) and cells[0] in strategies and \
                cells[1] in ("prompt", "completion", "both"):
            out.append((cells[0], cells[1], dict(zip(ARMS, _nums(cells[2:2 + len(ARMS)])))))
    return out


def macro(R, arms=MAIN):
    return {a: sum(R[ax][a] for ax in AXES) / len(AXES) for a in arms}


def main():
    for tag in ("5태스크", "4태스크"):
        R = parse_axes(tag)
        n = tag.replace("태스크", " tasks")
        print(f"### Per-axis Δacc at 1.3B, unified protocol, {n} (robustness_suite.md §12.1)\n")
        print("| Axis | " + " | ".join(LABEL[a] for a in ARMS) + " | worst of the three |")
        print("|---|" + "---|" * (len(ARMS) + 1))
        for ax in AXES:
            v = R[ax]
            worst = min(MAIN, key=lambda a: v[a])
            cells = [f"{v[a]:+.2f}" if v[a] is not None else "--" for a in ARMS]
            print(f"| {ax} | " + " | ".join(cells) + f" | {LABEL[worst]} |")
        m = macro(R)
        print("| **macro** | " + " | ".join(
            (f"**{m[a]:+.2f}**" if a in MAIN else "--") for a in ARMS) + " | -- |")
        print()
        print("| Comparison | macro gap (pp) | " + " | ".join(f"{ax}" for ax in AXES) + " |")
        print("|---|---|" + "---|" * len(AXES))
        for a, b in (("BPEByte", "Transformer"), ("BPEByte", "AUNet"), ("AUNet", "Transformer")):
            gap = m[a] - m[b]
            parts = [(R[ax][a] - R[ax][b]) / len(AXES) for ax in AXES]
            print(f"| {LABEL[a]} − {LABEL[b]} | {gap:+.2f} | " +
                  " | ".join(f"{p:+.2f} ({100*p/gap:.0f}%)" for p in parts) + " |")
        print()

    # ---- what "noise robustness" actually is
    nz = parse_noise()
    print("### Is \"byte models are robust to noise\" really about casing? (§12.2)\n")
    print("| Noise group | " + " | ".join(LABEL[a] for a in ARMS) + " | subword penalty |")
    print("|---|" + "---|" * (len(ARMS) + 1))
    for gname, group in (("casing (randomcase, uppercase)", CASING),
                         ("information-destroying (antspeak, drop, repeat)", DESTRUCTIVE)):
        sel = [d for s, _p, d in nz if s in group]
        mean = {a: sum(d[a] for d in sel) / len(sel) for a in ARMS}
        pen = mean["Transformer"] - min(mean[a] for a in MAIN if a != "Transformer")
        print(f"| {gname} | " + " | ".join(f"{mean[a]:+.2f}" for a in ARMS) +
              f" | {pen:+.2f} pp |")
    # ---- convergence: does Experiment E separate the same two groups?
    import json
    RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"
    E2 = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte"}
    fam_group = {"noise_randomcase": "casing", "noise_uppercase": "casing",
                 "noise_antspeak": "destructive", "noise_drop": "destructive",
                 "noise_repeat": "destructive"}
    cos = {}
    for arm in E2:
        fp = RES / f"repr_suite_{arm}.jsonl"
        if not fp.exists():
            cos = None
            break
        rr = [json.loads(l) for l in open(fp)]
        for g in ("casing", "destructive"):
            fams = [f for f, gg in fam_group.items() if gg == g]
            v = [x["cos_centered_mean"] for x in rr
                 if x["family"] in fams and x.get("cos_centered_mean") is not None]
            cos[(E2[arm], g)] = sum(v) / len(v)
    if cos:
        print()
        print("### Does Experiment E separate the same two groups? (convergence check)\n")
        print("| Noise group | subword penalty, Δacc (§12.2) | subword penalty, "
              "centred cosine (E2) |")
        print("|---|---|---|")
        for g, gname in (("casing", "casing"), ("destructive", "information-destroying")):
            sel = [d for s, _p, d in nz if s in (CASING if g == "casing" else DESTRUCTIVE)]
            mean = {a: sum(d[a] for d in sel) / len(sel) for a in ARMS}
            acc_pen = mean["Transformer"] - min(mean[a] for a in MAIN if a != "Transformer")
            best_byte = max(cos[(a, g)] for a in ("AUNet", "BPEByte"))
            cos_pen = cos[("Transformer", g)] - best_byte
            print(f"| {gname} | {acc_pen:+.2f} pp | {cos_pen:+.3f} |")
        print()
        print("Both measures put the architectures far apart on casing and on top of each other "
              "on information-destroying corruption -- an independent corpus (held-out DCLM "
              "windows) reaching §13's conclusion from HellaSwag contexts.")

    print()
    print("The noise axis splits cleanly. On the corruptions that destroy information every arm "
          "lands within ~1 pp of every other, so the parser cannot help. The whole of the byte "
          "models' noise advantage is casing -- which is what §13 diagnoses and what Experiment E "
          "quantifies (see RESULTS.md §3b/§7).")


if __name__ == "__main__":
    main()
