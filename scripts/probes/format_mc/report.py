#!/usr/bin/env python3
"""Aggregate format_mc runs into reports/format_robustness/{summary.md,summary.json}.

Inputs: <in_dir>/<model>/results.json (lingua runs) and <in_dir>/<model>*.json (ext runs, possibly
sharded by task / variant group). Per-item bits are re-aggregated here, so shards merge exactly.
Primary metric = acc (raw total logprob argmax), as in the despace_mc / brittle_mc grids.
Delta = perturbed - clean on the same items; 95% CI from a paired item bootstrap of the 5-task macro.
"""
import argparse, glob, json, os, random, statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_mc as F  # noqa: E402

MODELS = [("aunet", "AU-Net"), ("bpebyte", "BPEByte rg"), ("blt", "BLT-1B"), ("hnet", "H-Net 1st")]
TNAME = {"hellaswag": "HS", "arc_easy": "ARC-E", "arc_challenge": "ARC-C", "piqa": "PIQA", "boolq": "BoolQ"}


def load(in_dir, model):
    files = glob.glob(f"{in_dir}/{model}/results.json") + sorted(glob.glob(f"{in_dir}/{model}*.json"))
    rows = {}
    for f in files:
        j = json.load(open(f))
        r = j["results"]
        rows.update({k: v for k, v in r.items() if k.startswith("fmt_") and "bits" in v})
    return rows


PAPER_ACC = ("piqa", "boolq")   # metric "paper": acc on PIQA/BoolQ, acc_norm elsewhere (main-table convention)


def bits(rows, t, v, metric):
    r = rows.get(f"fmt_{t}_{v}")
    if metric == "paper":
        metric = "acc" if t in PAPER_ACC else "acc_norm"
    return None if r is None else r["bits"][metric]


def macro_delta(rows, v, metric, tasks, B=1000, seed=0):
    """(clean macro, perturbed macro, delta, ci_lo, ci_hi, changed frac, delta on changed items)."""
    per = []
    for t in tasks:
        c, p = bits(rows, t, "clean", metric), bits(rows, t, v, metric)
        if c is None or p is None:
            return None
        ch = rows[f"fmt_{t}_{v}"]["bits"].get("changed", [1] * len(c))
        per.append((c, p, ch))
    cm = st.mean(st.mean(c) for c, _, _ in per)
    pm = st.mean(st.mean(p) for _, p, _ in per)
    import numpy as np
    rng = np.random.default_rng(seed)
    ds = np.zeros(B)
    for c, p, _ in per:
        d = np.asarray(p, dtype=np.float32) - np.asarray(c, dtype=np.float32)
        for b0 in range(0, B, 100):                    # 100 resamples at a time keeps memory small
            ds[b0:b0 + 100] += d[rng.integers(0, len(d), size=(min(100, B - b0), len(d)))].mean(axis=1)
    ds = list(ds / len(per))
    ds.sort()
    chg = st.mean(st.mean(ch) for _, _, ch in per)
    sub = [st.mean([p[i] - c[i] for i in range(len(c)) if ch[i]]) for c, p, ch in per if any(ch)]
    return cm, pm, pm - cm, ds[int(0.025 * B)], ds[int(0.975 * B)], chg, (st.mean(sub) if sub else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metric", default="acc")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    data = {m: load(a.in_dir, m) for m, _ in MODELS}
    models = [(m, n) for m, n in MODELS if data[m]]
    V = F.all_variants()
    tasks = list(F.TASKS)
    out = {"metric": a.metric, "tasks": tasks, "macro": {}, "per_task": {}}
    L = []
    pp = lambda x: f"{100 * x:+.1f}"
    L.append(f"# Format / whitespace / punctuation robustness ({a.metric}, 5-task macro, ≤2000 items/task)\n")
    L.append("Δ = perturbed − clean (pt), 5-task macro (HS/ARC-E/ARC-C/PIQA/BoolQ); [95% paired bootstrap CI]. "
             "`chg` = share of items whose prompt the variant actually changed (Δ restricted to those items: `delta_changed` in summary.json).\n")
    hdr = "| variant | family | " + " | ".join(n for _, n in models) + " | chg |"
    L.append(hdr)
    L.append("|" + "---|" * (len(models) + 3))
    clean_row = "| clean | | " + " | ".join(
        f"{100 * st.mean(st.mean(bits(data[m], t, 'clean', a.metric)) for t in tasks):.1f}" for m, _ in models) + " | |"
    L.append(clean_row)
    for v in V[1:]:
        cells, chg = [], None
        for m, _ in models:
            r = macro_delta(data[m], v, a.metric, tasks, B=a.B)
            if r is None:
                cells.append("–")
                continue
            out["macro"].setdefault(m, {})[v] = dict(zip(["clean", "pert", "delta", "lo", "hi", "changed", "delta_changed"], r))
            cells.append(f"{pp(r[2])} [{pp(r[3])},{pp(r[4])}]")
            chg = r[5]
        L.append(f"| {v} | {F.family_of(v)} | " + " | ".join(cells) + f" | {'' if chg is None else f'{chg:.2f}'} |")
    # FormatSpread spread
    L.append("\n## FormatSpread: accuracy spread over the 10 sampled formats (+ default)\n")
    L.append("| model | default | mean | min | max | spread (max−min) |")
    L.append("|---|---|---|---|---|---|")
    fsv = [v for v in V if v.startswith("fs")]
    for m, n in models:
        accs = []
        for v in ["clean"] + fsv:
            b = [bits(data[m], t, v, a.metric) for t in tasks]
            if all(x is not None for x in b):
                accs.append(st.mean(st.mean(x) for x in b))
        if len(accs) == len(fsv) + 1:
            out.setdefault("formatspread", {})[m] = {"default": accs[0], "formats": accs[1:]}
            L.append(f"| {n} | {100 * accs[0]:.1f} | {100 * st.mean(accs[1:]):.1f} | {100 * min(accs):.1f} | "
                     f"{100 * max(accs):.1f} | {100 * (max(accs) - min(accs)):.1f} |")
    L.append("\nFormats: " + "; ".join(f"`fs{i:02d}`={tuple(f)!r}" for i, f in enumerate(F.formatspread_formats())))
    # per-task deltas for the frequency sweep
    L.append("\n## Per-task Δ (pt)\n")
    for m, n in models:
        L.append(f"\n### {n}\n")
        L.append("| variant | " + " | ".join(TNAME[t] for t in tasks) + " |")
        L.append("|" + "---|" * (len(tasks) + 1))
        L.append("| clean acc | " + " | ".join(
            f"{100 * st.mean(bits(data[m], t, 'clean', a.metric)):.1f}" if bits(data[m], t, 'clean', a.metric) else "–"
            for t in tasks) + " |")
        for v in V[1:]:
            cells = []
            for t in tasks:
                c, p = bits(data[m], t, "clean", a.metric), bits(data[m], t, v, a.metric)
                cells.append("–" if c is None or p is None else pp(st.mean(p) - st.mean(c)))
                out["per_task"].setdefault(m, {}).setdefault(v, {})[t] = None if c is None or p is None else st.mean(p) - st.mean(c)
            L.append(f"| {v} | " + " | ".join(cells) + " |")
    os.makedirs(a.out_dir, exist_ok=True)
    sfx = "" if a.metric == "acc" else f"_{a.metric}"
    open(f"{a.out_dir}/summary{sfx}.md", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{a.out_dir}/summary{sfx}.json", "w"), indent=1)
    print("\n".join(L[:50]))


if __name__ == "__main__":
    main()
