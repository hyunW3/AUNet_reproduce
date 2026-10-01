#!/usr/bin/env python3
"""Aggregate long-context suite results -> reports/longctx/{summary.md, *.png}.

Reads results/<file>_<tag>.jsonl from run_longctx.py (one row per item) and
reports mean score with a percentile-bootstrap 95% CI per cell.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = [("subword_llama", "Llama (subword)"), ("aunet_static", "AU-Net"),
          ("byte_greedyroot", "BPEByte-rg")]
COLOR = {"subword_llama": "#2a78d6", "aunet_static": "#eb6834", "byte_greedyroot": "#1baf7a"}
MARK = {"subword_llama": "o", "aunet_static": "s", "byte_greedyroot": "^"}


def ci(xs, n_boot=2000, seed=0):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    m = sum(xs) / len(xs)
    bs = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return m, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]


def load(res_dir):
    rows, seen = [], set()
    for f in sorted(glob.glob(f"{res_dir}/*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if (r["tag"], r["id"]) not in seen:   # a shard re-run on another node
                    seen.add((r["tag"], r["id"]))
                    rows.append(r)
    return rows


def cell(rows, **kw):
    return [r["score"] for r in rows if all(r.get(k) == v for k, v in kw.items())]


def fmt(xs):
    if not xs:
        return "—"
    m, lo, hi = ci(xs)
    return f"{100 * m:.1f} [{100 * lo:.0f}–{100 * hi:.0f}]"


def table(rows, task, cond_key, conds, models, extra=None, **kw):
    hdr = "| " + cond_key + " | " + " | ".join(lbl for _, lbl in models) + (" | " + extra[0] if extra else "") + " |"
    out = [hdr, "|" + "---|" * (len(models) + 1 + (1 if extra else 0))]
    for c in conds:
        line = f"| {c} | " + " | ".join(fmt(cell(rows, task=task, tag=t, **{cond_key: c}, **kw)) for t, _ in models)
        if extra:
            line += f" | {extra[1](c)}"
        out.append(line + " |")
    return "\n".join(out)


def style(ax, title, xlabel, ylabel="score (%)"):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylim(-3, 103)


def line_panel(ax, rows, models, xs, xkey, logx=False, **kw):
    for t, lbl in models:
        pts = [(x, ci(cell(rows, tag=t, **{xkey: x}, **kw))) for x in xs]
        pts = [(x, c) for x, c in pts if c[0] == c[0]]
        if not pts:
            continue
        X = [x for x, _ in pts]
        ax.plot(X, [100 * c[0] for _, c in pts], color=COLOR[t], lw=2, marker=MARK[t], ms=6, label=lbl)
        ax.fill_between(X, [100 * c[1] for _, c in pts], [100 * c[2] for _, c in pts],
                        color=COLOR[t], alpha=0.12, lw=0)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])


def llama_tok_counter(lingua_dir, tok_path):
    try:
        import sys
        sys.path.insert(0, lingua_dir)
        from lingua.tokenizer import build_tokenizer
        tok = build_tokenizer("tiktoken", tok_path)
        return lambda s: len(tok.encode(s, add_bos=False, add_eos=False))
    except Exception:
        return None


def copy_baseline(data_dir, task, length, k):
    """Recall a trivial copier gets by emitting the first k distinct words of the scored
    list (CWE/FWE). If the models do not beat it, the task is not measuring aggregation."""
    rec = []
    for line in open(f"{data_dir}/ruler_agg.jsonl"):
        r = json.loads(line)
        if r["task"] != task or r["length"] != length:
            continue
        body = r["prompt"].split("\n\n")[-1].split("\nQuestion:")[0]
        if task == "cwe":
            words = [w.split(". ", 1)[1] for w in
                     __import__("re").findall(r"\d+\. [a-z]+", body)]
        else:
            words = [w for w in body.split("coded words. ", 1)[1].split() if w != "...."]
        first = []
        for w in words:
            if w not in first:
                first.append(w)
            if len(first) == k:
                break
        rec.append(sum(a in first for a in r["answers"]) / len(r["answers"]))
    return 100 * sum(rec) / len(rec) if rec else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/longctx/results")
    ap.add_argument("--data", default="data/longctx")
    ap.add_argument("--out", default="reports/longctx")
    ap.add_argument("--lingua", default="lingua", help="lingua checkout (for the Llama-3 token counts)")
    ap.add_argument("--tok", default="tokenizer/llama3/tokenizer.model")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(a.results)
    tags = {r["tag"] for r in rows}
    models = [(t, l) for t, l in MODELS if t in tags]
    md = ["# Long-context suite (1.3B: Llama / AU-Net / BPEByte-rg)", "",
          "Greedy free generation scored on the first `len(answer)+24` generated BYTES "
          "(identical for every family); ICL SST-2/TREC-coarse/AG News by label log-likelihood. "
          "Cells: mean % [bootstrap 95% CI]. Prompts ≤ 7.4 KB so no byte-model truncation; rows "
          "that exceed either byte model's 3200-patch trunk window are dropped for all models "
          "(KV k≤60, 2 FWE@6144 rows).", ""]
    n_by = collections.Counter((r["tag"], r["task"]) for r in rows)
    md += ["Rows scored: " + ", ".join(f"{t}/{k}={v}" for (t, k), v in sorted(n_by.items())), ""]

    # --- 1. needle types ---------------------------------------------------
    vts = ["num7", "num20", "uuid", "hex32", "alnum12", "ident", "nonce", "word", "han4", "hangul4"]
    count = llama_tok_counter(a.lingua, a.tok)
    vt_tok = {}
    if count and os.path.exists(f"{a.data}/needle_types.jsonl"):
        acc = collections.defaultdict(list)
        for l in open(f"{a.data}/needle_types.jsonl"):
            r = json.loads(l)
            acc[r["cond"]].append(count(" " + r["answers"][0]))
        vt_tok = {k: sum(v) / len(v) for k, v in acc.items()}
    md += ["## 1. Needle value-type sweep (S-NIAH, DCLM essay haystack, 1–6 KB, 5 depths)", "",
           table(rows, "needle_types", "cond", vts, models,
                 extra=("Llama tok/value", lambda c: f"{vt_tok.get(c, float('nan')):.1f}")), ""]
    lens = [1024, 2048, 4096, 6144]
    md += ["Per length (all value types pooled):", "",
           table(rows, "needle_types", "length", lens, models), ""]
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharey=True)
    for ax, vt in zip(axes.flat, vts):
        line_panel(ax, rows, models, lens, "length", task="needle_types", cond=vt)
        style(ax, vt, "haystack bytes")
        ax.set_xticks(lens)
        ax.set_xticklabels(["1K", "2K", "4K", "6K"])
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Needle retrieval by value type (greedy gen, substring in first len+24 bytes)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "needle_types.png", dpi=150)
    plt.close(fig)

    # --- 2. KV lost-in-the-middle -----------------------------------------
    ks = [5, 10, 25, 50, 60]
    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    md += ["## 2. Lost-in-the-Middle KV retrieval (JSON of k UUID→UUID pairs)", "",
           "By #pairs (all gold positions pooled):", "",
           table(rows, "kv_litm", "length", ks, models), "",
           "By gold position, k ≥ 25 pooled:", ""]
    kvbig = [r for r in rows if r["task"] == "kv_litm" and r["length"] >= 25]
    md += [table(kvbig, "kv_litm", "pos", pos, models), ""]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    line_panel(axes[0], rows, models, ks, "length", task="kv_litm")
    style(axes[0], "KV retrieval vs #pairs", "#pairs k")
    line_panel(axes[1], kvbig, models, pos, "pos", task="kv_litm")
    style(axes[1], "KV retrieval vs gold position (k≥25)", "relative position of gold pair")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "kv_litm.png", dpi=150)
    plt.close(fig)

    # --- 3. RULER aggregation ---------------------------------------------
    md += ["## 3. RULER VT / CWE / FWE (recall of the gold set in the answer window)", ""]
    panels = [("vt", "c1h4", lens, "VT 1 chain × 4 hops"), ("vt", "c2h4", lens, "VT 2 chains × 4 hops"),
              ("cwe", "cwe", lens[1:], "CWE top-10 common words"), ("fwe", "fwe", lens[1:], "FWE top-3 frequent coded words")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (task, cond, L, title) in zip(axes, panels):
        if task in ("cwe", "fwe") and os.path.exists(f"{a.data}/ruler_agg.jsonl"):
            k = 10 if task == "cwe" else 3
            base = {x: copy_baseline(a.data, task, x, k) for x in L}
            md += [f"**{title}**", "", table(rows, task, "length", L, models, cond=cond,
                                             extra=(f"copy-first-{k} baseline", lambda x, b=base: f"{b[x]:.1f}")), ""]
            ax.plot(L, [base[x] for x in L], color="#8a8a85", lw=1.5, ls="--", label=f"copy-first-{k}")
        else:
            md += [f"**{title}**", "", table(rows, task, "length", L, models, cond=cond), ""]
        line_panel(ax, rows, models, L, "length", task=task, cond=cond)
        style(ax, title, "prompt bytes")
        ax.set_xticks(L)
        ax.set_xticklabels([f"{x // 1024}K" for x in L])
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes[2:]:
        h = [l for l in ax.get_lines() if l.get_label().startswith("copy-first")]
        ax.legend(handles=h, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ruler_agg.png", dpi=150)
    plt.close(fig)

    # --- 4. many-shot ICL --------------------------------------------------
    shots = [1, 2, 4, 8, 16, 32, 64]
    md += ["## 4. Many-shot ICL (accuracy vs #shots; identical prompts for all models)", ""]
    sets = [("sst2", "SST-2 (2-way, rank)"), ("trec_coarse", "TREC coarse (6-way, rank)"),
            ("agnews", "AG News (4-way, rank)"), ("trec_fine", "TREC fine (50-way, gen exact)")]
    ptoks = {}
    if count and os.path.exists(f"{a.data}/icl.jsonl"):
        acc = collections.defaultdict(list)
        for l in open(f"{a.data}/icl.jsonl"):
            r = json.loads(l)
            if r["id"].split("-")[2] in ("0", "1", "2", "3", "4"):   # 5 items per cell suffice for a mean
                acc[(r["cond"], r["length"])].append((r["prompt_bytes"], count(r["prompt"])))
        ptoks = {k: (sum(b for b, _ in v) / len(v), sum(t for _, t in v) / len(v)) for k, v in acc.items()}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (ds, title) in zip(axes, sets):
        S = [k for k in shots if any(r["task"] == "icl" and r["cond"] == ds and r["length"] == k for r in rows)]
        md += [f"**{title}**", "",
               table(rows, "icl", "length", S, models, cond=ds,
                     extra=("prompt B / Llama tok", lambda k, ds=ds: "{:.0f} / {:.0f}".format(*ptoks.get((ds, k), (float('nan'),) * 2)))), ""]
        line_panel(ax, rows, models, S, "length", logx=True, task="icl", cond=ds)
        style(ax, title, "#shots")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "icl.png", dpi=150)
    plt.close(fig)

    md += ["Figures: `needle_types.png`, `kv_litm.png`, `ruler_agg.png`, `icl.png`.", ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
