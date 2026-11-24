#!/usr/bin/env python3
"""Patch statistics of the AU-Net and BPEByte parsers on the FLORES source sentences.

Motivation: on FLORES X->en, BPEByte greedy_root beats AU-Net2 by +1.24 / +1.12 Latin-5 BLEU
(dev / devtest), and the gap is far from uniform across languages -- largest on German, then
Dutch, smallest on Spanish and French. German and Dutch are the two compounding Germanic
languages in the set, which suggests a boundary explanation: AU-Net pools on whitespace, so a
compound noun becomes ONE long patch, while BPEByte's BPE-trie walk splits it.

This measures exactly that, on the same 200 dev sentences per language that the eval scored,
using the parsers from scripts/probes/compression_stability.py (CPU only).
"""
import json, math, os, statistics as st, sys
from pathlib import Path

ROOT = Path("/mnt/ssd2/hyun2/AUNet")
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))
from compression_stability import build_parser  # noqa: E402

# Ordered by morphological type so the table reads as an axis, not a list:
#   compounding Germanic -> agglutinative -> fusional -> Romance -> isolating -> no-space scripts
LANGS = {"de": "sentence_deu_Latn", "nl": "sentence_nld_Latn", "sv": "sentence_swe_Latn",
         "fi": "sentence_fin_Latn", "tr": "sentence_tur_Latn", "hu": "sentence_hun_Latn",
         "ru": "sentence_rus_Cyrl", "it": "sentence_ita_Latn", "fr": "sentence_fra_Latn",
         "es": "sentence_spa_Latn", "vi": "sentence_vie_Latn", "ko": "sentence_kor_Hang"}
# Latin-5 BLEU deltas (BPEByte - AU-Net2) from reports_afterAAAIsub/flores/results.jsonl
DELTA = {"de": (2.36, 3.47), "fr": (0.56, 0.89), "es": (0.19, -0.71),
         "it": (1.71, 0.82), "nl": (1.40, 1.11), "ko": (0.44, 0.55)}   # measured; new langs: n/a


def stats(parser, texts):
    lens = []
    for t in texts:
        nb = len(t.encode())
        s = parser(t)
        lens += [b - a for a, b in zip(s, s[1:] + [nb])]
    lens = [x for x in lens if x > 0]
    lens_sorted = sorted(lens)
    p = lambda q: lens_sorted[min(len(lens_sorted) - 1, int(q * len(lens_sorted)))]
    return {"bytes_per_patch": sum(lens) / len(lens), "std": st.pstdev(lens),
            "p95": p(0.95), "max": max(lens),
            "pct_bytes_in_gt12": 100 * sum(x for x in lens if x > 12) / sum(lens)}


def main():
    from datasets import load_dataset
    ds = load_dataset("facebook/flores", "all")["dev"]
    n = int(os.environ.get("N", "200"))
    parsers = {"aunet": build_parser("aunet"), "bpebyte": build_parser("bpebyte")}
    print(f"FLORES dev, first {n} sentences per language, source side\n")
    print(f"{'lang':5s} {'AU-Net B/patch':>14s} {'BPEByte B/patch':>15s} {'ratio':>6s} "
          f"{'AUNet p95':>9s} {'BPE p95':>8s} {'AUNet %>12B':>11s} {'BPE %>12B':>10s} {'ΔBLEU dev/dt':>13s}")
    rows = []
    for lg, col in LANGS.items():
        texts = ds[col][:n]
        a, b = stats(parsers["aunet"], texts), stats(parsers["bpebyte"], texts)
        r = a["bytes_per_patch"] / b["bytes_per_patch"]
        d = DELTA.get(lg, (float("nan"), float("nan")))
        print(f"{lg:5s} {a['bytes_per_patch']:14.2f} {b['bytes_per_patch']:15.2f} {r:6.2f} "
              f"{a['p95']:9d} {b['p95']:8d} {a['pct_bytes_in_gt12']:10.1f}% {b['pct_bytes_in_gt12']:9.1f}% "
              f"{d[0]:+6.2f}/{d[1]:+5.2f}")
        rows.append({"lang": lg, "aunet": a, "bpebyte": b, "ratio": r, "delta_bleu": d})
    out = ROOT / "reports_afterAAAIsub/flores/patch_stats.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nWROTE {out}")
    lat = [r for r in rows if r["lang"] != "ko"]
    xs = [r["ratio"] for r in lat]
    for i, lbl in ((0, "dev"), (1, "devtest")):
        ys = [r["delta_bleu"][i] for r in lat]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        print(f"Pearson r(patch-length ratio, ΔBLEU {lbl}) over Latin-5 = {num/den:+.3f}")


if __name__ == "__main__":
    main()
