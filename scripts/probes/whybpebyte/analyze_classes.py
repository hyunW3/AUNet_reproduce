#!/usr/bin/env python3
"""Experiment C: what kinds of byte positions become boundaries?

Design section 6 -- diagnostic, not a superiority claim. For each arm we report P(class | B=1)
and the enrichment E(c) = P(c | B=1) / P(c) over the corpus frequency of the class, so a class
that is simply common does not look informative.

    python analyze_classes.py --domain en
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
CACHE = ROOT / "reports_afterAAAIsub/whybpebyte/cache"
RES = ROOT / "reports_afterAAAIsub/whybpebyte/results"

CLASSES = ["space", "newline", "word_initial", "word_interior", "digit",
           "punct", "code_delim", "utf8_lead", "utf8_cont", "other"]
_PUNCT = set(b".,!?;:'\"-")          # sentence punctuation
_CODE = set(b"(){}[]<>=+*/\\|&%#@$~^_`")   # bracket / operator delimiters


def classify(raw: np.ndarray) -> np.ndarray:
    """One class index per byte (mutually exclusive, in the order of CLASSES)."""
    n = len(raw)
    cls = np.full(n, CLASSES.index("other"), dtype=np.int8)
    b = raw.astype(np.int64)
    is_cont = (b >= 0x80) & (b < 0xC0)
    is_lead = b >= 0xC0
    is_alpha = ((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122))
    is_digit = (b >= 48) & (b <= 57)
    is_space = (b == 0x20) | (b == 0x09) | (b == 0x0D)
    is_nl = b == 0x0A
    is_punct = np.isin(b, np.fromiter(_PUNCT, dtype=np.int64))
    is_code = np.isin(b, np.fromiter(_CODE, dtype=np.int64))

    prev_alpha = np.concatenate(([False], is_alpha[:-1]))
    cls[is_alpha & ~prev_alpha] = CLASSES.index("word_initial")
    cls[is_alpha & prev_alpha] = CLASSES.index("word_interior")
    cls[is_digit] = CLASSES.index("digit")
    cls[is_punct] = CLASSES.index("punct")
    cls[is_code] = CLASSES.index("code_delim")
    cls[is_space] = CLASSES.index("space")
    cls[is_nl] = CLASSES.index("newline")
    cls[is_lead] = CLASSES.index("utf8_lead")
    cls[is_cont] = CLASSES.index("utf8_cont")
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="en")
    a = ap.parse_args()

    from dump import load_windows, unpack
    from segmenters import starts_to_mask
    wins = load_windows(a.domain)
    seg = np.load(CACHE / f"seg_{a.domain}.npz")
    arms = [k[len("starts_"):] for k in seg.files if k.startswith("starts_")]

    corpus = np.zeros(len(CLASSES))
    per_arm = {arm: np.zeros(len(CLASSES)) for arm in arms}
    for wi, w in enumerate(wins):
        raw = np.frombuffer(w["text"].encode(), dtype=np.uint8)
        n = len(raw)
        c = classify(raw)
        np.add.at(corpus, c[1:], 1.0)
        for arm in arms:
            st = unpack(seg[f"starts_{arm}"], seg[f"offs_{arm}"])[wi]
            m = starts_to_mask(st, n)
            m[0] = False
            np.add.at(per_arm[arm], c[m], 1.0)

    pc = corpus / corpus.sum()
    out = {"domain": a.domain, "classes": CLASSES, "corpus_freq": pc.tolist(), "arms": {}}
    for arm in arms:
        p = per_arm[arm] / per_arm[arm].sum()
        out["arms"][arm] = {"p_given_boundary": p.tolist(),
                            "enrichment": np.where(pc > 0, p / np.maximum(pc, 1e-12), np.nan).tolist(),
                            "n_boundaries": int(per_arm[arm].sum())}
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / f"classes_{a.domain}.json", "w"), indent=1)

    w = max(len(c) for c in CLASSES) + 1
    print(f"{'class':{w}s} {'corpus':>7s} " + " ".join(f"{arm:>17s}" for arm in arms))
    for i, c in enumerate(CLASSES):
        cells = " ".join(f"{out['arms'][arm]['p_given_boundary'][i]*100:8.2f}%"
                         f"{out['arms'][arm]['enrichment'][i]:8.2f}x" for arm in arms)
        print(f"{c:{w}s} {pc[i]*100:6.2f}% {cells}")
    print(f"-> {RES}/classes_{a.domain}.json")


if __name__ == "__main__":
    main()
