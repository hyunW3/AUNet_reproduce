#!/usr/bin/env python3
"""Experiment E: representation locality at the global transformer's input.

Two regimes:

  --regime single   one character edit per example. Cosine similarity between the clean and
                    perturbed trunk-input vectors, resolved by distance from the edit. This is what
                    "the damage stays confined" has to mean operationally.
  --regime suite    the paper's own corruption families (despace / typo / noise) applied to the
                    whole span, which is the condition the downstream robustness numbers are
                    measured under. Whole-sequence summary, no distance axis.

Comparability. Cosine similarities are not on a common scale across architectures, so every number
is also reported normalised against each model's OWN null -- the mean cosine between two unrelated
slots of the same clean sequence:

    sim_norm = (cos - null) / (1 - null)        1 = identical, 0 = as unrelated as two random units

Alongside it, a purely discrete measure that needs no normalisation at all: the Levenshtein
distance between the clean and perturbed UNIT sequences (token ids for subword, patch byte-strings
for the byte arms), normalised by the clean unit count.

    python analyze_repr.py --regime single --arm bpebyte --ckpt <consolidated> --out <jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lingua"))

MAXD = 256
BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 10 ** 9)]
_WS_RE = re.compile(r"[ \t]+")


# ---------------------------------------------------------------- corruption suite
def suite_fns():
    """The paper's families, imported rather than re-implemented (eval_typo / eval_robust_aligned)."""
    from apps.aunet.eval_typo import _CHAR_FNS
    from apps.aunet.eval_robust_aligned import _NOISE_FNS
    fns = {f"typo_{k}": v for k, v in _CHAR_FNS.items()}
    fns.update({f"noise_{k}": v for k, v in _NOISE_FNS.items()})
    fns["despace"] = lambda t, rng: "\n".join(_WS_RE.sub("", ln) for ln in t.split("\n"))
    return fns


def byte_align(a: bytes, b: bytes) -> np.ndarray:
    """align[i] = index of clean byte i in `b`, or -1 when it was replaced or deleted."""
    out = np.full(len(a), -1, dtype=np.int64)
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            out[i1:i2] = np.arange(j1, j2)
    return out


# ---------------------------------------------------------------- matching + metrics
def slot_of_byte(ref: np.ndarray, n_bytes: int) -> np.ndarray:
    """For every raw byte, the index of the FIRST trunk-input vector whose causal read position
    is >= that byte -- i.e. the first thing the global transformer holds that contains it.

    One rule for both families, which is what makes the comparison honest. For the subword arm
    `ref` is each token's last byte, so this resolves to the token containing the byte; for the
    byte arms it is the pooled read position, so it resolves to the patch vector that closes over
    the byte. Comparing in BYTE coordinates (design 1.2 / 8) is also what keeps a changed unit in
    the measurement: matching units by span identity would silently drop exactly the units the
    perturbation destroyed, which are the ones the claim is about.
    """
    s = np.searchsorted(ref, np.arange(n_bytes), side="left")
    return np.minimum(s, len(ref) - 1)


def cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    b = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return (a * b).sum(1)


def center(Hc: np.ndarray, *Hs):
    """Subtract the clean sequence's mean vector from every representation.

    Byte-encoder states share a large common component (mean random-pair cosine ~0.44 here)
    while subword embeddings are nearly orthogonal (~0.05), so a raw cosine is not comparable
    across the two. Centering removes the shared offset and leaves the part that carries which
    unit this is. The SAME centre (from the clean run) is applied to the perturbed run, so the
    comparison is not shifted by the perturbation itself.
    """
    mu = Hc.mean(0, keepdims=True)
    return tuple(H - mu for H in (Hc,) + Hs)


def null_cos(H: np.ndarray, rng: np.random.Generator, n: int = 256) -> float:
    """Mean cosine between two UNRELATED slots of the same sequence -- the model's own floor."""
    m = len(H)
    if m < 4:
        return 0.0
    i = rng.integers(0, m, n)
    j = rng.integers(0, m, n)
    k = i != j
    return float(cos_rows(H[i[k]], H[j[k]]).mean()) if k.any() else 0.0


def unit_symbols(arm: str, text: str, seg) -> list:
    st = seg(text)
    b = text.encode()
    ends = list(st[1:]) + [len(b)]
    return [b[s:e] for s, e in zip(st, ends)]


def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["llama", "aunet", "bpebyte"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regime", default="single", choices=["single", "suite"])
    ap.add_argument("--domain", default="en")
    ap.add_argument("--span", type=int, default=640)
    ap.add_argument("--sites", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--families", nargs="*", default=None)
    a = ap.parse_args()

    import torch
    import perturb as P
    from dump import load_windows
    from segmenters import build
    from repr_extract import ReprExtractor, ARM_FAMILY

    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed)
    wins = load_windows(a.domain, a.limit)
    seg = build(a.arm)
    fam_model = ARM_FAMILY[a.arm]

    # ---- build the example set deterministically, identical for every arm
    ex = []
    if a.regime == "single":
        fams = a.families or ["sub_key", "insert", "delete", "swap", "ws_delete", "case_flip"]
        for wi, w in enumerate(wins):
            cs = P._char_starts(w["text"])
            if int(cs[-1]) < a.span + 16:
                continue
            for _ in range(a.sites):
                fam = rng.choice(fams)
                c0 = rng.randrange(0, len(w["text"]) - 1)
                b0 = int(cs[c0])
                if b0 + a.span > int(cs[-1]):
                    continue
                c1 = int(np.searchsorted(cs, b0 + a.span))
                span = w["text"][c0:c1]
                sites = P.sample_edits(span, fam, 8, rng, margin=32)
                if not sites:
                    continue
                r = P.apply_edit(span, fam, rng.choice(sites), rng)
                if r is None:
                    continue
                pert, align, touched = r
                ex.append({"window": wi, "family": fam, "clean": span, "pert": pert,
                           "align": align, "edit_lo": int(touched[0]), "edit_hi": int(touched[-1])})
    else:
        FN = suite_fns()
        fams = a.families or sorted(FN)
        for wi, w in enumerate(wins):
            cs = P._char_starts(w["text"])
            if int(cs[-1]) < a.span + 16:
                continue
            c1 = int(np.searchsorted(cs, a.span))
            span = w["text"][:c1]
            for fam in fams:
                r = random.Random(hash((a.seed, wi, fam)) & 0xFFFFFFFF)
                pert = FN[fam](span, r)
                if pert == span or not pert.strip():
                    continue
                ex.append({"window": wi, "family": fam, "clean": span, "pert": pert,
                           "align": None, "edit_lo": None, "edit_hi": None})
    print(f"{len(ex)} examples ({a.regime})", flush=True)

    extractor = ReprExtractor(a.arm, a.ckpt)
    rows = []
    agg_sim = np.zeros(MAXD + 1)
    agg_n = np.zeros(MAXD + 1)
    # Second curve for the subword arm only: its hidden state after as many layers as the byte
    # arms' local encoder. Without it the comparison is open to "you put a lookup table against a
    # 3-layer encoder"; with it, any remaining gap is about discrete symbols, not depth.
    agg_sim_d = np.zeros(MAXD + 1)
    agg_n_d = np.zeros(MAXD + 1)
    for i, e in enumerate(ex):
        cl = extractor.extract(e["clean"])
        pe = extractor.extract(e["pert"])
        align = e["align"] if e["align"] is not None else byte_align(e["clean"].encode(),
                                                                    e["pert"].encode())
        nb_c = len(e["clean"].encode())
        sc_idx = slot_of_byte(cl["ref"], nb_c)
        sp_idx = slot_of_byte(pe["ref"], len(e["pert"].encode()))
        ok = align >= 0
        if len(align) > nb_c:
            ok = ok[:nb_c]
            align_c = align[:nb_c]
        else:
            align_c = align
        bi = np.flatnonzero(ok)
        null = null_cos(cl["H"], nrng)
        row = {"arm": a.arm, "window": e["window"], "family": e["family"],
               "n_units_clean": int(len(cl["H"])), "n_units_pert": int(len(pe["H"])),
               "n_bytes": nb_c, "n_aligned_bytes": int(len(bi)), "null_cos": null}
        # discrete: unit-sequence edit distance (needs no normalisation)
        sc = unit_symbols(a.arm, e["clean"], seg)
        sp = unit_symbols(a.arm, e["pert"], seg)
        row["unit_levenshtein"] = int(levenshtein(sc, sp))
        row["unit_levenshtein_norm"] = row["unit_levenshtein"] / max(len(sc), 1)

        if len(bi):
            Hc, Hp = center(cl["H"], pe["H"])
            null_c = null_cos(Hc, nrng)
            c = cos_rows(cl["H"][sc_idx[bi]], pe["H"][sp_idx[align_c[bi]]])
            cc = cos_rows(Hc[sc_idx[bi]], Hp[sp_idx[align_c[bi]]])
            cn = cc                                   # centred cosine is the headline
            row["null_cos_centered"] = null_c
            row["cos_mean"] = float(c.mean())
            row["cos_centered_mean"] = float(cc.mean())
            row["sim_norm_mean"] = float(((c - null) / max(1.0 - null, 1e-9)).mean())
            row["frac_bytes_below_095"] = float((cn < 0.95).mean())
            if e["edit_lo"] is not None:
                d = np.where(bi < e["edit_lo"], e["edit_lo"] - bi,
                             np.maximum(bi - e["edit_hi"], 0))
                d = np.minimum(d, MAXD)
                np.add.at(agg_sim, d, cn)
                np.add.at(agg_n, d, 1.0)
                far = d > 8
                row["sim_norm_gt8"] = float(cn[far].mean()) if far.any() else None
                bad = cn < 0.95
                row["repr_radius_095"] = int(d[bad].max()) if bad.any() else 0
                row["n_bytes_below_095"] = int(bad.sum())
            if "Hdepth" in cl and "Hdepth" in pe:
                Hcd, Hpd = center(cl["Hdepth"], pe["Hdepth"])
                ccd = cos_rows(Hcd[sc_idx[bi]], Hpd[sp_idx[align_c[bi]]])
                row["cos_depth_centered_mean"] = float(ccd.mean())
                row["frac_bytes_depth_below_095"] = float((ccd < 0.95).mean())
                if e["edit_lo"] is not None:
                    np.add.at(agg_sim_d, d, ccd)
                    np.add.at(agg_n_d, d, 1.0)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ex)}", flush=True)
        del cl, pe
    del extractor
    torch.cuda.empty_cache()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if agg_n.sum():
        curve = (agg_sim / np.maximum(agg_n, 1))[:MAXD + 1]
        np.save(Path(a.out).with_suffix(".curve.npy"), np.vstack([curve, agg_n]))
        if agg_n_d.sum():
            np.save(Path(a.out).with_suffix(".curve_depth.npy"),
                    np.vstack([(agg_sim_d / np.maximum(agg_n_d, 1))[:MAXD + 1], agg_n_d]))
        binned = [float(agg_sim[lo:min(hi, MAXD) + 1].sum() /
                        max(agg_n[lo:min(hi, MAXD) + 1].sum(), 1)) for lo, hi in BINS]
        print("  sim_norm by distance: " +
              "  ".join(f"{lo}-{hi if hi < 10**9 else ''}:{v:.3f}" for (lo, hi), v in zip(BINS, binned)))
    lev = np.array([r["unit_levenshtein_norm"] for r in rows])
    print(f"[{a.arm}/{a.regime}] n={len(rows)}  unit-Lev/unit={lev.mean():.4f}  "
          f"null_cos={np.mean([r['null_cos'] for r in rows]):.3f}  -> {a.out}")


if __name__ == "__main__":
    main()
