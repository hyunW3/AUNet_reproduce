#!/usr/bin/env python3
"""Per-window structural metrics over cached probe surprisal + boundary masks.

Everything is computed as per-window SUFFICIENT STATISTICS (sums and counts) so the corpus
aggregate is a ratio of sums and a document-level bootstrap is a resample of rows -- design
section 12 ("bootstrap over documents/examples, not individual byte positions").

Byte 0 is excluded from every statistic: it is a forced boundary for every arm (and for every
control), and it is the one byte the probe scores with no left context, so including it would add
the same large constant to every arm's boundary mean.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

PEAK_KS = (2, 4, 8)
SHIFTS = tuple(range(-8, 9))   # offset-swept boundary gap; see window_stats
WS_BYTES = frozenset((0x20, 0x09, 0x0A, 0x0D))
PROFILE_K = 8          # boundary-centred window, offsets -K..+K
REL_BINS = 10          # within-unit normalised-position bins
ABS_OFFSETS = 8        # within-unit absolute offsets 0..ABS_OFFSETS-1


def window_stats(s: np.ndarray, mask: np.ndarray, raw: np.ndarray = None) -> Dict[str, float]:
    """Sufficient statistics for one window. `s` bits/byte, `mask` bool patch-start mask.

    `raw` (uint8 bytes) enables the whitespace stratification. It is needed because every
    production scheme here is SPACE-LEADING -- a llama3/AU-Net/OnlineBPE unit is " word", not
    "word " -- so the byte at offset 0 of a unit is usually the separating space, the single most
    predictable byte in English text. Taking the design's Delta S literally therefore reads the
    valley rather than the peak. Two corrections are recorded alongside it: the offset-swept gap
    Delta S(delta) (the peak lands at delta=+1) and this whitespace-excluded split.
    """
    n = len(s)
    m = mask.copy()
    m[0] = True
    valid = np.ones(n, bool)
    valid[0] = False                       # drop the forced, context-free first byte

    b = m & valid
    i_ = (~m) & valid
    out = {
        "n_bytes": float(n),
        "n_units": float(m.sum()),
        "nb": float(b.sum()), "sb": float(s[b].sum()),
        "ni": float(i_.sum()), "si": float(s[i_].sum()),
    }

    # ---- offset-swept boundary gap: Sb(d) = E[s_{i+d} | B_i], Si(d) = E[s_{j+d} | not B_j]
    for d in SHIFTS:
        lo, hi = max(0, -d), min(n, n - d)
        v = np.zeros(n, bool); v[lo:hi] = True; v[0] = False
        bb, ii2 = m & v, (~m) & v
        out[f"nb{d:+d}"] = float(bb.sum()); out[f"sb{d:+d}"] = float(s[np.flatnonzero(bb) + d].sum())
        out[f"ni{d:+d}"] = float(ii2.sum()); out[f"si{d:+d}"] = float(s[np.flatnonzero(ii2) + d].sum())

    # ---- whitespace stratification (see docstring)
    if raw is not None:
        ws = np.isin(raw, np.fromiter(WS_BYTES, dtype=np.uint8))
        out["nb_ws"] = float((b & ws).sum())
        nb_, ni_ = b & ~ws, i_ & ~ws
        out["nb_nws"] = float(nb_.sum()); out["sb_nws"] = float(s[nb_].sum())
        out["ni_nws"] = float(ni_.sum()); out["si_nws"] = float(s[ni_].sum())

    # ---- unit coherence C(u) = s_start - mean(interior)   (units of length >= 2)
    starts = np.flatnonzero(m)
    ends = np.append(starts[1:], n)
    lens = ends - starts
    keep = (lens >= 2) & (starts > 0)      # unit at byte 0 has a context-free head
    if keep.any():
        cs = np.concatenate(([0.0], np.cumsum(s, dtype=np.float64)))
        st, en, ln = starts[keep], ends[keep], lens[keep]
        interior_mean = (cs[en] - cs[st + 1]) / (ln - 1)
        c = s[st] - interior_mean
        out["n_units_c"] = float(len(c))
        out["sum_c_unit"] = float(c.sum())
        out["sum_c_byte"] = float((c * ln).sum())
        out["sum_len_c"] = float(ln.sum())
        # length-conditioned coherence, binned
        for lo, hi, tag in ((2, 2, "L2"), (3, 4, "L34"), (5, 8, "L58"),
                            (9, 16, "L916"), (17, 10 ** 9, "L17p")):
            sel = (ln >= lo) & (ln <= hi)
            out[f"n_c_{tag}"] = float(sel.sum())
            out[f"sum_c_{tag}"] = float(c[sel].sum())
    else:
        out["n_units_c"] = out["sum_c_unit"] = out["sum_c_byte"] = out["sum_len_c"] = 0.0

    # ---- boundary peak score P_b (surprisal minus the local +-k mean, boundary byte excluded)
    cs = np.concatenate(([0.0], np.cumsum(s, dtype=np.float64)))
    for k in PEAK_KS:
        idx = starts[(starts >= k) & (starts < n - k)]
        if idx.size:
            tot = cs[idx + k + 1] - cs[idx - k]        # sum over [i-k, i+k]
            neigh = (tot - s[idx]) / (2 * k)
            p = s[idx] - neigh
            out[f"n_peak{k}"] = float(len(p))
            out[f"sum_peak{k}"] = float(p.sum())
        else:
            out[f"n_peak{k}"] = out[f"sum_peak{k}"] = 0.0

    # ---- surprisal-regime change D_i = |mean(s[i:i+k]) - mean(s[i-k:i])|, boundary vs interior
    for k in (4, 8):
        lo, hi = k, n - k
        if hi > lo:
            ii = np.arange(lo, hi)
            L = (cs[ii] - cs[ii - k]) / k
            R = (cs[ii + k] - cs[ii]) / k
            D = np.abs(R - L)
            mb = m[ii]
            out[f"nDb{k}"] = float(mb.sum()); out[f"sDb{k}"] = float(D[mb].sum())
            out[f"nDi{k}"] = float((~mb).sum()); out[f"sDi{k}"] = float(D[~mb].sum())
        else:
            out[f"nDb{k}"] = out[f"sDb{k}"] = out[f"nDi{k}"] = out[f"sDi{k}"] = 0.0
    return out


def boundary_profile(s: np.ndarray, mask: np.ndarray, K: int = PROFILE_K):
    """Sum and count of surprisal at each offset -K..+K around a patch start (x=0 = first byte
    of the new unit). Returns (sums[2K+1], counts[2K+1])."""
    n = len(s)
    idx = np.flatnonzero(mask)
    idx = idx[(idx >= K) & (idx < n - K)]
    sums = np.zeros(2 * K + 1)
    cnts = np.zeros(2 * K + 1)
    if idx.size:
        win = s[idx[:, None] + np.arange(-K, K + 1)[None, :]]
        sums = win.sum(axis=0)
        cnts[:] = len(idx)
    return sums, cnts


def within_unit_profile(s: np.ndarray, mask: np.ndarray, min_len: int = 3):
    """Mean surprisal vs normalised position inside a unit (REL_BINS bins) and vs absolute
    offset from the unit start. Only units of length >= min_len enter the relative profile, so a
    bin is not dominated by 1-2 byte units."""
    n = len(s)
    m = mask.copy(); m[0] = True
    starts = np.flatnonzero(m)
    ends = np.append(starts[1:], n)
    lens = ends - starts
    rel_s = np.zeros(REL_BINS); rel_n = np.zeros(REL_BINS)
    abs_s = np.zeros(ABS_OFFSETS); abs_n = np.zeros(ABS_OFFSETS)
    sel = (lens >= min_len) & (starts > 0)
    if sel.any():
        st, ln = starts[sel], lens[sel]
        within = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
        idx = np.repeat(st, ln) + within
        r = (within / np.repeat(ln, ln) * REL_BINS).astype(np.int64).clip(0, REL_BINS - 1)
        np.add.at(rel_s, r, s[idx])
        np.add.at(rel_n, r, 1.0)

    sel2 = starts > 0
    if sel2.any():
        st2, ln2 = starts[sel2], lens[sel2]
        for k in range(ABS_OFFSETS):
            take = st2[ln2 > k] + k
            abs_s[k] += s[take].sum()
            abs_n[k] += len(take)
    return rel_s, rel_n, abs_s, abs_n


# ------------------------------------------------------------------ retrieval metrics
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with tie correction (scipy's C rankdata; the pure-Python tie walk cost
    ~0.5 s per call at 1.2M byte positions and the bootstrap makes ~1600 of them)."""
    from scipy.stats import rankdata
    pos = labels.astype(bool)
    npos, nneg = int(pos.sum()), int(labels.size - pos.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (step-wise), descending score."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    npos = tp[-1]
    return float((prec * y).sum() / npos) if npos > 0 else float("nan")


def precision_at_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """Precision when exactly as many positions are predicted as the arm has boundaries."""
    k = int(labels.sum())
    if k == 0:
        return float("nan")
    idx = np.argpartition(-scores, k - 1)[:k]
    return float(labels[idx].mean())
