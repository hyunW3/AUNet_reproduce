#!/usr/bin/env python3
"""Single-edit perturbations with an EXACT clean<->perturbed byte alignment.

Design section 7.1-7.2. One edit per sample, so the causal reading of the locality curves is
unambiguous, and the alignment is known in closed form rather than recovered by Levenshtein
(a min-edit alignment is not unique on repetitive text, and any tie-breaking would silently
become part of the measured disturbance).

Edits are CHARACTER-level on the decoded text, matching the paper's robustness suite
(apps/aunet/eval_typo.py: delete / swap / key / insert on alphabetic characters) and keeping the
perturbed stream valid UTF-8, which the subword arm needs in order to tokenize at all. Two extra
families from design 7.1 are added: whitespace deletion and case flip.

`align` maps a clean byte index to its perturbed index, or -1 when the byte was edited away.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lingua"))
from apps.aunet.eval_typo import _QWERTY, _sub_key                      # noqa: E402

FAMILIES = ["sub_key", "insert", "delete", "swap", "ws_delete", "case_flip"]
_WS = " \t"


def _char_starts(text: str) -> np.ndarray:
    """Byte offset of each character (plus a final sentinel at len(bytes))."""
    lens = np.fromiter((len(c.encode()) for c in text), dtype=np.int64, count=len(text))
    return np.concatenate(([0], np.cumsum(lens)))


def _eligible(text: str, family: str) -> np.ndarray:
    a = np.fromiter((c.isalpha() and c.lower() in _QWERTY for c in text), bool, len(text))
    if family in ("sub_key", "insert", "delete"):
        return np.flatnonzero(a)
    if family == "swap":
        ok = a[:-1] & a[1:]
        return np.flatnonzero(ok)                      # swap chars i, i+1
    if family == "ws_delete":
        return np.flatnonzero(np.fromiter((c in _WS for c in text), bool, len(text)))
    if family == "case_flip":
        return np.flatnonzero(np.fromiter((c.isalpha() and c.lower() != c.upper()
                                           for c in text), bool, len(text)))
    raise ValueError(family)


def apply_edit(text: str, family: str, ci: int, rng: random.Random
               ) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Edit character `ci`. Returns (perturbed_text, align, edited_clean_bytes).

    align[i] = index of clean byte i in the perturbed stream, or -1 if it was removed.
    edited_clean_bytes = clean byte positions the edit touched (the distance-0 reference set).
    """
    cs = _char_starts(text)
    b0, b1 = int(cs[ci]), int(cs[ci + 1])
    n = int(cs[-1])
    if family == "sub_key":
        rep = _sub_key(text[ci], rng)
        if rep == text[ci]:
            return None
        new = text[:ci] + rep + text[ci + 1:]
        touched = np.arange(b0, b1)
    elif family == "case_flip":
        c = text[ci]
        rep = c.lower() if c.isupper() else c.upper()
        if rep == c:
            return None
        new = text[:ci] + rep + text[ci + 1:]
        touched = np.arange(b0, b1)
    elif family == "insert":
        rep = _sub_key(text[ci], rng)
        new = text[:ci + 1] + rep + text[ci + 1:]
        touched = np.arange(b0, b1)
    elif family in ("delete", "ws_delete"):
        new = text[:ci] + text[ci + 1:]
        touched = np.arange(b0, b1)
    elif family == "swap":
        b2 = int(cs[ci + 2])
        if text[ci] == text[ci + 1]:
            return None                                 # a no-op swap
        new = text[:ci] + text[ci + 1] + text[ci] + text[ci + 2:]
        touched = np.arange(b0, b2)
    else:
        raise ValueError(family)

    # ---- exact alignment: every edit is confined to [b0, b_end); bytes outside shift by delta
    nb_new = len(new.encode())
    delta = nb_new - n
    align = np.arange(n, dtype=np.int64)
    end = int(touched[-1]) + 1
    align[end:] += delta
    align[touched] = -1                                 # edited bytes have no clean counterpart
    return new, align, touched


def sample_edits(text: str, family: str, n_sites: int, rng: random.Random,
                 margin: int = 64) -> List[int]:
    """Character indices to edit. `margin` keeps edits away from the window edges so the
    distance bins are not truncated by the boundary of the window itself."""
    cs = _char_starts(text)
    n = int(cs[-1])
    elig = _eligible(text, family)
    elig = elig[(cs[elig] >= margin) & (cs[elig] < n - margin - 4)]
    if elig.size == 0:
        return []
    if elig.size <= n_sites:
        return elig.tolist()
    return sorted(rng.sample(list(elig), n_sites))
