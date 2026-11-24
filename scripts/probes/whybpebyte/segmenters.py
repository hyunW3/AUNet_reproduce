#!/usr/bin/env python3
"""Stage 1 of the structured-locality suite: one `segment(text) -> boundary starts` per arm.

Every arm returns SORTED patch-START byte indices, always beginning at 0, so patch k spans
[start_k, start_{k+1}) and the patches tile the raw byte sequence exactly (the convention already
used by scripts/probes/compression_stability.py and the paper's tab:zh_stats).

Real systems reuse compression_stability.build_parser so this suite and the compression-stability
sweep cannot drift apart:
    llama    conventional subword tokenizer (Llama-3 tiktoken) token starts
    aunet    AU-Net word/regex pooling (RegexPool word1)
    bpebyte  OnlineBPE causal byte-trie walk (RegexPool bpe_br, greedy, root placement)
    blt      entropy patcher -- CIRCULAR against a byte-LM probe, kept only as a labelled reference

Controls (design section 11) are built here because they are defined RELATIVE to a real arm:
    fixed_k         chunk every k bytes, k = the arm's mean bytes/unit
    random_matched  the arm's own segment-length multiset, randomly re-ordered (Control A)
    whitespace      natural word starts, a diagnostic reference (Control 3)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(ROOT / "scripts/probes"))

REAL_ARMS = ["llama", "aunet", "bpebyte"]


def build(name: str, **kw):
    """f(text) -> list of patch-start byte indices (starts with 0)."""
    if name == "whitespace":
        def ws_starts(text: str) -> List[int]:
            b = text.encode()
            # a word starts at the first non-space byte after any whitespace run
            starts, prev_ws = [0], True
            for i, v in enumerate(b):
                cur_ws = v in (0x20, 0x09, 0x0A, 0x0D)
                if prev_ws and not cur_ws and i > 0:
                    starts.append(i)
                prev_ws = cur_ws
            return starts
        return ws_starts

    from compression_stability import build_parser
    return build_parser(name, **kw)


# --------------------------------------------------------------------------- controls
def fixed_k_starts(n_bytes: int, k: int) -> np.ndarray:
    """Control 1: fixed-width byte chunks."""
    return np.arange(0, n_bytes, max(1, int(k)), dtype=np.int64)


def random_matched_starts(lengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Control 2 / Control A: the SAME multiset of segment lengths, randomly re-ordered.

    Preserves the arm's exact bytes-per-unit distribution and unit count, so any surviving
    boundary/interior contrast cannot be an artefact of how long the arm's units are.
    """
    perm = rng.permutation(lengths)
    return np.concatenate(([0], np.cumsum(perm)[:-1])).astype(np.int64)


def starts_to_mask(starts, n_bytes: int) -> np.ndarray:
    m = np.zeros(n_bytes, dtype=bool)
    s = np.asarray(starts, dtype=np.int64)
    m[s[(s >= 0) & (s < n_bytes)]] = True
    m[0] = True
    return m


def starts_to_lengths(starts, n_bytes: int) -> np.ndarray:
    s = np.asarray(starts, dtype=np.int64)
    return np.diff(np.concatenate([s, [n_bytes]]))
