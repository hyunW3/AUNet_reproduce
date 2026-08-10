#!/usr/bin/env python3
"""Shared loader + greedy scorer for the state-tracking / recall probes.

Reuses fflm_probe.build_generator (same 3 family loaders). Scoring is
teacher-forced greedy exact-match: a request is (context, answer); we check
whether the model's argmax reproduces the answer token(s) verbatim == it would
greedily generate it. Tokenization-agnostic across subword / aunet(byte)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fflm"))
from fflm_probe import build_generator, DEFAULT_TOK  # noqa: E402,F401

# Per-tag checkpoint dir (relative to AUNET_ROOT). Each is overridable via a
# CKPT_<TAG> env var so a debug run can point a tag at a locally-present
# checkpoint without editing this file; unset => the canonical 1.3B path.
CKPT = {
    "subword_llama":   ("subword", os.environ.get("CKPT_SUBWORD_LLAMA",   "runs/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated")),
    "aunet_static":    ("aunet",   os.environ.get("CKPT_AUNET_STATIC",    "runs/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated")),
    "byte_bt":         ("aunet",   os.environ.get("CKPT_BYTE_BT",         "runs/main/1.3B/bpebyte_br_bt_1.3B/checkpoints/0000180000/consolidated")),
    "byte_greedyroot": ("aunet",   os.environ.get("CKPT_BYTE_GREEDYROOT", "runs/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated")),
    # BLT-style entropy patching. 100M ONLY (no 1.3B run exists) — not scale-matched to the
    # tags above; compare with the 100M ablation caveat.
    "blt_entropy":     ("aunet",   os.environ.get("CKPT_BLT_ENTROPY",     "runs/small/aunet_100M_entropy_low/checkpoints/0000001672/consolidated")),
}
L = os.environ.get("AUNET_ROOT", "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet")


def ckpt_for(tag):
    fam, rel = CKPT[tag]
    return fam, f"{L}/{rel}"


def fmt_pair(prompt, answer, family, variant=None):
    """Family-appropriate (context, continuation); the graded tokens isolate the answer.

    variant=None keeps the family-default cut. The two PBP variants place one space
    at the prompt/answer cut on opposite sides (identical total text):
      canonical : (prompt, " " + answer)   -- space leads the answer
      space     : (prompt + " ", answer)   -- space committed into the prompt
    A byte model sees identical bytes (delta == 0); a subword model re-tokenizes
    the prompt tail, so exact-match can move -- that is the prompt-boundary problem.
    """
    if variant == "canonical":
        return prompt.rstrip(" "), " " + answer
    if variant == "space":
        return prompt.rstrip(" ") + " ", answer
    if family == "subword":
        return prompt, " " + answer
    return prompt + " ", answer          # bytes: fold the separating space into context


def score(generator, tokenizer, pairs, batch=8):
    """[(exact_match, n_correct_tok, n_tok, ll_sum)] per (context, answer) pair."""
    out = []
    old = generator.max_gen_len
    generator.max_gen_len = 1
    try:
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            inputs = [c + k for c, k in chunk]
            _, lls, greedy = generator.generate(inputs)
            for (ctx, _k), ll, gr in zip(chunk, lls, greedy):
                p = len(tokenizer.encode(ctx, add_bos=False, add_eos=False))
                g = gr[p:]
                n = int(g.numel())
                # empty slice = the answer was truncated (prompt exceeded the model
                # window). NEVER a vacuous pass — score it as a miss.
                exact = bool(n > 0 and g.all().item())
                out.append((exact, int(g.sum().item()), n, ll[p:].sum().item()))
    finally:
        generator.max_gen_len = old
    return out
