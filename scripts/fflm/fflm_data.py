#!/usr/bin/env python3
"""Flip-Flop Language Modeling (FFLM) data generator.

Reproduces the synthetic task of Liu et al. 2023, "Exposing Attention Glitches
with Flip-Flop Language Modeling" (arXiv:2306.00946).

A sequence is a stream of (instruction, bit) pairs over the alphabet {w,r,i,0,1}:
  - w b : WRITE bit b to the 1-bit memory (sets state := b)
  - i b : IGNORE (state unchanged; b is a random distractor bit)
  - r b : READ  (b MUST equal the current state = the most recent write)

Sequences begin with a write and end with a read.  The family FFL(p_i) sets
p_w = p_r = (1 - p_i)/2.  Regimes used in the paper:
  - dense   FFL(0.1)   : frequent writes/reads, short-range
  - in-dist FFL(0.8)   : training distribution
  - sparse  FFL(0.98)  : long-range dependencies (exposes "attention glitches")

We emit the token list + the (instruction_index, correct_bit) of every read so a
probe can score next-token prediction at exactly the deterministic read bits.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

WRITE, READ, IGNORE = "w", "r", "i"


@dataclass
class FFLMSequence:
    tokens: List[str]                       # ["w","0","i","1","r","1", ...]
    reads: List[Tuple[int, str]]            # [(instr_index, correct_bit), ...]
    p_i: float

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def context_and_bit(self, read_pos: int) -> Tuple[str, str, str]:
        """Return (prefix_up_to_and_incl_read_instr, correct_bit, wrong_bit).

        `prefix` ends at the "r" instruction token (no trailing space); the probe
        appends the family-appropriate continuation (" 0"/" 1" for subword,
        " " already implied for bytes)."""
        instr_index, correct = read_pos, None
        for idx, bit in self.reads:
            if idx == read_pos:
                correct = bit
                break
        assert correct is not None, f"{read_pos} is not a read position"
        prefix = " ".join(self.tokens[: 2 * instr_index + 1])   # ends at "r"
        wrong = "1" if correct == "0" else "0"
        return prefix, correct, wrong


def generate_sequence(num_instr: int, p_i: float, rng: random.Random) -> FFLMSequence:
    """Generate one FFL(p_i) sequence with `num_instr` instructions.

    First instruction is always a write; last is always a read.  Total symbol
    length T = 2 * num_instr."""
    assert num_instr >= 2
    p_w = p_r = (1.0 - p_i) / 2.0
    instrs_pool = [WRITE, READ, IGNORE]
    weights = [p_w, p_r, p_i]

    tokens: List[str] = []
    reads: List[Tuple[int, str]] = []
    state = None

    for k in range(num_instr):
        if k == 0:
            instr = WRITE                       # must start with a write
        elif k == num_instr - 1:
            instr = READ                        # must end with a read
        else:
            instr = rng.choices(instrs_pool, weights=weights, k=1)[0]

        if instr == WRITE:
            bit = rng.choice("01")
            state = bit
        elif instr == IGNORE:
            bit = rng.choice("01")              # distractor, state unchanged
        else:  # READ
            bit = state                         # deterministic
            reads.append((k, bit))

        tokens.extend([instr, bit])

    return FFLMSequence(tokens=tokens, reads=reads, p_i=p_i)


def generate_dataset(n: int, num_instr: int, p_i: float, seed: int) -> List[FFLMSequence]:
    rng = random.Random(seed)
    return [generate_sequence(num_instr, p_i, rng) for _ in range(n)]


# ---- regimes ---------------------------------------------------------------
REGIMES = {"dense": 0.1, "indist": 0.8, "sparse": 0.98}


def _validate(seqs: List[FFLMSequence]) -> None:
    """Assert every read bit equals the most-recent write bit (task invariant)."""
    for s in seqs:
        state = None
        rd = dict(s.reads)
        for k in range(len(s.tokens) // 2):
            instr, bit = s.tokens[2 * k], s.tokens[2 * k + 1]
            if instr == WRITE:
                state = bit
            elif instr == READ:
                assert bit == state, "read bit != state"
                assert rd.get(k) == bit, "reads metadata mismatch"
        assert s.tokens[0] == WRITE and s.tokens[-2] == READ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=list(REGIMES), default=None,
                    help="dense/indist/sparse; omit to self-test all three")
    ap.add_argument("--num_instr", type=int, default=64,
                    help="instructions per sequence (T = 2*num_instr symbols)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSONL to this path")
    args = ap.parse_args()

    if args.regime is None:
        # Self-test: generate all regimes, validate invariants, print stats.
        for name, p_i in REGIMES.items():
            seqs = generate_dataset(args.n, args.num_instr, p_i, args.seed)
            _validate(seqs)
            avg_reads = sum(len(s.reads) for s in seqs) / len(seqs)
            print(f"[{name:6s}] p_i={p_i:<4} n={len(seqs)} T={2*args.num_instr} "
                  f"avg_reads/seq={avg_reads:5.1f}  ex: {seqs[0].text[:60]}...")
        print("OK: all invariants hold.")
        return

    p_i = REGIMES[args.regime]
    seqs = generate_dataset(args.n, args.num_instr, p_i, args.seed)
    _validate(seqs)
    if args.out:
        with open(args.out, "w") as f:
            for s in seqs:
                f.write(json.dumps({"tokens": s.tokens, "reads": s.reads,
                                    "p_i": s.p_i}) + "\n")
        print(f"wrote {len(seqs)} sequences -> {args.out}")
    else:
        print(seqs[0].text)


if __name__ == "__main__":
    main()
