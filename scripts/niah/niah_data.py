#!/usr/bin/env python3
"""S-NIAH-1 (single needle-in-a-haystack) data, following RULER (Hsieh et al. 2024).

A "magic number" needle is hidden in a haystack of repeated noise sentences; the
model must retrieve the number for a given key. This is the simplest RULER NIAH:
  - haystack  : repeated noise sentence (not an essay)
  - needle    : "One of the special magic numbers for {key} is {value}."
  - value     : a random N-digit number (default 7, as in RULER)

We vary total length (iso-byte across models) and needle depth (position).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

L = os.environ.get("AUNET_ROOT", "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet")
DCLM_GLOB = f"{L}/data/dclm_baseline_1.0_2shards_shuffled/*.jsonl"

NOISE = ("The grass is green. The sky is blue. The sun is yellow. "
         "Here we go. There and back again. ")

# RULER task presets: (haystack_type, value_type)
TASKS = {
    "1": ("noise", "number"),   # S-NIAH-1: repeated noise + magic number
    "2": ("essay", "number"),   # S-NIAH-2: natural essay + magic number
    "3": ("essay", "uuid"),     # S-NIAH-3: natural essay + magic UUID
    "4": ("essay", "word"),     # S-NIAH-4: natural essay + unseen magic WORD (absent from haystack)
    "5": ("noise", "word"),     # S-NIAH-5: repeated noise + unseen magic WORD
}

_essay_cache = {}


def _essay_pool(min_bytes: int = 3_000_000) -> str:
    """A big pool of natural English text (DCLM, the models' train distribution)
    to sample haystack windows from."""
    if "pool" in _essay_cache:
        return _essay_cache["pool"]
    path = sorted(glob.glob(DCLM_GLOB))[0]
    texts, total = [], 0
    with open(path) as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "").strip()
            except Exception:
                continue
            if len(t) < 200:
                continue
            texts.append(t)
            total += len(t)
            if total >= min_bytes:
                break
    _essay_cache["pool"] = "\n\n".join(texts)
    return _essay_cache["pool"]


def _uuid(rng: random.Random) -> str:
    h = "0123456789abcdef"
    return "-".join("".join(rng.choice(h) for _ in range(n)) for n in (8, 4, 4, 4, 12))


# Distinctive words for the "word" value type: the retrieved value is a WORD (not a
# number/UUID), and we require it to be ABSENT from the haystack so retrieval is a
# genuine copy of an unseen lexical string rather than something the haystack primes.
_MAGIC_WORDS = [
    "quokka", "sassafras", "zeugma", "obsidian", "zephyr", "kumquat", "marzipan",
    "gargoyle", "xylophone", "quagmire", "syzygy", "pemmican", "zydeco", "kerfuffle",
    "brouhaha", "vellum", "cummerbund", "persimmon", "tamarind", "saffron", "juniper",
    "peregrine", "albatross", "manticore", "basilisk", "obelisk", "ziggurat", "pergola",
    "trellis", "filigree", "sarsaparilla", "flibbertigibbet", "collywobbles", "gobbledygook",
]
_CONS, _VOW = "bcdfghjklmnpqrstvwz", "aeiou"


def _magic_word(rng: random.Random, avoid: str = "") -> str:
    """A distinctive word guaranteed not to occur (case-insensitively) in `avoid`
    (the haystack + instruction + keys). Falls back to a pronounceable nonce."""
    av = avoid.lower()
    cands = _MAGIC_WORDS[:]
    rng.shuffle(cands)
    for w in cands:
        if w.lower() not in av:
            return w
    for _ in range(50):  # nonce fallback: consonant-vowel syllables + final consonant
        w = "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(rng.randint(2, 3))) + rng.choice(_CONS)
        if w.lower() not in av:
            return w
    return w

# generic keys so the model can't lean on a memorized association
KEYS = ["forest", "harbor", "candle", "marble", "engine", "meadow", "lantern",
        "planet", "river", "cactus", "violin", "pebble", "glacier", "orchard",
        "compass", "beacon", "thicket", "canyon", "willow", "anchor"]

INSTRUCTION = ("Some special magic numbers are hidden within the following text. "
               "Make sure to memorize them. I will quiz you about the numbers "
               "afterwards.")


def make_sample(target_bytes: int, depth: float, value_digits: int,
                rng: random.Random, haystack_type: str = "noise",
                value_type: str = "number", control: bool = False,
                needle_sep: str = "") -> dict:
    """control=True: build the SAME prompt but DO NOT insert the needle, so the
    (teacher-forced) value is absent from the context. A model that genuinely
    retrieves must score ~0 here; if it still scores high the metric is leaking.

    needle_sep: separator between "is" and the value INSIDE the needle only
    (e.g. ":" restores the original RULER template "...is: {value}"). The query
    is untouched, so this isolates the effect of the colon in the context."""
    key = rng.choice(KEYS)
    noun = {"uuid": "UUID", "word": "word"}.get(value_type, "number")

    # Build the haystack body first, so a "word" value can be chosen to be absent
    # from it (unseen retrieval target). ~80 B covers the longest needle template.
    body_target = max(target_bytes, 80 + 200)
    if haystack_type == "essay":
        pool = _essay_pool()
        start = rng.randint(0, max(0, len(pool) - body_target - 1))
        body = pool[start:start + body_target]
    else:
        body = (NOISE * (body_target // len(NOISE) + 3))[:body_target]

    if value_type == "uuid":
        value = _uuid(rng)
    elif value_type == "word":
        value = _magic_word(rng, avoid=body + " " + INSTRUCTION + " " + " ".join(KEYS))
    else:
        value = "".join(rng.choice("0123456789") for _ in range(value_digits))
    needle = f"One of the special magic {noun}s for {key} is{needle_sep} {value}. "

    cut = int(len(body) * depth)
    while 0 < cut < len(body) and body[cut] != " ":   # snap to a word boundary
        cut += 1
    haystack = body[:cut] + ("" if control else needle) + body[cut:]

    query = (f"What is the special magic {noun} for {key} mentioned in the "
             f"provided text? The special magic {noun} for {key} is")
    prompt = f"{INSTRUCTION}\n\n{haystack}\n\n{query}"
    return {"prompt": prompt, "key": key, "value": value, "depth": depth,
            "value_type": value_type, "haystack_type": haystack_type,
            "target_bytes": target_bytes, "prompt_bytes": len(prompt.encode())}


def make_dataset(target_bytes, depths, n_per_cell, value_digits, seed,
                 haystack_type="noise", value_type="number", control=False,
                 needle_sep=""):
    rng = random.Random(seed)
    out = []
    for d in depths:
        for _ in range(n_per_cell):
            out.append(make_sample(target_bytes, d, value_digits, rng,
                                   haystack_type, value_type, control,
                                   needle_sep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_bytes", type=int, default=1024)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--value_digits", type=int, default=7)
    ap.add_argument("--task", choices=list(TASKS), default="1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ht, vt = TASKS[args.task]
    s = make_sample(args.target_bytes, args.depth, args.value_digits,
                    random.Random(args.seed), ht, vt)
    print(f"[S-NIAH-{args.task}] haystack={ht} value={vt} "
          f"key={s['key']} value={s['value']} depth={s['depth']} "
          f"prompt_bytes={s['prompt_bytes']}")
    print("----- prompt -----")
    print(s["prompt"][:400], "...")
    print("----- ends -----")
    print("..." + s["prompt"][-120:])


if __name__ == "__main__":
    main()
