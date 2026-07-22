#!/usr/bin/env python3
"""Extended long-context retrieval tasks in the iso-byte, teacher-forced
exact-match protocol (same contract as niah_probe/niah_data), for a controlled
cross-tokenizer comparison (Llama / AU-Net / BPEByte):

  mv      RULER multi-value  : ONE key -> K values, retrieve them all (in text order)
  mq      RULER multi-query  : K keys -> one value each, retrieve per queried key
  seq     Sequential-NIAH    : K step-labeled needles inserted at SHUFFLED depths;
                               retrieve the values in STEP order (reorder, not read-order)
  nolima  NoLiMa-lite        : needle keyed by a noun, query keyed by a zero-overlap
                               synonym (paraphrase) -> retrieval needs latent association,
                               not literal string matching. A 'literal' control queries
                               with the needle's own noun; NoLiMa effect = literal - paraphrase.

Design choices that keep the comparison clean:
  * every value is a 7-digit magic number (RULER default) -> retrieval is exact STRING recall;
  * needle and query use the SAME no-colon format ("... is <value>" / query ends "... are"),
    so the prompt-boundary (partial-token) confound is removed at the source and all three
    families see an identical, boundary-consistent prompt (uniform pair construction);
  * needles are spread over evenly-spaced depths; mq/seq shuffle the depth assignment so
    insertion order != answer order.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from niah_data import _essay_pool, NOISE, KEYS, INSTRUCTION  # noqa: E402


def _num(rng: random.Random, d: int = 7) -> str:
    return "".join(rng.choice("0123456789") for _ in range(d))


def _distinct_nums(rng: random.Random, k: int, d: int = 7):
    vs = []
    while len(vs) < k:
        v = _num(rng, d)
        if v not in vs:
            vs.append(v)
    return vs


def _body(target_bytes: int, rng: random.Random, haystack: str = "essay") -> str:
    bt = max(target_bytes, 400)
    if haystack == "essay":
        pool = _essay_pool()
        start = rng.randint(0, max(0, len(pool) - bt - 1))
        return pool[start:start + bt]
    return (NOISE * (bt // len(NOISE) + 3))[:bt]


def _insert(body: str, needles_depths):
    """Insert each (needle, depth) at int(len(body)*depth), snapped to a word
    boundary. Insert from the LARGEST offset first, using the ORIGINAL body
    offsets -- inserting past an offset never shifts smaller offsets, so all cut
    points stay valid."""
    cuts = []
    for needle, depth in needles_depths:
        cut = int(len(body) * depth)
        while 0 < cut < len(body) and body[cut] != " ":
            cut += 1
        cuts.append((cut, needle))
    s = body
    for cut, needle in sorted(cuts, key=lambda x: x[0], reverse=True):
        s = s[:cut] + needle + s[cut:]
    return s


def _spread(K: int):
    return [(i + 1) / (K + 1) for i in range(K)]


def _wrap(hay: str, query: str) -> str:
    return f"{INSTRUCTION}\n\n{hay}\n\n{query}"


def make_mv(target_bytes, K, rng, haystack="essay") -> dict:
    body = _body(target_bytes, rng, haystack)
    key = rng.choice(KEYS)
    values = _distinct_nums(rng, K)                      # answer order = text order (depth asc)
    needles = [f"One of the special magic numbers for {key} is {v}. " for v in values]
    hay = _insert(body, list(zip(needles, _spread(K))))
    query = (f"What are all {K} special magic numbers for {key} mentioned in the "
             f"provided text? The special magic numbers for {key} are")
    p = _wrap(hay, query)
    return {"prompt": p, "values": values, "task": "mv", "K": K,
            "prompt_bytes": len(p.encode())}


def make_mq(target_bytes, K, rng, haystack="essay") -> dict:
    body = _body(target_bytes, rng, haystack)
    keys = rng.sample(KEYS, K)
    values = _distinct_nums(rng, K)                      # values[i] belongs to keys[i]
    order = list(range(K)); rng.shuffle(order)           # insert at shuffled depths
    depths = _spread(K)
    nd = [(f"One of the special magic numbers for {keys[order[i]]} is {values[order[i]]}. ",
           depths[i]) for i in range(K)]
    hay = _insert(body, nd)
    kl = ", ".join(keys)
    query = (f"What are the special magic numbers for {kl}? The magic numbers for "
             f"{kl}, in that order, are")
    p = _wrap(hay, query)
    return {"prompt": p, "values": values, "task": "mq", "K": K,   # answer order = queried key order
            "prompt_bytes": len(p.encode())}


def make_seq(target_bytes, K, rng, haystack="essay") -> dict:
    body = _body(target_bytes, rng, haystack)
    values = _distinct_nums(rng, K)                      # values[i] == step i+1
    order = list(range(K)); rng.shuffle(order)           # step labels inserted at shuffled depths
    depths = _spread(K)
    nd = [(f"Step {order[i] + 1}: the secret code is {values[order[i]]}. ", depths[i])
          for i in range(K)]
    hay = _insert(body, nd)
    query = (f"There are {K} secret codes in the text, each labeled with a step number. "
             f"List the secret codes in order from step 1 to step {K}. The secret codes "
             f"in order are")
    p = _wrap(hay, query)
    return {"prompt": p, "values": values, "task": "seq", "K": K,  # answer order = step order 1..K
            "prompt_bytes": len(p.encode())}


# Synonym pairs (needle noun, query noun): zero shared word, high-frequency so a 1.3B
# model can plausibly associate them. NoLiMa-lite queries with the paraphrase side.
SYN_PAIRS = [
    ("physician", "doctor"), ("automobile", "car"), ("canine", "dog"), ("feline", "cat"),
    ("residence", "house"), ("vessel", "ship"), ("infant", "baby"), ("attorney", "lawyer"),
    ("beverage", "drink"), ("currency", "money"), ("summit", "peak"), ("blaze", "fire"),
    ("tempest", "storm"), ("voyage", "trip"), ("monarch", "king"), ("cavern", "cave"),
    ("jewel", "gem"), ("garment", "clothing"), ("spectacles", "glasses"), ("labyrinth", "maze"),
]


def make_nolima(target_bytes, rng, literal=False, n_distract=3, haystack="essay") -> dict:
    body = _body(target_bytes, rng, haystack)
    pairs = rng.sample(SYN_PAIRS, n_distract + 1)
    target = pairs[0]
    used = [(target[0], _num(rng))]                      # (needle noun, value); target first
    for d in pairs[1:]:
        used.append((d[0], _num(rng)))
    tval = used[0][1]
    K = len(used)
    depths = _spread(K); rng.shuffle(depths)
    nd = [(f"The special magic number for the {noun} is {v}. ", depths[i])
          for i, (noun, v) in enumerate(used)]
    hay = _insert(body, nd)
    qnoun = target[0] if literal else target[1]          # literal: needle noun; paraphrase: synonym
    query = (f"What is the special magic number for the {qnoun} mentioned in the text? "
             f"The special magic number for the {qnoun} is")
    p = _wrap(hay, query)
    return {"prompt": p, "values": [tval], "task": "nolima",
            "mode": "literal" if literal else "paraphrase",
            "qnoun": qnoun, "target_noun": target[0], "K": K,
            "prompt_bytes": len(p.encode())}


def make_dataset(task, target_bytes, K, n, seed, haystack="essay",
                 literal=False, n_distract=3):
    rng = random.Random(seed)
    fn = {"mv": make_mv, "mq": make_mq, "seq": make_seq}
    out = []
    for _ in range(n):
        if task == "nolima":
            out.append(make_nolima(target_bytes, rng, literal, n_distract, haystack))
        else:
            out.append(fn[task](target_bytes, K, rng, haystack))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mv", "mq", "seq", "nolima"], default="mv")
    ap.add_argument("--target_bytes", type=int, default=2048)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--literal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    s = make_dataset(args.task, args.target_bytes, args.K, 1, args.seed,
                     literal=args.literal)[0]
    print(f"[{args.task}] K={s.get('K')} values={s['values']} "
          f"{'mode=' + s['mode'] + ' qnoun=' + s['qnoun'] + ' target=' + s['target_noun'] if args.task == 'nolima' else ''} "
          f"prompt_bytes={s['prompt_bytes']}")
    print("----- prompt head -----"); print(s["prompt"][:350], "...")
    print("----- prompt tail -----"); print("..." + s["prompt"][-260:])


if __name__ == "__main__":
    main()
