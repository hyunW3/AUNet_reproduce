#!/usr/bin/env python3
"""Third batch (EXPERIMENTS_3.md): what makes BLT fall for echoed answer strings, and how far it goes.
Same MC items / options / scoring as build_tasks2.py `echo` (clean baseline = echo/clean); only the
context changes. Rows go to data/probe3.jsonl (task "probe3", cond = variant, length = MC task, pos = idx).

  E1  clean_fill | echo_all_near | echo_all_far | echo_wrong_near | echo_wrong_far
        ~700 B neutral filler; near = filler + echo + question, far = echo + filler + question
        (same bytes, order differs; far puts the echo outside BLT's 512-B entropy/local window)
  A   dist1 | dist2 | dist4 (irrelevant general-knowledge sentences) | indomain (another item's question)
  B   soft_wrong | student_wrong | neg_wrong | neg_gold
  C   pre25 | pre50 | pre75 | suf50 | bow   (Hint: part of a wrong option / its words shuffled)
  D   list_gold_first | list_gold_last | list_two

  python scripts/probes/blt_weak/build_tasks3.py --mc reports/blt_weak/data/mc_items.jsonl --out reports/blt_weak/data/probe3.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "niah"))
sys.path.insert(0, str(HERE))
from niah_data import _essay_pool  # noqa: E402
from build_tasks2 import split_q, rank_row  # noqa: E402

FACTS = [
    "The Pacific Ocean is the largest ocean on Earth.", "Honey never spoils if it is stored properly.",
    "Mount Everest is located in the Himalayas.", "A group of crows is called a murder.",
    "The Eiffel Tower was completed in 1889.", "Octopuses have three hearts.",
    "Bananas are botanically classified as berries.", "The Great Wall of China is thousands of miles long.",
    "Venus is the hottest planet in the solar system.", "Sharks existed before trees did.",
    "The human body contains 206 bones.", "Saturn has the most known moons of any planet.",
    "The Amazon is the largest rainforest in the world.", "Light from the Sun takes about eight minutes to reach Earth.",
    "Penguins live almost exclusively in the Southern Hemisphere.", "The piano was invented in Italy.",
    "Hummingbirds can fly backwards.", "Iceland has no native mosquitoes.",
    "The Sahara is the largest hot desert on Earth.", "A day on Mars lasts about 24 hours and 39 minutes.",
]
L = "ABCDEFGH"


def _seed(*p):
    return zlib.crc32(repr(p).encode())


def filler(rng, n=700):
    pool = _essay_pool()
    s = rng.randrange(0, len(pool) - 2 * n)
    t = pool[s:s + 2 * n]
    t = t[t.find(" ") + 1:]                              # start at a word
    t = t.encode()[:n].decode(errors="ignore")
    return t[: t.rfind(" ")].replace("\n", " ") + "."


def word_prefix(s, frac):
    w = s.split(" ")
    k = max(1, round(len(w) * frac))
    return " ".join(w[:k]) if len(w) > 1 else s[: max(1, round(len(s) * frac))]


def word_suffix(s, frac):
    w = s.split(" ")
    k = max(1, round(len(w) * frac))
    return " ".join(w[-k:]) if len(w) > 1 else s[-max(1, round(len(s) * frac)):]


def shuffled_words(s, rng):
    """Same words, different order (retried so a multi-word string never comes back verbatim)."""
    w = s.split(" ")
    for _ in range(20):
        o = rng.sample(w, len(w))
        if o != w or len(set(w)) < 2:
            break
    return " ".join(o)


def options_line(ch):
    return "Options: " + " ".join(f"({L[i]}) {c}" for i, c in enumerate(ch))


def variants(it, items_by_task, rng):
    ch, g, ctx = it["choices"], it["gold"], it["context"]
    wi = rng.choice([i for i in range(len(ch)) if i != g])
    wrong, gold = ch[wi], ch[g]
    fill = filler(rng)
    opts = options_line(ch)
    hw = f"Hint: {wrong}"
    v = {
        # E1
        "clean_fill": f"{fill}\n{ctx}",
        "echo_all_near": f"{fill}\n{opts}\n{ctx}",
        "echo_all_far": f"{opts}\n{fill}\n{ctx}",
        "echo_wrong_near": f"{fill}\n{hw}\n{ctx}",
        "echo_wrong_far": f"{hw}\n{fill}\n{ctx}",
        # A
        "dist1": " ".join(rng.sample(FACTS, 1)) + "\n" + ctx,
        "dist2": " ".join(rng.sample(FACTS, 2)) + "\n" + ctx,
        "dist4": " ".join(rng.sample(FACTS, 4)) + "\n" + ctx,
        "indomain": split_q(rng.choice([x for x in items_by_task[it["task"]] if x["idx"] != it["idx"]]))[0]
                    + "\n" + ctx,
        # B
        "soft_wrong": f"Some people think the answer is: {wrong}\n{ctx}",
        "student_wrong": f"A student answered: {wrong}\n{ctx}",
        "neg_wrong": f"It is NOT true that the answer is: {wrong}\n{ctx}",
        "neg_gold": f"It is NOT true that the answer is: {gold}\n{ctx}",
        # C
        "pre25": f"Hint: {word_prefix(wrong, 0.25)}\n{ctx}",
        "pre50": f"Hint: {word_prefix(wrong, 0.50)}\n{ctx}",
        "pre75": f"Hint: {word_prefix(wrong, 0.75)}\n{ctx}",
        "suf50": f"Hint: {word_suffix(wrong, 0.50)}\n{ctx}",
        "bow": f"Hint: {shuffled_words(wrong, rng)}\n{ctx}",
    }
    # D: listing order (scored option order is unchanged)
    others = [i for i in range(len(ch)) if i != g]
    rng.shuffle(others)
    first = [g] + others
    last = others + [g]
    two = [g, wi]
    rng.shuffle(two)
    lst = lambda order: "Options: " + " ".join(f"({L[k]}) {ch[i]}" for k, i in enumerate(order))
    v["list_gold_first"] = f"{lst(first)}\n{ctx}"
    v["list_gold_last"] = f"{lst(last)}\n{ctx}"
    v["list_two"] = f"{lst(two)}\n{ctx}"
    return v, wi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.mc)]
    by_task = {}
    for it in items:
        by_task.setdefault(it["task"], []).append(it)
    rows = []
    for it in items:
        rng = random.Random(_seed("p3", it["task"], it["idx"]))
        vs, wi = variants(it, by_task, rng)
        for name, p in vs.items():
            r = rank_row(f"p3-{name}-{it['task']}-{it['idx']}", "probe3", name, it["task"], it["idx"],
                         p, [" " + c for c in it["choices"]], it["gold"])
            r["wrong_idx"] = wi
            rows.append(r)
    big = [r["id"] for r in rows if 1 + r["prompt_bytes"] + max(len(o.encode()) for o in r["options"]) > 4096]
    assert not big, big[:3]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows;", dict(Counter(r["cond"] for r in rows)))


if __name__ == "__main__":
    main()
