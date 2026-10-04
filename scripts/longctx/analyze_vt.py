#!/usr/bin/env python3
"""Error anatomy of RULER Variable Tracking generations (why does one model beat another?).

Every 5-letter upper-case name in a model's answer window is classified against the scored
instance's chains: target chain / distractor chain / copied from the one-shot example /
absent from the prompt (hallucinated). Also: did the answer list the target chain in chain
order, and how many names were emitted.
"""
import argparse
import collections
import glob
import json
import re

NAME = re.compile(r"\b[A-Z]{5}\b")
ASSIGN = re.compile(r"VAR ([A-Z]{5}) = (?:VAR ([A-Z]{5})|(\d+))\.")


def chains_of(text):
    """{root_value: [names in chain order]} from 'VAR A = 123.' / 'VAR B = VAR A.' statements."""
    parent, value = {}, {}
    for name, src, num in ASSIGN.findall(text):
        if num:
            value[name] = num
        else:
            parent[name] = src
    out = collections.defaultdict(list)
    for root, v in value.items():
        chain, cur = [root], root
        nxt = {p: c for c, p in parent.items()}
        while cur in nxt:
            cur = nxt[cur]
            chain.append(cur)
        out[v] = chain
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longctx/ruler_agg.jsonl")
    ap.add_argument("--results", default="reports/longctx/results")
    ap.add_argument("--extra_data", default="data/longctx/fit4k.jsonl")
    a = ap.parse_args()
    prompts = {}
    for f in (a.data, a.extra_data):
        try:
            for l in open(f):
                r = json.loads(l)
                if r["task"] == "vt":
                    prompts[r["id"]] = r
        except FileNotFoundError:
            pass
    res = [json.loads(l) for f in glob.glob(f"{a.results}/*.jsonl") for l in open(f)]
    res = [r for r in res if r["task"] == "vt" and r.get("score") is not None and r["id"] in prompts]
    stats = collections.defaultdict(lambda: collections.Counter())
    for r in res:
        p = prompts[r["id"]]["prompt"]
        example, scored = p.split("\n\nMemorize and track", 1) if "\n\nMemorize and track" in p else ("", p)
        q = re.search(r"assigned the value (\d+) in the text above", scored).group(1)
        ch = chains_of(scored)
        target = ch[q]
        distract = {n for v, c in ch.items() if v != q for n in c}
        ex_names = set(NAME.findall(example))
        names = [n for n in NAME.findall(r["gen"]) if n != "VAR"]
        k = (r["tag"], r["cond"], r["length"])
        s = stats[k]
        s["n"] += 1
        s["recall"] += r["score"]
        seen = set()
        for n in names:
            if n in target:
                s["dup" if n in seen else "target"] += 1
                seen.add(n)
            elif n in distract:
                s["distractor"] += 1
            elif n in ex_names:
                s["example"] += 1
            elif n in scored:
                s["other_in_prompt"] += 1
            else:
                s["hallucinated"] += 1
        s["emitted"] += len(names)
        got = [n for n in names if n in target]
        s["all5"] += int(set(target) <= set(names))
        s["any_distractor"] += int(any(n in distract for n in names))
        s["in_order"] += int(got[:len(target)] == target[:len(got)] and len(got) > 0)
        s["first_is_root"] += int(bool(names) and names[0] == target[0])
        # window-free view: recall over the first len(target) names only (an answer that
        # lists BOTH chains gets more names into the byte window and so more recall)
        s["recall_first"] += sum(t in names[:len(target)] for t in target) / len(target)
        s["var_prefix"] += int(r["gen"].lstrip().startswith("VAR"))
    order = ["subword_llama", "aunet_static", "byte_greedyroot", "blt_1b", "hnet_1stage_XL"]
    print("| model | cond | len | n | recall (window) | recall (first 5 names) | all-5 | rows w/ distractor | "
          "names/answer | 'VAR ' prefix | target | distractor | example-copy | hallucinated | dup |")
    print("|" + "---|" * 15)
    for k in sorted(stats, key=lambda k: (k[1], str(k[2]), order.index(k[0]) if k[0] in order else 9)):
        s = stats[k]
        n, e = s["n"], max(1, s["emitted"])
        print(f"| {k[0]} | {k[1]} | {k[2]} | {n} | {100 * s['recall'] / n:.1f} | {100 * s['recall_first'] / n:.1f} | "
              f"{100 * s['all5'] / n:.0f}% | "
              f"{100 * s['any_distractor'] / n:.0f}% | {s['emitted'] / n:.1f} | {100 * s['var_prefix'] / n:.0f}% | "
              f"{100 * s['target'] / e:.0f}% | "
              f"{100 * s['distractor'] / e:.0f}% | {100 * s['example'] / e:.0f}% | "
              f"{100 * s['hallucinated'] / e:.0f}% | {100 * s['dup'] / e:.0f}% |")


if __name__ == "__main__":
    main()
