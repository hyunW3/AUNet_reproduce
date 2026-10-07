#!/usr/bin/env python3
"""CUTE multiple-choice table (build_items.py --cute) across models, next to the generative CUTE.

  python summarize_cute_mc.py --run Llama=<results.json> ... [--gen Llama=20.1,...] --out cute_mc.md

Per split: acc / acc_norm, chance; for the edit splits also how often the unchanged input wins and the
accuracy over the remaining choices (input dropped from the argmax). Category means follow CUTE.
"""
import argparse, json, statistics as st

GROUPS = {"composition": ["spell", "spell_inverse", "contains_char", "contains_word"],
          "orthographic": ["orth", "sem"],
          "sequence": ["ins_char", "ins_word", "del_char", "del_word", "sub_char", "sub_word",
                       "swap_char", "swap_word"]}
SPLITS = [s for g in GROUPS.values() for s in g]


def ll(s):
    return [float(x[0][0]) if isinstance(x[0], (list, tuple)) else float(x[0]) for x in s["resps"]]


def split_stats(res, samples, sp):
    t = f"cute_{sp}_cloze"
    row = {"acc": 100 * res[t]["acc,none"], "acc_norm": 100 * res[t]["acc_norm,none"]}
    pick_in = keep_hit = keep_n = n_in = 0
    for s in samples.get(t, []):
        d, lls = s["doc"], ll(s)
        if not d.get("input") or d["input"] not in d["choices"]:
            continue
        n_in += 1
        top = max(range(len(lls)), key=lambda i: lls[i])
        pick_in += d["choices"][top] == d["input"]
        keep = [i for i, c in enumerate(d["choices"]) if c != d["input"]]
        if d["gold"] in keep:
            keep_n += 1
            keep_hit += max(keep, key=lambda i: lls[i]) == d["gold"]
    if n_in:
        row["pick_input"] = 100 * pick_in / n_in
        row["nocopy_acc"] = 100 * keep_hit / max(keep_n, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path")
    ap.add_argument("--items", default="/mnt/ssd2/hyun2/AUNet/reports/charbench/items")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json_out", default=None)
    a = ap.parse_args()
    runs = {}
    for kv in a.run:
        k, p = kv.split("=", 1)
        r = json.load(open(p))
        runs[k] = (r["results"], r.get("samples", {}))
    chance = {sp: 100 / len(json.loads(open(f"{a.items}/data/cute_{sp}.jsonl").readline())["choices"])
              for sp in SPLITS}
    M = list(runs)
    tab = {m: {sp: split_stats(*runs[m], sp) for sp in SPLITS} for m in M}
    f = lambda v: "–" if v is None else f"{v:.1f}"
    L = ["| category | split | chance | " + " | ".join(M) + " |", "|---|---|--:|" + "--:|" * len(M)]
    for g, sps in GROUPS.items():
        for sp in sps:
            L.append(f"| {g} | {sp} | {chance[sp]:.0f} | " + " | ".join(
                f"{f(tab[m][sp]['acc'])} / {f(tab[m][sp]['acc_norm'])}" for m in M) + " |")
        L.append(f"| **{g} avg** | | {st.mean(chance[s] for s in sps):.1f} | " + " | ".join(
            f"{st.mean(tab[m][s]['acc'] for s in sps):.1f} / {st.mean(tab[m][s]['acc_norm'] for s in sps):.1f}"
            for m in M) + " |")
    L.append(f"| **all 14** | | {st.mean(chance.values()):.1f} | " + " | ".join(
        f"{st.mean(tab[m][s]['acc'] for s in SPLITS):.1f} / {st.mean(tab[m][s]['acc_norm'] for s in SPLITS):.1f}"
        for m in M) + " |")
    L += ["", "Cells: acc / acc_norm. Copy diagnostics (edit splits): picks unchanged input % ; acc without it.", "",
          "| split | " + " | ".join(M) + " |", "|---|" + "--:|" * len(M)]
    for sp in SPLITS:
        if "pick_input" in tab[M[0]][sp]:
            L.append(f"| {sp} | " + " | ".join(
                f"{f(tab[m][sp]['pick_input'])} ; {f(tab[m][sp]['nocopy_acc'])}" for m in M) + " |")
    open(a.out, "w").write("\n".join(L) + "\n")
    if a.json_out:
        json.dump(tab, open(a.json_out, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
