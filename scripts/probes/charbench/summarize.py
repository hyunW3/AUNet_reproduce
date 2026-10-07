#!/usr/bin/env python3
"""Tables for the CharBench / Strawberry runs (trio results.json + external JSONs).

  python summarize.py --run llama=<dir>/llama/results.json --run blt134=X.json ... --out summary.md

Per model: gen exact match and cloze acc per task, suite means. Extra cloze readings:
  * strawberry "no-copy": accuracy over the 3 non-identity choices (the unchanged input is dropped
    from the argmax), i.e. can the model tell the correct edit from a wrong/corrupted one when it is
    not allowed to fall back on copying the input;
  * charbench gen accuracy by word length (4..10).
"""
import argparse, collections, json

CB = ["freq", "unique", "first", "last"]


def load(path):
    r = json.load(open(path))
    return r["results"], r.get("samples", {})


def metric(res, task):
    row = res.get(task, {})
    for k, v in row.items():
        if (k.startswith("exact_match,") or k == "acc,none") and "stderr" not in k:
            return 100 * float(v)
    return float("nan")


def ll(s):
    return [float(x[0][0]) if isinstance(x[0], (list, tuple)) else float(x[0]) for x in s["resps"]]


def nocopy_acc(samples):
    hit = n = 0
    for s in samples:
        d, lls = s["doc"], ll(s)
        keep = [i for i, c in enumerate(d["choices"]) if c != d["input"]]
        if d["gold"] not in keep:
            continue
        n += 1
        hit += max(keep, key=lambda i: lls[i]) == d["gold"]
    return 100 * hit / max(n, 1)


def copy_rate(samples):
    """share of gen outputs that are the input verbatim (stripped)."""
    c = sum(1 for s in samples if str(s["filtered_resps"][0]).strip() == s["doc"]["input"].strip())
    return 100 * c / max(len(samples), 1)


def by_len(samples):
    acc = collections.defaultdict(list)
    for s in samples:
        acc[s["doc"]["word_len"]].append(float(s["exact_match"]))
    return {L: 100 * sum(v) / len(v) for L, v in sorted(acc.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=path")
    ap.add_argument("--items", default="/mnt/ssd2/hyun2/AUNet/reports/charbench/items")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json_out", default=None)
    a = ap.parse_args()
    names = json.load(open(f"{a.items}/task_lists.json"))
    sp = [t[len("strawberry_"):-len("_gen")] for t in names["strawberry_gen"]]
    runs = {}
    for kv in a.run:
        k, p = kv.split("=", 1)
        runs[k] = load(p)
    M = list(runs)
    base = {}   # CharBench reference rows: always-majority answer (gen), uniform guess (cloze)
    for t in CB:
        it = [json.loads(l) for l in open(f"{a.items}/data/charbench_{t}.jsonl")]
        c = collections.Counter(x["answer"] for x in it)
        base[f"charbench_{t}_gen"] = 100 * c.most_common(1)[0][1] / len(it)
        base[f"charbench_{t}_cloze"] = 100 * sum(1 / len(x["choices"]) for x in it) / len(it)
    L, J = [], {}
    hdr = "| task | " + " | ".join(M) + " |\n|---|" + "--:|" * len(M)

    def block(title, rows):
        L.append(f"\n### {title}\n\n{hdr}")
        for lab, vals in rows:
            L.append(f"| {lab} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")

    for mode in ("gen", "cloze"):
        rows, means = [], [[] for _ in M]
        for t in CB:
            v = [metric(runs[m][0], f"charbench_{t}_{mode}") for m in M]
            rows.append((f"{t} (baseline {base[f'charbench_{t}_{mode}']:.1f})", v)); [means[i].append(x) for i, x in enumerate(v)]
        bl = sum(base[f"charbench_{t}_{mode}"] for t in CB) / len(CB)
        rows.append((f"**avg** (baseline {bl:.1f})", [sum(x) / len(x) for x in means]))
        J[f"charbench_{mode}"] = dict(zip(M, rows[-1][1]))
        block(f"CharBench — {mode} ({'exact match' if mode == 'gen' else 'acc over 0..len(word)'})", rows)
    for mode in ("gen", "cloze"):
        rows, means = [], [[] for _ in M]
        for t in sp:
            v = [metric(runs[m][0], f"strawberry_{t}_{mode}") for m in M]
            rows.append((t, v)); [means[i].append(x) for i, x in enumerate(v)]
        rows.append(("**avg**", [sum(x) / len(x) for x in means]))
        J[f"strawberry_{mode}"] = dict(zip(M, rows[-1][1]))
        block(f"Strawberry — {mode} ({'exact match' if mode == 'gen' else 'acc, 4 choices (chance 25)'})", rows)
    rows = []
    for lab, fn, mode in (("cloze no-copy acc (3 choices, chance 33.3)", nocopy_acc, "cloze"),
                          ("gen: output == input (copy rate)", copy_rate, "gen")):
        v = []
        for m in M:
            s = runs[m][1]
            per = [fn(s[f"strawberry_{t}_{mode}"]) for t in sp if s.get(f"strawberry_{t}_{mode}")]
            v.append(sum(per) / len(per) if per else float("nan"))
        rows.append((lab, v))
        J[f"strawberry_{fn.__name__}"] = dict(zip(M, v))
    block("Strawberry — copy diagnostics (mean over 20 tasks)", rows)
    rows = []
    for t in CB:
        per = {m: by_len(runs[m][1].get(f"charbench_{t}_gen", [])) for m in M}
        for Ln in range(4, 11):
            rows.append((f"{t} len={Ln}", [per[m].get(Ln, float("nan")) for m in M]))
    block("CharBench gen by word length", rows)
    open(a.out, "w").write("\n".join(L) + "\n")
    if a.json_out:
        json.dump(J, open(a.json_out, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
