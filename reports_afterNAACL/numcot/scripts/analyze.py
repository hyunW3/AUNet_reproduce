#!/usr/bin/env python3
"""Score NUMCoT generations and write tables.

paper   = the NUMCoT repo's scoring (src/code/experience_*/run_*.py::score), ported verbatim in behaviour:
          num2word: lower-case, drop " and ", ",", "-", " " from gold and response; correct if gold is a substring.
          numeric (word2num, units easy/medium, MWP): drop ",", "，", " ", and the narrow no-break space from the
          response; correct if gold occurs not preceded by a digit and not followed by a digit or ".digit".
          units hard: both numbers of the gold tuple occur that way.
          It credits a gold string anywhere in the response (incl. intermediate CoT steps).
strict  = only the first answer counts. Answer span = text after the last "answer is" if present, else the whole
          response. num2word: normalised span (letters only) equals normalised gold. Numeric: the first number in the
          span equals gold (units hard: the first two numbers).
CIs: item bootstrap, B=10,000, seed 0; model differences paired by item.
Section A: models with every item (the trio). Section B: models covering items_ext.jsonl (all five), on those keys.

  python reports_afterNAACL/numcot/scripts/analyze.py   -> scored/<model>.jsonl, tables.md
"""
import ast, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1]
MODELS = [("llama", "Transformer"), ("aunet", "AU-Net"), ("bpebyte", "BPEByte"), ("blt", "BLT†"), ("hnet", "H-Net†")]
TASKS = [("num2word", ["easy", "medium", "hard_fraction", "hard_decimal"]),
         ("word2num", ["easy", "medium", "hard_fraction", "hard_decimal"]),
         ("units", ["easy", "medium", "hard"]), ("mwp", ["AddSub", "SingleOp", "MultiArith"])]
CONDS = ["zeroshot", "cot_zeroshot", "fewshot", "cot_fewshot", "numcot"]
CN = {"zeroshot": "0-shot", "cot_zeroshot": "0-shot CoT", "fewshot": "few-shot", "cot_fewshot": "few-shot CoT",
      "numcot": "few-shot CoT + knowledge"}
B = 10000


def num_pat(g):
    return rf"(?<!\d){re.escape(g)}(?!\d)(?!\.\d)"


def wnorm(s):
    return s.lower().replace(" and ", " ").replace(",", "").replace("-", "").replace(" ", "")


def paper(it, resp):
    g = it["gold"]
    if it["task"] == "num2word":
        return wnorm(g) in wnorm(resp)
    r = resp.replace(",", "").replace("，", "").replace(" ", "").replace(" ", "")
    if it["task"] == "units" and it["split"] == "hard":
        t = ast.literal_eval(g)
        return bool(re.search(num_pat(str(t[0])), r) and re.search(num_pat(str(t[1])), r))
    return bool(re.search(num_pat(g), r))


NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def span(resp):
    m = list(re.finditer(r"answer is", resp, re.I))
    return resp[m[-1].end():] if m else resp


def same_num(a, b):
    if "/" in a or "/" in b:
        return a == b
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def strict(it, resp):
    s, g = span(resp), it["gold"]
    if it["task"] == "num2word":
        s = re.split(r"[.;\n]", s.strip())[0]
        return re.sub(r"[^a-z]", "", wnorm(s)) == re.sub(r"[^a-z]", "", wnorm(g))
    nums = NUM.findall(s.replace(",", ""))
    if it["task"] == "units" and it["split"] == "hard":
        t = ast.literal_eval(g)
        return len(nums) >= 2 and same_num(nums[0], str(t[0])) and same_num(nums[1], str(t[1]))
    return bool(nums) and same_num(nums[0], g)


def boot_mean(v, rng):
    idx = rng.integers(0, len(v), (B, len(v)))
    bs = v[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 100 * v.mean(), 100 * lo, 100 * hi, min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))


def section(L, title, items, S, models, rng):
    """Per-split tables, task totals with CIs, and BPEByte-paired differences on the given items (dict key -> item)."""
    P = L.append
    P(f"\n# {title}\n")
    P("Models: " + ", ".join(n for _, n in models) + f"; {len(items)} generations.\n")
    cell = lambda task, c, sp=None: [k for k, it in items.items() if it["task"] == task and it["cond"] == c
                                     and (sp is None or it["split"] == sp)]
    for metric, mi in (("paper", 0), ("strict", 1)):
        P(f"\n## Per split, metric = {metric}\n")
        for task, splits in TASKS:
            conds = [c for c in CONDS if cell(task, c)]
            if not conds:
                continue
            P(f"\n### {task}\n")
            P("| model | cond | " + " | ".join(splits) + " | all |")
            P("|---|---|" + "---|" * (len(splits) + 1))
            for c in conds:
                for tag, name in models:
                    cells = [f"{100 * np.mean([S[tag][k][mi] for k in cell(task, c, sp)]):.1f}" for sp in splits]
                    P(f"| {name} | {CN[c]} | " + " | ".join(cells)
                      + f" | {100 * np.mean([S[tag][k][mi] for k in cell(task, c)]):.1f} |")
    for metric, mi in (("paper", 0), ("strict", 1)):
        P(f"\n## Task totals with 95% CI, metric = {metric}\n")
        P("| task | cond | n | " + " | ".join(n for _, n in models) + " |")
        P("|---|---|---|" + "---|" * len(models))
        for task, _ in TASKS:
            for c in CONDS:
                ks = cell(task, c)
                if not ks:
                    continue
                row = []
                for tag, _ in models:
                    m, lo, hi, _ = boot_mean(np.array([S[tag][k][mi] for k in ks], float), rng)
                    row.append(f"{m:.1f} [{lo:.1f}, {hi:.1f}]")
                P(f"| {task} | {CN[c]} | {len(ks)} | " + " | ".join(row) + " |")
        if "bpebyte" in dict(models) and len(models) > 1:
            P(f"\n### Paired differences (pp, 95% CI, bootstrap p), metric = {metric}\n")
            others = [(t, n) for t, n in models if t != "bpebyte"]
            P("| task | cond | " + " | ".join(f"BPEByte − {n}" for _, n in others) + " |")
            P("|---|---|" + "---|" * len(others))
            for task, _ in TASKS:
                for c in CONDS:
                    ks = cell(task, c)
                    if not ks:
                        continue
                    row = []
                    for tag, _ in others:
                        d = np.array([S["bpebyte"][k][mi] - S[tag][k][mi] for k in ks], float)
                        m, lo, hi, p = boot_mean(d, rng)
                        row.append(f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.3f}")
                    P(f"| {task} | {CN[c]} | " + " | ".join(row) + " |")


def main():
    items = {r["key"]: r for r in map(json.loads, open(D / "items.jsonl"))}
    ext = {r["key"] for r in map(json.loads, open(D / "items_ext.jsonl"))}
    S = {}
    (D / "scored").mkdir(exist_ok=True)
    for tag, _ in MODELS:
        f = D / f"out/{tag}.jsonl"
        if not f.exists():
            continue
        out = {r["key"]: r["output"] for r in map(json.loads, open(f))}
        print(f"{tag}: {len(out)} outputs ({'full' if set(out) >= set(items) else 'ext subset' if set(out) >= ext else 'INCOMPLETE'})")
        S[tag] = {}
        with open(D / f"scored/{tag}.jsonl", "w") as g:
            for k, o in out.items():
                it = items[k]
                p, s = paper(it, o), strict(it, o)
                S[tag][k] = (int(p), int(s))
                g.write(json.dumps({"key": k, "paper": int(p), "strict": int(s), "output": o}, ensure_ascii=False) + "\n")
    full = [(t, n) for t, n in MODELS if t in S and set(S[t]) >= set(items)]
    shared = [(t, n) for t, n in MODELS if t in S and set(S[t]) >= ext]
    L = ["# NUMCoT on the 1B models: tables\n",
         "Accuracy %. `paper` = NUMCoT scoring (gold anywhere in the response); `strict` = first answer only. "
         "† external models (BLT: items whose prompt + budget exceed 4,096 bytes are left-truncated). "
         "CIs: item bootstrap B=10,000, seed 0; differences paired by item.\n"]
    rng = np.random.default_rng(0)
    if full:
        section(L, "A. Full protocol (every item, all conditions)", items, S, full, rng)
    if shared:
        section(L, "B. Shared subset (all models): few-shot on every item, few-shot CoT (+ knowledge) on 50 "
                   "questions per split", {k: items[k] for k in items if k in ext}, S, shared, rng)
    (D / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
