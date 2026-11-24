#!/usr/bin/env python3
"""Constrained-format QA: answer correctness AND verifiable format-instruction compliance, scored separately.

Tasks with gold answers that 1B base models partly solve (BoolQ, ARC-Easy, LAMBADA), each under a plain
condition (no format instruction) and two verifiable format instructions, at 0 and 3 shots (demonstrations
follow the instruction). Greedy, stop at the first newline. Per item:
  format  - the stripped first line matches the instruction's format exactly (regex)
  correct - the answer extracted leniently from the response equals the gold (format-agnostic)
  joint   - format and correct
Plain conditions have no format, so only `correct` is reported for them.

  python cfqa_bench.py --model bpebyte_1.3b --out reports/cfqa/bpebyte_1.3b/cfqa.json [--n 500] [--shots 0 3]
Run through run_cfqa.sh. IFEVAL_EAGER_MASK=1 avoids the compiled create_block_mask crash of the
lingua_aunet generators (see ifeval_bench.py); the outputs are identical.
"""
import argparse, json, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models  # noqa: E402
sys.path.insert(0, f"{models.REPO}/scripts/probes/ext_ci")
import run_ext  # noqa: E402

LETTERS = "ABCDE"

# ------------------------------------------------------------------------------------------- tasks
def load_tasks(n, seed):
    from datasets import load_dataset
    rng = random.Random(seed)
    out = {}
    bq = load_dataset("aps/super_glue", "boolq")
    test = list(bq["validation"]); train = list(bq["train"])
    out["boolq"] = {"test": [{"body": f"Passage: {d['passage']}\nQuestion: {d['question']}?", "gold": "yes" if d["label"] else "no"}
                             for d in rng.sample(test, n)],
                    "train": [{"body": f"Passage: {d['passage']}\nQuestion: {d['question']}?", "gold": "yes" if d["label"] else "no"}
                              for d in rng.sample([d for d in train if len(d["passage"]) < 600], 3)]}

    def arc(d):
        labels = d["choices"]["label"]
        idx = {l: i for i, l in enumerate(labels)}
        opts = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(d["choices"]["text"]))
        return {"body": f"Question: {d['question']}\n{opts}", "gold": LETTERS[idx[d["answerKey"]]],
                "options": d["choices"]["text"]}
    ae = load_dataset("allenai/ai2_arc", "ARC-Easy")
    out["arc_easy"] = {"test": [arc(d) for d in rng.sample(list(ae["test"]), n)],
                       "train": [arc(d) for d in rng.sample(list(ae["train"]), 3)]}

    lb = list(load_dataset("EleutherAI/lambada_openai", "default")["test"])
    def lam(d):
        ctx, w = d["text"].rsplit(" ", 1)
        return {"body": f"Passage: {ctx} ___", "gold": w}
    picked = rng.sample(range(len(lb)), n + 3)
    out["lambada"] = {"test": [lam(lb[i]) for i in picked[:n]], "train": [lam(lb[i]) for i in picked[n:]]}
    return out


# conditions: name -> (instruction or None, render gold for demos, strict format regex or None)
CONDS = {
    "boolq": {
        "plain": (None, lambda g, d: g, None),
        "lower": ('Answer with exactly one word, "yes" or "no", in lowercase.', lambda g, d: g, r"(yes|no)"),
        "caps_quote": ('Answer with exactly one word in capital letters wrapped in double quotes, "YES" or "NO".',
                       lambda g, d: f'"{g.upper()}"', r'"(YES|NO)"'),
    },
    "arc_easy": {
        "plain": (None, lambda g, d: g, None),
        "letter": ("Answer with only the letter of the correct option.", lambda g, d: g, r"[A-E]"),
        "bracket": ("Answer with only the letter of the correct option wrapped in square brackets, e.g. [A].",
                    lambda g, d: f"[{g}]", r"\[[A-E]\]"),
    },
    "lambada": {
        "plain": (None, lambda g, d: g, None),
        "upper": ("Write only the missing last word of the passage, in capital letters.",
                  lambda g, d: g.upper(), r"[A-Z0-9'’-]+"),
        "quote": ("Write only the missing last word of the passage, wrapped in double quotes.",
                  lambda g, d: f'"{g}"', r'"[^"\s]+"'),
    },
}
INTRO = {"boolq": "Read the passage and answer the question.",
         "arc_easy": "Answer the multiple-choice science question.",
         "lambada": "Fill in the missing last word of the passage."}


def prompt(task, cond, item, demos):
    instr, render, _ = CONDS[task][cond]
    head = INTRO[task] + (" " + instr if instr else "")
    shots = "".join(f"{d['body']}\nAnswer: {render(d['gold'], d)}\n\n" for d in demos)
    return f"{head}\n\n{shots}{item['body']}\nAnswer:"


# ------------------------------------------------------------------------------------------ scoring
def extract(task, resp, item):
    """Lenient, format-agnostic answer extraction from the first line."""
    if task == "boolq":
        m = re.search(r"\b(yes|no)\b", resp, re.I)
        return m.group(1).lower() if m else None
    if task == "arc_easy":
        m = re.search(r"(?<![A-Za-z])([A-E])(?![a-z])", resp)
        if m:
            return m.group(1)
        low = resp.strip().lower()
        for i, t in enumerate(item["options"]):          # option text instead of the letter
            if low and low.startswith(t.strip().lower()):
                return LETTERS[i]
        return None
    m = re.search(r"[\w'’-]+", resp)
    return m.group(0).lower() if m else None


def score(task, cond, resp, item):
    line = resp.split("\n")[0].strip()
    ans = extract(task, line, item)
    gold = item["gold"].lower() if task != "arc_easy" else item["gold"]
    correct = ans is not None and ans == gold
    fmt_re = CONDS[task][cond][2]
    fmt = None if fmt_re is None else bool(re.fullmatch(fmt_re, line))
    return {"line": line, "answer": ans, "correct": int(correct),
            "format": None if fmt is None else int(fmt), "joint": None if fmt is None else int(fmt and correct)}


class Req:
    def __init__(self, args):
        self.args = args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--max_gen_toks", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    spec = models.resolve(a.model)
    run_ext._lm_eval_compat()
    if os.environ.get("IFEVAL_EAGER_MASK") and spec["family"] == "lingua_aunet":
        import apps.aunet.generate as _g
        _cbm = _g.create_block_mask
        _g.create_block_mask = lambda *x, **kw: _cbm(*x, **{**kw, "_compile": False})
    lm, _ = models.build(spec, a.max_len)

    tasks = load_tasks(a.n, a.seed)
    reqs, meta = [], []
    for task, d in tasks.items():
        for cond in CONDS[task]:
            for k in a.shots:
                for i, item in enumerate(d["test"]):
                    reqs.append(Req((prompt(task, cond, item, d["train"][:k]),
                                     {"until": ["\n"], "max_gen_toks": a.max_gen_toks, "do_sample": False, "temperature": 0.0})))
                    meta.append((task, cond, k, i))
    print(f"[cfqa] {len(reqs)} generations", file=sys.stderr, flush=True)
    resps = lm.generate_until(reqs)
    rows = []
    for (task, cond, k, i), r, q in zip(meta, resps, reqs):
        rows.append({"task": task, "cond": cond, "shots": k, "i": i, "gold": tasks[task]["test"][i]["gold"],
                     "prompt_tail": q.args[0][-200:], "response": r, **score(task, cond, r, tasks[task]["test"][i])})
    summ = {}
    for r in rows:
        s = summ.setdefault(f"{r['task']}/{r['cond']}/{r['shots']}", {"n": 0, "correct": 0, "format": 0, "joint": 0})
        s["n"] += 1
        for m in ("correct", "format", "joint"):
            s[m] += r[m] or 0
    for s in summ.values():
        for m in ("correct", "format", "joint"):
            s[m] = 100 * s[m] / s["n"]
    out = {"model": a.model, "n": a.n, "seed": a.seed, "shots": a.shots, "max_gen_toks": a.max_gen_toks,
           "summary": summ, "rows": rows, "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"\n{a.model}  constrained-format QA  n={a.n}/condition")
    print(f"  {'task/cond/shots':28s} {'correct':>8s} {'format':>8s} {'joint':>8s}")
    for key, s in summ.items():
        f, j = ("-", "-") if "/plain/" in key else (f"{s['format']:.1f}", f"{s['joint']:.1f}")
        print(f"  {key:28s} {s['correct']:8.1f} {f:>8s} {j:>8s}")
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()
