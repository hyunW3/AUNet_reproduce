#!/usr/bin/env python3
"""Answer surface-form robustness: the same BoolQ / LAMBADA items answered in 9 surface forms of the gold
word (as-is, lower, Title, UPPER, "quoted", [bracketed], period., **bold**, s p a c e d), 3 shots whose
demonstrations use the same form, greedy generation to the first newline. Per item: format (exact regex),
correct (answer recovered after undoing the form, case-insensitive), joint. The as-is form is the baseline,
so form cost = correct(form) - correct(as-is) on the same items.

Each row also records whether the Transformer (Llama-3) tokens of " " + form(gold) differ from those of
" " + gold (`tok_changed`), to separate forms that change the subword answer from forms that do not.

  python surface_bench.py --model bpebyte_1.3b --out reports/surface/bpebyte_1.3b/surface_n500.json
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models  # noqa: E402
import cfqa_bench as C  # noqa: E402

FORMS = {  # name -> (instruction, transform, format regex)
    "asis": (None, lambda w: w, None),
    "lower": ("Write the answer in lowercase letters.", str.lower, r"[a-z0-9'’-]+"),
    "title": ("Write the answer with only its first letter capitalized.", lambda w: w[:1].upper() + w[1:].lower(),
              r"[A-Z0-9][a-z0-9'’-]*"),
    "upper": ("Write the answer in capital letters.", str.upper, r"[A-Z0-9'’-]+"),
    "quote": ("Wrap the answer in double quotes.", lambda w: f'"{w}"', r'"[^"\s]+"'),
    "bracket": ("Wrap the answer in square brackets.", lambda w: f"[{w}]", r"\[[^\]\s]+\]"),
    "period": ("End the answer with a period.", lambda w: f"{w}.", r"[^\s.]+\."),
    "bold": ("Wrap the answer in double asterisks.", lambda w: f"**{w}**", r"\*\*[^*\s]+\*\*"),
    "spaced": ("Write the answer with a space between every two letters.", lambda w: " ".join(w), r"\S( \S)+"),
}
LEAD = {"boolq": "Read the passage and answer the question with one word, yes or no.",
        "lambada": "Write only the missing last word of the passage."}


def prompt(task, form, item, demos):
    instr, tf, _ = FORMS[form]
    head = LEAD[task] + (" " + instr if instr else "")
    shots = "".join(f"{d['body']}\nAnswer: {tf(d['gold'])}\n\n" for d in demos)
    return f"{head}\n\n{shots}{item['body']}\nAnswer:"


def undo(form, line):
    s = line.strip()
    if form == "spaced":
        s = s.replace(" ", "")
    s = s.strip('"*[].').strip()
    m = re.match(r"[\w'’-]+", s)
    return m.group(0).lower() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_gen_toks", type=int, default=24)
    ap.add_argument("--llama_tok", default=None, help="consolidated dir of llama_1.3b (tokenizer for tok_changed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    spec = models.resolve(a.model)
    C.run_ext._lm_eval_compat()
    if os.environ.get("IFEVAL_EAGER_MASK") and spec["family"] == "lingua_aunet":
        import apps.aunet.generate as _g
        _cbm = _g.create_block_mask
        _g.create_block_mask = lambda *x, **kw: _cbm(*x, **{**kw, "_compile": False})
    lm, _ = models.build(spec, 4096)
    ltok = lm.generator.tokenizer if spec["family"] == "lingua_main" else None

    data = C.load_tasks(a.n, a.seed)
    data = {t: data[t] for t in ("boolq", "lambada")}
    reqs, meta = [], []
    for task, d in data.items():
        for form in FORMS:
            for i, item in enumerate(d["test"]):
                reqs.append(C.Req((prompt(task, form, item, d["train"][:3]),
                                   {"until": ["\n"], "max_gen_toks": a.max_gen_toks, "do_sample": False, "temperature": 0.0})))
                meta.append((task, form, i))
    print(f"[surface] {len(reqs)} generations", file=sys.stderr, flush=True)
    resps = lm.generate_until(reqs)
    rows = []
    for (task, form, i), r in zip(meta, resps):
        gold = data[task]["test"][i]["gold"]
        line = r.split("\n")[0].strip()
        rx = FORMS[form][2]
        fmt = None if rx is None else int(bool(re.fullmatch(rx, line)))
        correct = int(undo(form, line) == gold.lower())
        row = {"task": task, "form": form, "i": i, "gold": gold, "line": line, "correct": correct,
               "format": fmt, "joint": None if fmt is None else int(fmt and correct)}
        if ltok is not None:
            enc = lambda s: ltok.encode(s, add_bos=False, add_eos=False)
            ft = enc(" " + FORMS[form][1](gold))
            row["tok_changed"] = int(ft != enc(" " + gold))
            # the gold word's own subword tokens survive inside the form (only wrappers/punctuation added)
            row["core_kept"] = int(any(any(ft[j:j + len(s)] == s for j in range(len(ft) - len(s) + 1))
                                       for s in (enc(" " + gold), enc(gold))))
        rows.append(row)
    summ = {}
    for r in rows:
        s = summ.setdefault(f"{r['task']}/{r['form']}", {"n": 0, "correct": 0, "format": 0, "joint": 0})
        s["n"] += 1
        for m in ("correct", "format", "joint"):
            s[m] += r[m] or 0
    for s in summ.values():
        for m in ("correct", "format", "joint"):
            s[m] = 100 * s[m] / s["n"]
    out = {"model": a.model, "n": a.n, "seed": a.seed, "summary": summ, "rows": rows, "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"\n{a.model}  surface-form  n={a.n}/form, 3-shot")
    for k, s in summ.items():
        print(f"  {k:18s} correct {s['correct']:5.1f}  format {s['format']:5.1f}  joint {s['joint']:5.1f}")
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()
