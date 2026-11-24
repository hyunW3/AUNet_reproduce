#!/usr/bin/env python3
"""Prompt-format robustness by log-likelihood scoring (the EvalHarnessLM.loglikelihood path of the
standard benchmarks). Same items, many prompt formats; per format the task metric per item.

Modes
  grid         FormatSpread-style grid (Sclar et al. 2023): field names x name/value separator x field
               join, a fixed sample of formats always containing the canonical one, at 0 and 3 shots.
  promptsource every PromptSource template of the dataset that scores the original task with
               answer_choices (Bach et al. 2022), 0-shot, continuation = " " + choice (T0 / lm-eval style).
  demos        canonical format, 3 shots: 5 demonstration sets in a fixed order, and all 6 orders of set 0.
  cloze        LAMBADA in the GPT-3 cloze frame "<passage> <blank> <cue>" + " <word>" (Brown et al. 2020),
               blank marker x answer cue (CLOZE_BLANKS x CLOZE_CUES, "none" = omitted; none/none is the
               standard format) at 0/3/5 shots. Items: the first n test passages (same items as the
               reports/std_bench/lambada template ablation); demonstrations: 5 fixed passages after them.
  clozek       the cue contrast "____. ->" vs "____. Answer:" with demonstrations drawn per item (as lm-eval
               does), k in CLOZEK_K, 3 seeds, from a pool of 200 passages after the test items: separates a
               shot-count effect from the idiosyncrasy of one fixed demonstration set.
  clozefull    the full blank x cue grid of `cloze` with per-item demonstrations (k in CLOZEFULL_K, 2 seeds;
               the same draw for every frame, so frame contrasts are paired).

Tasks: BoolQ (acc), PIQA (acc_norm), LAMBADA (greedy exact match of " word"); HellaSwag (acc_norm) in the
promptsource mode. n items per task, fixed seed, identical across models and formats.

Every (mode, task, config) chunk is appended to --out_jsonl as it finishes and skipped on rerun, so a crash
loses one chunk (run_fmt_robust.sh restarts). For the Transformer each request also records whether the
prompt's tokens are a prefix of the prompt+answer tokens (`aligned`); when they are not, the subword
scoring boundary moves (the PBP effect) and the format is reported separately.

  python fmt_robust.py --model llama_1.3b --mode grid --out_jsonl reports/fmt_robust/llama_1.3b/grid.jsonl
"""
import argparse, itertools, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models  # noqa: E402
sys.path.insert(0, f"{models.REPO}/scripts/probes/ext_ci")
import run_ext  # noqa: E402

METRIC = {"boolq": "acc", "piqa": "acc_norm", "lambada": "greedy", "hellaswag": "acc_norm"}


# ------------------------------------------------------------------------------------------- data
def load_data(n, seed, tasks):
    from datasets import load_dataset
    rng = random.Random(seed)
    out = {}
    if "boolq" in tasks:
        bq = load_dataset("aps/super_glue", "boolq")
        conv = lambda d: {"passage": d["passage"], "question": d["question"], "choices": ["no", "yes"],
                          "gold": int(d["label"]), "raw": dict(d)}
        out["boolq"] = {"test": [conv(d) for d in rng.sample(list(bq["validation"]), n)],
                        "pool": [conv(d) for d in rng.sample([d for d in bq["train"] if len(d["passage"]) < 600], 15)]}
    if "piqa" in tasks:
        pq = load_dataset("baber/piqa")
        conv = lambda d: {"question": d["goal"], "choices": [d["sol1"], d["sol2"]], "gold": int(d["label"]), "raw": dict(d)}
        out["piqa"] = {"test": [conv(d) for d in rng.sample(list(pq["validation"]), n)],
                       "pool": [conv(d) for d in rng.sample(list(pq["train"]), 15)]}
    if "lambada" in tasks:
        lb = list(load_dataset("EleutherAI/lambada_openai", "default")["test"])
        def conv(d):
            ctx, w = d["text"].rsplit(" ", 1)
            return {"question": ctx + " ___", "choices": [w], "gold": 0, "raw": dict(d)}
        picked = rng.sample(range(len(lb)), n + 15)
        out["lambada"] = {"test": [conv(lb[i]) for i in picked[:n]], "pool": [conv(lb[i]) for i in picked[n:]]}
    if "hellaswag" in tasks:
        hs = load_dataset("Rowan/hellaswag")
        conv = lambda d: {"choices": d["endings"], "gold": int(d["label"]), "raw": dict(d)}
        out["hellaswag"] = {"test": [conv(d) for d in rng.sample(list(hs["validation"]), n)], "pool": []}
    return out


# ---------------------------------------------------------------------------------------- formats
NAMES = [("Question", "Answer"), ("Q", "A"), ("QUESTION", "ANSWER"), ("question", "answer"), ("Input", "Output")]
SEPS = [":", " -", " ->", " ="]
JOINS = ["\n", " ", "\n\n"]
CANON = (0, 0, 0)
CLOZE_BLANKS = {"____.": "____.", "___": "___", "BLANK.": "BLANK.", "[blank].": "[blank].", "none": ""}
CLOZEK_K = (1, 2, 3, 4, 5, 8)
CLOZEFULL_K = (1, 3, 5)
CLOZE_CUES = {"->": "->", "Answer:": "Answer:", "=>": "=>", "=": "=", "Word:": "Word:", "none": ""}


def cloze_ctx(passage, blank, cue):
    return " ".join([passage] + [x for x in (CLOZE_BLANKS[blank], CLOZE_CUES[cue]) if x])


def grid_formats(k=12, seed=0):
    allf = [f for f in itertools.product(range(len(NAMES)), range(len(SEPS)), range(len(JOINS))) if f != CANON]
    return [CANON] + random.Random(seed).sample(allf, k - 1)


def fmt_name(f):
    (q, a), s, j = NAMES[f[0]], SEPS[f[1]], JOINS[f[2]]
    return f"{q}{s}|{a}{s}|join={j!r}"


def render(task, item, f):
    """-> (body ending right before the answer, list of continuations)"""
    (qn, an), sep, join = NAMES[f[0]], SEPS[f[1]], JOINS[f[2]]
    if task == "boolq":
        body = f"{item['passage']}{join}{qn}{sep} {item['question']}?{join}{an}{sep}"
    else:
        body = f"{qn}{sep} {item['question']}{join}{an}{sep}"
    return body, [" " + c for c in item["choices"]]


def build_ctx(task, item, f, demos):
    shots = ""
    for d in demos:
        b, conts = render(task, d, f)
        shots += b + conts[d["gold"]] + "\n\n"
    b, conts = render(task, item, f)
    return shots + b, conts


# ----------------------------------------------------------------------------------- promptsource
def load_promptsource(path):
    """Parse a PromptSource templates.yaml (custom !Template tags) without the promptsource package."""
    import yaml

    class L(yaml.SafeLoader):
        pass
    L.add_multi_constructor("!", lambda loader, suffix, node: loader.construct_mapping(node, deep=True))
    y = yaml.load(open(path), Loader=L)
    out = []
    for tid, t in y["templates"].items():
        md = t.get("metadata") or {}
        if t.get("answer_choices") and md.get("original_task", False) and "|||" in t["jinja"]:
            out.append({"id": tid, "name": t["name"], "jinja": t["jinja"], "answer_choices": t["answer_choices"]})
    return out


def ps_render(tmpl, doc):
    from jinja2.sandbox import SandboxedEnvironment
    env = SandboxedEnvironment()
    choices = [c.strip() for c in env.from_string(tmpl["answer_choices"]).render(**doc).split("|||")]
    inp, tgt = [s.strip() for s in env.from_string(tmpl["jinja"]).render(**doc, answer_choices=choices).split("|||")[:2]]
    return inp, choices, tgt


# -------------------------------------------------------------------------------------------- run
class Req:
    def __init__(self, args):
        self.args = args


def score_chunk(lm, task, items, ctxs_conts, tok=None):
    reqs, owner = [], []
    for i, (ctx, conts) in enumerate(ctxs_conts):
        for c in conts:
            reqs.append(Req((ctx, c)))
            owner.append(i)
    res = lm.loglikelihood(reqs)
    rows = [{"lls": [], "greedy": [], "lens": [], "aligned": []} for _ in ctxs_conts]
    for (ll, gr), o, rq in zip(res, owner, reqs):
        r = rows[o]
        r["lls"].append(float(ll)); r["greedy"].append(bool(gr)); r["lens"].append(len(rq.args[1].encode()))
        if tok is not None:
            pc = tok.encode(rq.args[0], add_bos=False, add_eos=False)
            full = tok.encode(rq.args[0] + rq.args[1], add_bos=False, add_eos=False)
            r["aligned"].append(full[:len(pc)] == pc)
    out = []
    for it, r, (ctx, conts) in zip(items, rows, ctxs_conts):
        gold = it["gold"]
        acc = int(max(range(len(conts)), key=lambda j: r["lls"][j]) == gold)
        accn = int(max(range(len(conts)), key=lambda j: r["lls"][j] / r["lens"][j]) == gold)
        m = {"acc": acc, "acc_norm": accn, "greedy": int(r["greedy"][gold])}
        out.append({"correct": m[METRIC[task]], **m, "aligned": all(r["aligned"]) if r["aligned"] else None,
                    "ctx_tail": ctx[-120:]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["grid", "promptsource", "demos", "cloze", "clozek", "clozefull"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_formats", type=int, default=12)
    ap.add_argument("--ps_dir", default=None, help="dir with <dataset>/templates.yaml from PromptSource")
    ap.add_argument("--max_tokens", type=int, default=None, help="lingua generator packing budget override")
    ap.add_argument("--out_jsonl", required=True)
    a = ap.parse_args()
    spec = models.resolve(a.model, max_tokens=a.max_tokens)
    run_ext._lm_eval_compat()
    if os.environ.get("IFEVAL_EAGER_MASK") and spec["family"] == "lingua_aunet":
        # same compiled create_block_mask illegal-memory-access as in ifeval_bench.py; it also hits the
        # loglikelihood prefill (bpebyte_1.3b, grid/piqa/1-2-2/3shot, every time). Eager = identical mask.
        import apps.aunet.generate as _g
        _cbm = _g.create_block_mask
        _g.create_block_mask = lambda *x, **kw: _cbm(*x, **{**kw, "_compile": False})
    lm, _ = models.build(spec, 4096)
    if os.environ.get("FLEX_CAP"):
        # torch 2.7 picks flex-attention block configs by compute capability; on sm_120 (RTX PRO 6000
        # Blackwell, ~99 KB shared memory) the >= (9, 0) H100 branch needs 107 KB and fails. Report the
        # given capability (e.g. 8,6 = the RTX A5000 configs of the local runs) to that selection only.
        import torch
        import torch._inductor.lowering  # noqa: F401  (import order: avoids a circular import)
        import torch._inductor.kernel.flex_attention as _fa
        _cap = tuple(int(x) for x in os.environ["FLEX_CAP"].split(","))
        _nv = _fa._get_nv_config

        def _nv_cap(*x, **kw):
            saved = torch.cuda.get_device_capability
            torch.cuda.get_device_capability = lambda *_a, **_k: _cap
            try:
                return _nv(*x, **kw)
            finally:
                torch.cuda.get_device_capability = saved
        _fa._get_nv_config = _nv_cap
    tok = lm.generator.tokenizer if spec["family"] == "lingua_main" else None
    done = set()
    if os.path.exists(a.out_jsonl):
        done = {json.loads(l)["chunk"] for l in open(a.out_jsonl) if l.strip()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out_jsonl)), exist_ok=True)

    chunks = []  # (chunk id, task, items, ctxs_conts, meta)
    if a.mode in ("grid", "demos"):
        data = load_data(a.n, a.seed, ["boolq", "piqa", "lambada"])
        for task, d in data.items():
            if a.mode == "grid":
                for f in grid_formats(a.n_formats):
                    for k in (0, 3):
                        cid = f"grid/{task}/{f[0]}-{f[1]}-{f[2]}/{k}shot"
                        chunks.append((cid, task, d["test"], lambda d=d, f=f, k=k, task=task:
                                       [build_ctx(task, it, f, d["pool"][:k]) for it in d["test"]],
                                       {"format": fmt_name(f), "fmt": f, "shots": k, "canonical": f == CANON}))
            else:
                configs = [(s, (0, 1, 2)) for s in range(5)] + [(0, p) for p in itertools.permutations(range(3)) if p != (0, 1, 2)]
                for s, p in configs:
                    demos = [d["pool"][3 * s + j] for j in p]
                    cid = f"demos/{task}/set{s}/order{''.join(map(str, p))}"
                    chunks.append((cid, task, d["test"], lambda d=d, demos=demos, task=task:
                                   [build_ctx(task, it, CANON, demos) for it in d["test"]],
                                   {"set": s, "order": list(p)}))
    elif a.mode == "cloze":
        from datasets import load_dataset
        lb = list(load_dataset("EleutherAI/lambada_openai", "default")["test"])
        split = lambda d: d["text"].rsplit(" ", 1)
        test = [{"passage": split(d)[0], "word": split(d)[1], "choices": [split(d)[1]], "gold": 0} for d in lb[:a.n]]
        pool = [lb[i] for i in random.Random(a.seed).sample(range(a.n, len(lb)), 5)]
        pool = [{"passage": split(d)[0], "word": split(d)[1]} for d in pool]
        for blank in CLOZE_BLANKS:
            for cue in CLOZE_CUES:
                for k in (0, 3, 5):
                    def mk(blank=blank, cue=cue, k=k):
                        shots = "".join(cloze_ctx(d["passage"], blank, cue) + " " + d["word"] + "\n\n" for d in pool[:k])
                        return [(shots + cloze_ctx(it["passage"], blank, cue), [" " + it["word"]]) for it in test]
                    chunks.append((f"cloze/lambada/{blank}|{cue}/{k}shot", "lambada", test, mk,
                                   {"blank": blank, "cue": cue, "shots": k}))
    elif a.mode == "clozefull":
        from datasets import load_dataset
        lb = list(load_dataset("EleutherAI/lambada_openai", "default")["test"])
        split = lambda d: d["text"].rsplit(" ", 1)
        test = [{"passage": split(d)[0], "word": split(d)[1], "choices": [split(d)[1]], "gold": 0} for d in lb[:a.n]]
        pool = [{"passage": split(lb[i])[0], "word": split(lb[i])[1]}
                for i in random.Random(a.seed).sample(range(a.n, len(lb)), 200)]
        for seed in range(2):
            for k in CLOZEFULL_K:
                draws = [random.Random(seed * 100003 + i).sample(range(len(pool)), k) for i in range(len(test))]
                for blank in CLOZE_BLANKS:
                    for cue in CLOZE_CUES:
                        def mk(blank=blank, cue=cue, draws=draws):
                            out = []
                            for it, dr in zip(test, draws):
                                shots = "".join(cloze_ctx(pool[j]["passage"], blank, cue) + " " + pool[j]["word"] + "\n\n" for j in dr)
                                out.append((shots + cloze_ctx(it["passage"], blank, cue), [" " + it["word"]]))
                            return out
                        chunks.append((f"clozefull/lambada/seed{seed}/k{k}/{blank}|{cue}", "lambada", test, mk,
                                       {"blank": blank, "cue": cue, "shots": k, "seed": seed}))
    elif a.mode == "clozek":
        from datasets import load_dataset
        lb = list(load_dataset("EleutherAI/lambada_openai", "default")["test"])
        split = lambda d: d["text"].rsplit(" ", 1)
        test = [{"passage": split(d)[0], "word": split(d)[1], "choices": [split(d)[1]], "gold": 0} for d in lb[:a.n]]
        pool = [{"passage": split(lb[i])[0], "word": split(lb[i])[1]}
                for i in random.Random(a.seed).sample(range(a.n, len(lb)), 200)]
        for seed in range(3):
            for k in CLOZEK_K:
                # per-item demonstration draw, shared by both cues so the contrast is paired
                draws = [random.Random(seed * 100003 + i).sample(range(len(pool)), k) for i in range(len(test))]
                for cue in ("->", "Answer:"):
                    def mk(cue=cue, draws=draws):
                        out = []
                        for it, dr in zip(test, draws):
                            shots = "".join(cloze_ctx(pool[j]["passage"], "____.", cue) + " " + pool[j]["word"] + "\n\n" for j in dr)
                            out.append((shots + cloze_ctx(it["passage"], "____.", cue), [" " + it["word"]]))
                        return out
                    chunks.append((f"clozek/lambada/seed{seed}/k{k}/{cue}", "lambada", test, mk,
                                   {"cue": cue, "k": k, "seed": seed}))
    else:
        data = load_data(a.n, a.seed, ["boolq", "piqa", "hellaswag"])
        ds_dir = {"boolq": "super_glue/boolq", "piqa": "piqa", "hellaswag": "hellaswag"}
        for task, d in data.items():
            for t in load_promptsource(os.path.join(a.ps_dir, ds_dir[task], "templates.yaml")):
                def mk(d=d, t=t):
                    cc, keep = [], True
                    for it in d["test"]:
                        inp, choices, tgt = ps_render(t, it["raw"])
                        if tgt not in choices:
                            raise ValueError(f"target {tgt!r} not in choices {choices}")
                        it["ps_gold"] = choices.index(tgt)
                        cc.append((inp, [" " + c for c in choices]))
                    return cc
                cid = f"promptsource/{task}/{t['id']}"
                chunks.append((cid, task, d["test"], mk, {"template": t["name"]}))

    todo = [c for c in chunks if c[0] not in done]
    print(f"[fmt_robust] {a.mode}: {len(chunks)} chunks, {len(done)} done, {len(todo)} to go", file=sys.stderr, flush=True)
    for cid, task, items, mk, meta in todo:
        t0 = time.time()
        try:
            cc = mk()
        except Exception as e:  # a PromptSource template that does not render on these items
            print(f"[fmt_robust] skip {cid}: {e}", file=sys.stderr, flush=True)
            with open(a.out_jsonl, "a") as fo:
                fo.write(json.dumps({"chunk": cid, "task": task, "skipped": str(e)[:300], **meta}) + "\n")
            continue
        its = [dict(it, gold=it.get("ps_gold", it["gold"])) for it in items] if a.mode == "promptsource" else items
        rows = score_chunk(lm, task, its, cc, tok)
        rec = {"chunk": cid, "task": task, **meta, "n": len(rows), "acc": sum(r["correct"] for r in rows) / len(rows),
               "rows": [{"i": i, **r} for i, r in enumerate(rows)], "seconds": round(time.time() - t0)}
        with open(a.out_jsonl, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(f"[fmt_robust] {cid}  {100 * rec['acc']:.1f}  ({rec['seconds']}s)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
