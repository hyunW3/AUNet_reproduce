#!/usr/bin/env python3
"""Export the exact byte strings each benchmark feeds the models, for patch statistics.

One row per scored item: the gold continuation appended to its context (ctx + cont), i.e. the
sequence a byte model patches when it scores the correct answer. BOS is not included; each
model's own measurement script adds it where the model does.

Sources (every text is byte-identical to what the paper's evaluations scored):
  dclm        reports/bpb_windows.jsonl                       (BPB column, 320 x 3840 B)
  hellaswag / arc_easy / arc_challenge / piqa / boolq
              runs/robustness_paper1p3b{,_ext}/llama samples   (lm-eval clean tasks, limit 2000)
  winogrande / mmlu_text
              lm-eval 0.4 requests, 0-shot                      (not in the robustness runs;
              mmlu_text = the paper's cloze MMLU: bare question + "Answer:" + answer text)
  sniah1/2/3  reports/niah/sniah123_n250_le4096_pairs.jsonl    (tab:sniah_full, 4 lengths x 250)
  noise / typo
              the same robustness samples for every variant    (first --pert_limit items per variant)
  despace     apps.aunet.eval_despace_mc builder, variant despaceall (limit 2000)

Run with the lingua venv (lm-eval + apps.aunet):
  lingua/.venv/bin/python scripts/probes/patch_stats/export_texts.py
"""
import argparse, json, os, sys

L = "/mnt/ssd2/hyun2/AUNet"
OUT = f"{L}/reports/patch_stats/texts.jsonl"
ROB = ("robustness_paper1p3b", "robustness_paper1p3b_ext")
ROB_TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")


def gold_text(sample):
    """lm-eval sample -> ctx + gold continuation."""
    args, tgt = sample["arguments"], sample["target"]
    if isinstance(args, dict):                      # older lm-eval serialisation
        args = [(a["arg_0"], a["arg_1"]) for a in args.values()]
    ctx, cont = args[int(tgt)]
    return ctx + cont


def robustness_samples():
    S = {}
    for d in ROB:
        S.update(json.load(open(f"{L}/runs/{d}/llama/results.json"))["samples"])
    return S


def lm_eval_texts(task_names, limit):
    """ctx + gold continuation from lm-eval request construction (0-shot)."""
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager(include_path=f"{L}/lingua/eval_tasks")   # mmlu_text lives here
    out = []

    def leaves(d, prefix):
        for k, v in d.items():
            name = getattr(k, "group_name", k) if not isinstance(k, str) else k
            if isinstance(v, dict):
                yield from leaves(v, prefix)
            else:
                yield name, v

    for tn in task_names:
        for name, task in leaves(get_task_dict([tn], tm), tn):
            task.build_all_requests(limit=limit, rank=0, world_size=1)
            by_doc = {}
            for inst in task.instances:
                by_doc.setdefault(inst.doc_id, []).append(inst)
            for doc_id, insts in sorted(by_doc.items()):
                doc = insts[0].doc
                if task.multiple_input:                  # winogrande: gold picks the context
                    gold = task.doc_to_text(doc)
                else:
                    gold = task.doc_to_target(doc)
                if isinstance(gold, str):
                    gold = task.doc_to_choice(doc).index(gold)
                inst = next(i for i in insts if i.idx == int(gold))
                ctx, cont = inst.arguments
                out.append((tn, f"{name}/{doc_id}", ctx + cont))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert_limit", type=int, default=400,
                    help="items per noise/typo variant (15 noise + 8 typo variants x 5 tasks)")
    ap.add_argument("--limit", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []

    for w in map(json.loads, open(f"{L}/reports/bpb_windows.jsonl")):
        rows.append(("dclm", "dclm", str(w["idx"]), w["text"]))

    S = robustness_samples()
    for t in ROB_TASKS:
        for s in sorted(S[t], key=lambda s: int(s["doc_id"]))[:a.limit]:
            rows.append(("downstream", t, str(s["doc_id"]), gold_text(s)))
    for t, key, text in lm_eval_texts(["winogrande", "mmlu_text"], None):
        rows.append(("downstream", t, key, text))

    for p in map(json.loads, open(f"{L}/reports/niah/sniah123_n250_le4096_pairs.jsonl")):
        rows.append(("sniah", p["probe"], p["cell"], p["prompt"] + " " + p["values"][0]))

    for axis in ("noise", "typo"):
        for k in sorted(S):
            if not any(k.startswith(f"{t}_{axis}_") for t in ROB_TASKS) or k.endswith("_avg"):
                continue
            t = next(t for t in ROB_TASKS if k.startswith(f"{t}_{axis}_"))
            for s in sorted(S[k], key=lambda s: int(s["doc_id"]))[:a.pert_limit]:
                rows.append((axis, t, f"{k}/{s['doc_id']}", gold_text(s)))

    sys.path.insert(0, f"{L}/lingua")
    os.environ["DESPACE_FULL"] = "1"      # despaceall exists only with the 100% endpoint enabled
    os.environ["DESPACE_ANSWER"] = "1"
    from apps.aunet.eval_despace_mc import _build, _items_for
    for t in ROB_TASKS:
        for ii, item in enumerate(_items_for(t, a.limit)):
            ctx, cont = _build(item, "despaceall", ii)[item["gold"]]
            rows.append(("despace", t, str(ii), ctx + cont))

    with open(OUT, "w") as f:
        for group, bench, key, text in rows:
            f.write(json.dumps({"group": group, "bench": bench, "key": key, "text": text}) + "\n")
    from collections import Counter
    c = Counter((g, b) for g, b, _, _ in rows)
    nb = Counter()
    for g, b, _, t in rows:
        nb[(g, b)] += len(t.encode())
    for k in sorted(c):
        print(f"{k[0]:10s} {k[1]:14s} items={c[k]:6d} bytes={nb[k]:10d}")
    print("total bytes", sum(nb.values()), "->", OUT)


if __name__ == "__main__":
    main()
