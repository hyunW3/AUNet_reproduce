#!/usr/bin/env python
"""results.json (log_samples=true) -> compact per-doc jsonl + soft metrics.

The raw lm-eval blob is ~58 MB per checkpoint for the 6-bench suite; this reduces it to
~3 MB of exactly the columns the scaling-law fit needs, and asserts on the way through
that our recomputation reproduces lm-eval's own per-doc acc / acc_norm.

Per-doc row:
    {task, doc_id, K, gold, ll[K], b_cont[K], b_choice[K], c_choice[K],
     is_greedy[K], acc, acc_norm, multiple_input}

  ll        filtered_resps[i][0], natural log, summed over the continuation
  b_cont    utf-8 bytes of arguments[i][1] -- INCLUDING the " " target delimiter,
            i.e. what the model actually scored
  b_choice  utf-8 bytes of the choice alone (what the patched acc_bytes uses)
  c_choice  characters of the choice alone (what acc_norm uses, task.py:1494)

Candidate soft metrics (plans/delphi_downstream_prediction_plan.md §4). Scores are
    z_c = ll_c * T / len_c
with len = c_choice (characters) for acc_norm tasks and b_cont (bytes) for acc tasks,
and T a reference length: t1 / t10 / t20, or `docmean` = the document's own mean choice
length. `docmean` is argmax-preserving w.r.t. the headline metric (scaling every z_c by
one positive constant cannot move the argmax), so its soft score stays consistent with
acc_norm. `raw` is the unnormalized sum-loglik, i.e. Delphi's stated MMLU metric, whose
argmax is `acc`. For each we emit X = mean_d -log softmax(z)[gold], p_gold = mean_d
softmax(z)[gold], and margin = mean_d [z_gold - max_{i!=gold} z_i]; plus bpb_gold =
mean_d [-ll_gold / (b_choice_gold * ln2)], Delphi's generative soft metric.

NOTHING IS COMMITTED HERE. A temperature sweep across two training stages of one 1.3B
run showed both ends of the scale are degenerate: at T=1 hellaswag X moves 1.280 ->
1.272 across a 19 pp accuracy gain and p_gold moves the WRONG WAY (-0.051), while at the
raw end the choice posterior is a near-delta and p_gold collapses onto accuracy. The fit
script selects among the panel by pre-registered inner CV on the fit set -- never on the
held-out 1.3B run.

Usage:
    python scripts/delphi_extract.py --dump_dir <dir> --out <per_doc.jsonl> [--summary s.json]
"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict

# The repo's canonical headline metric per task (code_appendix/eval_suite/paths.py).
ACC_NORM_TASKS = {"hellaswag", "arc_easy", "arc_challenge", "piqa"}
ACC_TASKS = {"winogrande", "boolq"}
LN2 = math.log(2.0)

# Reference lengths (characters for acc_norm tasks, continuation bytes for acc tasks)
# used as softmax temperatures.  None = the document's own mean choice length, which is
# argmax-preserving w.r.t. the headline metric.
TEMPERATURES = [("t1", 1.0), ("t10", 10.0), ("t20", 20.0), ("docmean", None)]


def load_results(dump_dir):
    for name in ("results.json", "results.json.gz"):
        p = os.path.join(dump_dir, name)
        if os.path.exists(p):
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt") as fh:
                return json.load(fh), p
    raise FileNotFoundError(f"no results.json[.gz] in {dump_dir}")


def choice_strings(arguments):
    """-> (choices, multiple_input).

    lm-eval's `multiple_input` tasks (winogrande) put the varying text in the CONTEXT
    and share one continuation, so the "choice" is arguments[i][0]. Everything else
    varies the continuation, which carries the target_delimiter (" ") as its first
    character (task.py:1390) -- strip exactly that one delimiter.
    """
    conts = [a[1] for a in arguments]
    if len(set(conts)) == 1 and len(conts) > 1:
        return [a[0] for a in arguments], True
    return [c[1:] if c.startswith(" ") else c for c in conts], False


def gold_index(rec, choices, multiple_input):
    t = rec.get("target")
    if isinstance(t, int):
        return t
    doc = rec.get("doc", {})
    if multiple_input and "answer" in doc:
        return int(doc["answer"]) - 1          # winogrande: "1"/"2"
    if isinstance(t, str):
        if t in choices:
            return choices.index(t)
        stripped = t[1:] if t.startswith(" ") else t
        if stripped in choices:
            return choices.index(stripped)
    raise ValueError(f"cannot resolve gold index (target={t!r}, multiple_input={multiple_input})")


def logsoftmax_at(z, i):
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return z[i] - lse


def extract(dump_dir, out_path, summary_path=None, strict=True, only_tasks=None):
    results, src = load_results(dump_dir)
    samples = results.get("samples")
    if not samples:
        raise SystemExit(f"{src} has no 'samples' -- was harness.log_samples true?")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    agg = defaultdict(lambda: defaultdict(float))
    counts, violations, skipped = defaultdict(int), [], []
    tails = defaultdict(list)

    with open(out_path, "w") as out:
        for task, recs in samples.items():
            if only_tasks and task not in only_tasks:
                continue
            # generate_until tasks (hellaswag_gen, arc_easy_gen, ...) store
            # filtered_resps as a list of STRINGS, not [loglik, is_greedy] pairs.
            # They carry no choice-set likelihoods, so they cannot yield a soft metric.
            if not recs or not isinstance(recs[0].get("filtered_resps", [None])[0], (list, tuple)):
                skipped.append(task)
                continue
            seen = set()
            for rec in recs:
                args = rec["arguments"]
                ll = [r[0] for r in rec["filtered_resps"]]
                greedy = [bool(r[1]) for r in rec["filtered_resps"]]
                choices, multi = choice_strings(args)
                gold = gold_index(rec, choices, multi)
                K = len(ll)

                b_cont = [len(a[1].encode()) for a in args]
                b_choice = [len(c.encode()) for c in choices]
                c_choice = [len(c) for c in choices]

                doc_id = rec["doc_id"]
                if doc_id in seen:
                    violations.append(f"{task}: duplicate doc_id {doc_id}")
                seen.add(doc_id)

                # --- self-check against lm-eval's own per-doc metrics ---------------
                acc, acc_norm = rec.get("acc"), rec.get("acc_norm")
                if acc is not None:
                    if (max(range(K), key=lambda i: ll[i]) == gold) != (acc == 1.0):
                        violations.append(f"{task}#{doc_id}: acc mismatch")
                if acc_norm is not None:
                    zn = [ll[i] / c_choice[i] if c_choice[i] else -math.inf for i in range(K)]
                    if (max(range(K), key=lambda i: zn[i]) == gold) != (acc_norm == 1.0):
                        violations.append(f"{task}#{doc_id}: acc_norm mismatch")

                out.write(json.dumps({
                    "task": task, "doc_id": doc_id, "K": K, "gold": gold,
                    "ll": ll, "b_cont": b_cont, "b_choice": b_choice,
                    "c_choice": c_choice, "is_greedy": greedy,
                    "acc": acc, "acc_norm": acc_norm, "multiple_input": multi,
                }) + "\n")

                # --- candidate soft metrics -----------------------------------------
                # No single variant is committed to here. A temperature sweep on two
                # training stages of the same 1.3B run showed that BOTH ends of the
                # scale are degenerate: at T=1 (pure per-character rate) hellaswag X
                # moves 1.280 -> 1.272 across a 19 pp accuracy gain and p_gold moves the
                # WRONG WAY (-0.051), while at the raw end the choice posterior is a
                # near-delta so p_gold collapses onto accuracy. Emit the panel; the fit
                # script picks by pre-registered inner CV on the fit set (never on 1.3B).
                a = agg[task]
                L = c_choice if task in ACC_NORM_TASKS else b_cont
                mean_len = sum(L) / K
                for tag, T in TEMPERATURES:
                    scale = mean_len if T is None else T
                    z = [ll[i] * scale / L[i] if L[i] else -math.inf for i in range(K)]
                    lsm = logsoftmax_at(z, gold)
                    a[f"X_{tag}"] += -lsm
                    a[f"p_gold_{tag}"] += math.exp(lsm)
                    a[f"margin_{tag}"] += z[gold] - max(z[i] for i in range(K) if i != gold)
                    if T is None:
                        tails[task].append(-lsm)
                # raw, unnormalized sum-loglik: Delphi's stated MMLU metric, and the
                # surrogate whose argmax IS `acc`
                lsm_raw = logsoftmax_at(ll, gold)
                a["X_raw"] += -lsm_raw
                a["p_gold_raw"] += math.exp(lsm_raw)
                a["margin_raw"] += ll[gold] - max(ll[i] for i in range(K) if i != gold)
                # Delphi's generative soft metric: bits/byte of the gold continuation
                a["logK"] += math.log(K)
                a["bpb_gold"] += (-ll[gold] / (b_choice[gold] * LN2)) if b_choice[gold] else 0.0
                if acc is not None:
                    a["acc"] += acc
                if acc_norm is not None:
                    a["acc_norm"] += acc_norm
                    a["_n_norm"] += 1
                counts[task] += 1

    summary = {}
    for task, a in agg.items():
        n = counts[task]
        row = {k: a[k] / n for k in a if not k.startswith("_")}
        row["n"] = n
        row["X_chance"] = a["logK"] / n
        row["X_excess_docmean"] = row["X_docmean"] - row["X_chance"]
        # tail diagnostic: mean(-log p) is heavy-tailed at every temperature, so report
        # how far the 1%-trimmed mean and the 99th percentile sit from the mean
        xs = sorted(tails[task]); k1 = max(1, len(xs) // 100)
        row["X_docmean_trim1"] = sum(xs[:len(xs) - k1]) / (len(xs) - k1)
        row["X_docmean_p99"] = xs[int(0.99 * len(xs))]
        row["acc_norm"] = (a["acc_norm"] / a["_n_norm"]) if a["_n_norm"] else None
        row["headline"] = row["acc_norm"] if task in ACC_NORM_TASKS else row["acc"]
        # cross-check the aggregate against lm-eval's own reported number
        rep = results.get("results", {}).get(task, {})
        row["reported_acc"] = rep.get("acc,none")
        row["reported_acc_norm"] = rep.get("acc_norm,none")
        summary[task] = row

    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w") as fh:
            json.dump({"dump_dir": dump_dir, "source": src, "tasks": summary}, fh, indent=1)

    for task, r in sorted(summary.items()):
        head = "acc_norm" if task in ACC_NORM_TASKS else "acc"
        rep = r["reported_acc_norm"] if task in ACC_NORM_TASKS else r["reported_acc"]
        drift = abs(r["headline"] - rep) * 100 if (rep is not None and r["headline"] is not None) else float("nan")
        print(f"  {task:15s} n={r['n']:5d}  bpb_gold={r['bpb_gold']:7.4f}  "
              f"p_gold[t10/t20/dm/raw]={r['p_gold_t10']:.4f}/{r['p_gold_t20']:.4f}/"
              f"{r['p_gold_docmean']:.4f}/{r['p_gold_raw']:.4f}  "
              f"X_dm={r['X_docmean']:8.3f} (trim1 {r['X_docmean_trim1']:8.3f}, "
              f"p99 {r['X_docmean_p99']:8.2f})  {head}={r['headline']*100:.2f}  "
              f"drift={drift:.3f}pp")

    if skipped:
        print(f"  (skipped {len(skipped)} non-multiple-choice task(s): {', '.join(sorted(skipped)[:6])}"
              f"{' ...' if len(skipped) > 6 else ''})")

    if violations:
        print(f"\n{len(violations)} INVARIANT VIOLATIONS (first 10):", file=sys.stderr)
        for v in violations[:10]:
            print("   ", v, file=sys.stderr)
        if strict:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary")
    ap.add_argument("--tasks", help="comma-separated subset to extract (default: all MC tasks)")
    ap.add_argument("--no-strict", action="store_true",
                    help="report invariant violations but exit 0")
    a = ap.parse_args()
    only = set(a.tasks.split(",")) if a.tasks else None
    sys.exit(extract(a.dump_dir, a.out, a.summary, strict=not a.no_strict, only_tasks=only))


if __name__ == "__main__":
    main()
