"""BLT left-truncates any BoolQ request with 1 (BOS) + bytes(context + continuation) > 4096 (eval_suite/blt_eval/harness.py,
max_len 4096). Count the truncated items per variant of the context-only BoolQ table (tab:boolq_robust) and recompute
BLT's delta without them (items truncated in the clean form or in any variant of the axis are dropped from clean and
perturbed alike). Contexts are rebuilt with the evaluation code and seed (1234): Noise boolq_noise_<s>_prompt, Typo
boolq_typoctx_<s>; Leet from reports/patch_stats_conditions/leet/texts.jsonl (the scored Leet strings); Despace only
removes spaces, so its contexts are never longer than the clean ones.
  PYTHONPATH=<lingua typo-leet-both> lingua/.venv/bin/python blt_truncation_check.py"""
import json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boolq_both_tasks as B  # noqa: E402

L = "/mnt/ssd2/hyun2/AUNet"
MAX = 4096
N = 2000
CONT = (" no", " yes")


def over(ctx):
    return any(1 + len((ctx + c).encode()) > MAX for c in CONT)


def task_over(t):
    docs = list(t.validation_docs())[:N]
    return {i for i, d in enumerate(docs) if over(t.doc_to_text(d))}


def main():
    B.install()
    from apps.aunet import eval_noise as NZ, eval_typo_ds as TD
    tl = NZ.expand_noise_tasks(["boolq", "boolq_noise"], base_seed=1234) + \
        TD.expand_typo_ds_tasks(["boolq_typoctx"], base_seed=1234)
    by = {(t if isinstance(t, str) else t.config.task): t for t in tl}
    from lm_eval.tasks import TaskManager
    clean_t = TaskManager().load("boolq")["tasks"]["boolq"]
    clean = task_over(clean_t)
    trunc = {"clean": clean}
    trunc["noise"] = {k: task_over(t) for k, t in by.items() if k.startswith("boolq_noise_")}
    trunc["typo"] = {k: task_over(t) for k, t in by.items() if k.startswith("boolq_typoctx_")}
    leet = set()
    for l in open(f"{L}/reports/patch_stats_conditions/leet/texts.jsonl"):
        r = json.loads(l)
        if r["bench"] == "boolq":
            ctx = r["text"].rsplit(" ", 1)[0]          # text = context + " " + gold label
            if over(ctx):
                leet.add(int(r["key"]))
    trunc["leet"] = {"nla_leet": leet}
    print("truncated items: clean", len(clean))
    for ax in ("noise", "typo", "leet"):
        print(f"  {ax:6s}", {k.replace('boolq_noise_', '').replace('boolq_typoctx_', ''): len(v)
                              for k, v in trunc[ax].items()})
    print("  despace <= clean (spaces removed only)")
    # recompute BLT deltas without the truncated items
    sys.path.insert(0, f"{L}/scripts/probes")
    import boolq_ctx_ci as CI
    res = json.load(open(f"{L}/runs/robustness_boolq_ctx/boolq_ctx_ci.json"))
    for m in ("blt", "blt1609"):
        d = CI.per_item(m)
        for ax in ("noise", "typo", "despace", "leet"):
            drop = set(clean)
            for v in trunc.get(ax, {}).values():
                drop |= v
            c, p = d[ax]
            ids = [i for i in c if i in p and i < N]
            keep = [i for i in ids if i not in drop]
            full = 100 * (st.mean(p[i] for i in ids) - st.mean(c[i] for i in ids))
            excl = 100 * (st.mean(p[i] for i in keep) - st.mean(c[i] for i in keep))
            print(f"  {m:8s} {ax:7s} delta all {full:+6.2f} (table {res[m][ax]['delta']:+6.2f})  "
                  f"without {len(ids) - len(keep):2d} truncated {excl:+6.2f}  change {excl - full:+.2f}")


if __name__ == "__main__":
    main()
