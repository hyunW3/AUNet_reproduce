#!/usr/bin/env python3
"""Count requests that BLT's 4096-byte harness left-truncates (BOS + context + continuation > 4096 bytes).

Builds exactly the lm-eval requests of run_ext.py (same tasks, shots, seeds, limits, perturbation expansion)
with a model-free LM that records byte lengths. Despace is not listed: it only deletes spaces, so its inputs are
never longer than the clean ones counted here.

  cd lingua; PYTHONPATH=. .venv/bin/python ../scripts/probes/ext_ci/check_blt_truncation.py
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ext
from lm_eval.api.model import LM

MAXB = 4096
STATS = defaultdict(lambda: {"req": 0, "over": 0, "docs": set(), "docs_over": set(), "max": 0})


class LenLM(LM):
    def loglikelihood(self, reqs, disable_tqdm=False):
        out = []
        for r in reqs:
            ctx, cont = r.args
            n = 1 + len((ctx + cont).encode("utf-8"))
            s = STATS[(TAG, r.task_name)]
            s["req"] += 1; s["max"] = max(s["max"], n); s["docs"].add(r.doc_id)
            if n > MAXB:
                s["over"] += 1; s["docs_over"].add(r.doc_id)
            out.append((0.0, False))
        return out

    def loglikelihood_rolling(self, reqs, disable_tqdm=False):
        return [0.0 for _ in reqs]

    def generate_until(self, reqs, disable_tqdm=False):
        return ["" for _ in reqs]


def main():
    global TAG
    run_ext._lm_eval_compat()
    sys.path.insert(0, run_ext.LINGUA); sys.modules.pop("apps", None)
    from apps.aunet.eval_noise import expand_noise_tasks
    from apps.aunet.eval_typo import expand_typo_tasks
    from apps.aunet.eval_typo_ds import expand_typo_ds_tasks
    lm = LenLM()
    R5 = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq"]
    runs = [("ds0", ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "boolq", "mmlu_text"], 0, None),
            ("ds3", ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "boolq"], 3, None),
            ("ds5", ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"], 5, None)]
    for t in R5:
        runs.append((f"noise_{t}", expand_noise_tasks([t, f"{t}_noise"], base_seed=run_ext.PERTURB_SEED), 0, 2000))
        tl = expand_typo_tasks([t, f"{t}_typo"], base_seed=run_ext.PERTURB_SEED)
        runs.append((f"typo_{t}", expand_typo_ds_tasks(tl, base_seed=run_ext.PERTURB_SEED), 0, 2000))
    only = set(sys.argv[1:])                       # optional subset of run tags
    for tag, tasks, shots, limit in runs:
        if only and tag not in only:
            continue
        TAG = tag
        run_ext.run_lm_eval(lm, tasks, shots, limit)
        print(tag, "done", flush=True)
    rows = []
    for (tag, task), s in sorted(STATS.items()):
        rows.append({"run": tag, "task": task, "requests": s["req"], "over": s["over"], "docs": len(s["docs"]),
                     "docs_over": len(s["docs_over"]), "max_bytes": s["max"],
                     "doc_ids_over": sorted(s["docs_over"])})
    out = "/mnt/ssd2/hyun2/AUNet/reports/ext_ci/blt_truncation.json"
    json.dump(rows, open(out, "w"), indent=1)
    for r in rows:
        if r["over"]:
            print({k: v for k, v in r.items() if k != "doc_ids_over"})
    print("wrote", out)


if __name__ == "__main__":
    main()
