"""Rebuild the exact lambada_openai prompts std_bench.py scored (same lm-eval call, same seeds) with a recording LM."""
import json, sys
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes/ext_ci")
import run_ext
from lm_eval import simple_evaluate
from lm_eval.api.model import LM


class Rec(LM):
    def __init__(self):
        super().__init__()
        self.reqs = []

    def loglikelihood(self, requests):
        self.reqs += [r.args for r in requests]
        return [(0.0, False) for _ in requests]

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        raise NotImplementedError


run_ext._lm_eval_compat()
out = {}
for k in (3,):
    lm = Rec()
    r = simple_evaluate(lm, tasks=["lambada_openai_gpt3"], num_fewshot=k, task_manager=run_ext.safe_task_manager(run_ext.TASKS_DIR),
                        log_samples=True, bootstrap_iters=0, fewshot_random_seed=1234, random_seed=0,
                        numpy_random_seed=1234, torch_random_seed=1234)
    S = sorted(r["samples"]["lambada_openai_gpt3"], key=lambda s: int(s["doc_id"]))
    out[k] = [{"doc_id": int(s["doc_id"]), "ctx": s["arguments"][0][0], "target": s["arguments"][0][1]} for s in S]
json.dump(out, open("/home/hyunw3/.claude/jobs/003c5e28/tmp/gpt3_prompts.json", "w"))
for k in (3,):
    L = sorted(len((x["ctx"] + x["target"]).encode()) for x in out[k])
    print(k, "bytes: mean", sum(L) / len(L), "p99", L[int(.99 * len(L))], "max", L[-1])
print(repr(out[3][0]["ctx"][-900:])); print(repr(out[3][0]["target"]))
