"""Greedy continuations for items a model gets right 0-shot but wrong 3-shot (exact std_bench prompts)."""
import json, sys
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes/std_bench")
import models
from lm_eval.api.instance import Instance

reg = sys.argv[1]
T = "/home/hyunw3/.claude/jobs/003c5e28/tmp/"
SB = "/mnt/ssd2/hyun2/AUNet/reports/std_bench/"
P = json.load(open(T + "lambada_prompts.json"))


def bits(k):
    j = json.load(open(f"{SB}{reg}/ds{k}_seed1234_lambada.json"))
    return {int(i.split(":")[1]): b for i, b in zip(j["items"]["lambada_openai"], j["bits"]["lambada_openai"])}


b0, b3 = bits(0), bits(3)
ids = [i for i in range(len(b0)) if b0[i] == 1 and b3[i] == 0]
spec = models.resolve(reg, None, None, None)
models.run_ext._lm_eval_compat() if hasattr(models, "run_ext") else None
lm, _ = models.build(spec)
ga = {"until": ["\n"], "max_gen_toks": 24, "temperature": 0.0, "do_sample": False}
out = {}
for k in ("0", "3"):
    reqs = [Instance(request_type="generate_until", doc={}, arguments=(P[k][i]["ctx"], dict(ga)), idx=0) for i in ids]
    out[k] = lm.generate_until(reqs)
json.dump({"ids": ids, "gen": out}, open(f"{T}gen_{reg}.json", "w"))
print(reg, len(ids), "items")
for n in range(8):
    i = ids[n]
    print(repr(P["0"][i]["target"]), "| 0-shot:", repr(out["0"][n][:30]), "| 3-shot:", repr(out["3"][n][:30]))
