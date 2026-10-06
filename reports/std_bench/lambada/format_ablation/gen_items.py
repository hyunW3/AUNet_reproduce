"""Greedy continuations of selected LAMBADA items for one model.
usage: gen_items.py <registry model> <prompts json> <shot key> <ids json> <out json>"""
import json, sys
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/.claude/worktrees/lambada-fewshot-avg6/scripts/probes/std_bench")
import models
from lm_eval.api.instance import Instance

reg, pfile, key, idfile, out = sys.argv[1:6]
P = json.load(open(pfile))[key]
ids = json.load(open(idfile))
spec = models.resolve(reg, None, None, None)
lm, _ = models.build(spec)
ga = {"until": ["\n"], "max_gen_toks": 24, "temperature": 0.0, "do_sample": False}
reqs = [Instance(request_type="generate_until", doc={}, arguments=(P[i]["ctx"], dict(ga)), idx=0) for i in ids]
gen = lm.generate_until(reqs)
json.dump({"model": reg, "ids": ids, "gen": gen}, open(out, "w"))
for i, g in list(zip(ids, gen))[:12]:
    print(repr(P[i]["target"]), "->", repr(g[:40]))
