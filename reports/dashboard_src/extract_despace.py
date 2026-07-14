#!/usr/bin/env python3
"""Despace robustness (space-strip) per 1.3B model -> reports/despace.json.
Merges despace_mc_full (arc_easy, hellaswag) + despace_mc_more (arc_challenge, piqa, boolq, winogrande).
Each benchmark: clean->degraded (delta) under acc AND acc_norm. Avg = macro-mean of delta over the
benchmarks present."""
import json, os
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
MODELS={"llama":"llama_1.8B_paper","aunet(word)":"aunet2_1.3B",
        "bpebyte_rg":"bpebyte_br_greedy_root_1.3B","hybrid":"hybrid_1p3B_leaf_B3"}
BENCH=["arc_easy","arc_challenge","piqa","boolq","hellaswag","winogrande"]

def load(run):
    r={}
    for sub in ("despace_mc_full","despace_mc_more"):
        p=f"{ROOT}/runs/main/1.3B/{run}/evals_pbp/{sub}/results.json"
        if os.path.exists(p):
            r.update(json.load(open(p))["results"])
    return r

def pc(x): return round(x*100,1) if x is not None else None
def pd(x): return round(x*100,2) if x is not None else None

out={}
for lab,run in MODELS.items():
    R=load(run); by={}; da=[]; dn=[]
    for b in BENCH:
        c=R.get(f"despace_mc_{b}_clean",{}); s=R.get(f"despace_mc_{b}_despace",{})
        if not s: by[b]=None; continue
        by[b]={"clean":pc(c.get("acc")),"degraded":pc(s.get("acc")),"delta":pd(s.get("delta_acc")),
               "clean_norm":pc(c.get("acc_norm")),"degraded_norm":pc(s.get("acc_norm")),"delta_norm":pd(s.get("delta_acc_norm"))}
        if s.get("delta_acc") is not None: da.append(s["delta_acc"])
        if s.get("delta_acc_norm") is not None: dn.append(s["delta_acc_norm"])
    out[lab]={"by_bench":by,
              "avg_delta":round(sum(da)/len(da)*100,2) if da else None,
              "avg_delta_norm":round(sum(dn)/len(dn)*100,2) if dn else None,
              "n_bench":len(da)}
json.dump(out,open(f"{ROOT}/reports/despace.json","w"),indent=1)
for lab,v in out.items():
    print(f"{lab:12} avg Δacc={v['avg_delta']}  Δacc_norm={v['avg_delta_norm']}  ({v['n_bench']} benches)")
    for b in BENCH:
        o=v['by_bench'].get(b)
        if o: print(f"    {b:14} acc {o['clean']}->{o['degraded']} ({o['delta']}) | norm {o['clean_norm']}->{o['degraded_norm']} ({o['delta_norm']})")
