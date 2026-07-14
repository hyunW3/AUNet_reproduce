#!/usr/bin/env python3
"""Noise + PBP (prompt-boundary) robustness summaries per 1.3B model -> reports/robustness.json"""
import json, os, statistics
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
MODELS={"llama":"llama_1.8B_paper","aunet(word)":"aunet2_1.3B","bpebyte_rg":"bpebyte_br_greedy_root_1.3B","hybrid":"hybrid_1p3B_leaf_B3"}
def firstjson(paths):
    for p in paths:
        if os.path.exists(p): return json.load(open(p))["results"], p
    return None,None
def noise(run):
    r,_=firstjson([f"{ROOT}/runs/main/1.3B/{run}/evals_noise/results.json",
                   f"{ROOT}/runs/main/1.3B/{run}/evals_noise/hellaswag_noise/results.json"])
    if not r: return None
    a=[v.get("acc_norm,none") for v in r.values() if v.get("acc_norm,none") is not None]
    return round(100*statistics.mean(a),1) if a else None
def pbp(run):
    # pbp_mc_avg.delta_acc + pbp_overall.delta_bpc, from flat or nested layout
    mc,_=firstjson([f"{ROOT}/runs/main/1.3B/{run}/evals_pbp/pbp_mc_full/results.json",  # FULL (limit=20000)
                    f"{ROOT}/runs/main/1.3B/{run}/evals_pbp/results.json",
                    f"{ROOT}/runs/main/1.3B/{run}/evals_pbp/pbp_mc/results.json"])
    ov,_=firstjson([f"{ROOT}/runs/main/1.3B/{run}/evals_pbp/results.json",
                    f"{ROOT}/runs/main/1.3B/{run}/evals_pbp/pbp/results.json"])
    dmc=(mc or {}).get("pbp_mc_avg",{}).get("delta_acc")
    dbpc=(ov or {}).get("pbp_overall",{}).get("delta_bpc")
    # per-benchmark Δacc under space-perturbed prompt
    BENCH=["curated","arc_easy","arc_challenge","hellaswag"]
    by={}
    def pc(x): return round(x*100,1) if x is not None else None
    def pd(x): return round(x*100,2) if x is not None else None
    for b in BENCH:
        can=(mc or {}).get(f"pbp_mc_{b}_canonical",{})
        sp=(mc or {}).get(f"pbp_mc_{b}_space",{})
        by[b]={"canon":pc(can.get("acc")),"degraded":pc(sp.get("acc")),"delta":pd(sp.get("delta_acc")),
               "canon_norm":pc(can.get("acc_norm")),"degraded_norm":pc(sp.get("acc_norm")),"delta_norm":pd(sp.get("delta_acc_norm"))}
    return (round(dmc*100,2) if dmc is not None else None, round(dbpc,4) if dbpc is not None else None, by)
out={}
for lab,run in MODELS.items():
    dmc,dbpc,by=pbp(run)
    out[lab]={"noise_hs":noise(run),"pbp_mc_dacc":dmc,"pbp_dbpc":dbpc,"pbp_by_bench":by,"clean_hs":None}
# attach clean HS (from downstream json) for the noise delta
dd=json.load(open(f"{ROOT}/reports/downstream_data.json"))["data"]
cmap={"llama":"llama","aunet(word)":"aunet(word)","bpebyte_rg":"bpebyte_rg","hybrid":"hyb_greedyQ"}
for lab in out:
    c=dd.get(cmap[lab],{}).get("1.3B",{}).get("hellaswag")
    out[lab]["clean_hs"]=c
json.dump(out,open(f"{ROOT}/reports/robustness.json","w"),indent=1)
for lab,v in out.items(): print(f"{lab:12} noise_HS={v['noise_hs']} (clean {v['clean_hs']})  pbp_mc_Δacc={v['pbp_mc_dacc']}  Δbpc={v['pbp_dbpc']}")
