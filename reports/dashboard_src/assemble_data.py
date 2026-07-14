#!/usr/bin/env python3
"""Single source of truth for the downstream table / figures / dashboard.
Emits downstream_data.json = {series: {scale: {bench: val}}} + avg helpers."""
import csv, json, os
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
BEN=["hellaswag","arc_easy","arc_challenge","piqa","boolq","winogrande","mmlu_text"]
MET={"hellaswag":"acc_norm","arc_easy":"acc_norm","arc_challenge":"acc_norm","piqa":"acc_norm",
     "boolq":"acc","winogrande":"acc","mmlu_text":"acc"}
BASE=["llama","aunet(word)","bpebyte_rg"]
data={s:{} for s in BASE+["hyb_leafQ","hyb_greedyQ"]}
def put(ser,scale,bench,val):
    data[ser].setdefault(scale,{})[bench]=round(val,1)
def rd(p,bench):
    if not os.path.exists(p): return None
    r=json.load(open(p))["results"]; m=MET[bench]
    v=r.get(bench,{}).get(m+",none", r.get(bench,{}).get("acc,none"))
    return round(v*100,1) if v is not None else None

# --- base 100M/300M/1.3B from scaling_data.csv ---
for r in csv.DictReader(open(f"{ROOT}/reports/scaling_data.csv")):
    if r["model"] in BASE and r["scale"]!="760M":
        put(r["model"],r["scale"],r["benchmark"],float(r["value_pct"]))
# --- base 760M: HS/ARCe/ARCc/PIQA from cmp_g10 (ratio-matched); BoolQ/WinoG from main (scaling_data.csv 760M) ---
for m,run in {"llama":"llama_760M","aunet(word)":"aunet_760M","bpebyte_rg":"rg_760M"}.items():
    for b in ["hellaswag","arc_easy","arc_challenge","piqa"]:
        for sub in ["eval_scaling","evals_5bench"]:
            v=rd(f"{ROOT}/runs/small/cmp_g10/{run}/{sub}/results.json",b)
            if v is not None: put(m,"760M",b,v); break
for r in csv.DictReader(open(f"{ROOT}/reports/scaling_data.csv")):
    if r["model"] in BASE and r["scale"]=="760M" and r["benchmark"] in ("boolq","winogrande"):
        put(r["model"],"760M",r["benchmark"],float(r["value_pct"]))  # main 760M (†)
# --- base 1.3B MMLU-text ---
for m,v in {"llama":31.9,"aunet(word)":32.2,"bpebyte_rg":32.3}.items(): put(m,"1.3B","mmlu_text",v)

# --- hybrid leafQ/greedyQ: 100M/300M from NEW full canonical evals ---
for scale,run in {"100M":"main/100M/hybrid_100M","300M":"main/300M/hybrid_300M"}.items():
    for ser,reg in {"hyb_leafQ":"leafQ","hyb_greedyQ":"greedyQ"}.items():
        for b in BEN[:-1]:
            v=rd(f"{ROOT}/runs/{run}/evals_{reg}/results.json",b)
            if v is not None: put(ser,scale,b,v)
# --- hybrid 1.3B from hybrid_1p3B_leaf_B3 (curated) ---
T13={"hyb_leafQ":  dict(hellaswag=62.0,arc_easy=58.8,arc_challenge=35.8,piqa=74.2,boolq=64.3,winogrande=61.4,mmlu_text=30.7),
     "hyb_greedyQ":dict(hellaswag=62.6,arc_easy=65.0,arc_challenge=37.0,piqa=73.8,boolq=63.6,winogrande=61.5,mmlu_text=32.0)}
for ser,d in T13.items():
    for b,v in d.items(): put(ser,"1.3B",b,v)

# --- LAW-recipe override for base families @100M/300M (2026-07-13). 100M = complete (21GB);
#     300M = PARTIAL (aunet/rg @40k≈21B, llama @17.8k≈9B — under-trained, still training). ---
LAW={
 ("llama","100M"):dict(hellaswag=33.2,arc_easy=45.1,arc_challenge=23.8,boolq=54.9,piqa=64.0,winogrande=50.6),
 ("aunet(word)","100M"):dict(hellaswag=31.8,arc_easy=36.0,arc_challenge=24.4,boolq=54.8,piqa=59.1,winogrande=49.6),
 ("bpebyte_rg","100M"):dict(hellaswag=30.7,arc_easy=34.6,arc_challenge=23.5,boolq=39.1,piqa=59.4,winogrande=50.2),
 ("aunet(word)","300M"):dict(hellaswag=32.3,arc_easy=37.1,arc_challenge=23.8,boolq=49.7,piqa=61.0,winogrande=52.3),
 ("bpebyte_rg","300M"):dict(hellaswag=31.5,arc_easy=34.7,arc_challenge=23.9,boolq=38.7,piqa=61.7,winogrande=51.6),
 ("llama","300M"):dict(hellaswag=39.3,arc_easy=48.0,arc_challenge=24.7,boolq=59.2,piqa=67.4,winogrande=52.2),
}
for (ser,sc),d in LAW.items():
    for b,v in d.items(): put(ser,sc,b,v)

def avg(cells,keys):
    vs=[cells[k] for k in keys if cells.get(k) is not None]
    return round(sum(vs)/len(vs),1) if vs else None
out={"data":data,
     "meta":{"benches":BEN,"scales":["100M","300M","760M","1.3B"],
             "labels":{"llama":"Llama","aunet(word)":"AU-Net","bpebyte_rg":"BPEByte rg",
                       "hyb_leafQ":"BPEByte hybrid (leafQ)","hyb_greedyQ":"BPEByte hybrid (greedyQ)"}}}
# attach avgs
for ser,scales in data.items():
    for s,cells in scales.items():
        cells["_avg3"]=avg(cells,["hellaswag","arc_easy","piqa"])
        cells["_avgall"]=avg(cells,["hellaswag","arc_easy","arc_challenge","piqa","boolq","winogrande","mmlu_text"])
json.dump(out,open(f"{ROOT}/reports/downstream_data.json","w"),indent=1)
print("wrote reports/downstream_data.json")
# quick print
for ser in ["llama","aunet(word)","bpebyte_rg","hyb_leafQ","hyb_greedyQ"]:
    for s in ["100M","300M","760M","1.3B"]:
        c=data[ser].get(s,{})
        if c: print(f"{ser:14} {s:5} HS={c.get('hellaswag','—')} avg3={c.get('_avg3')} avgall={c.get('_avgall')}")
