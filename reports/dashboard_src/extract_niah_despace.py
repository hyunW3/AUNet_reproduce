#!/usr/bin/env python3
"""S-NIAH despace robustness (verbatim retrieval) per family -> reports/niah_despace.json.
Reads reports/niah/despace/results_S{1,2,3}.jsonl. Headline = exact_match (mean over lengths x depths);
tok_frac kept as diagnostic. Family tags in the jsonl: subword/aunet/byte -> dashboard keys."""
import json, collections, os
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
B=f"{ROOT}/reports/niah/despace"
DESC={"1":"noise + number","2":"essay + number","3":"essay + UUID"}
FAM={"subword":"llama","aunet":"aunet(word)","byte":"bpebyte_rg"}

def load(T):
    d=collections.defaultdict(dict)
    p=f"{B}/results_S{T}.jsonl"
    if not os.path.exists(p): return None
    for line in open(p):
        r=json.loads(line); d[r["tag"]][r["target_bytes"]]=(r["exact_match"],r["tok_frac"])
    return d

def mean(d,tag,i):
    v=[d[tag][L][i] for L in d.get(tag,{})]
    return sum(v)/len(v) if v else None

out={"tasks":{}}
for T in ["1","2","3"]:
    d=load(T)
    if d is None: continue
    row={"desc":DESC[T]}
    for ftag,key in FAM.items():
        ec,ed=mean(d,ftag+"_clean",0),mean(d,ftag+"_despace",0)
        tc,td=mean(d,ftag+"_clean",1),mean(d,ftag+"_despace",1)
        if ec is None or ed is None: continue
        row[key]={"clean":round(ec*100,1),"despace":round(ed*100,1),"delta":round((ed-ec)*100,1),
                  "clean_tf":round(tc*100,1) if tc is not None else None,
                  "despace_tf":round(td*100,1) if td is not None else None}
    out["tasks"][T]=row
json.dump(out,open(f"{ROOT}/reports/niah_despace.json","w"),indent=1)
for T,row in out["tasks"].items():
    print(f"S-NIAH-{T} ({row['desc']})")
    for k in ("llama","aunet(word)","bpebyte_rg"):
        if k in row: o=row[k]; print(f"   {k:12} exact {o['clean']}->{o['despace']} ({o['delta']:+}) tf {o['clean_tf']}->{o['despace_tf']}")
