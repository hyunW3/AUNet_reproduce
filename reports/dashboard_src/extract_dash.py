#!/usr/bin/env python3
"""Extract training-BPB curves (downsampled) + model configs for the dashboard."""
import csv, json, os, math, re
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
LN2=math.log(2); BPT=4.5483
NPTS=180  # downsample target

def curve(run_rel, llama):
    f=f"{ROOT}/runs/{run_rel}/metrics.jsonl"
    if not os.path.exists(f): return None
    den=LN2*(BPT if llama else 1)
    pts=[]  # (step, loss, tokens)
    for line in open(f):
        try:
            x=json.loads(line)
            if x.get("loss/out") and x.get("global_step"):
                pts.append((x["global_step"], x["loss/out"], x.get("optim/total_tokens")))
        except: pass
    if not pts: return None
    n=len(pts); w=max(1,n//NPTS)   # block-average each window -> smooth
    xs,ys,tk=[],[],[]
    for b in range(0,n,w):
        chunk=pts[b:b+w]
        xs.append(chunk[-1][0]); tk.append(chunk[-1][2])
        ys.append(round(sum(p[1] for p in chunk)/len(chunk)/den,4))
    return {"steps":xs,"tokens":tk,"bpb":ys}

curves={}
for r in csv.DictReader(open(f"{ROOT}/reports/scaling_bpb.csv")):
    key=f"{r['model']}|{r['scale']}"
    c=curve(r["run"], r["model"]=="llama")
    if c: c.update(model=r["model"],scale=r["scale"],final_bpb=float(r["bpb"]),run=r["run"]); curves[key]=c
json.dump(curves,open(f"{ROOT}/reports/training_bpb_curves.json","w"))
print("curves:",len(curves),"| pts/curve ~",[len(c["steps"]) for c in list(curves.values())[:3]])

# --- model configs (proper YAML) ---
import yaml
TEMPL={"llama":"llama","aunet(word)":"aunet","bpebyte_rg":"bpebyte_rg","hybrid":"bpebyte_rg_hybrid_leaf"}
BYTE={"aunet(word)","bpebyte_rg","hybrid"}
def cfg(model, scale, run_rel, laststep):
    # prefer the run's own config; for byte models the g10 ladder shares arch+HP, so fall back to the
    # same-scale AU-Net run config (NOT the portable_aunetlaw recipe template, a different budget).
    y=None
    cands=[f"{ROOT}/runs/{run_rel}/config.yaml"]
    if model in BYTE: cands.append(f"{ROOT}/runs/main/{scale}/aunet_{scale}/config.yaml")
    cands.append(f"{ROOT}/lingua/portable_aunetlaw/{TEMPL.get(model,model)}_{scale}.yaml")  # arch-only last resort
    for p in cands:
        if os.path.exists(p):
            try:
                yy=yaml.safe_load(open(p))
                if yy: y=yy; break
            except Exception: pass
    if y is None: return {}
    m=y.get("model",{}) or {}
    d=y.get("data",{}) or {}
    reg=(d.get("regex",{}) or {})
    dims=m.get("dimensions") or m.get("dim")
    lays=m.get("layers") or m.get("n_layers")
    tok=((d.get("tokenizer",{}) or {}).get("name")) or m.get("tokenizer")
    return {
      "tokenizer": tok,
      "dims": dims if isinstance(dims,(int,str)) else ("×".join(map(str,dims)) if dims else None),
      "layers": lays if isinstance(lays,(int,str)) else ("×".join(map(str,lays)) if lays else None),
      "vocab": m.get("vocab_size"),
      "steps": laststep,   # actual trained steps (scaling_bpb laststep), not the recipe's planned value
      "batch": d.get("batch_size"),
      "grad_acc": y.get("grad_acc_steps"),
      "seq": d.get("seq_len") or d.get("max_seqlen"),
      "lr": (y.get("optim",{}) or {}).get("lr"),
      "prefill": reg.get("bpe_hybrid_prefill"),
      "boundary": reg.get("bpe_hybrid_boundary"),
    }
def ckpt_of(run):
    d=f"{ROOT}/runs/{run}/checkpoints"
    if os.path.isdir(d):
        st=sorted(x for x in os.listdir(d) if x.isdigit())
        if st: return f"runs/{run}/checkpoints/{st[-1]}", True
    return f"runs/{run}/checkpoints/", False   # not local (weights on ece)
configs={}
for r in csv.DictReader(open(f"{ROOT}/reports/scaling_bpb.csv")):
    ck,loc=ckpt_of(r["run"])
    configs[f"{r['model']}|{r['scale']}"]={**cfg(r["model"],r["scale"],r["run"],r["laststep"]),
        "params":r["params"],"run":r["run"],"final_bpb":r["bpb"],"ckpt":ck,"ckpt_local":loc}
json.dump(configs,open(f"{ROOT}/reports/model_config.json","w"),indent=1)
print("configs:",len(configs))
for k,v in list(configs.items())[:4]: print(" ",k,"dims=",v.get("dims"),"layers=",v.get("layers"),"tok=",v.get("tokenizer"))
