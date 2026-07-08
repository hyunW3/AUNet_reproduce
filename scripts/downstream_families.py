#!/usr/bin/env python
"""Downstream scaling law + 7B estimate for BOTH families (AU-Net word, Llama subword).
Trunk-matched -> common trunk-N axis. Loss units differ (byte vs token) so Stage-2
(loss->acc) is fit within each family; comparison is on downstream acc vs trunk-N."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
N=np.array([98_591_488,296_659_712,714_426_880,1_324_203_008],float)   # trunk (both families)
tasks=["hellaswag","arc_easy","arc_challenge","piqa"]; NT=7.0e9
FAM={
 "AU-Net":dict(loss=np.array([0.8065,0.7087,0.6474,0.6047]),
   acc={"hellaswag":[28.5,37.2,52.4,59.1],"arc_easy":[33.7,43.9,55.9,65.7],
        "arc_challenge":[23.3,25.6,31.7,37.9],"piqa":[58.3,64.9,71.1,74.2]}),
 "Llama":dict(loss=np.array([3.400,2.995,2.853,2.743]),
   acc={"hellaswag":[31.3,42.4,55.8,62.2],"arc_easy":[42.1,48.8,62.0,65.4],
        "arc_challenge":[23.7,26.0,31.5,38.2],"piqa":[64.3,68.3,73.7,75.3]}),
}
for f in FAM:
    a=FAM[f]["acc"]; a={k:np.array(v,float) for k,v in a.items()}
    a["AVERAGE"]=np.mean([a[t] for t in tasks],0); FAM[f]["acc"]=a
ALLT=tasks+["AVERAGE"]
def logit(p): p=np.clip(p/100,1e-3,1-1e-3); return np.log(p/(1-p))
def inv(x): return 100/(1+np.exp(-x))
def fitL(Ns,Ls):
    best=None
    for E in np.linspace(0,min(Ls)-1e-3,500):
        y=np.log(Ls-E); x=np.log(Ns); b,a=np.polyfit(x,y,1); r=np.sum((y-(a+b*x))**2)
        if best is None or r<best[0]: best=(r,E,np.exp(a),-b)
    return best[1],best[2],best[3]
Lof=lambda n,p:p[0]+p[1]*n**(-p[2])

results={}
for fam,D in FAM.items():
    loss=D["loss"]; acc=D["acc"]; pL=fitL(N,loss); L7=Lof(NT,pL)
    est={}; loo={t:[] for t in ALLT}; holo={}
    for t in ALLT:
        b,a=np.polyfit(loss,logit(acc[t]),1); est[t]=inv(a+b*L7)
        # hold-out predict 1.3B from <=760M
        b3,a3=np.polyfit(loss[:3],logit(acc[t][:3]),1); holo[t]=(acc[t][3],inv(a3+b3*loss[3]))
    for i in range(4):
        m=np.ones(4,bool); m[i]=False; pLi=fitL(N[m],loss[m]); L7i=Lof(NT,pLi)
        for t in ALLT:
            b,a=np.polyfit(loss[m],logit(acc[t][m]),1); loo[t].append(inv(a+b*L7i))
    results[fam]=dict(pL=pL,L7=L7,est=est,loo=loo,holo=holo,loss=loss,acc=acc)

# ---- print ----
for fam in FAM:
    R=results[fam]; mae=np.mean([abs(R["holo"][t][1]-R["holo"][t][0]) for t in tasks])
    print(f"\n===== {fam}  (L7={R['L7']:.3f}, hold-out 1.3B MAE={mae:.1f}pp) =====")
    print(f"{'task':14s} {'7B est':>7s} {'LOO':>13s}   {'(1.3B act/pred)':>15s}")
    for t in ALLT:
        lo,hi=min(R['loo'][t]),max(R['loo'][t])
        print(f"{t:14s} {R['est'][t]:6.1f}%  [{lo:4.0f}-{hi:4.0f}]     {R['holo'][t][0]:4.1f}/{R['holo'][t][1]:.1f}")

# ---- plot ----
fig,ax=plt.subplots(1,2,figsize=(14,5.4))
col={"AU-Net":"C0","Llama":"C3"}; Ng=np.logspace(np.log10(N[0]*0.85),np.log10(8e9),120)
for fam in FAM:
    R=results[fam]; b,a=np.polyfit(R["loss"],logit(R["acc"]["AVERAGE"]),1)
    accg=inv(a+b*Lof(Ng,R["pL"]))
    ax[0].plot(N/1e6,R["acc"]["AVERAGE"],"o",color=col[fam],ms=8,label=f"{fam} (data)")
    ax[0].plot(Ng/1e6,accg,"-",color=col[fam],lw=1.6)
    ax[0].plot(NT/1e6,R["est"]["AVERAGE"],"*",color=col[fam],ms=20)
    lo=min(R["loo"]["AVERAGE"]);hi=max(R["loo"]["AVERAGE"])
    ax[0].plot([NT/1e6,NT/1e6],[lo,hi],color=col[fam],lw=2)
ax[0].axvline(NT/1e6,ls=":",c="gray"); ax[0].set(xscale="log",title="Average acc_norm vs trunk params (★=7B estimate)",
    xlabel="trunk params (M)",ylabel="avg acc_norm %"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
x=np.arange(len(ALLT)); w=0.38
for k,fam in enumerate(FAM):
    R=results[fam]; e=[R["est"][t] for t in ALLT]
    err=[[R["est"][t]-min(R["loo"][t]) for t in ALLT],[max(R["loo"][t])-R["est"][t] for t in ALLT]]
    ax[1].bar(x+(k-0.5)*w,e,w,yerr=err,capsize=3,color=col[fam],label=fam)
ax[1].set_xticks(x); ax[1].set_xticklabels([t[:8] for t in ALLT],rotation=30,fontsize=8)
ax[1].set(title="Estimated 7B acc_norm by task",ylabel="acc_norm %"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3,axis="y")
fig.suptitle("7B downstream estimate: AU-Net (byte) vs Llama (subword), trunk-matched ladders")
fig.tight_layout(); fig.savefig(f"{ROOT}/reports/downstream_families.png",dpi=120)
print("\nsaved: reports/downstream_families.png")
