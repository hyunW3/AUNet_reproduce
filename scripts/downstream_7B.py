#!/usr/bin/env python
"""Estimate 7B downstream from the 4-point AU-Net ladder (incl. 1.3B).
Full-data two-stage fit + uncertainty: leave-one-out spread, a loss-based vs
direct-log(N) functional-form comparison, and the hold-out undershoot correction."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
scale = ["100M","300M","760M","1.3B"]
N    = np.array([98_591_488,296_659_712,714_426_880,1_324_203_008], float)
loss = np.array([0.8065,0.7087,0.6474,0.6047])
tasks= ["hellaswag","arc_easy","arc_challenge","piqa"]
acc = {"hellaswag":np.array([28.5,37.2,52.4,59.1]),"arc_easy":np.array([33.7,43.9,55.9,65.7]),
       "arc_challenge":np.array([23.3,25.6,31.7,37.9]),"piqa":np.array([58.3,64.9,71.1,74.2])}
acc["AVERAGE"]=np.mean([acc[t] for t in tasks],axis=0)
ALLT=tasks+["AVERAGE"]
NT=7.0e9
def logit(p): p=np.clip(p/100,1e-3,1-1e-3); return np.log(p/(1-p))
def inv(x): return 100/(1+np.exp(-x))
def fitL(Ns,Ls):
    best=None
    for E in np.linspace(0,min(Ls)-1e-3,500):
        y=np.log(Ls-E); x=np.log(Ns); b,a=np.polyfit(x,y,1); r=np.sum((y-(a+b*x))**2)
        if best is None or r<best[0]: best=(r,E,np.exp(a),-b)
    return best[1],best[2],best[3]
def Lof(n,p): E,A,al=p; return E+A*n**(-al)

# ---- full-data fit ----
pL=fitL(N,loss); L7=Lof(NT,pL)
print(f"Stage1 full: L(N)={pL[0]:.3f}+{pL[1]:.3g}*N^-{pL[2]:.3f}  ->  L(7B)={L7:.3f}")
est={}
for t in ALLT:
    b,a=np.polyfit(loss,logit(acc[t]),1); est[t]=inv(a+b*L7)

# ---- uncertainty: leave-one-out (drop each ladder point, refit both stages, predict 7B) ----
loo={t:[] for t in ALLT}
for i in range(4):
    m=np.ones(4,bool); m[i]=False
    pLi=fitL(N[m],loss[m]); L7i=Lof(NT,pLi)
    for t in ALLT:
        b,a=np.polyfit(loss[m],logit(acc[t][m]),1); loo[t].append(inv(a+b*L7i))
# ---- alt functional form: acc directly vs log10(N) (skip loss) ----
altN={}
for t in ALLT:
    b,a=np.polyfit(np.log10(N),logit(acc[t]),1); altN[t]=inv(a+b*np.log10(NT))

print(f"\n7B downstream estimate (all 4 points; NT=7B):")
print(f"{'task':14s} {'via loss':>9s} {'LOO range':>14s} {'via log(N)':>10s}")
for t in ALLT:
    lo,hi=min(loo[t]),max(loo[t])
    print(f"{t:14s} {est[t]:7.1f}%  [{lo:4.1f}-{hi:4.1f}]   {altN[t]:8.1f}%")
print("\nNote: hold-out (predict 1.3B from <=760M) UNDERSHOT by ~3pp -> these are likely conservative (+~3pp).")

# ---- plot: acc vs loss with fits extended to L(7B) ----
fig,ax=plt.subplots(1,2,figsize=(14,5.2))
Lg=np.linspace(L7-0.02, loss.max()+0.03, 120)
for t in tasks:
    b,a=np.polyfit(loss,logit(acc[t]),1); c=ax[0].plot(loss,acc[t],"o",ms=7,label=t)[0].get_color()
    ax[0].plot(Lg,inv(a+b*Lg),"-",color=c,lw=1.2); ax[0].plot(L7,est[t],"*",color=c,ms=15)
b,a=np.polyfit(loss,logit(acc["AVERAGE"]),1)
ax[0].plot(loss,acc["AVERAGE"],"ks",ms=8,label="AVERAGE"); ax[0].plot(Lg,inv(a+b*Lg),"k-",lw=1.8)
ax[0].plot(L7,est["AVERAGE"],"k*",ms=18)
ax[0].axvline(L7,ls=":",c="gray"); ax[0].text(L7,20,f" L(7B)={L7:.3f}",fontsize=8,color="gray")
ax[0].invert_xaxis(); ax[0].set(title="acc_norm = logistic(loss), extrapolated to 7B (stars)",
    xlabel="final loss (→ better)",ylabel="acc_norm %"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
# bar: 7B estimate per task with LOO band
x=np.arange(len(ALLT)); e=[est[t] for t in ALLT]; err=[[est[t]-min(loo[t]) for t in ALLT],[max(loo[t])-est[t] for t in ALLT]]
ax[1].bar(x,e,0.6,yerr=err,capsize=4,color="C1")
for i,t in enumerate(ALLT): ax[1].text(i,e[i]+1,f"{e[i]:.0f}",ha="center",fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels([t[:8] for t in ALLT],rotation=30,fontsize=8)
ax[1].set(title="Estimated 7B acc_norm (bar=point, whisker=leave-one-out)",ylabel="acc_norm %"); ax[1].grid(alpha=.3,axis="y")
fig.suptitle(f"7B downstream estimate from 4-point AU-Net ladder  —  L(7B)≈{L7:.2f}, avg acc_norm≈{est['AVERAGE']:.0f}% (likely +3pp)")
fig.tight_layout(); fig.savefig(f"{ROOT}/reports/downstream_7B.png",dpi=120)
print("saved: reports/downstream_7B.png")
