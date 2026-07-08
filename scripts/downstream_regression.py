#!/usr/bin/env python
"""Regression line + extrapolation for downstream acc vs scale, both families.
Direct linear regression: logit(acc) = a + b*log10(N).  Solid = fitted data range
(100M-1.3B); dashed = extrapolation to 7B.  Left: natural units; right: linearized."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
N=np.array([98_591_488,296_659_712,714_426_880,1_324_203_008],float)
tasks=["hellaswag","arc_easy","arc_challenge","piqa"]; NT=7.0e9
FAM={
 "AU-Net":{"hellaswag":[28.5,37.2,52.4,59.1],"arc_easy":[33.7,43.9,55.9,65.7],
           "arc_challenge":[23.3,25.6,31.7,37.9],"piqa":[58.3,64.9,71.1,74.2]},
 "Llama":{"hellaswag":[31.3,42.4,55.8,62.2],"arc_easy":[42.1,48.8,62.0,65.4],
          "arc_challenge":[23.7,26.0,31.5,38.2],"piqa":[64.3,68.3,73.7,75.3]},
}
AVG={f:np.mean([FAM[f][t] for t in tasks],0) for f in FAM}
def logit(p): p=np.clip(np.asarray(p)/100,1e-3,1-1e-3); return np.log(p/(1-p))
def inv(x): return 100/(1+np.exp(-x))
col={"AU-Net":"C0","Llama":"C3"}
x=np.log10(N); xT=np.log10(NT)

fig,ax=plt.subplots(1,2,figsize=(14,5.4))
xr_fit=np.linspace(x.min(),x.max(),50); xr_ext=np.linspace(x.max(),xT,50)
print(f"{'family':8s} {'slope':>7s} {'R^2':>6s} {'7B avg':>7s}  per-task 7B")
for f in FAM:
    y=logit(AVG[f]); b,a=np.polyfit(x,y,1)
    yhat=a+b*x; r2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
    e7=inv(a+b*xT)
    # natural-units: solid fit range, dashed extrapolation
    ax[0].plot(N/1e6,AVG[f],"o",color=col[f],ms=8,label=f"{f} data")
    ax[0].plot(10**xr_fit/1e6,inv(a+b*xr_fit),"-",color=col[f],lw=1.8)
    ax[0].plot(10**xr_ext/1e6,inv(a+b*xr_ext),"--",color=col[f],lw=1.8)
    ax[0].plot(NT/1e6,e7,"*",color=col[f],ms=20)
    ax[0].annotate(f"{e7:.0f}%",(NT/1e6,e7),textcoords="offset points",xytext=(-4,8),fontsize=9,color=col[f])
    # linearized: literal straight regression line
    ax[1].plot(x,y,"o",color=col[f],ms=8,label=f"{f}  (R²={r2:.3f})")
    ax[1].plot(xr_fit,a+b*xr_fit,"-",color=col[f],lw=1.8)
    ax[1].plot(xr_ext,a+b*xr_ext,"--",color=col[f],lw=1.8)
    ax[1].plot(xT,a+b*xT,"*",color=col[f],ms=18)
    pt=" ".join(f"{t[:4]}={inv(np.polyval(np.polyfit(x,logit(FAM[f][t]),1),xT)):.0f}" for t in tasks)
    print(f"{f:8s} {b:7.3f} {r2:6.3f} {e7:6.1f}%  {pt}")

ax[0].axvline(NT/1e6,ls=":",c="gray"); ax[0].set(xscale="log",
    title="avg acc_norm vs trunk N — solid=fit, dashed=extrapolation, ★=7B",
    xlabel="trunk params (M)",ylabel="avg acc_norm %"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
ax[1].axvline(xT,ls=":",c="gray")
for xv,lab in [(x[0],"100M"),(x[3],"1.3B"),(xT,"7B")]: ax[1].annotate(lab,(xv,ax[1].get_ylim()[0]),fontsize=7,color="gray")
ax[1].set(title="Linearized: logit(acc) = a + b·log₁₀(N)  (regression line)",
    xlabel="log₁₀(trunk params)",ylabel="logit(avg acc_norm)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
fig.suptitle("Downstream regression + extrapolation to 7B: AU-Net vs Llama")
fig.tight_layout(); fig.savefig(f"{ROOT}/reports/downstream_regression.png",dpi=120)
print("saved: reports/downstream_regression.png")
