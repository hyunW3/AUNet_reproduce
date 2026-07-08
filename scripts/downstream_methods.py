#!/usr/bin/env python
"""Compare 3 downstream-extrapolation methods for the 7B estimate:
  (A) loss-based two-stage:  N -> L(N)=E+A N^-a  -> logit(acc)=c+d*L
  (B) direct logit-logN:     logit(acc)=a+b*log10(N)
  (C) N-weighted Huber (from scaling_laws/scaling_laws_plan.md, linear-in-acc)
Overlay A & B lines; mark C's 7B points; report which flips the AU-Net vs Llama order."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
N=np.array([98_591_488,296_659_712,714_426_880,1_324_203_008],float); NT=7.0e9
tasks=["hellaswag","arc_easy","arc_challenge","piqa"]
FAM={
 "AU-Net":dict(loss=np.array([0.8065,0.7087,0.6474,0.6047]),
   acc={"hellaswag":[28.5,37.2,52.4,59.1],"arc_easy":[33.7,43.9,55.9,65.7],"arc_challenge":[23.3,25.6,31.7,37.9],"piqa":[58.3,64.9,71.1,74.2]}),
 "Llama":dict(loss=np.array([3.400,2.995,2.853,2.743]),
   acc={"hellaswag":[31.3,42.4,55.8,62.2],"arc_easy":[42.1,48.8,62.0,65.4],"arc_challenge":[23.7,26.0,31.5,38.2],"piqa":[64.3,68.3,73.7,75.3]}),
}
for f in FAM: FAM[f]["avg"]=np.mean([FAM[f]["acc"][t] for t in tasks],0)
HUBER7B={"AU-Net":70.9,"Llama":70.2,"rg":72.2}   # from scaling_laws_plan.md
def logit(p): p=np.clip(np.asarray(p)/100,1e-3,1-1e-3); return np.log(p/(1-p))
def inv(x): return 100/(1+np.exp(-x))
def fitL(Ns,Ls):
    best=None
    for E in np.linspace(0,min(Ls)-1e-3,600):
        y=np.log(Ls-E); x=np.log(Ns); b,a=np.polyfit(x,y,1); r=np.sum((y-(a+b*x))**2)
        if best is None or r<best[0]: best=(r,E,np.exp(a),-b)
    return best[1],best[2],best[3]
col={"AU-Net":"C0","Llama":"C3"}
Ng=np.logspace(np.log10(N[0]*0.85),np.log10(8e9),160); xg=np.log10(Ng); xT=np.log10(NT)

fig,ax=plt.subplots(1,2,figsize=(14,5.6))
table={}
for f in FAM:
    loss=FAM[f]["loss"]; av=FAM[f]["avg"]
    # (A) loss-based
    pL=fitL(N,loss); L=lambda n:pL[0]+pL[1]*n**(-pL[2])
    d,c=np.polyfit(loss,logit(av),1); accA=inv(c+d*L(Ng)); A7=inv(c+d*L(NT))
    # (B) direct logN
    b,a=np.polyfit(np.log10(N),logit(av),1); accB=inv(a+b*xg); B7=inv(a+b*xT)
    table[f]=(A7,B7,HUBER7B[f])
    ax[0].plot(N/1e6,av,"o",color=col[f],ms=8,label=f"{f} data")
    ax[0].plot(Ng/1e6,accB,"-",color=col[f],lw=1.7)          # direct: solid
    ax[0].plot(Ng/1e6,accA,":",color=col[f],lw=2.2)          # loss-based: dotted
    ax[0].plot(NT/1e6,B7,"*",color=col[f],ms=18); ax[0].plot(NT/1e6,A7,"D",color=col[f],ms=8)
    ax[0].plot(NT/1e6,HUBER7B[f],"X",color=col[f],ms=10,mec="k")
ax[0].axvline(NT/1e6,ls=":",c="gray")
ax[0].plot([],[],"k-",label="(B) direct logit–logN"); ax[0].plot([],[],"k:",lw=2.2,label="(A) loss-based 2-stage")
ax[0].plot([],[],"k*",ms=14,label="B @7B"); ax[0].plot([],[],"kD",ms=7,label="A @7B"); ax[0].plot([],[],"kX",ms=9,label="Huber(plan) @7B")
ax[0].set(xscale="log",title="avg acc_norm vs trunk N — method overlay",xlabel="trunk params (M)",ylabel="avg acc_norm %")
ax[0].legend(fontsize=7,ncol=2); ax[0].grid(alpha=.3)
# bar comparison
labs=["(A) loss-2stage","(B) direct logN","(C) Huber plan"]; x=np.arange(3); w=0.35
for k,f in enumerate(FAM):
    vals=table[f]; ax[1].bar(x+(k-0.5)*w,vals,w,color=col[f],label=f)
    for i,v in enumerate(vals): ax[1].text(x[i]+(k-0.5)*w,v+0.3,f"{v:.1f}",ha="center",fontsize=8)
ax[1].axhline(HUBER7B["rg"],ls="--",c="gray",lw=1); ax[1].text(2.3,HUBER7B["rg"]+0.2,"rg(plan)=72.2",fontsize=7,color="gray")
ax[1].set_xticks(x); ax[1].set_xticklabels(labs,fontsize=8); ax[1].set(ylim=(60,75),title="7B avg acc_norm by method",ylabel="%")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3,axis="y")
fig.suptitle("Which extrapolation method changes the conclusion? (AU-Net vs Llama @7B)")
fig.tight_layout(); fig.savefig(f"{ROOT}/reports/downstream_methods.png",dpi=120)

print(f"{'method':22s} {'AU-Net':>7s} {'Llama':>7s} {'winner':>10s}")
for j,lab in enumerate(labs):
    au=table['AU-Net'][j]; ll=table['Llama'][j]
    w=("AU-Net" if au>ll else "Llama")+f" (+{abs(au-ll):.1f})"
    print(f"{lab:22s} {au:6.1f}% {ll:6.1f}% {w:>12s}")
print("saved: reports/downstream_methods.png")
