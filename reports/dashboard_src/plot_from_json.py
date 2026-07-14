#!/usr/bin/env python3
"""Downstream scaling figures from reports/downstream_data.json (single source of truth).
5 series: Llama / AU-Net / BPEByte-rg (solid) + Hybrid leafQ / greedyQ (dashed, full canonical)."""
import json, os, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT="/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
OUT=f"{ROOT}/reports/downstream_hybrid_variants"; os.makedirs(OUT,exist_ok=True)
D=json.load(open(f"{ROOT}/reports/downstream_data.json"))["data"]
SERIES=[("llama","Llama (BPE)","#0072B2","-"),("aunet(word)","AU-Net (word)","#E69F00","-"),
        ("bpebyte_rg","BPEByte-rg (online)","#009E73","-"),
        ("hyb_leafQ","Hybrid leafQ","#D55E00","--"),("hyb_greedyQ","Hybrid greedyQ","#CC79A7","--")]
SCALES=["100M","300M","760M","1.3B"]; SX={"100M":100e6,"300M":300e6,"760M":760e6,"1.3B":1300e6}
FL={"100M":3.4e18,"300M":2.9e19,"760M":1.6e20,"1.3B":5.5e20}
BEN=["hellaswag","arc_easy","arc_challenge","piqa","boolq","winogrande"]
TT={"hellaswag":"HellaSwag","arc_easy":"ARC-Easy","arc_challenge":"ARC-Challenge","piqa":"PIQA","boolq":"BoolQ","winogrande":"WinoGrande"}
CH={"hellaswag":25,"arc_easy":25,"arc_challenge":25,"piqa":50,"boolq":50,"winogrande":50}
YM={"hellaswag":"acc_norm","arc_easy":"acc_norm","arc_challenge":"acc_norm","piqa":"acc_norm","boolq":"acc","winogrande":"acc"}
INK,MUT,GR="#1a1a1a","#6b6b6b","#dcdcdc"
def sci(v): e=int(math.floor(math.log10(v)));return f"{v/10**e:.1f}×10"+str(e).translate(str.maketrans("0123456789","⁰¹²³⁴⁵⁶⁷⁸⁹"))
def style(ax,b):
    ax.set_xscale("log");ax.set_xticks(list(SX.values()))
    ax.set_xticklabels([f"{s}\n{sci(FL[s])}" for s in SCALES]);ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_:""))
    ax.tick_params(which="minor",length=0);ax.set_xlim(80e6,1.7e9);ax.grid(True,which="major",color=GR,lw=0.8,zorder=0)
    for sp in("top","right"):ax.spines[sp].set_visible(False)
    for sp in("left","bottom"):ax.spines[sp].set_color(MUT)
    ax.tick_params(colors=INK,labelsize=10)
    for l in ax.get_xticklabels():l.set_fontsize(9)
    ax.set_xlabel("Model scale (params)  ·  total training FLOPs",fontsize=10,color=INK)
    ax.set_ylabel(f"{YM[b]} (%)",fontsize=10,color=INK)
    ax.axhline(CH[b],color=MUT,lw=1,ls=(0,(2,3)),zorder=1)
    ax.annotate(f"chance {CH[b]}%",xy=(80e6,CH[b]),xytext=(0,3),textcoords="offset points",fontsize=8,color=MUT,va="bottom")
def draw(ax,b):
    style(ax,b)
    for key,lab,col,ls in SERIES:
        xs=[SX[s] for s in SCALES if D.get(key,{}).get(s,{}).get(b) is not None]
        ys=[D[key][s][b] for s in SCALES if D.get(key,{}).get(s,{}).get(b) is not None]
        if xs: ax.plot(xs,ys,color=col,lw=2,ls=ls,marker="o",ms=7,mfc=col,mec="white",mew=1.2,label=lab,zorder=5,clip_on=False)
    ax.set_title(TT[b],fontsize=12,color=INK,fontweight="bold",pad=8)
plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":130,"savefig.dpi":160,"axes.facecolor":"white","figure.facecolor":"white"})
for b in BEN:
    fig,ax=plt.subplots(figsize=(6.2,4.6));draw(ax,b);ax.legend(frameon=False,fontsize=8.5,loc="upper left");fig.tight_layout()
    fig.savefig(f"{OUT}/scaling_{b}.png",bbox_inches="tight");plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(16,9.5))
for ax,b in zip(axes.ravel(),BEN):draw(ax,b)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,frameon=False,fontsize=11,ncol=5,loc="lower center",bbox_to_anchor=(0.5,-0.005))
fig.suptitle("Downstream accuracy vs model scale (0-shot, FULL) — base 3 + Hybrid leafQ / greedyQ",fontsize=14,fontweight="bold",y=0.995)
fig.tight_layout(rect=(0,0.03,1,0.98));fig.savefig(f"{OUT}/scaling_grid.png",bbox_inches="tight");plt.close(fig)
print("wrote",len(BEN),"panels + grid to",OUT)
