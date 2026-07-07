#!/usr/bin/env python
"""Ratio-10 (gamma=10) scaling law — BPEByte root_greedy vs AU-Net(word) vs Llama(subword).

Constant data-to-model ratio (gamma=10 ~ 46 LLaMa-tok/param) across 100M/300M/760M/1.3B, so the
final-budget BPB is a clean iso-ratio scaling law. Fits are N-WEIGHTED HUBER (Chinchilla Approach-3
style): weight each point proportional to N so the larger, more-reliable models dominate the fit and
the 7B extrapolation is anchored on them; Huber loss makes it robust to the noisier small-scale/acc
points. Fits BPB(N)=E+A*N^-alpha and downstream acc(N)=C-A*N^-b, extrapolates to 7B, writes
scaling_laws/scaling_laws_plan.md + 3 figures.

Reads BPB/compute from each run's metrics.jsonl and downstream from eval_scaling/results.json
(1.3B downstream from the measured 4-bench; rg 1.3B from the native-greedy confirm eval).
"""
import json, math, statistics
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path("/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet")
RUNS = ROOT/"runs"
LN2 = math.log(2.0); BPT = 4.5483
SCALES = ["100M","300M","760M","1.3B"]
N = {"100M":98.59e6,"300M":296.0e6,"760M":679.6e6,"1.3B":1.30e9}
N7 = 7.0e9                                   # extrapolation target
COL = {"rg":"#1f4ed8","aunet":"#2e9e3f","llama":"#c81e1e"}
MK  = {"rg":"o","aunet":"s","llama":"^"}
LAB = {"rg":"BPEByte root_greedy","aunet":"AU-Net (word)","llama":"Llama (subword)"}

REG = {
 "rg":{
  "100M":("cmp_g10/v4_root_greedy","byte","cmp_g10/v4_root_greedy/eval_scaling"),
  "300M":("cmp_g10/rg_300M","byte","cmp_g10/rg_300M/eval_scaling"),
  "760M":("cmp_g10/rg_760M","byte","cmp_g10/rg_760M/eval_scaling"),
  "1.3B":("1.3B/bpebyte_br_greedy_root_1.3B","byte","1.3B/bpebyte_br_greedy_root_1.3B/eval_confirm_4bench")},
 "aunet":{
  "100M":("cmp_g10/aunet_100M","byte","cmp_g10/aunet_100M/eval_scaling"),
  "300M":("cmp_g10/aunet_300M","byte","cmp_g10/aunet_300M/eval_scaling"),
  "760M":("cmp_g10/aunet_760M","byte","cmp_g10/aunet_760M/eval_scaling"),
  "1.3B":("aunet2_1.3B","byte",None)},
 "llama":{
  "100M":("cmp_g10/llama_100M","llama","cmp_g10/llama_100M/eval_scaling"),
  "300M":("cmp_g10/llama_300M","llama","cmp_g10/llama_300M/eval_scaling"),
  "760M":("cmp_g10/llama_760M","llama","cmp_g10/llama_760M/eval_scaling"),
  "1.3B":("llama_1.8B_paper","llama",None)},
}
ACC_1p3B = {"rg":60.3,"aunet":59.75,"llama":60.28}
# 5-shot 4-bench (HS/ARC-E/ARC-C/PIQA) acc_norm means, measured on ece-agpu11 (num_fewshot=5).
FIVE = {"rg":{"100M":36.3,"300M":43.9,"760M":55.4,"1.3B":62.9},
        "aunet":{"100M":36.0,"300M":44.5,"760M":55.6,"1.3B":62.6},
        "llama":{"100M":41.0,"300M":48.3,"760M":57.4,"1.3B":61.9}}
TASKS = ["hellaswag","arc_easy","arc_challenge","piqa"]

def load(mdir, typ):
    p = RUNS/mdir/"metrics.jsonl"
    if not p.exists(): return None
    bpt = 1.0 if typ=="byte" else BPT
    rows=[]; fr=[]
    for line in open(p):
        try: d=json.loads(line)
        except: continue
        F,w=d.get("speed/FLOPS"),d.get("speed/wps")
        if F and w: fr.append(F/w)
        l,t,s=d.get("loss/out"),d.get("optim/total_tokens"),d.get("global_step")
        if None in (l,t,s): continue
        rows.append((s,l,t))
    if not rows: return None
    rows.sort()
    bpb=statistics.mean(r[1] for r in rows[-200:])/(LN2*bpt)
    fpt=statistics.median(fr) if fr else float("nan")
    return {"bpb":bpb,"C":rows[-1][2]*fpt/1e21,"gb":rows[-1][2]*bpt/1e9}

def acc(evaldir):
    if not evaldir: return None
    d=RUNS/evaldir
    for p in [d/"results.json"]+list(d.glob("results*.json"))+list(d.rglob("results*.json")):
        if p.exists():
            r=json.load(open(p)).get("results",{})
            vals=[r[t]["acc_norm,none"]*100 for t in TASKS if t in r and "acc_norm,none" in r[t]]
            if len(vals)==len(TASKS): return statistics.mean(vals)
    return None

# ---- N-weighted Huber fitters (x in units of 1e6 params) ----
def _wr2(model, x, y, w, p):
    resid = model(x,*p)-y
    ss_res=float((resid**2).sum()); ss_tot=float(((y-y.mean())**2).sum())
    return 1-ss_res/ss_tot if ss_tot>0 else float("nan")
def fit_bpb(Ns, ys):
    x=np.array(Ns,float)/1e6; y=np.array(ys,float); w=np.array(Ns,float)/np.mean(Ns)
    m=lambda x,E,A,a: E+A*np.power(x,-a)
    res=least_squares(lambda p: np.sqrt(w)*(m(x,*p)-y),[0.6,3,0.3],loss="huber",f_scale=0.02,
                      bounds=([0,0,0],[3,1e9,3]),max_nfev=20000)
    E,A,a=res.x
    return {"E":E,"A":A,"a":a,"R2":_wr2(m,x,y,w,res.x),"m":m,
            "pred7B":E+A*(N7/1e6)**-a}
def fit_acc(Ns, ys):
    x=np.array(Ns,float)/1e6; y=np.array(ys,float); w=np.array(Ns,float)/np.mean(Ns)
    m=lambda x,C,A,b: C-A*np.power(x,-b)
    res=least_squares(lambda p: np.sqrt(w)*(m(x,*p)-y),[80,140,0.18],loss="huber",f_scale=1.5,
                      bounds=([50,0,0],[100,1e7,3]),max_nfev=20000)
    C,A,b=res.x
    return {"C":C,"A":A,"b":b,"R2":_wr2(m,x,y,w,res.x),"m":m,
            "pred7B":C-A*(N7/1e6)**-b}

# ---------------- collect ----------------
D={m:{} for m in REG}
for m in REG:
    for sc in SCALES:
        mdir,typ,ed=REG[m][sc]
        r=load(mdir,typ) or {}
        a=acc(ed);  a=ACC_1p3B[m] if (a is None and sc=="1.3B") else a
        r["acc"]=a; r["N"]=N[sc]; D[m][sc]=r

FB={m:fit_bpb([D[m][sc]["N"] for sc in SCALES],[D[m][sc]["bpb"] for sc in SCALES]) for m in REG}
FA={m:fit_acc([D[m][sc]["N"] for sc in SCALES],[D[m][sc]["acc"] for sc in SCALES]) for m in REG}

print(f"{'scale':6} {'N':>8} {'model':>7} {'BPB':>7} {'acc%':>6}")
for sc in SCALES:
    for m in ("rg","aunet","llama"):
        r=D[m][sc]; print(f"{sc:6} {N[sc]/1e6:7.0f}M {m:>7} {r['bpb']:7.4f} {r['acc']:6.1f}")
print("\n=== N-weighted Huber fits + 7B extrapolation ===")
for m in ("rg","aunet","llama"):
    print(f"  {LAB[m]:22} BPB: E={FB[m]['E']:.3f} a={FB[m]['a']:.3f} R2={FB[m]['R2']:.3f} -> BPB@7B={FB[m]['pred7B']:.3f}"
          f"  | acc b={FA[m]['b']:.3f} -> acc@7B={FA[m]['pred7B']:.1f}")

# ---------------- Fig 1: BPB vs N (weighted Huber, extrapolated to 7B) ----------------
fig,ax=plt.subplots(figsize=(9,6.2))
xs=np.logspace(np.log10(min(N.values())),np.log10(N7),200)
for m in ("rg","aunet","llama"):
    Ns=[D[m][sc]["N"] for sc in SCALES]; ys=[D[m][sc]["bpb"] for sc in SCALES]
    ax.scatter(Ns,ys,color=COL[m],marker=MK[m],s=80,zorder=6,edgecolor="white")
    f=FB[m]; ax.plot(xs,f["m"](xs/1e6,f["E"],f["A"],f["a"]),color=COL[m],lw=2,
                     label=f"{LAB[m]}: E={f['E']:.3f}, alpha={f['a']:.3f} -> BPB@7B={f['pred7B']:.3f}")
    ax.scatter([N7],[f["pred7B"]],color=COL[m],marker="*",s=200,zorder=6,edgecolor="black")
ax.axvline(N7,color="gray",ls=":",alpha=.6); ax.text(N7*0.62,ax.get_ylim()[1],"7B (extrap.)",fontsize=8,color="gray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("non-embedding trunk params N"); ax.set_ylabel("ratio-10 train BPB")
ax.set_title("Ratio-10 BPB scaling law (N-weighted Huber) + 7B extrapolation")
ax.grid(True,which="both",alpha=.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(RUNS/"ratio10_bpb_vs_n.png",dpi=120)

# ---------------- Fig 2: BPB vs compute ----------------
fig2,bx=plt.subplots(figsize=(8.5,6))
for m in ("rg","aunet","llama"):
    pts=[(D[m][sc]["C"],D[m][sc]["bpb"]) for sc in SCALES if not math.isnan(D[m][sc].get("C",float("nan")))]
    if len(pts)<2: continue
    Cs,ys=zip(*pts); bx.plot(Cs,ys,color=COL[m],marker=MK[m],lw=1.8,label=LAB[m])
bx.set_xscale("log"); bx.set_yscale("log")
bx.set_xlabel("training compute C (ZFLOP)"); bx.set_ylabel("ratio-10 train BPB")
bx.set_title("Ratio-10 BPB vs compute"); bx.grid(True,which="both",alpha=.3); bx.legend(fontsize=9)
plt.tight_layout(); plt.savefig(RUNS/"ratio10_bpb_vs_compute.png",dpi=120)

# ---------------- Fig 3: downstream acc vs N (weighted Huber, extrapolated to 7B) ----------------
fig3,cx=plt.subplots(figsize=(9,6.2))
for m in ("rg","aunet","llama"):
    Ns=[D[m][sc]["N"] for sc in SCALES]; ys=[D[m][sc]["acc"] for sc in SCALES]
    cx.scatter(Ns,ys,color=COL[m],marker=MK[m],s=80,zorder=6,edgecolor="white")
    f=FA[m]; cx.plot(xs,f["m"](xs/1e6,f["C"],f["A"],f["b"]),color=COL[m],lw=2,
                     label=f"{LAB[m]}: b={f['b']:.3f} -> acc@7B={f['pred7B']:.1f}%")
    cx.scatter([N7],[f["pred7B"]],color=COL[m],marker="*",s=200,zorder=6,edgecolor="black")
cx.axvline(N7,color="gray",ls=":",alpha=.6); cx.text(N7*0.62,cx.get_ylim()[0],"7B (extrap.)",fontsize=8,color="gray")
cx.set_xscale("log")
cx.set_xlabel("non-embedding trunk params N"); cx.set_ylabel("0-shot acc_norm mean (HS/ARC-E/ARC-C/PIQA) %")
cx.set_title("Ratio-10 downstream accuracy scaling law (N-weighted Huber) + 7B extrapolation")
cx.grid(True,which="both",alpha=.3); cx.legend(fontsize=9,loc="lower right")
plt.tight_layout(); plt.savefig(RUNS/"ratio10_acc_vs_n.png",dpi=120)

# ---------------- writeup ----------------
def tbl(get):
    return [f"| {sc} | {N[sc]/1e6:.0f}M | {get('rg',sc)} | {get('aunet',sc)} | {get('llama',sc)} |" for sc in SCALES]
L=["# Ratio-10 (gamma=10) scaling law — BPEByte rg vs AU-Net vs Llama\n",
   "Constant data-to-model ratio gamma=10 (~46 LLaMa-tok/param), all scales. Fits are **N-weighted",
   "Huber** (weight prop. N, so the large models anchor the extrapolation; robust to small-scale noise).",
   "Auto-generated by `runs/plot_ratio10_ladder.py`.\n",
   "## Methods (fitting)\n",
   "- **Models** (x = N in units of 1e6 params): BPB(x)=E+A·x^(−α) (floor E, exponent α);"
   " acc(x)=C−A·x^(−b) (ceiling C, exponent b).",
   "- **N-weighting:** weighted least-squares with weight wᵢ = Nᵢ/mean(N), folded into"
   " `scipy.optimize.least_squares` as the residual √wᵢ·(m(xᵢ)−yᵢ). So the largest model counts"
   " ~13× the smallest → the big, reliable points anchor the fit and the 7B extrapolation.",
   "- **Huber robust loss:** `loss='huber'` with `f_scale=δ` (δ=0.02 BPB, 1.5 acc-pts): residuals"
   " below δ are ordinary squared error, above δ switch to linear → outliers (e.g. the 300M rg acc"
   " point) are down-weighted rather than dominating.",
   "- **Objective:** minimize Σᵢ ρ_Huber( √wᵢ·(m(Nᵢ)−yᵢ) ; δ ) over the fit params; bounds keep"
   " E∈[0,3], A≥0, α/b∈[0,3].",
   "- **Caveat:** 4 points / 3 params (1 dof) → residuals stay well under δ, so Huber is near-"
   "quadratic here (mild); the N-weighting is the choice that actually sets the extrapolation.\n",
   "## Train BPB\n","| scale | N | rg | AU-Net | Llama |","|---|---|---|---|---|",
   *tbl(lambda m,sc: f"{D[m][sc]['bpb']:.4f}"),
   "\n## Downstream acc_norm (0-shot 4-bench mean)\n","| scale | N | rg | AU-Net | Llama |","|---|---|---|---|---|",
   *tbl(lambda m,sc: f"{D[m][sc]['acc']:.1f}"),
   "\n## Downstream acc_norm — 5-shot (4-bench mean, num_fewshot=5)\n","| scale | N | rg | AU-Net | Llama |","|---|---|---|---|---|",
   *[f"| {sc} | {N[sc]/1e6:.0f}M | {FIVE['rg'][sc]:.1f} | {FIVE['aunet'][sc]:.1f} | {FIVE['llama'][sc]:.1f} |" for sc in SCALES],
   "\n**5-shot flips the ranking at 1.3B:** 0-shot Llama 60.3 = rg 60.3 > AU-Net 59.8; **5-shot rg 62.9 >",
   "AU-Net 62.6 > Llama 61.9** — the byte models exploit the exemplars more (Δ from 0-shot: rg +2.6,",
   "AU-Net +2.8 vs Llama +1.6). Up to 760M Llama still leads on 5-shot; the crossover is at 1.3B."
   " Per-benchmark at 1.3B 5-shot: rg wins ARC-Challenge (41.9), AU-Net wins ARC-Easy (72.4), Llama PIQA (75.7).",
   "\n## Fitted laws (N-weighted Huber)\n",
   "| model | BPB: E | alpha | R2 | acc: b | R2 |","|---|---|---|---|---|---|"]
for m in ("rg","aunet","llama"):
    L.append(f"| {LAB[m]} | {FB[m]['E']:.4f} | {FB[m]['a']:.4f} | {FB[m]['R2']:.3f} | {FA[m]['b']:.4f} | {FA[m]['R2']:.3f} |")
L+=["\n## 7B extrapolation (anchored on the large models)\n",
    "| model | BPB@7B | downstream@7B |","|---|---|---|"]
for m in ("rg","aunet","llama"):
    L.append(f"| {LAB[m]} | {FB[m]['pred7B']:.3f} | {FA[m]['pred7B']:.1f}% |")
L+=["\n**Reading it:** on BPB, Llama keeps a small edge to 7B (its low irreducible floor persists, ~0.04);",
    "on downstream, the byte models' steeper slope carries rg slightly *ahead* of Llama by 7B. So byte-level",
    "rg matches/overtakes subword on downstream while staying marginally behind on raw BPB.",
    "\n**Caveat:** 4-point fits extrapolated 5.4x past the largest data point — the weighting anchors on the",
    "reliable large models but cannot manufacture certainty; treat 7B as a hypothesis, not a measurement.\n",
    "## Figures\n","- ![BPB vs N](ratio10_bpb_vs_n.png)","- ![BPB vs compute](ratio10_bpb_vs_compute.png)",
    "- ![Acc vs N](ratio10_acc_vs_n.png)"]
(ROOT/"scaling_laws/scaling_laws_plan.md").write_text("\n".join(L)+"\n")
print("\nsaved 3 figures + scaling_laws/scaling_laws_plan.md")
