#!/usr/bin/env python3
"""참조 런이 실제로 무엇을 벌어주는지 — τ맞춤 100M / 비맞춤 760M / 참조없음 3자 비교."""
import numpy as np
from scipy.optimize import curve_fit

def load(path, idx=0):
    blocks, cur = [], None
    for line in open(path):
        if line.startswith('###'): cur=[]; blocks.append(cur); continue
        s,l = line.split(); cur.append((int(s), float(l)))
    d={}
    for s,l in blocks[idx]: d[s]=l
    st=np.array(sorted(d)); return st, np.array([d[s] for s in st])

def smooth(s,l,win):
    return np.array([np.median(l[(s>=s[i]-win)&(s<=s[i]+win)]) for i in range(len(l))])

import os
D=os.path.dirname(os.path.abspath(__file__))
C=os.path.join(D,"curves.txt")
rs,rl = load(C,0); ts,tl = load(C,1)
s7,l7 = load(os.path.join(D,"c760.txt"),0)
RT,TT,T7 = 80841,180000,29200
rl=smooth(rs,rl,400); tl=smooth(ts,tl,900); l7=smooth(s7,l7,150)
WARM=0.12
L_true = np.interp(1.0, ts/TT, tl)

def affine_pred(frac, refs, refl, refT):
    m=(ts/TT>=WARM)&(ts/TT<=frac)
    x=ts[m]/TT; y=tl[m]
    Lr=np.interp(x, refs/refT, refl)
    A=np.vstack([np.ones_like(Lr),Lr]).T
    (a,b),*_=np.linalg.lstsq(A,y,rcond=None)
    return a + b*np.interp(1.0, refs/refT, refl), np.sqrt(((y-(a+b*Lr))**2).mean())

def powerlaw_pred(frac):          # 참조 없이 대상 커브만으로: L = c + k*t^-alpha
    m=(ts/TT>=WARM)&(ts/TT<=frac)
    x=ts[m]/TT; y=tl[m]
    f=lambda t,c,k,a: c + k*t**(-a)
    try:
        p,_=curve_fit(f,x,y,p0=[0.5,0.1,0.3],maxfev=40000)
        return f(1.0,*p), np.sqrt(((y-f(x,*p))**2).mean())
    except Exception as e:
        return np.nan, np.nan

print(f"1.3B 실제 최종 loss = {L_true:.4f}\n")
hdr=f"{'관측':>6} | {'100M τ맞춤':>20} | {'760M 비맞춤':>20} | {'참조없음 멱법칙':>20}"
print(hdr); print('-'*len(hdr))
for frac in (0.20,0.30,0.40,0.50,0.70):
    row=f"{frac*100:5.0f}% |"
    for pred,rms in (affine_pred(frac,rs,rl,RT), affine_pred(frac,s7,l7,T7), powerlaw_pred(frac)):
        row += f" {pred:7.4f} ({(pred-L_true)/L_true*100:+6.2f}%) |"
    print(row)
