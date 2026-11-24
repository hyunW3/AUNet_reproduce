#!/usr/bin/env python3
"""arm 이 다르고 τ 는 맞은 칸 — 이게 빠져 있던 대조군이다.

지금까지 결과는 (같은 arm & τ 맞음) = 성공, (다른 arm | τ 어긋남) = 실패 뿐이라
두 원인이 완전히 교락돼 있었다. 논문 AU-Net2 1.3B 가 τ=0.0337, TPP=213.8 로
BPEByte 100M 참조와 정확히 같으면서 아키텍처만 다르므로, 이 칸을 채운다.
"""
import os,numpy as np
from scipy.optimize import curve_fit
D=os.path.dirname(os.path.abspath(__file__))
def load(p,i=0):
    b=[];c=None
    for ln in open(p):
        if ln.startswith('###'): c=[];b.append(c);continue
        s,l=ln.split();c.append((int(s),float(l)))
    d={}
    for s,l in b[i]: d[s]=l
    st=np.array(sorted(d));return st,np.array([d[s] for s in st])
def sm(s,l,w): return np.array([np.median(l[(s>=s[i]-w)&(s<=s[i]+w)]) for i in range(len(l))])
W=0.12
rs,rl=load(f"{D}/curves.txt",0);RT=80841;rl=sm(rs,rl,400)      # 참조: BPEByte 100M, τ=0.0337

T={}
s,l=load(f"{D}/curves.txt",1);   T["BPEByte 1.3B 재학습"]=(s,sm(s,l,900),180000,"같은","0.0337")
s,l=load(f"{D}/aunet2_paper.txt",0); T["AU-Net2 1.3B 논문"]=(s,sm(s,l,900),180000,"다름","0.0337")
s,l=load(f"{D}/paper.txt",2);    T["BPEByte 1.3B 논문"]  =(s,sm(s,l,900),180000,"같은","0.0337")
s,l=load(f"{D}/paper.txt",1);    T["llama 1.8B 논문"]    =(s,sm(s,l,300),60000, "다름","0.1684")
s,l=load(f"{D}/cllama.txt",0);   T["llama 1.8B 재학습"]  =(s,sm(s,l,300),59589, "다름","0.1695")

def affine(ts,tl,TT,frac):
    m=(ts/TT>=W)&(ts/TT<=frac); x=ts[m]/TT; y=tl[m]
    Lr=np.interp(x,rs/RT,rl); A=np.vstack([np.ones_like(Lr),Lr]).T
    (a,b),*_=np.linalg.lstsq(A,y,rcond=None)
    return a+b*np.interp(1.,rs/RT,rl), np.sqrt(((y-(a+b*Lr))**2).mean())
def powerlaw(ts,tl,TT,frac):
    m=(ts/TT>=W)&(ts/TT<=frac); x=ts[m]/TT; y=tl[m]
    f=lambda t,c,k,a: c+k*t**(-a)
    try:
        p,_=curve_fit(f,x,y,p0=[y.min()*0.9,0.1,0.3],maxfev=60000); return f(1.,*p)
    except Exception: return float('nan')

print("참조: BPEByte 100M (τ=0.0337, TPP=214.9), 8h34m / 2 GPU\n")
print(f"{'대상':<20}{'arm':<6}{'tau':<8}{'실제':>7} | {'20%':>17}{'30%':>17}{'50%':>17} | {'RMS':>7} | {'멱법칙30%':>15}")
print("-"*118)
for k,(ts,tl,TT,arm,tau) in T.items():
    true=np.interp(1.,ts/TT,tl)
    cells=""
    for f in (0.20,0.30,0.50):
        p,rms=affine(ts,tl,TT,f); cells+=f"{p:8.4f}({(p-true)/true*100:+6.2f}%)"
    _,rms=affine(ts,tl,TT,0.30)
    pw=powerlaw(ts,tl,TT,0.30)
    print(f"{k:<20}{arm:<6}{tau:<8}{true:7.4f} | {cells} | {rms:7.4f} | {pw:7.4f}({(pw-true)/true*100:+6.2f}%)")
