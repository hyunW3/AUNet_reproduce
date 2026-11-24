#!/usr/bin/env python3
"""AU-Net2 1.3B 와 BPEByte 1.3B 의 정규화 loss 곡선이 같은가 — 직접 대조.

이 두 논문 런은 레시피가 완전히 동일하다:
  B/step 1,572,864 (12x8192x4x4 = 24x8192x2x4), steps 180,000, warmup 10,000,
  lr 0.00165, wd 0.1, lr_min_ratio 0.01  ->  tau 0.0337, TPP 213.8
다른 것은 아키텍처뿐이고, 둘 다 byte 를 예측하므로 loss 단위도 같다(nats/byte).
따라서 정규화 전 원곡선끼리도 바로 비교할 수 있다.
"""
import numpy as np, os
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

T=180000; g=np.linspace(0.12,1.0,300)
runs={}
s,l=load(f"{D}/aunet2_paper.txt",0); runs["AU-Net2 1.3B 논문"]=(s,sm(s,l,900))
s,l=load(f"{D}/paper.txt",2);        runs["BPEByte 1.3B 논문"]=(s,sm(s,l,900))
s,l=load(f"{D}/curves.txt",1);       runs["BPEByte 1.3B 재학습"]=(s,sm(s,l,900))

C={k:(np.interp(g,s/T,l), float(np.interp(1.,s/T,l))) for k,(s,l) in runs.items()}
print("원곡선 (nats/byte, 둘 다 byte 예측이라 단위가 같다)")
print(f"{'t_hat':>7} " + " ".join(f"{k:>20}" for k in C))
for t in (0.12,0.2,0.3,0.5,0.7,0.9,1.0):
    i=np.abs(g-t).argmin()
    print(f"{g[i]:7.2f} " + " ".join(f"{C[k][0][i]:20.4f}" for k in C))

A,B="AU-Net2 1.3B 논문","BPEByte 1.3B 논문"
d=C[A][0]-C[B][0]
print(f"\n원곡선 차이 (A−B): 평균 {d.mean():+.4f}  최대 {d.max():+.4f}  최소 {d.min():+.4f}")

def norm(c,f,h): return (c-h)/(f-h)
def mismatch(k1,k2):
    c1,f1=C[k1]; c2,f2=C[k2]
    best=min((np.abs(norm(c1,f1,h1)-norm(c2,f2,h2)).max(),h1,h2)
             for h1 in np.linspace(0,c1.min()*.99,150) for h2 in np.linspace(0,c2.min()*.99,150))
    return best
print("\n정규화 곡선 mismatch (L̂ 는 [0, min L) 로 제한)")
ks=list(C)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        v,h1,h2=mismatch(ks[i],ks[j])
        print(f"  {ks[i]:>20} vs {ks[j]:<20} {v:.4f}   (L̂ {h1:.3f} / {h2:.3f})")

print("\nL̂=0 으로 고정했을 때 (즉 l = L(t)/L_final) 구간별 차이")
n={k:C[k][0]/C[k][1] for k in C}
dd=n[A]-n[B]
print(f"  전체 최대 |차이| {np.abs(dd).max():.4f} at t_hat={g[np.abs(dd).argmax()]:.2f}")
for lo,hi in [(0.12,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.0)]:
    m=(g>=lo)&(g<=hi)
    print(f"  t_hat {lo:.2f}-{hi:.2f}: 최대 {np.abs(dd[m]).max():.4f}  평균 {dd[m].mean():+.4f}")

print("\n한쪽을 다른 쪽에 아핀으로 맞췄을 때 (같다면 a≈0, b≈1)")
for k1,k2 in [(A,B),(B,A),(A,"BPEByte 1.3B 재학습")]:
    c1,_=C[k1]; c2,_=C[k2]
    M=np.vstack([np.ones_like(c2),c2]).T
    (a,b),*_=np.linalg.lstsq(M,c1,rcond=None)
    r=np.sqrt(((c1-(a+b*c2))**2).mean())
    print(f"  {k1} ≈ a + b·{k2}:  a={a:+.4f}  b={b:.4f}  잔차RMS={r:.4f}")
