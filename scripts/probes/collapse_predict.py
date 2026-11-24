#!/usr/bin/env python3
# 입력: reports_afterAAAIsub/curves.txt (100M 참조 + 1.3B 대상), c760.txt (760M).
#   재생성: ssh ece-agpu11 grep -ah "step:" <train.log> | sed -n "s/.*step: *\([0-9]*\).*loss: *\([0-9.]*\).*/\1 \2/p"
"""Scaling-with-Collapse 예측 — τ/TPP/스케줄을 맞춘 100M 참조 런으로 1.3B 최종 loss 예측.

Eq.1  l(t_hat) = (L(t_hat*T*) - L_hat) / (L(T*) - L_hat)
τ, TPP, LR 스케줄 모양을 맞추면 이 l(t_hat) 이 스케일에 무관하게 겹친다는 주장이다.
=> 대상 런에서는 L(t_hat) = L_hat + (L_final - L_hat)*l_ref(t_hat) 가 성립하고,
   이건 미지수 두 개(L_hat, L_final-L_hat)에 대해 **선형**이다.
   앞부분 f 만 보고 최소제곱으로 풀면 L_final = A + B (l_ref(1)=1 이므로).
"""
import numpy as np, sys

def load(path, marker_idx):
    blocks, cur = [], None
    for line in open(path):
        if line.startswith('###'):
            cur = []; blocks.append(cur); continue
        s, l = line.split()
        cur.append((int(s), float(l)))
    b = blocks[marker_idx]
    d = {}
    for s, l in b:            # 재개로 중복된 step 은 마지막 것을 쓴다
        d[s] = l
    st = np.array(sorted(d)); ls = np.array([d[s] for s in st])
    return st, ls

P = sys.argv[1]
rs, rl = load(P, 0); ts, tl = load(P, 1)
RT, TT = 80841, 180000
print(f"참조 100M: step {rs.min()}..{rs.max()}, {len(rs)}점")
print(f"대상 1.3B: step {ts.min()}..{ts.max()}, {len(ts)}점")

def smooth(s, l, win):        # step 축 기준 이동 중앙값 (loss 로그는 마이크로배치 단위라 시끄럽다)
    out = np.empty_like(l)
    for i in range(len(l)):
        m = (s >= s[i]-win) & (s <= s[i]+win)
        out[i] = np.median(l[m])
    return out

rl_s = smooth(rs, rl, 400)
tl_s = smooth(ts, tl, 900)

# warmup 구간(10%)은 제외한다 — 스케줄 모양은 같지만 초기 과도구간은 collapse 가정이 약하다
WARM = 0.12
rt_hat = rs / RT
grid = np.linspace(WARM, 1.0, 400)
l_ref_raw = np.interp(grid, rt_hat, rl_s)
L_ref_final = np.interp(1.0, rt_hat, rl_s)

def predict(frac, L_hat_free=True):
    """대상 런의 앞 frac 만 보고 최종 loss 예측."""
    m = (ts/TT >= WARM) & (ts/TT <= frac)
    x_t = ts[m]/TT; y = tl_s[m]
    # 참조 커브를 정규화하려면 L_hat_ref 가 필요한데, 그것 자체가 미지수다.
    # 대신 참조의 '모양'만 쓴다: L_tgt(t) = a + b * L_ref(t) 로 두면
    # a,b 는 (L_hat, scale) 의 아핀 재매개변수화이고 예측값은 동일하다.
    L_ref_t = np.interp(x_t, rt_hat, rl_s)
    A = np.vstack([np.ones_like(L_ref_t), L_ref_t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    pred = a + b * L_ref_final
    resid = y - (a + b*L_ref_t)
    return pred, a, b, np.sqrt((resid**2).mean())

L_true = np.interp(1.0, ts/TT, tl_s)
print(f"\n1.3B 실제 최종 loss (평활): {L_true:.4f}   (마지막 원시값 {tl[-1]:.4f})")
print(f"100M 참조 최종 loss (평활): {L_ref_final:.4f}")
print("\n앞부분만 보고 예측한 1.3B 최종 loss")
print(f"{'관측구간':>10} {'예측':>8} {'오차':>8} {'상대오차':>9} {'적합RMS':>9}")
for f in (0.20, 0.30, 0.40, 0.50, 0.70, 1.00):
    p, a, b, rms = predict(f)
    print(f"{f*100:9.0f}% {p:8.4f} {p-L_true:+8.4f} {(p-L_true)/L_true*100:+8.2f}% {rms:9.5f}")

# 정규화 곡선 mismatch: 두 런의 l(t_hat) 이 실제로 얼마나 겹치나
# L_hat 을 자유 파라미터로 두고, 두 정규화 곡선의 최대 절대차를 최소화한다
def mismatch(L_hat_r, L_hat_t):
    lr_ = (np.interp(grid, rt_hat, rl_s) - L_hat_r) / (L_ref_final - L_hat_r)
    lt_ = (np.interp(grid, ts/TT, tl_s) - L_hat_t) / (L_true - L_hat_t)
    return np.abs(lr_ - lt_).max()

best = min(((mismatch(hr, ht), hr, ht)
            for hr in np.linspace(0.0, 0.70, 71)
            for ht in np.linspace(0.0, 0.60, 61)), key=lambda z: z[0])
print(f"\n정규화 곡선 mismatch (L_hat 최적화): {best[0]:.4f}  at L_hat_ref={best[1]:.2f}, L_hat_tgt={best[2]:.2f}")
print(f"  L_hat=0 고정: {mismatch(0.0, 0.0):.4f}")

print("\n=== mismatch 세부 ===")
from scipy.optimize import minimize
def mm_curve(hr, ht):
    lr_ = (np.interp(grid, rt_hat, rl_s) - hr) / (L_ref_final - hr)
    lt_ = (np.interp(grid, ts/TT, tl_s) - ht) / (L_true - ht)
    return lr_ - lt_
res = minimize(lambda p: np.abs(mm_curve(*p)).max(), [0.0, 0.0], method='Nelder-Mead')
hr, ht = res.x
print(f"L_hat 자유(음수 허용): mismatch {res.fun:.4f}  L_hat_ref={hr:.3f}  L_hat_tgt={ht:.3f}")
c = mm_curve(hr, ht)
print(f"  최대 |차이| 위치 t_hat={grid[np.abs(c).argmax()]:.2f}")
for lo, hi in [(0.12,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.0)]:
    m=(grid>=lo)&(grid<hi if hi<1 else grid<=hi)
    print(f"  t_hat {lo:.2f}-{hi:.2f}: 최대 {np.abs(c[m]).max():.4f}  평균 {np.abs(c[m]).mean():.4f}")

print("\n=== 재개 지점 점검(1.3B) ===")
d1 = np.diff(tl_s); idx = np.argsort(-np.abs(d1))[:8]
for i in sorted(idx): print(f"  step {ts[i]:7d} (t_hat {ts[i]/TT:.3f})  d(loss)={d1[i]:+.4f}")
