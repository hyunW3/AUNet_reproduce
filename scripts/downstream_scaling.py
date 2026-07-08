#!/usr/bin/env python
"""Two-stage downstream scaling law for the AU-Net(word) ladder.
Stage 1: scale N -> pretraining loss L(N).
Stage 2: loss L -> downstream acc_norm  (logistic: logit(acc) = a + b*L).
Compose to estimate downstream at any scale; validate by holding out the 1.3B.
Data: final loss + existing acc_norm (hellaswag/arc_easy/arc_challenge/piqa) at 4 scales."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
scale = ["100M", "300M", "760M", "1.3B"]
N     = np.array([98_591_488, 296_659_712, 714_426_880, 1_324_203_008], float)
loss  = np.array([0.8065, 0.7087, 0.6474, 0.6047])
tasks = ["hellaswag", "arc_easy", "arc_challenge", "piqa"]
acc = {  # acc_norm (%)
    "hellaswag":     np.array([28.5, 37.2, 52.4, 59.1]),
    "arc_easy":      np.array([33.7, 43.9, 55.9, 65.7]),
    "arc_challenge": np.array([23.3, 25.6, 31.7, 37.9]),
    "piqa":          np.array([58.3, 64.9, 71.1, 74.2]),
}
acc["AVERAGE"] = np.mean([acc[t] for t in tasks], axis=0)
ALLT = tasks + ["AVERAGE"]

def logit(p): p = np.clip(p/100.0, 1e-3, 1-1e-3); return np.log(p/(1-p))
def inv_logit(x): return 100.0/(1+np.exp(-x))

# ---- Stage 1: L(N) = E + A * N^-alpha  (fit via log-grid over E) ----
def fit_L(Ns, Ls):
    best = None
    for E in np.linspace(0.0, min(Ls)-1e-3, 400):
        y = np.log(Ls - E); x = np.log(Ns)
        b, a = np.polyfit(x, y, 1)          # log(L-E) = a + b*log N
        res = np.sum((y - (a + b*x))**2)
        if best is None or res < best[0]: best = (res, E, np.exp(a), -b)
    _, E, A, alpha = best
    return E, A, alpha
E, A, alpha = fit_L(N, loss)
Lhat = lambda n: E + A * n**(-alpha)

# ---- Stage 2: logit(acc) = a + b*L, fit per task ----
def fit_acc(Ls, accs):
    b, a = np.polyfit(Ls, logit(accs), 1)   # logit = a + b*L
    return a, b
fits = {t: fit_acc(loss, acc[t]) for t in ALLT}
predA = lambda t, L: inv_logit(fits[t][0] + fits[t][1]*L)

# ---- Hold-out validation: fit on 100/300/760M, predict 1.3B ----
tr = slice(0, 3)
print("=== Stage 1: L(N) = %.4f + %.3g * N^-%.3f ===" % (E, A, alpha))
print("   L(1.3B) predicted from law: %.4f  (actual %.4f)\n" % (Lhat(N[3]), loss[3]))
print("=== Stage 2 hold-out: fit on 100M/300M/760M -> predict 1.3B acc_norm ===")
print(f"{'task':14s} {'actual':>7s} {'pred(L)':>8s} {'err':>6s}   {'pred via L(N)':>13s}")
holdout = {}
for t in ALLT:
    a, b = fit_acc(loss[tr], acc[t][tr])
    p_fromL = inv_logit(a + b*loss[3])            # uses ACTUAL 1.3B loss
    p_fromN = inv_logit(a + b*Lhat(N[3]))         # uses PREDICTED loss from N (full compose)
    holdout[t] = (acc[t][3], p_fromL, p_fromN)
    print(f"{t:14s} {acc[t][3]:6.1f}% {p_fromL:7.1f}% {p_fromL-acc[t][3]:+5.1f}   {p_fromN:12.1f}%")

# ---- Compose: estimate a hypothetical larger scale ----
print("\n=== Extrapolation example (full-data fit) ===")
for n in [3.0e9, 7.0e9]:
    L = Lhat(n)
    print(f"   N={n/1e9:.0f}B -> L={L:.3f} -> AVG acc_norm ~ {predA('AVERAGE', L):.1f}%  "
          + " ".join(f"{t[:4]}={predA(t,L):.0f}%" for t in tasks))

# ---- Plots ----
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# (1) Stage 1
ng = np.logspace(np.log10(N[0]*0.8), np.log10(8e9), 100)
ax[0].plot(N/1e6, loss, "o", ms=9); ax[0].plot(ng/1e6, Lhat(ng), "-", lw=1.5)
for i,s in enumerate(scale): ax[0].annotate(s, (N[i]/1e6, loss[i]), fontsize=8)
ax[0].set(xscale="log", title=f"Stage 1: L(N)=E+A·N^-α  (E={E:.3f}, α={alpha:.3f})",
          xlabel="params (M)", ylabel="final train loss"); ax[0].grid(alpha=.3)
# (2) Stage 2: acc vs loss
Lg = np.linspace(loss.min()-0.03, loss.max()+0.03, 100)
for t in tasks:
    c = ax[1].plot(loss, acc[t], "o", ms=7, label=t)[0].get_color()
    ax[1].plot(Lg, predA(t, Lg), "-", color=c, lw=1.2)
ax[1].plot(loss, acc["AVERAGE"], "ks", ms=8, label="AVERAGE"); ax[1].plot(Lg, predA("AVERAGE", Lg), "k-", lw=1.8)
ax[1].invert_xaxis(); ax[1].set(title="Stage 2: acc_norm = logistic(loss)", xlabel="final loss (→ better)", ylabel="acc_norm %")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
# (3) hold-out: predicted vs actual 1.3B
names = ALLT; act = [holdout[t][0] for t in names]; prd = [holdout[t][1] for t in names]
x = np.arange(len(names)); w=0.38
ax[2].bar(x-w/2, act, w, label="actual 1.3B"); ax[2].bar(x+w/2, prd, w, label="predicted (from ≤760M)")
ax[2].set_xticks(x); ax[2].set_xticklabels([n[:8] for n in names], rotation=30, fontsize=8)
ax[2].set(title="Hold-out: predict 1.3B downstream from ≤760M", ylabel="acc_norm %"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
fig.suptitle("Downstream scaling law (AU-Net word): scale → loss → benchmark accuracy")
fig.tight_layout(); fig.savefig(f"{ROOT}/reports/downstream_scaling.png", dpi=115)
mae = np.mean([abs(holdout[t][1]-holdout[t][0]) for t in tasks])
print(f"\nHold-out MAE across 4 tasks (loss-based): {mae:.1f} pp ; AVG task err {holdout['AVERAGE'][1]-holdout['AVERAGE'][0]:+.1f} pp")
print("saved: reports/downstream_scaling.png")
PY = None
