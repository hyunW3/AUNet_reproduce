#!/usr/bin/env python
"""Normalized training loss curves (TLC), per "Scaling with Collapse" (2509.25087).
ell(t_hat) = L(t_hat*T) / L_final  vs  t_hat = step/T.  If runs collapse, normalized
curves overlay onto a universal curve. We also report a collapse residual (mean
pointwise spread across a group on a common t_hat grid)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"

def load(path):
    steps, loss = [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "global_step" in d and "loss/out" in d:
                steps.append(d["global_step"]); loss.append(d["loss/out"])
    return np.array(steps, float), np.array(loss, float)

def ema(x, alpha=0.05):
    y = np.empty_like(x); acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc; y[i] = acc
    return y

def norm_curve(path):
    s, l = load(path)
    order = np.argsort(s); s, l = s[order], l[order]
    ls = ema(l, 0.05)
    T = s.max()
    that = s / T
    lfinal = ls[-max(1, len(ls)//50):].mean()   # mean of last ~2%
    return that, ls / lfinal, lfinal, T

def residual(curves, grid):
    """mean over grid of (max-min) across group's normalized curves (skip warmup)."""
    interp = []
    for (that, ell, _, _) in curves:
        m = that >= grid[0]
        interp.append(np.interp(grid, that[m], ell[m]))
    A = np.vstack(interp)
    spread = A.max(0) - A.min(0)
    return spread.mean(), spread.max()

# ---- group A: AU-Net (word) scale ladder ----
A_iso = [  # cmp_g10 => iso-TPP (gamma=10)
    ("100M (g10)", "runs/small/cmp_g10/aunet_100M/metrics.jsonl"),
    ("300M (g10)", "runs/small/cmp_g10/aunet_300M/metrics.jsonl"),
    ("760M (g10)", "runs/small/cmp_g10/aunet_760M/metrics.jsonl"),
]
A_extra = [("1.3B (TPP~218)", "runs/1.3B/aunet2_1.3B/metrics.jsonl")]

# ---- group B: collapse_tau 100M (tau effect on normalized shape) ----
B = [
    ("aunet base tau=0.75", "runs/collapse_tau_metrics/aunet_tau0748.jsonl"),
    ("aunet low  tau=0.034", "runs/collapse_tau_metrics/aunet_tau0034.jsonl"),
    ("bpe   base tau=0.75", "runs/collapse_tau_metrics/bpebyte_tau0748.jsonl"),
    ("bpe   low  tau=0.034", "runs/collapse_tau_metrics/bpebyte_tau0034.jsonl"),
]

def build(group):
    out = []
    for label, rel in group:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            out.append((label,) + norm_curve(p))
    return out

# ===== Figure 1: AU-Net scale ladder =====
gA = build([(l, r) for l, r in A_iso]) + build(A_extra)
grid = np.linspace(0.20, 1.0, 100)  # post-warmup bulk (warmup~10%)
iso_curves = [(t, e, lf, T) for (lab, t, e, lf, T) in gA if "g10" in lab]
res_mean, res_max = residual(iso_curves, grid)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for lab, t, e, lf, T in gA:
    ax[0].plot(t, e * lf, label=f"{lab}  (Lf={lf:.3f})", lw=1.3)
    ls = "--" if "1.3B" in lab else "-"
    ax[1].plot(t, e, label=lab, lw=1.5, linestyle=ls)
ax[0].set(title="Raw training loss", xlabel="t / T", ylabel="loss/out"); ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
ax[1].set(title=f"Normalized TLC  L/Lfinal  (iso-TPP collapse residual: mean {res_mean:.3f}, max {res_max:.3f})",
          xlabel="t / T", ylabel="L / L_final"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle("AU-Net (word) scale ladder — normalized loss-curve collapse")
fig.tight_layout(); fig.savefig(f"{ROOT}/reports/tlc_scale_ladder.png", dpi=130)
print(f"[ladder] iso-TPP(100/300/760M) collapse residual: mean={res_mean:.4f} max={res_max:.4f}")
for lab, t, e, lf, T in gA:
    print(f"   {lab:16s} T={int(T):>6d} Lfinal={lf:.4f}")

# ===== Figure 2: collapse_tau tau effect =====
gB = build(B)
res_all, _ = residual([(t, e, lf, T) for (_, t, e, lf, T) in gB], grid)
base = [(t, e, lf, T) for (lab, t, e, lf, T) in gB if "base" in lab]
low  = [(t, e, lf, T) for (lab, t, e, lf, T) in gB if "low"  in lab]
r_base, _ = residual(base, grid); r_low, _ = residual(low, grid)

fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
for lab, t, e, lf, T in gB:
    c = "C0" if "aunet" in lab else "C1"; ls = "-" if "base" in lab else "--"
    ax2[0].plot(t, e * lf, label=f"{lab} (Lf={lf:.3f})", color=c, linestyle=ls, lw=1.3)
    ax2[1].plot(t, e, label=lab, color=c, linestyle=ls, lw=1.6)
ax2[0].set(title="Raw loss (100M)", xlabel="t / T", ylabel="loss/out"); ax2[0].set_yscale("log"); ax2[0].legend(fontsize=8)
ax2[1].set(title=f"Normalized TLC — tau deforms shape\n(within-tau residual: base={r_base:.3f}, low={r_low:.3f}; base-vs-low differ)",
           xlabel="t / T", ylabel="L / L_final"); ax2[1].legend(fontsize=8); ax2[1].grid(alpha=.3)
fig2.suptitle("collapse_tau 100M — normalized TLC: baseline (tau=0.75) vs 1.3B-matched (tau=0.034)")
fig2.tight_layout(); fig2.savefig(f"{ROOT}/reports/tlc_collapse_tau.png", dpi=130)
print(f"[collapse_tau] within-group residual base(tau0.75)={r_base:.4f} low(tau0.034)={r_low:.4f}")
for lab, t, e, lf, T in gB:
    print(f"   {lab:22s} Lfinal={lf:.4f}")

# ===== Figure 3: 100M vs 1.3B at matched tau (across-scale collapse) =====
NPARAM = {"1.3B": 1_324_203_008, "100M": 98_591_488}  # actual param counts (from train logs)
def last_tokens(path):
    tok = None
    with open(path) as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            if "optim/total_tokens" in d: tok = d["optim/total_tokens"]
    return tok

C = [
    ("1.3B aunet", "runs/1.3B/aunet2_1.3B/metrics.jsonl",        "0.034", "1.3B"),
    ("1.3B bpe",   "runs/1.3B/bpebyte_br_bt_1.3B/metrics.jsonl", "0.034", "1.3B"),
    ("100M aunet", "runs/collapse_tau_metrics/aunet_tau0034.jsonl",   "0.034", "100M"),
    ("100M bpe",   "runs/collapse_tau_metrics/bpebyte_tau0034.jsonl", "0.034", "100M"),
    ("100M aunet", "runs/collapse_tau_metrics/aunet_tau0748.jsonl",   "0.75",  "100M"),
    ("100M bpe",   "runs/collapse_tau_metrics/bpebyte_tau0748.jsonl", "0.75",  "100M"),
]
gC = []
for base, rel, tk, sc in C:
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        tpp = last_tokens(p) / NPARAM[sc]
        lab = f"{base}  τ={tk} TPP={tpp:.0f}"
        gC.append((lab, tk, sc) + norm_curve(p))

# across-scale matched-tau residual: 100M-low vs 1.3B (tau=0.034)
mlow = [(t, e, lf, T) for (lab, tk, sc, t, e, lf, T) in gC if tk == "0.034"]
r_scale, r_scale_max = residual(mlow, grid)
# contrast: 100M-low(0.034) vs 100M-base(0.75)
c034 = [(t, e, lf, T) for (lab, tk, sc, t, e, lf, T) in gC if sc == "100M" and tk == "0.034"]
c075 = [(t, e, lf, T) for (lab, tk, sc, t, e, lf, T) in gC if sc == "100M" and tk == "0.75"]
def cross(a, b):
    ea = np.mean([np.interp(grid, t[t>=grid[0]], e[t>=grid[0]]) for (t,e,_,_) in a], 0)
    eb = np.mean([np.interp(grid, t[t>=grid[0]], e[t>=grid[0]]) for (t,e,_,_) in b], 0)
    return np.abs(ea-eb).mean()
d_tau = cross(c034, c075)

fig3, ax3 = plt.subplots(1, 2, figsize=(13, 5))
# color by (scale, tau) group; all solid lines. arch (aunet/bpe) differs by lw.
gcol = {("1.3B", "0.034"): "C0", ("100M", "0.034"): "C2", ("100M", "0.75"): "C3"}
for lab, tk, sc, t, e, lf, T in gC:
    c = gcol[(sc, tk)]; lw = 2.0 if "aunet" in lab else 1.2
    ax3[0].plot(t, e * lf, color=c, lw=lw, label=f"{lab} (Lf={lf:.3f})")
    ax3[1].plot(t, e, color=c, lw=lw, label=lab)
ax3[0].set(title="Raw loss", xlabel="t / T", ylabel="loss/out"); ax3[0].set_yscale("log"); ax3[0].legend(fontsize=7)
ax3[1].set(title=f"Normalized TLC — 1.3B (blue, TPP214) vs 100M (green/red, TPP18)\nmatched-τ but TPP-mismatched: across-scale residual={r_scale:.3f}; 100M τ-gap (0.034 vs 0.75)={d_tau:.3f}",
           xlabel="t / T", ylabel="L / L_final"); ax3[1].legend(fontsize=7); ax3[1].grid(alpha=.3)
fig3.suptitle("Across-scale collapse at matched τ=0.034: 1.3B (blue) vs 100M τ=0.034 (green) vs 100M τ=0.75 (red)")
fig3.tight_layout(); fig3.savefig(f"{ROOT}/reports/tlc_100M_vs_1p3B.png", dpi=130)
print(f"[across-scale] matched-tau 100M-vs-1.3B residual={r_scale:.4f} (max {r_scale_max:.4f}); 100M tau-gap 0.034-vs-0.75={d_tau:.4f}")
for lab, tk, sc, t, e, lf, T in gC:
    print(f"   {lab:22s} T={int(T):>6d} Lfinal={lf:.4f}")
# ===== Figure 4: SIX-CURVE across-scale collapse test (all tau=0.034) =====
# 1.3B (TPP214) vs 100M matched-TPP=209 (new) vs 100M TPP=18 (old collapse_tau).
D6 = [
    ("1.3B aunet",     "runs/1.3B/aunet2_1.3B/metrics.jsonl",                 "1.3B", "C0"),
    ("1.3B bpe",       "runs/1.3B/bpebyte_br_greedy_root_1.3B/metrics.jsonl", "1.3B", "C0"),
    ("100M aunet new", "runs/collapse_tau_metrics/scale_aunet_word.jsonl",    "100M", "C2"),
    ("100M bpe new",   "runs/collapse_tau_metrics/scale_bpebyte_rg.jsonl",    "100M", "C2"),
    ("100M aunet old", "runs/collapse_tau_metrics/aunet_tau0034.jsonl",       "100M", "C3"),
    ("100M bpe old",   "runs/collapse_tau_metrics/bpebyte_tau0034.jsonl",     "100M", "C3"),
]
g6 = []
for base, rel, sc, col in D6:
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        tpp = last_tokens(p) / NPARAM[sc]
        g6.append((base, sc, col, tpp) + norm_curve(p))
def cross_group(a, b):
    ea = np.mean([np.interp(grid, t[t >= grid[0]], e[t >= grid[0]]) for (t, e, _, _) in a], 0)
    eb = np.mean([np.interp(grid, t[t >= grid[0]], e[t >= grid[0]]) for (t, e, _, _) in b], 0)
    d = np.abs(ea - eb); return d.mean(), d.max()
b13 = [(t, e, lf, T) for (base, sc, col, tpp, t, e, lf, T) in g6 if sc == "1.3B"]
new = [(t, e, lf, T) for (base, sc, col, tpp, t, e, lf, T) in g6 if col == "C2"]
old = [(t, e, lf, T) for (base, sc, col, tpp, t, e, lf, T) in g6 if col == "C3"]
r_new, _ = cross_group(new, b13)
r_old, _ = cross_group(old, b13)

fig4, ax4 = plt.subplots(1, 2, figsize=(13, 5))
for base, sc, col, tpp, t, e, lf, T in g6:
    lw = 2.3 if "aunet" in base else 1.2
    ax4[0].plot(t, e * lf, color=col, lw=lw, label=f"{base}  TPP={tpp:.0f} (Lf={lf:.3f})")
    ax4[1].plot(t, e, color=col, lw=lw, label=f"{base}  TPP={tpp:.0f}")
ax4[0].set(title="Raw loss (all τ=0.034)", xlabel="t / T", ylabel="loss/out"); ax4[0].set_yscale("log"); ax4[0].legend(fontsize=7)
ax4[1].set(title=f"Normalized TLC (all τ=0.034)\n100M matched-TPP(209) vs 1.3B residual={r_new:.3f}   |   100M TPP18 vs 1.3B={r_old:.3f}",
           xlabel="t / T", ylabel="L / L_final"); ax4[1].legend(fontsize=7); ax4[1].grid(alpha=.3)
fig4.suptitle("Across-scale collapse test — 1.3B (blue) vs 100M matched TPP=209 (green) vs 100M TPP=18 (red); all τ=0.034, 1.3B schedule")
fig4.tight_layout(); fig4.savefig(f"{ROOT}/reports/tlc_six_curve.png", dpi=130)
print(f"[SIX-CURVE] 100M matched-TPP(209) vs 1.3B residual={r_new:.4f} ; 100M TPP18(old) vs 1.3B residual={r_old:.4f}")
for base, sc, col, tpp, t, e, lf, T in g6:
    print(f"   {base:16s} TPP={tpp:6.1f}  Lfinal={lf:.4f}")

print("saved: reports/tlc_scale_ladder.png , reports/tlc_collapse_tau.png , reports/tlc_100M_vs_1p3B.png , reports/tlc_six_curve.png")
