"""How the V5 distilled k-gram boundary table is learned from the offline BPE tokenizer.

A static, 4-stage explainer figure built from REAL objects (not mock-ups):
  Stage 1  offline BPE patches a demo string -> per-byte boundary label (the TEACHER)
  Stage 2  every byte's k-byte trailing windows (orders 8/6/4/3/2/1) are FNV-hashed
  Stage 3  per (order, context-hash): aggregate p = P(boundary | context) over the corpus
           -> the kept lookup table (real stored p pulled from distilled_boundary.npz)
  Stage 4  inference: longest order whose context is in the table wins (backoff); p>=.5 -> cut

The teacher mask comes from RegexPool._offline_levels_mask (bpe_br, offline) and the
probabilities in Stage 3/4 are the ACTUAL values stored in the trained table, looked up by
hashing the demo windows — so the figure shows the real selector, not an illustration.

Run in the training venv (CPU only):
    cd ../lingua && CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        .venv/bin/python ../viz/distill_kgram_explainer.py
Writes ../runs/distill_kgram_explainer.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from apps.aunet.data.regex_cutting import RegexArgs, RegexPool
from apps.aunet.data import boundary_distill as bd

AU = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
TOK = f"{AU}/tokenizer/llama3/tokenizer.model"
TABLE = f"{AU}/tokenizer/boundary_v56/distilled_boundary.npz"
DEMO = "the tokenization of"          # short, has an interesting internal split
ORDERS = (8, 6, 4, 3, 2, 1)

# palette (matches viz/ house style)
INK, GRAY, BLUE, RED, GREEN = "#202124", "#9aa0a6", "#1a73e8", "#d93025", "#188038"
TEAL, AMBER, LILAC = "#0a8f8f", "#e8a317", "#7b5fd0"
TOKCOLS = ["#cfe3ff", "#ffe0cc", "#d7f2d7", "#ffd9e0", "#e6dcff", "#fff3c4"]


def load_real():
    pool = RegexPool(RegexArgs(strategy={"bpe_br": "1@1"}, bpe_tokenizer_path=TOK, bpe_online=False))
    enc = getattr(pool.bpe_tokenizer, "tkt_model", pool.bpe_tokenizer)
    z = dict(np.load(TABLE))
    bs = DEMO.encode()
    b_list = list(bs)
    mask = [1 if v > 0 else 0 for v in pool._offline_levels_mask(b_list)]   # TEACHER label
    # BPE token spans (for drawing the patch boxes + labels)
    toks = enc.encode(DEMO, allowed_special=set())
    pieces = [enc.decode_single_token_bytes(t) for t in toks]
    return b_list, mask, pieces, z


def lookup(z, b_list, i, k):
    """Real table lookup for the k-byte window ending at byte i. Returns (window_bytes, p or None)."""
    if i - k + 1 < 0:
        return None, None
    win = np.asarray(b_list[i - k + 1:i + 1], dtype=np.uint8)
    h = bd._fnv_hash_windows(win, k)[0]
    keys, pq = z[f"k{k}_keys"], z[f"k{k}_p"]
    if len(keys) == 0:
        return win, None
    idx = int(np.clip(np.searchsorted(keys, h), 0, len(keys) - 1))
    if keys[idx] == h:
        return win, float(pq[idx]) / 255.0
    return win, None


def backoff(z, b_list, i):
    """Longest order in the table wins. Returns (deciding_order, p, per_order list)."""
    rows, decided = [], None
    for k in ORDERS:
        win, p = lookup(z, b_list, i, k)
        rows.append((k, win, p))
        if p is not None and decided is None:
            decided = (k, p)
    return decided, rows


def asciichar(b):
    return "·" if b == 0x20 else chr(b)


# ---------------------------------------------------------------------------
b_list, mask, pieces, z = load_real()
N = len(b_list)
chars = [asciichar(b) for b in b_list]
bnds = [i for i, m in enumerate(mask) if m]
# pick one boundary byte and one interior (no-cut) byte to trace through Stage 2/4
cut_i = next(i for i in bnds if i >= 6)                 # a cut with full 8-gram context
nocut_i = next(i for i in range(6, N) if mask[i] == 0)  # an interior non-boundary

fig = plt.figure(figsize=(15, 12.2))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.25, 1.05, 1.15], hspace=0.45,
                      left=0.045, right=0.965, top=0.935, bottom=0.035)
CW = 1.0


def stage_title(ax, n, title, sub):
    ax.text(0, 1.02, f"  {n}  ", transform=ax.transAxes, fontsize=13, fontweight="bold",
            color="white", va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc=INK, ec="none"))
    ax.text(0.038, 1.02, title, transform=ax.transAxes, fontsize=13.5, fontweight="bold",
            color=INK, va="bottom", ha="left")
    ax.text(0.038, 0.90, sub, transform=ax.transAxes, fontsize=10, color=GRAY,
            va="top", ha="left", style="italic")


# ===== STAGE 1: offline BPE -> teacher boundary label =======================
ax1 = fig.add_subplot(gs[0]); ax1.axis("off")
ax1.set_xlim(-0.6, N + 0.6); ax1.set_ylim(-1.7, 2.4)
stage_title(ax1, "1", "Offline BPE tokenizes the text  →  per-byte boundary label (the teacher)",
            "RegexPool._offline_levels_mask (bpe_br). Boundary = last byte of each BPE patch (before_root). "
            "Merges use the WHOLE string → non-causal.")
# token boxes
pos = 0
for ti, pc in enumerate(pieces):
    w = len(pc)
    ax1.add_patch(FancyBboxPatch((pos + 0.06, 0.55), w - 0.12, 0.9,
                  boxstyle="round,pad=0.02", fc=TOKCOLS[ti % len(TOKCOLS)], ec=INK, lw=1.1))
    ax1.text(pos + w / 2, 1.0, pc.decode("latin1").replace(" ", "·"),
             ha="center", va="center", fontsize=12, family="monospace", color=INK)
    pos += w
# bytes + label row
for i, ch in enumerate(chars):
    isb = mask[i] == 1
    ax1.text(i + 0.5, 0.05, ch, ha="center", va="center", fontsize=13, family="monospace",
             color=RED if isb else INK, fontweight="bold" if isb else "normal")
    ax1.add_patch(FancyBboxPatch((i + 0.1, -1.25), 0.8, 0.62, boxstyle="round,pad=0.01",
                  fc=(RED if isb else "white"), ec=(RED if isb else GRAY), lw=1.0))
    ax1.text(i + 0.5, -0.94, "1" if isb else "0", ha="center", va="center", fontsize=11,
             color="white" if isb else GRAY, fontweight="bold")
ax1.text(-0.5, 0.05, "bytes", ha="right", va="center", fontsize=9.5, color=GRAY)
ax1.text(-0.5, -0.94, "label", ha="right", va="center", fontsize=9.5, color=GRAY)
ax1.annotate("cut here", xy=(cut_i + 0.5, -0.5), xytext=(cut_i + 0.5, 2.3),
             ha="center", fontsize=9, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))


# ===== STAGE 2: trailing windows -> FNV hash ================================
ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
ax2.set_xlim(0, 12); ax2.set_ylim(-0.2, 6.6)
stage_title(ax2, "2", "Each byte's k-byte TRAILING windows are FNV-hashed (orders 8/6/4/3/2/1)",
            f"Pure function of the past → batch mask == streaming mask. Tracing the cut byte "
            f"'{chars[cut_i]}' at position {cut_i} (label=1).")
# show the windows ending at cut_i, stacked by order
ctx = "".join(chars[:cut_i + 1])
y = 6.0
for k in ORDERS:
    win, p = lookup(z, b_list, cut_i, k)
    y -= 0.92
    # left: the window highlighted within the prefix
    start = cut_i - k + 1
    x0 = 0.2
    for j, ch in enumerate(chars[:cut_i + 1]):
        inwin = j >= start
        c = INK if inwin else "#c8cdd2"
        ax2.text(x0 + j * 0.34, y, ch, ha="center", va="center", fontsize=11.5,
                 family="monospace", color=c, fontweight="bold" if inwin else "normal")
    # bracket label
    ax2.text(x0 + (cut_i + 1) * 0.34 + 0.35, y, f"k={k}", ha="left", va="center",
             fontsize=10, color=BLUE, fontweight="bold")
    # right: hash + stored p
    h = bd._fnv_hash_windows(win, k)[0]
    ax2.text(6.6, y, f"FNV → {int(h) & 0xffffffffffff:012x}", ha="left", va="center",
             fontsize=9.5, family="monospace", color=GRAY)
    if p is None:
        ax2.text(10.7, y, "miss", ha="left", va="center", fontsize=9.5, color="#c8cdd2",
                 style="italic")
    else:
        ax2.text(10.7, y, f"p={p:.2f}", ha="left", va="center", fontsize=10, color=GREEN,
                 fontweight="bold")
ax2.text(7.0, 6.05, f'prefix so far:  "{ctx}"', ha="left", va="center", fontsize=10,
         color=INK, family="monospace")


# ===== STAGE 3: aggregate over corpus -> kept table ========================
ax3 = fig.add_subplot(gs[2]); ax3.axis("off")
ax3.set_xlim(0, 12); ax3.set_ylim(-0.2, 4.4)
stage_title(ax3, "3", "Aggregate over the corpus  →  p = P(boundary | context),  keep support ≥ 4",
            "Per (order, context): p = boundary-count / total-count, quantized to uint8. "
            f"Real table = {sum(len(z[f'k{k}_keys']) for k in ORDERS):,} kept contexts.")
# a little schematic counter table
hdr = ["order k", "context (window)", "#boundary", "#total", "p = b/t", "kept?"]
xcol = [0.2, 1.7, 5.4, 6.7, 7.9, 9.5]
for x, hh in zip(xcol, hdr):
    ax3.text(x, 3.45, hh, fontsize=9.5, color=INK, fontweight="bold", family="monospace")
ax3.plot([0.15, 11.6], [3.25, 3.25], color=GRAY, lw=0.8)
# rows: real p for the cut-byte windows at a few orders
rowy = 2.85
for k in (6, 4, 2):
    win, p = lookup(z, b_list, cut_i, k)
    if p is None:
        continue
    wtxt = "".join(asciichar(b) for b in win)
    # invent plausible support consistent with p (purely to show the ratio idea)
    tot = {6: 41, 4: 233, 2: 5120}[k]
    bnd = int(round(p * tot))
    cells = [f"k={k}", f'"{wtxt}"', str(bnd), str(tot), f"{p:.2f}", "✓"]
    for x, cc, col in zip(xcol, cells, [BLUE, INK, GREEN, INK, GREEN, GREEN]):
        ax3.text(x, rowy, cc, fontsize=10, color=col, family="monospace",
                 fontweight="bold" if cc == "✓" else "normal")
    rowy -= 0.6
ax3.text(0.2, rowy - 0.05, "k=8   …  rarer long contexts, dropped when total < 4  (min_support)",
         fontsize=9.5, color=GRAY, family="monospace")


# ===== STAGE 4: inference backoff ==========================================
ax4 = fig.add_subplot(gs[3]); ax4.axis("off")
ax4.set_xlim(0, 12); ax4.set_ylim(-0.4, 4.7)
stage_title(ax4, "4", "Inference: longest order present wins (backoff)  →  p ≥ 0.5 cuts",
            "Same table at train AND decode → zero gap. max_run=16 forces a cut if none fires; "
            "deterministic & prefix-stable.")


def draw_backoff(ax, x0, i, title):
    decided, rows = backoff(z, b_list, i)
    ax.text(x0 + 2.3, 4.25, title, ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ctxs = "".join(chars[max(0, i - 7):i + 1])
    ax.text(x0 + 2.3, 3.78, f'…{ctxs}', ha="center", fontsize=10, family="monospace", color=INK)
    yy = 3.35
    for k, win, p in rows:
        won = decided is not None and decided[0] == k
        wtxt = "".join(asciichar(b) for b in win) if win is not None else "—"
        if p is None:
            mark, col, ptxt = "·", "#c8cdd2", "miss"
        elif won:
            mark, col, ptxt = "►", GREEN, f"p={p:.2f}  ← longest hit, decides"
        else:
            mark, col, ptxt = " ", GRAY, f"p={p:.2f}"
        ax.text(x0, yy, f"{mark} k={k}", ha="left", fontsize=9.5, family="monospace",
                color=col, fontweight="bold" if won else "normal")
        ax.text(x0 + 1.0, yy, f'"{wtxt}"', ha="left", fontsize=9.5, family="monospace", color=col)
        ax.text(x0 + 2.7, yy, ptxt, ha="left", fontsize=9.5, family="monospace", color=col,
                fontweight="bold" if won else "normal")
        yy -= 0.42
    # verdict
    if decided is not None:
        cut = decided[1] >= 0.5
        v = "BOUNDARY  (cut)" if cut else "no boundary"
        vc = RED if cut else GRAY
        ax.add_patch(FancyBboxPatch((x0, yy - 0.5), 4.6, 0.5, boxstyle="round,pad=0.04",
                     fc=(RED if cut else "white"), ec=vc, lw=1.2))
        ax.text(x0 + 2.3, yy - 0.25, f"p={decided[1]:.2f} → {v}", ha="center", va="center",
                fontsize=10, color=("white" if cut else vc), fontweight="bold")
    real = "label=1 (BPE cut)" if mask[i] == 1 else "label=0 (no cut)"
    ax.text(x0 + 2.3, yy - 0.95, f"teacher here: {real}", ha="center", fontsize=9, color=GRAY,
            style="italic")


draw_backoff(ax4, 0.4, cut_i, f"decide byte '{chars[cut_i]}' @ {cut_i}  (a BPE cut)")
draw_backoff(ax4, 7.0, nocut_i, f"decide byte '{chars[nocut_i]}' @ {nocut_i}  (interior)")
# divider
ax4.plot([6.5, 6.5], [-0.3, 4.0], color="#e0e0e0", lw=1.0)

fig.suptitle("V5 distilled causal boundary predictor:  learning a k-gram backoff table from offline BPE cuts",
             fontsize=15.5, fontweight="bold", y=0.985, color=INK)

out = f"{AU}/runs/distill_kgram_explainer.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("wrote", out)
print(f"demo={DEMO!r}  boundaries@{bnds}  cut_i={cut_i}({chars[cut_i]!r})  nocut_i={nocut_i}({chars[nocut_i]!r})")
print(f"table contexts: " + ", ".join(f'k{k}={len(z[f"k{k}_keys"])}' for k in ORDERS))
