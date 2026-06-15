"""Animated sweep: the V5 distilled predictor deciding each byte left-to-right -> GIF/MP4.

One frame per byte position. At each step the selector looks ONLY at the trailing window
(<=8 bytes), backs off longest-order-first through the REAL distilled_boundary.npz table,
and commits cut / no-cut (p>=0.5, with max_run=16). Bars accumulate as the causal selector
runs; a tick row shows where its cut matches the offline-BPE teacher. Everything is the real
table + real offline mask (see distill_kgram_explainer.py for the static 4-stage version).

Run in the training venv (CPU only):
    cd ../lingua && CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        .venv/bin/python ../viz/distill_kgram_sweep.py
Writes ../viz/distill_kgram_sweep.gif (+ .mp4 if ffmpeg), and key frame PNGs to /var/tmp.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter

from apps.aunet.data.regex_cutting import RegexArgs, RegexPool
from apps.aunet.data import boundary_distill as bd

AU = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
TOK = f"{AU}/tokenizer/llama3/tokenizer.model"
TABLE = f"{AU}/tokenizer/boundary_v56/distilled_boundary.npz"
DEMO = "the tokenization"
ORDERS = (8, 6, 4, 3, 2, 1)
MAX_RUN = 16

INK, GRAY, BLUE, RED, GREEN, AMBER = "#202124", "#9aa0a6", "#1a73e8", "#d93025", "#188038", "#e8a317"
CW, CH, Y = 1.0, 1.2, 0.0


def load():
    pool = RegexPool(RegexArgs(strategy={"bpe_br": "1@1"}, bpe_tokenizer_path=TOK, bpe_online=False))
    z = dict(np.load(TABLE))
    b_list = list(DEMO.encode())
    teacher = [1 if v > 0 else 0 for v in pool._offline_levels_mask(b_list)]
    return b_list, teacher, z


def asciichar(b):
    return "·" if b == 0x20 else chr(b)


def lookup(z, b_list, i, k):
    if i - k + 1 < 0:
        return None, None
    win = np.asarray(b_list[i - k + 1:i + 1], dtype=np.uint8)
    h = bd._fnv_hash_windows(win, k)[0]
    keys, pq = z[f"k{k}_keys"], z[f"k{k}_p"]
    if len(keys) == 0:
        return win, None
    idx = int(np.clip(np.searchsorted(keys, h), 0, len(keys) - 1))
    return win, (float(pq[idx]) / 255.0 if keys[idx] == h else None)


def backoff(z, b_list, i, prior):
    rows, decided = [], None
    for k in ORDERS:
        win, p = lookup(z, b_list, i, k)
        rows.append((k, win, p))
        if p is not None and decided is None:
            decided = (k, p)
    return (decided if decided is not None else (0, prior)), rows


b_list, teacher, z = load()
N = len(b_list)
chars = [asciichar(b) for b in b_list]
prior = float(z["prior"])

# Precompute the causal run: per byte the backoff rows, decided (order,p), and the committed
# selector cut (p>=0.5 OR max_run). This is exactly DistilledBoundaryPredictor.mask01's logic.
STEPS, run = [], 0
for i in range(N):
    (dk, dp), rows = backoff(z, b_list, i, prior)
    run += 1
    raw = dp >= 0.5
    cut = bool(raw or run >= MAX_RUN)
    forced = cut and not raw
    if cut:
        run = 0
    STEPS.append(dict(rows=rows, dk=dk, dp=dp, cut=cut, forced=forced))

sel_cuts = [s["cut"] for s in STEPS]

fig, ax = plt.subplots(figsize=(13.5, 7.6))
fig.patch.set_facecolor("white")
X0 = max(0.0, (16 - N) / 2)          # center short strings


def draw(i):
    ax.clear()
    ax.set_xlim(-1.4, max(N, 16) + 0.4)
    ax.set_ylim(-7.4, 2.7)
    ax.axis("off")
    s = STEPS[i]

    ax.text(-1.2, 2.35, "V5 distilled selector — causal sweep", fontsize=14, fontweight="bold",
            color=INK)
    ax.text(max(N, 16) + 0.3, 2.35, f'"{DEMO}"', fontsize=12, ha="right", family="monospace",
            color=GRAY)

    # ---- byte cells -------------------------------------------------------
    for j, ch in enumerate(chars):
        x = X0 + j
        if j < i:                                    # already decided
            fc, ec, tc, fw = "#f5f6f7", GRAY, INK, "normal"
        elif j == i:                                 # current decision byte
            fc, ec, tc, fw = "#fff3d6", AMBER, INK, "bold"
        else:                                        # future (unseen by a causal selector)
            fc, ec, tc, fw = "white", "#dfe1e3", "#c8cdd2", "normal"
        ax.add_patch(FancyBboxPatch((x + 0.06, Y + 0.06), CW - 0.12, CH - 0.12,
                     boxstyle="round,pad=0.02,rounding_size=0.08", lw=2.2 if j == i else 1.4,
                     edgecolor=ec, facecolor=fc))
        ax.text(x + 0.5, Y + CH / 2, ch, ha="center", va="center", fontsize=19,
                family="monospace", color=tc, fontweight=fw)
        # committed selector cut = bar on the cell's right edge
        if j < i and sel_cuts[j]:
            xb = X0 + j + 1
            ax.plot([xb, xb], [Y - 0.06, Y + CH + 0.06], color=RED, lw=5, solid_capstyle="round")
        # agreement tick vs teacher, under decided bytes
        if j < i:
            ok = (sel_cuts[j] == (teacher[j] == 1))
            ax.text(x + 0.5, Y - 0.42, "✓" if ok else "✗", ha="center", va="center",
                    fontsize=10, color=GREEN if ok else RED)
    ax.text(X0 - 0.55, Y - 0.42, "vs BPE", ha="right", va="center", fontsize=8.5, color=GRAY)

    # trailing-window brace under the current byte (looks back <=8)
    if i < N:
        a = X0 + max(0, i - 7)
        b = X0 + i + 1
        yb = Y - 0.95
        ax.plot([a, b], [yb, yb], color=BLUE, lw=1.6)
        ax.plot([a, a], [yb, yb + 0.12], color=BLUE, lw=1.6)
        ax.plot([b, b], [yb, yb + 0.12], color=BLUE, lw=1.6)
        ax.text((a + b) / 2, yb - 0.22, "trailing window (≤8 bytes) — the only input",
                ha="center", va="top", fontsize=9, color=BLUE)

    # ---- backoff ladder for the current byte ------------------------------
    lx = -1.2
    ax.text(lx, -1.95, f"decide byte  '{chars[i]}'  @ {i}   →   backoff (longest order present wins)",
            fontsize=11.5, fontweight="bold", color=INK)
    yy = -2.55
    for k, win, p in s["rows"]:
        won = (s["dk"] == k)
        wtxt = "".join(asciichar(c) for c in win) if win is not None else "(need %d bytes)" % k
        if win is None:
            mark, col = "  ", "#c8cdd2"
            ptxt = "—"
        elif p is None:
            mark, col = "  ", "#c8cdd2"
            ptxt = "miss"
        elif won:
            mark, col = "► ", GREEN
            ptxt = f"p={p:.2f}   ← decides"
        else:
            mark, col = "  ", GRAY
            ptxt = f"p={p:.2f}"
        ax.text(lx, yy, f"{mark}k={k}", fontsize=10.5, family="monospace", color=col,
                fontweight="bold" if won else "normal")
        ax.text(lx + 1.35, yy, f'"{wtxt}"', fontsize=10.5, family="monospace", color=col,
                fontweight="bold" if won else "normal")
        ax.text(lx + 5.3, yy, ptxt, fontsize=10.5, family="monospace", color=col,
                fontweight="bold" if won else "normal")
        yy -= 0.52

    # backoff verdict box
    if s["dk"] == 0:
        src = "no k-gram hit → prior"
    else:
        src = f"k={s['dk']} context"
    cut, forced, dp = s["cut"], s["forced"], s["dp"]
    if cut and not forced:
        vt, vc = f"p={dp:.2f} ≥ 0.5  →  CUT (patch boundary)", RED
    elif forced:
        vt, vc = f"p={dp:.2f} < 0.5  but run hit max_run=16  →  forced CUT", AMBER
    else:
        vt, vc = f"p={dp:.2f} < 0.5  →  no cut", GRAY
    ax.add_patch(FancyBboxPatch((lx, yy - 0.62), 8.7, 0.56, boxstyle="round,pad=0.04",
                 fc=(RED if (cut and not forced) else ("#fff3d6" if forced else "white")),
                 ec=vc, lw=1.3))
    ax.text(lx + 0.18, yy - 0.34, vt, va="center", fontsize=10.5, family="monospace",
            color=("white" if (cut and not forced) else INK), fontweight="bold")
    ax.text(lx, yy - 1.15, f"source: {src}     (same rule used at training AND decode → zero gap)",
            fontsize=9, color=GRAY, style="italic")

    # ---- running tally (right) -------------------------------------------
    rx = 9.6
    decided_n = i
    cuts_so_far = sum(sel_cuts[:i])
    agree = sum(1 for j in range(i) if sel_cuts[j] == (teacher[j] == 1))
    ax.text(rx, -2.55, "running (causal, left→right)", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(rx, -3.15, f"bytes decided : {decided_n}/{N}", fontsize=10.5, family="monospace", color=INK)
    ax.text(rx, -3.7, f"patches closed: {cuts_so_far}", fontsize=10.5, family="monospace", color=RED)
    acc = (agree / decided_n) if decided_n else 1.0
    ax.text(rx, -4.25, f"agree w/ BPE  : {agree}/{decided_n}" + (f"  ({acc:.0%})" if decided_n else ""),
            fontsize=10.5, family="monospace", color=GREEN)
    ax.text(rx, -5.15, "bar = committed cut", fontsize=9, color=RED)
    ax.text(rx, -5.6, "✓/✗ = match vs offline BPE", fontsize=9, color=GRAY)
    ax.text(rx, -6.05, "nothing is ever revised", fontsize=9, color=GRAY, style="italic")

    return ax.patches


# hold each cut frame a touch longer, plus a final settle frame
order = []
for i in range(N):
    order.append(i)
    if STEPS[i]["cut"]:
        order.append(i)
order += [N - 1, N - 1]

anim = FuncAnimation(fig, lambda f: draw(order[f]), frames=len(order), interval=750, blit=False)
out_gif = f"{AU}/viz/distill_kgram_sweep.gif"
anim.save(out_gif, writer=PillowWriter(fps=1.4))
print("wrote", out_gif)
try:
    anim.save(out_gif.replace(".gif", ".mp4"), writer="ffmpeg", fps=1.4, dpi=120)
    print("wrote", out_gif.replace(".gif", ".mp4"))
except Exception as e:
    print("mp4 skipped:", e)

for fi in (3, 8, N - 1):
    draw(fi)
    fig.savefig(f"/var/tmp/sweep_frame_{fi}.png", dpi=110, bbox_inches="tight")
    print(f"wrote /var/tmp/sweep_frame_{fi}.png")
print(f"demo={DEMO!r}  teacher_cuts={[j for j in range(N) if teacher[j]]}  "
      f"selector_cuts={[j for j in range(N) if sel_cuts[j]]}")
