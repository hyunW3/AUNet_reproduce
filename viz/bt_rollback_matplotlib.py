"""Animated before_root:bt rollback trace -> GIF/MP4 (matplotlib fallback for manim).

Frames are the REAL captured trace of prompt="hello", generate=" world" through
apps/aunet/generate_bt.online_bt_loop with the llama3 BPE parser (rb=1). See the
manim version in bt_rollback_manim.py. Run in the training venv:

    cd ../lingua && CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        .venv/bin/python ../viz/bt_rollback_matplotlib.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter

PROMPT_LEN = 5  # "hello"

# (string_so_far, bars_after_idx, drop_idxs, caption) — straight from the real trace.
FRAMES = [
    ("hello",       [],     [],                "prompt 'hello' fed — tail boundary held speculative (commit_margin=2)"),
    ("hello ",      [],     [],                "feed '·'  -> ACCEPT"),
    ("hello w",     [],     [],                "feed 'w'  -> ACCEPT"),
    ("hello wo",    [],     [],                "feed 'o'  -> ACCEPT"),
    ("hello wor",   [],     [],                "feed 'r'  -> ACCEPT   (no boundary committed yet)"),
    ("hello worl",  [4],    [5,6,7,8,9],       "feed 'l': 'hello|' boundary commits (change before frontier) -> '·worl' was off-policy"),
    ("hello",       [4],    [],                "ROLLBACK (rb=1): drop '·worl', restore parser, re-sample from 'hello|'"),
    ("hello ",      [4],    [],                "feed '·'  -> ACCEPT"),
    ("hello w",     [4],    [],                "feed 'w'  -> ACCEPT"),
    ("hello wo",    [4],    [],                "feed 'o'  -> ACCEPT"),
    ("hello wor",   [4],    [],                "feed 'r'  -> ACCEPT"),
    ("hello worl",  [4],    [],                "feed 'l'  -> ACCEPT   (stable now, no rollback)"),
    ("hello world", [4],    [],                "feed 'd'  -> ACCEPT"),
    ("hello world", [4,10], [],                "finalize(): commit trailing token  ==  offline get_levels_mask  OK"),
]

CW, CH, Y = 1.0, 1.2, 0.0          # cell width/height, baseline
GRAY, BLUE, RED = "#9aa0a6", "#1a73e8", "#d93025"
BAR = "#d93025"

fig, ax = plt.subplots(figsize=(12, 3.4))
fig.patch.set_facecolor("white")


def draw(i):
    ax.clear()
    ax.set_xlim(-0.6, 13.2)
    ax.set_ylim(-2.3, 2.2)
    ax.axis("off")
    s, bars, drop, cap = FRAMES[i]
    rb = sum(1 for j in range(i + 1) if FRAMES[j][3].startswith("ROLLBACK"))
    ax.text(-0.4, 1.9, "before_root:bt   'hello' + ' world'",
            fontsize=13, fontweight="bold", family="monospace")
    ax.text(13.1, 1.9, f"rollbacks: {rb}", fontsize=11, ha="right", color=RED, family="monospace")
    for k, ch in enumerate(s):
        is_gen = k >= PROMPT_LEN
        if k in drop:
            fc, ec, tc = "#fde7e6", RED, RED
        elif is_gen:
            fc, ec, tc = "#e8f0fe", BLUE, BLUE
        else:
            fc, ec, tc = "#f1f3f4", GRAY, "#202124"
        x = k * CW
        ax.add_patch(FancyBboxPatch((x + 0.06, Y + 0.06), CW - 0.12, CH - 0.12,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=2, edgecolor=ec, facecolor=fc))
        glyph = "·" if ch == " " else ch
        ax.text(x + CW / 2, Y + CH / 2, glyph, ha="center", va="center",
                fontsize=20, family="monospace", color=tc,
                fontweight="bold" if is_gen else "normal")
    # patch-boundary bars (thick red line on the right edge of the boundary cell)
    for b in bars:
        x = (b + 1) * CW
        ax.plot([x, x], [Y - 0.12, Y + CH + 0.12], color=BAR, linewidth=5, solid_capstyle="round")
        ax.text(x, Y + CH + 0.28, "patch end", ha="center", fontsize=8, color=BAR)
    # legend strip
    ax.add_patch(Rectangle((0, -1.15), 0.32, 0.32, facecolor="#f1f3f4", edgecolor=GRAY))
    ax.text(0.42, -1.0, "prompt", fontsize=9, va="center", color="#202124")
    ax.add_patch(Rectangle((2.0, -1.15), 0.32, 0.32, facecolor="#e8f0fe", edgecolor=BLUE))
    ax.text(2.42, -1.0, "generated", fontsize=9, va="center", color=BLUE)
    ax.add_patch(Rectangle((4.4, -1.15), 0.32, 0.32, facecolor="#fde7e6", edgecolor=RED))
    ax.text(4.82, -1.0, "rolled back", fontsize=9, va="center", color=RED)
    ax.plot([6.9, 6.9], [-1.18, -0.83], color=BAR, linewidth=5, solid_capstyle="round")
    ax.text(7.1, -1.0, "patch boundary", fontsize=9, va="center", color=BAR)
    # caption
    is_roll = cap.startswith("ROLLBACK") or "off-policy" in cap
    ax.text(-0.4, -1.85, cap, fontsize=12, family="monospace",
            color=RED if is_roll else "#202124",
            fontweight="bold" if is_roll else "normal")
    return ax.patches


# hold the rollback (5,6) and final (13) frames longer by repeating them
order = []
for i in range(len(FRAMES)):
    order.append(i)
    if i in (5, 6, 13):
        order += [i, i]

anim = FuncAnimation(fig, lambda f: draw(order[f]), frames=len(order), interval=850, blit=False)
out_gif = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/viz/bt_rollback.gif"
anim.save(out_gif, writer=PillowWriter(fps=1.25))
print("wrote", out_gif)
try:
    out_mp4 = out_gif.replace(".gif", ".mp4")
    anim.save(out_mp4, writer="ffmpeg", fps=1.25, dpi=120)
    print("wrote", out_mp4)
except Exception as e:
    print("mp4 skipped:", e)

# verification PNGs of the key rollback frames
for fi in (5, 6, 13):
    draw(fi)
    fig.savefig(f"/var/tmp/bt_frame_{fi}.png", dpi=110, bbox_inches="tight")
    print("wrote /var/tmp/bt_frame_%d.png" % fi)
