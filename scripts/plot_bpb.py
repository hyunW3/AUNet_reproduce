#!/usr/bin/env python
"""BPB (bits-per-byte) training-loss curves for the matched 760M runs.

BPB is tokenizer-agnostic (same DCLM data), so byte and subword models share a y-axis:
  byte models (`bytes` tokenizer, 1 token == 1 byte):  BPB = loss/out / ln2
  Llama (llama3 tiktoken subword):                     BPB = loss/out / (ln2 * bytes_per_token)
X-axis = training BYTES seen (the only fair shared unit): byte runs log bytes directly in
optim/total_tokens; Llama logs subword tokens -> x bytes_per_token.

Outputs (top-level, committable; runs/ is gitignored):
  bpb_compare.png         — redraw: BPEByte(new) vs AU-Net-word vs Llama
  bpebyte_newold_bpb.png  — new-code vs old-code (4ed607e) BPEByte, the down-projection/eval-fix impact
"""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent   # repo root (script lives in scripts/)
RUNS = REPO / "runs"
LN2 = math.log(2.0)
BPT = 4.5376  # measured bytes/token, llama3 tiktoken on DCLM (matches model_results_760M.md)

RUNDEFS = {
    "bpebyte_new": ("bpebyte_root_greedy_760M",                 1.0,  "#2f6db5", "BPEByte root_greedy (new code)"),
    "bpebyte_old": ("bpebyte_root_greedy_760M_oldcode_4ed607e", 1.0,  "#b23b3b", "BPEByte root_greedy (old code, 4ed607e)"),
    "aunet_word":  ("aunet2_760M",                              1.0,  "#1d7a44", "AU-Net word"),
    "llama":       ("llama_760M",                               BPT,  "#b07a18", "Llama (subword)"),
}

def load(run, bpt):
    """Return (x_bytes_GB, bpb) deduped by step, light EMA-smoothed."""
    by_step = {}
    with open(RUNS / run / "metrics.jsonl") as f:
        for line in f:
            try: d = json.loads(line)
            except json.JSONDecodeError: continue
            s, loss, tok = d.get("global_step"), d.get("loss/out"), d.get("optim/total_tokens")
            if s is None or loss is None or tok is None: continue
            by_step[s] = (tok * bpt, loss / (LN2 * bpt))   # bytes seen, BPB
    xs = [by_step[s][0] / 1e9 for s in sorted(by_step)]    # -> GB
    ys = [by_step[s][1] for s in sorted(by_step)]
    # centered rolling-mean smoothing (window ~ steps) so near-identical curves separate
    w = max(5, min(301, len(ys) // 40) | 1)                # odd window, ~2.5% of run
    half = w // 2
    sm = []
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        sm.append(sum(ys[lo:hi]) / (hi - lo))
    return xs, sm, ys[-1]                                   # final = canonical last-step BPB (matches results doc)

data = {k: load(*v[:2]) for k, v in RUNDEFS.items()}
fin = {k: data[k][2] for k in data}

# ---- Plot 1: redraw — byte vs subword (BPEByte new / AU-Net word / Llama) ----
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for k in ("bpebyte_new", "aunet_word", "llama"):
    xs, ys, f = data[k]; c, lbl = RUNDEFS[k][2], RUNDEFS[k][3]
    ax.plot(xs, ys, color=c, lw=1.8, label=f"{lbl}  (BPB {f:.4f})")
ax.set_xlabel("training bytes seen (GB)"); ax.set_ylabel("BPB (bits / byte)")
ax.set_title("760M matched-compute — BPB vs bytes seen (lower = better)")
ax.set_ylim(0.90, 1.05); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(REPO / "reports/bpb_compare.png", dpi=140); plt.close(fig)

# ---- Plot 2: new vs old code BPEByte (the change of interest) ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2, 1]})
for k in ("bpebyte_new", "bpebyte_old"):
    xs, ys, f = data[k]; c, lbl = RUNDEFS[k][2], RUNDEFS[k][3]
    axL.plot(xs, ys, color=c, lw=1.8, label=f"{lbl}  (final {f:.4f})")
axL.set_xlabel("training bytes seen (GB)"); axL.set_ylabel("BPB (bits / byte)")
axL.set_title("BPEByte 760M — new vs old code, BPB vs bytes seen")
axL.set_ylim(0.915, 1.02); axL.grid(alpha=0.25); axL.legend(frameon=False, fontsize=9)
# zoom on the final tail
for k in ("bpebyte_new", "bpebyte_old"):
    xs, ys, f = data[k]; c = RUNDEFS[k][2]
    n = int(len(xs) * 0.8)
    axR.plot(xs[n:], ys[n:], color=c, lw=1.8)
d = fin["bpebyte_new"] - fin["bpebyte_old"]
axR.set_title(f"final tail (zoom)\nΔBPB = new − old = {d:+.4f}")
axR.set_xlabel("bytes seen (GB)"); axR.grid(alpha=0.25)
axR.axhline(fin["bpebyte_new"], color="#2f6db5", ls="--", lw=0.9)
axR.axhline(fin["bpebyte_old"], color="#b23b3b", ls="--", lw=0.9)
fig.suptitle(f"Down-projection / eval-fix impact: new code reaches lower BPB ({fin['bpebyte_new']:.4f} < {fin['bpebyte_old']:.4f})",
             fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig(REPO / "reports/bpebyte_newold_bpb.png", dpi=140, bbox_inches="tight"); plt.close(fig)

print("final BPB:", {k: round(v, 4) for k, v in fin.items()})
print("new-old ΔBPB:", round(d, 4))
print("wrote reports/bpb_compare.png, reports/bpebyte_newold_bpb.png")
