#!/usr/bin/env python
"""Bits-Per-Byte (BPB) loss-curve comparison: AU-Net (byte model) vs Llama (subword transformer).

Both runs train on the same data (DCLM dclm_baseline_1.0_2shards_shuffled), so BPB is a
tokenizer-agnostic, apples-to-apples loss. The training loss `loss/out` is mean cross-entropy
in *nats per token*:

  - AU-Net uses a `bytes` tokenizer -> 1 token == 1 byte, so   BPB = loss/out / ln(2)
  - Llama uses the llama3 tiktoken (subword) tokenizer, so     BPB = loss/out / (ln(2) * bytes_per_token)

`bytes_per_token` is measured empirically by tokenizing a 5000-doc DCLM sample with the same
llama3 tokenizer (total UTF-8 bytes / total tokens). bos/eos add <0.2% tokens over these long
docs and are ignored. Re-measure with measure_bpt() if the tokenizer or corpus changes.

X-axis is *training bytes seen* (the only fair shared unit): AU-Net's optim/total_tokens is
already bytes; Llama's is subword tokens x bytes_per_token.
"""
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs")
AUNET = RUNS / "aunet2_1.3B" / "metrics.jsonl"
LLAMA = RUNS / "llama_1B_dm10" / "metrics.jsonl"
AUNET_VAL = RUNS / "aunet2_1.3B" / "eval" / "validation.json"
LLAMA_VAL = RUNS / "llama_1B_dm10" / "eval" / "validation.json"
OUT = RUNS / "bpb_compare.png"
OUT_FLOPS = RUNS / "bpb_vs_flops.png"

LN2 = math.log(2.0)
BYTES_PER_TOKEN_LLAMA = 4.5483  # measured on 5000 DCLM docs with llama3 tiktoken (see measure_bpt)


def load(path, bytes_per_token):
    """Return dict of arrays deduped by global_step (last write wins).

    Keys: gb (bytes seen, billions), bpb (training loss as bits/byte),
    zflops (cumulative compute, zettaflops = 1e21 FLOPs).

    Cumulative FLOPs = flops_per_token * total_tokens, where flops_per_token is
    derived per-run as median(speed/FLOPS / speed/wps) (both are logged; FLOPS is
    throughput = wps * flops_per_token, so the ratio recovers the constant).
    """
    by_step = {}
    fpt_ratios = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            F, w = d.get("speed/FLOPS"), d.get("speed/wps")
            if F and w:
                fpt_ratios.append(F / w)
            loss = d.get("loss/out")
            tok = d.get("optim/total_tokens")
            step = d.get("global_step")
            if loss is None or tok is None or step is None:
                continue
            bpb = loss / (LN2 * bytes_per_token)
            bytes_seen = tok * bytes_per_token
            by_step[step] = (bytes_seen, bpb, tok)
    flops_per_token = statistics.median(fpt_ratios)
    steps = sorted(by_step)
    return {
        "gb": [by_step[s][0] / 1e9 for s in steps],
        "bpb": [by_step[s][1] for s in steps],
        "zflops": [by_step[s][2] * flops_per_token / 1e21 for s in steps],
        "flops_per_token": flops_per_token,
    }


def heldout_bpb(val_json):
    """Held-out BPB = -nll_per_byte / ln2 (nll_per_byte is mean log-prob per UTF-8 byte)."""
    d = json.load(open(val_json))
    src = next(iter(d.values()))
    return -src["nll_per_byte"] / LN2


def ema(xs, alpha=0.01):
    out, m = [], xs[0]
    for x in xs:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def main():
    # AU-Net: byte tokenizer, so bytes_per_token = 1 (loss/out is already nats/byte).
    a = load(AUNET, bytes_per_token=1.0)
    l = load(LLAMA, bytes_per_token=BYTES_PER_TOKEN_LLAMA)
    a["ema"] = ema(a["bpb"], 0.01)
    l["ema"] = ema(l["bpb"], 0.01)

    # Held-out BPB on the shared dclm *.val.jsonl set, identical UTF-8 byte accounting.
    a_ho = heldout_bpb(AUNET_VAL)
    l_ho = heldout_bpb(LLAMA_VAL)

    # ---- Figure 1: BPB vs training bytes seen ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    ax1.plot(a["gb"], a["bpb"], color="#a0c4ff", lw=0.3, alpha=0.4)
    ax1.plot(a["gb"], a["ema"], color="#1f4ed8", lw=1.8, label=f"AU-Net 2 1.3B (byte)  — train BPB {a['ema'][-1]:.3f}")
    ax1.plot(l["gb"], l["bpb"], color="#ffadad", lw=0.3, alpha=0.4)
    ax1.plot(l["gb"], l["ema"], color="#c81e1e", lw=1.8, label=f"Llama 1B (subword)  — train BPB {l['ema'][-1]:.3f}")
    ax1.set_xlabel("training data seen (GB, bytes)")
    ax1.set_ylabel("Bits Per Byte (BPB)")
    ax1.set_ylim(0.7, 2.2)
    ax1.set_title("BPB loss curve — AU-Net (byte) vs Llama transformer (subword), same DCLM data")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)

    xmax = min(a["gb"][-1], l["gb"][-1])
    ax2.plot(a["gb"], a["ema"], color="#1f4ed8", lw=1.8, label="AU-Net 2 1.3B (byte)")
    ax2.plot(l["gb"], l["ema"], color="#c81e1e", lw=1.8, label="Llama 1B (subword)")
    ax2.set_xlim(0, xmax)
    ax2.set_xlabel(f"training data seen (GB, bytes) — overlap region 0..{xmax:.1f} GB")
    ax2.set_ylabel("BPB (EMA α=0.01)")
    ax2.set_ylim(0.7, 1.4)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120)

    # ---- Figure 2: BPB vs cumulative compute (FLOPs), with held-out BPB markers ----
    fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(11, 9))
    bx1.plot(a["zflops"], a["bpb"], color="#a0c4ff", lw=0.3, alpha=0.4)
    bx1.plot(a["zflops"], a["ema"], color="#1f4ed8", lw=1.8, label="AU-Net 2 1.3B (byte) — train BPB")
    bx1.plot(l["zflops"], l["bpb"], color="#ffadad", lw=0.3, alpha=0.4)
    bx1.plot(l["zflops"], l["ema"], color="#c81e1e", lw=1.8, label="Llama 1B (subword) — train BPB")
    # held-out BPB as stars at each run's final compute
    bx1.scatter([a["zflops"][-1]], [a_ho], color="#1f4ed8", marker="*", s=240, zorder=5,
                edgecolor="white", label=f"AU-Net held-out BPB {a_ho:.3f}")
    bx1.scatter([l["zflops"][-1]], [l_ho], color="#c81e1e", marker="*", s=240, zorder=5,
                edgecolor="white", label=f"Llama held-out BPB {l_ho:.3f}")
    bx1.set_xlabel("cumulative training compute (ZFLOPs = 1e21 FLOPs)")
    bx1.set_ylabel("Bits Per Byte (BPB)")
    bx1.set_ylim(0.7, 2.2)
    bx1.set_title("BPB vs compute — AU-Net (byte) vs Llama transformer (subword); ★ = held-out BPB")
    bx1.grid(True, alpha=0.3)
    bx1.legend(loc="upper right", fontsize=9)

    fxmax = min(a["zflops"][-1], l["zflops"][-1])
    bx2.plot(a["zflops"], a["ema"], color="#1f4ed8", lw=1.8, label="AU-Net 2 1.3B (byte)")
    bx2.plot(l["zflops"], l["ema"], color="#c81e1e", lw=1.8, label="Llama 1B (subword)")
    bx2.set_xlim(0, fxmax)
    bx2.set_xlabel(f"cumulative compute (ZFLOPs) — overlap region 0..{fxmax:.2f}")
    bx2.set_ylabel("BPB (EMA α=0.01)")
    bx2.set_ylim(0.7, 1.4)
    bx2.grid(True, alpha=0.3)
    bx2.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_FLOPS, dpi=120)

    print(f"saved: {OUT}")
    print(f"saved: {OUT_FLOPS}")
    print(f"AU-Net : bytes={a['gb'][-1]:6.1f} GB  compute={a['zflops'][-1]:.3f} ZFLOP  "
          f"train BPB(EMA)={a['ema'][-1]:.4f}  HELD-OUT BPB={a_ho:.4f}  (fpt={a['flops_per_token']:.3e})")
    print(f"Llama  : bytes={l['gb'][-1]:6.1f} GB  compute={l['zflops'][-1]:.3f} ZFLOP  "
          f"train BPB(EMA)={l['ema'][-1]:.4f}  HELD-OUT BPB={l_ho:.4f}  (fpt={l['flops_per_token']:.3e})")


def measure_bpt(n_docs=5000):
    """Re-measure llama bytes-per-token on a DCLM sample. Run from the lingua venv."""
    import itertools
    from lingua.tokenizer import build_tokenizer
    tok = build_tokenizer("tiktoken", "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/tokenizer/llama3/tokenizer.model")
    src = RUNS.parent / "data/dclm_baseline_1.0_2shards_shuffled/dclm_baseline_1.0_2shards.chunk.00.jsonl"
    tb = tt = 0
    with open(src) as f:
        for line in itertools.islice(f, n_docs):
            d = json.loads(line)
            text = d.get("text") or d.get("content")
            if not text:
                continue
            tb += len(text.encode("utf-8"))
            tt += len(tok.encode(text, add_bos=False, add_eos=False))
    print(f"bytes_per_token = {tb / tt:.4f}  (docs={n_docs})")
    return tb / tt


if __name__ == "__main__":
    main()
