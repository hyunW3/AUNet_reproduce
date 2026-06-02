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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs")
AUNET = RUNS / "aunet2_1.3B" / "metrics.jsonl"
LLAMA = RUNS / "llama_1B_dm10" / "metrics.jsonl"
OUT = RUNS / "bpb_compare.png"

LN2 = math.log(2.0)
BYTES_PER_TOKEN_LLAMA = 4.5483  # measured on 5000 DCLM docs with llama3 tiktoken (see measure_bpt)


def load(path, bytes_per_token):
    """Return (bytes_seen[billions], bpb) deduped by global_step (last write wins)."""
    by_step = {}
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            loss = d.get("loss/out")
            tok = d.get("optim/total_tokens")
            step = d.get("global_step")
            if loss is None or tok is None or step is None:
                continue
            bpb = loss / (LN2 * bytes_per_token)
            bytes_seen = tok * bytes_per_token
            by_step[step] = (bytes_seen, bpb)
    steps = sorted(by_step)
    gb = [by_step[s][0] / 1e9 for s in steps]
    bpb = [by_step[s][1] for s in steps]
    return gb, bpb


def ema(xs, alpha=0.01):
    out, m = [], xs[0]
    for x in xs:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def main():
    # AU-Net: byte tokenizer, so bytes_per_token = 1 (loss/out is already nats/byte).
    a_gb, a_bpb = load(AUNET, bytes_per_token=1.0)
    l_gb, l_bpb = load(LLAMA, bytes_per_token=BYTES_PER_TOKEN_LLAMA)

    a_ema = ema(a_bpb, 0.01)
    l_ema = ema(l_bpb, 0.01)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))

    # Panel 1: BPB vs training bytes seen (fair shared axis)
    ax1.plot(a_gb, a_bpb, color="#a0c4ff", lw=0.3, alpha=0.4)
    ax1.plot(a_gb, a_ema, color="#1f4ed8", lw=1.8, label=f"AU-Net 2 1.3B (byte)  — final BPB {a_ema[-1]:.3f}")
    ax1.plot(l_gb, l_bpb, color="#ffadad", lw=0.3, alpha=0.4)
    ax1.plot(l_gb, l_ema, color="#c81e1e", lw=1.8, label=f"Llama 1B (subword)  — final BPB {l_ema[-1]:.3f}")
    ax1.set_xlabel("training data seen (GB, bytes)")
    ax1.set_ylabel("Bits Per Byte (BPB)")
    ax1.set_ylim(0.7, 2.2)
    ax1.set_title("BPB loss curve — AU-Net (byte) vs Llama transformer (subword), same DCLM data")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)

    # Panel 2: zoomed to the overlapping byte range, log-y for late-training detail
    xmax = min(a_gb[-1], l_gb[-1])
    ax2.plot(a_gb, a_ema, color="#1f4ed8", lw=1.8, label="AU-Net 2 1.3B (byte)")
    ax2.plot(l_gb, l_ema, color="#c81e1e", lw=1.8, label="Llama 1B (subword)")
    ax2.set_xlim(0, xmax)
    ax2.set_xlabel(f"training data seen (GB, bytes) — overlap region 0..{xmax:.1f} GB")
    ax2.set_ylabel("BPB (EMA α=0.01)")
    ax2.set_ylim(0.7, 1.4)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    print(f"saved: {OUT}")
    print(f"AU-Net : points={len(a_gb):>6}  bytes_seen={a_gb[-1]:7.1f} GB  final BPB(EMA)={a_ema[-1]:.4f}  raw last={a_bpb[-1]:.4f}")
    print(f"Llama  : points={len(l_gb):>6}  bytes_seen={l_gb[-1]:7.1f} GB  final BPB(EMA)={l_ema[-1]:.4f}  raw last={l_bpb[-1]:.4f}")
    print(f"(llama bytes_per_token = {BYTES_PER_TOKEN_LLAMA})")


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
