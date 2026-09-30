#!/usr/bin/env python3
"""BLT-1B entropy-patch lengths, via the HF transformers port of BLT's entropy model.

The bytelatent checkout + venv the paper's BLT evals used (lzspacebyte/blt) is gone from this host,
so the patcher is rebuilt from the HF port (itazap/blt-1b-hf, which ships the same released
threshold 1.335442066192627). Only the patcher's weights are loaded (not the 4.6B model).

Rule (identical to bytelatent's find_entropy_patch_start_ids and HF's
BltPatcher.patch_lengths_from_entropies): token 0 (BOS) and token 1 always start a patch; for
p >= 1, a patch starts at token p+1 when the next-byte entropy at position p exceeds the
threshold. Token t >= 1 is byte t-1; the lone BOS patch is excluded, like every other model's BOS.

Sliding window: the released entropy model (facebook/blt-entropy config) was trained with
attn_bias_type=local_block_causal, sliding_window=512 — each byte attends to at most the previous
512 positions. The HF port runs plain causal attention, and so did the paper-time bytelatent
harness call (its DCLM result, 1.151 B/patch, is reproduced exactly without the window). Past
position 512 the model is then out of distribution: mean entropy jumps from ~0.9 to ~2.5 nats and
>95% of bytes exceed the threshold, i.e. ~1 byte/patch. --window 512 (default) restores the trained
attention; --window 0 reproduces the paper-time behaviour (written as hist_blt_nowindow.json).

Entropies are computed per row (right-padded batches; the entropy model is causal, so padding
never changes a real position), which avoids the batch-mate dependence that bytelatent's
flattening entropy call has (see eval_suite/blt_eval/harness.py::_row_entropies).

  source scripts/probes/patch_stats/env.sh
  CUDA_VISIBLE_DEVICES=2 lingua/.venv/bin/python scripts/probes/patch_stats/measure_blt.py [--gate_only]
"""
import argparse, glob, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_common import Hist, check_gate, lengths_from_starts, load_texts  # noqa: E402

SNAP = glob.glob("/home/hyunw3/.cache/huggingface/hub/models--itazap--blt-1b-hf/snapshots/*")[0]
BOS, OFFSET = 1, 4          # HF/bytelatent BLT ids: BOS=1, byte b -> b + 4


def load_patcher(dtype):
    from transformers import BltConfig
    from transformers.models.blt.modeling_blt import BltPatcher
    cfg = BltConfig.from_pretrained(SNAP)
    patcher = BltPatcher(cfg.patcher_config)
    sd = {}
    with safe_open(f"{SNAP}/model.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith("model.patcher."):
                sd[k[len("model.patcher."):]] = f.get_tensor(k)
    missing, unexpected = patcher.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary" not in m]      # buffers, rebuilt from config
    assert not missing and not unexpected, (missing, unexpected)
    return patcher.to("cuda", dtype).eval(), float(cfg.patching_threshold)


@torch.inference_mode()
def entropies(patcher, batch_ids, window=0):
    """[B, L] next-token entropies. window > 0 adds the sliding-window restriction (key j visible to
    query i iff i - window < j <= i, as xformers make_local_attention); only rows longer than the
    window differ from plain causal attention."""
    L = max(len(x) for x in batch_ids)
    toks = torch.zeros((len(batch_ids), L), dtype=torch.long, device="cuda")
    for i, x in enumerate(batch_ids):
        toks[i, :len(x)] = torch.tensor(x, device="cuda")
    mask = None
    if window and L > window:
        i = torch.arange(L, device="cuda")
        allowed = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < window)
        mask = torch.zeros((1, 1, L, L), device="cuda", dtype=patcher.dtype)
        mask = mask.masked_fill(~allowed, float("-inf")).expand(len(batch_ids), 1, L, L)
    ent, _, _ = patcher(input_ids=toks, attention_mask=mask)   # [B, L] next-token entropy
    return ent.float().cpu()


def byte_starts(ent_row, n_bytes, threshold):
    """Patch start byte indices from one row's entropies (row length n_bytes + 1 incl. BOS)."""
    h = ent_row[1:n_bytes]                   # positions p = 1 .. n_bytes-1 (p = n_bytes predicts past the end)
    tok_starts = (torch.nonzero(h > threshold).flatten() + 2).tolist()   # p+1, with h[0] = position 1
    return [0] + [t - 1 for t in tok_starts]                  # token 1 (byte 0) is always a start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--tok_budget", type=int, default=65536, help="max padded tokens per batch")
    ap.add_argument("--window", type=int, default=512, help="entropy-model sliding window; 0 = none")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the released threshold (1.335442066192627); output tagged _t<theta>")
    a = ap.parse_args()
    patcher, thr = load_patcher(getattr(torch, a.dtype))
    if a.threshold is not None:
        thr = a.threshold
    print(f"[blt] threshold {thr:.6f}, patcher dtype {a.dtype}, window {a.window}", flush=True)
    tag = "blt" if a.window else "blt_nowindow"
    if a.threshold is not None:
        tag += f"_t{a.threshold:.4f}".replace(".", "")

    rows = load_texts(["dclm"] if a.gate_only else None)
    gate_rows = [r for r in rows if r["group"] == "dclm"]
    rest = sorted((r for r in rows if r["group"] != "dclm"), key=lambda r: len(r["text"].encode()))
    h, t0 = Hist(), time.time()
    for phase, rs in (("gate", gate_rows), ("bench", rest)):
        i = 0
        while i < len(rs):
            j, L = i, 0
            while j < len(rs):
                L2 = max(L, len(rs[j]["text"].encode()) + 1)
                if (j - i + 1) * L2 > a.tok_budget and j > i:
                    break
                L, j = L2, j + 1
            batch = rs[i:j]
            ids = [[BOS] + [b + OFFSET for b in r["text"].encode()] for r in batch]
            if a.window and max(len(x) for x in ids) > a.window:     # dense L x L mask: keep batches small
                ent = [entropies(patcher, [x], a.window)[0] for x in ids]   # one row at a time
            else:
                ent = entropies(patcher, ids)
            for k, r in enumerate(batch):
                nb = len(ids[k]) - 1
                h.add(r, lengths_from_starts(byte_starts(ent[k], nb, thr), nb))
            i = j
            if phase == "bench" and i % 5000 < len(batch):
                print(f"  {i}/{len(rs)}  {time.time() - t0:.0f}s", flush=True)
        if phase == "gate":
            if a.window or a.threshold is not None:   # no paper-time reference exists for the windowed patcher; report, don't gate
                print(f"[blt] DCLM bytes/patch with window {a.window}: {h.bpp('dclm/dclm'):.4f} "
                      f"(paper-time, no window: {1228802 / 1067826:.4f})", flush=True)
            elif not check_gate("blt", h):
                raise SystemExit("gate failed — HF patcher does not reproduce the bytelatent measurement")
            if a.gate_only:
                return
    print("wrote", h.save(tag, {"impl": "transformers BltPatcher (itazap/blt-1b-hf)",
                                "threshold": thr, "dtype": a.dtype, "sliding_window": a.window}))


if __name__ == "__main__":
    main()
