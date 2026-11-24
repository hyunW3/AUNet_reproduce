#!/usr/bin/env python3
"""Frozen byte-LM probe: per-byte surprisal s_i = -log2 p(x_i | x_<i) and entropy H_i.

Section 3 of bpebyte_structured_locality_experiment_design.md. Boundary quality must not be
scored with a model's own likelihood (a model trained under one segmentation is cheaper exactly
where that segmentation put its units), so every experiment reads byte uncertainty from ONE
frozen byte-level LM that never saw any of the three boundary schemes.

apps.aunet.data.entropy_patch.EntropyModel is close but (a) returns only the predictive entropy
H_i, not the REALIZED surprisal s_i the design makes its primary metric; (b) its CPU path
hardcodes lingua's BOS=256/offset=0 convention, which silently mis-feeds a foreign byte-LM such
as Meta's BLT entropy model (byte_offset=4, bos_id=1) -- measured 7.12 bpb instead of 1.45; and
(c) it never applies the checkpoint's sliding window, because apps.main.create_causal_mask only
honours `sliding_window` under attn_impl="fmha" while the default path is full-causal sdpa. The
BLT entropy model is trained with a 512-byte local window, so full-causal scoring costs it
3.79 bpb vs 1.45 with the window. This loader fixes all three.

Alignment: x = [bos, b_0 + off, ..., b_{n-1} + off]; logits[i] predicts x[i+1], so
    s_i = -log2 p(b_i | b_<i) = -log2 softmax(logits[i])[b_i + off].
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

LN2 = math.log(2.0)


class ByteLMProbe:
    """A frozen causal byte LM scored for realized surprisal.

    `window`: local-attention span in tokens. None -> auto: the checkpoint's `sliding_window`
    when params.json says it was trained with an xformers/fmha backend (which is the only one
    that applies the window), else full causal. Pass 0 to force full causal.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda", dtype: str = "fp32",
                 window: Optional[int] = None):
        import dataclasses
        import torch
        from apps.main.transformer import LMTransformer, LMTransformerArgs

        self._torch = torch
        self.ckpt_path = str(ckpt_path)
        ckpt = Path(ckpt_path)
        cfg = json.load(open(ckpt / "params.json"))
        mcfg = cfg["model"]
        self.byte_offset = int(cfg.get("byte_offset", 0))
        self.bos_id = int(cfg.get("bos_id", 256))
        fields = {f.name for f in dataclasses.fields(LMTransformerArgs)}
        margs = LMTransformerArgs(**{k: v for k, v in mcfg.items() if k in fields})
        self.model = LMTransformer(margs)
        st = torch.load(ckpt / "consolidated.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(st["model"])
        self.device = device
        self.model = self.model.to(device).float().eval()
        _DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if device != "cpu" and _DT[dtype] is not torch.float32:
            self.model = self.model.to(_DT[dtype])
        self.dtype = dtype if device != "cpu" else "fp32"
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.max_seqlen = int(mcfg.get("max_seqlen") or 8192)

        if window is None:
            trained_local = str(mcfg.get("attn_impl", "sdpa")) in ("xformers", "fmha")
            window = int(mcfg.get("sliding_window") or 0) if trained_local else 0
        self.window = int(window or 0)
        # left context that gives a scored byte its full receptive field
        self._ctx = self.window if self.window > 0 else (self.max_seqlen - 1)
        self._mask_cache: dict = {}

    def _mask(self, n: int):
        torch = self._torch
        if self.window <= 0:
            return "causal"
        m = self._mask_cache.get(n)
        if m is None:
            i = torch.arange(n, device=self.device)[:, None]
            j = torch.arange(n, device=self.device)[None, :]
            m = ((j <= i) & (j > i - self.window))[None, None]
            self._mask_cache = {n: m}          # windows are near-uniform; keep one
        return m

    # ------------------------------------------------------------------ scoring
    def _score_window(self, byte_seq: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        n = len(byte_seq)
        ids = [self.bos_id] + [int(b) + self.byte_offset for b in byte_seq]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x, mask=self._mask(n + 1), attn_impl="sdpa")   # (1, n+1, V)
            logp = torch.log_softmax(logits[0].float(), dim=-1)[:n]            # logp[i] ~ p(.|b_<i)
            s = -logp.gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)           # nats
            H = -(logp.exp() * logp).sum(dim=-1)                               # nats
        return (s.cpu().numpy() / LN2).astype(np.float32), (H.cpu().numpy() / LN2).astype(np.float32)

    def score(self, byte_seq) -> Tuple[np.ndarray, np.ndarray]:
        """Per-byte (surprisal_bits, entropy_bits), both length len(byte_seq)."""
        n = len(byte_seq)
        if n == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        cap = self.max_seqlen - 1
        if n <= cap:
            return self._score_window(list(byte_seq))
        # Slide with overlap >= receptive field so a kept byte never sees the chunk's fresh BOS.
        s_out, h_out = np.empty(n, np.float32), np.empty(n, np.float32)
        overlap = min(self._ctx, cap // 2)
        stride = cap - overlap
        start = 0
        while start < n:
            end = min(start + stride, n)
            lo = max(0, start - overlap)
            s, h = self._score_window(list(byte_seq[lo:end]))
            s_out[start:end] = s[start - lo:end - lo]
            h_out[start:end] = h[start - lo:end - lo]
            start = end
        return s_out, h_out

    def info(self) -> dict:
        return {"ckpt": self.ckpt_path, "n_params": int(self.n_params), "window": self.window,
                "byte_offset": self.byte_offset, "bos_id": self.bos_id, "dtype": self.dtype}


# Probe registry: primary = Meta's BLT entropy model (external, never trained on any of the three
# boundary schemes); secondary = a 50M byte LM trained in-repo, as an independence check.
PROBES = {
    "blt_entropy": "runs/small/entropy_model/blt_entropy/consolidated",
    "byte_50M": "runs/small/entropy_model/byte_50M/checkpoints/0000004600/consolidated",
}
