#!/usr/bin/env python3
"""What the GLOBAL transformer actually receives, per model family.

Experiment E (design section 8). Boundary agreement is a statement about segmentation; the claim
"a character perturbation stays confined" is a statement about the representation the main stack
reads. The two comparison points are:

  subword  tok_embeddings(ids)         -- the trunk input IS the embedding lookup; a token whose
                                          id changed contributes an unrelated vector
  aunet /  transitions[-1].down(...)   -- the local byte encoder's states, pooled at the patch
  bpebyte                                 boundaries; a one-byte edit moves this continuously

Both are captured from the SAME validated eval path (fflm_probe.build_generator -> prefill), not
from a re-implementation of the forward pass, so what is measured is what the model runs.

Each slot is returned with a reference byte position and a byte span, which is what lets clean and
perturbed runs be aligned in raw-byte coordinates:

  subword  span = the token's byte span; ref = its LAST byte
  byte     ref  = the byte the pooling gathered (a causal read position); span = the patch it closes

Tail caveat: subword tokens tile the byte sequence exactly, while the byte arms' final patch is
still open at the end of the span, so the last ~1 mean-patch-length bytes (~0.5% of a 640-byte
span) have no pooled vector that has read them and are clamped to the last slot. They sit at
maximum distance from the edit, where every arm is already at cosine 1, so the comparison is
unaffected.

`depth_ref` additionally captures the subword model's hidden state after as many layers as the byte
models' local encoder (3), so the comparison is not merely "a lookup table vs a 3-layer encoder".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/fflm"))
sys.path.insert(0, str(ROOT / "lingua"))

ARM_FAMILY = {"llama": "subword", "aunet": "aunet", "bpebyte": "aunet"}


class ReprExtractor:
    """extract(text) -> dict(H, ref, span_start, span_end) for the trunk-input representation."""

    def __init__(self, arm: str, ckpt: str, tok_path: str = None, max_tokens: int = 8192,
                 depth_ref: int = 3):
        import torch
        from fflm_probe import build_generator, DEFAULT_TOK
        self._torch = torch
        self.arm = arm
        self.family = ARM_FAMILY[arm]
        self.depth_ref = depth_ref
        self.gen, self.tok = build_generator(
            self.family, ckpt, tok_path or os.environ.get("AUNET_TOK", DEFAULT_TOK), max_tokens)
        self.gen.max_gen_len = 1
        self._cap = {}
        self._install()

    # ------------------------------------------------------------------ hooks
    def _install(self):
        cap = self._cap
        if self.family == "aunet":
            trans = self.gen.transitions[-1]
            orig = trans.down

            def down(x, mask, freq_cis, mask_idx=None, position=None, _o=orig):
                out = _o(x, mask, freq_cis, mask_idx, position)
                cap["H"] = out.detach()[0].float().cpu().numpy()
                cap["ref"] = (mask_idx.detach().flatten().cpu().numpy()
                              if mask_idx is not None else None)
                return out
            trans.down = down
        else:
            model = self.gen.model

            def emb_hook(_m, _i, o):
                cap["H"] = o.detach()[0].float().cpu().numpy()
            model.tok_embeddings.register_forward_hook(emb_hook)
            if self.depth_ref and len(model.layers) > self.depth_ref:
                def layer_hook(_m, _i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    cap["Hdepth"] = h.detach()[0].float().cpu().numpy()
                model.layers[self.depth_ref - 1].register_forward_hook(layer_hook)

    # ------------------------------------------------------------------ api
    def extract(self, text: str) -> dict:
        self._cap.clear()
        with self._torch.no_grad():
            self.gen.generate([text])
        H = self._cap["H"]
        nb = len(text.encode())

        if self.family == "aunet":
            # prefill feeds [BOS] + bytes, so token index t is byte t-1; slot 0 is the forced BOS
            # boundary and has no byte to anchor to.
            ref = self._cap["ref"].astype(np.int64) - 1
            keep = ref >= 0
            ref, H = ref[keep], H[:len(ref)][keep]
            order = np.argsort(ref, kind="stable")
            ref, H = ref[order], H[order]
            # patch k closes at its read position; its span runs from the previous read position
            span_start = np.concatenate(([0], ref[:-1] + 1))
            span_end = ref + 1
            return {"H": H, "ref": ref, "span_start": span_start, "span_end": span_end,
                    "n_bytes": nb}

        ids = self.tok.encode(text, add_bos=False, add_eos=False)
        parts = self.tok.tkt_model.decode_tokens_bytes(ids)
        lens = np.fromiter((len(p) for p in parts), np.int64, len(parts))
        en = np.cumsum(lens)
        st = np.concatenate(([0], en[:-1]))
        H = H[1:1 + len(ids)]                      # drop the BOS row
        out = {"H": H, "ref": en - 1, "span_start": st, "span_end": en, "n_bytes": nb,
               "ids": np.asarray(ids, np.int64)}
        if "Hdepth" in self._cap:
            out["Hdepth"] = self._cap["Hdepth"][1:1 + len(ids)]
        return out
