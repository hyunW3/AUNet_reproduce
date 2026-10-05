#!/usr/bin/env python3
"""Mechanism probe: how many patches do BLT and BPEByte spend on the spans the probes hinge on?

For every row of the probe files (build_tasks.py) the analysed text is prompt + gold continuation
(the bytes a teacher-forced scorer sees). Per span we report bytes, boundaries inside the span
(= patches, +-1 depending on start/end convention, identical across rows) and the longest run of
bytes without a boundary:
  answer   the gold continuation (value / copy output / count option)
  qkey1    first mention of the key in the query (d bytes after the needle) [NIAH-style tasks]
  qkey     last mention of the key in the query (primed by qkey1)           [NIAH-style tasks]
  nkey     first occurrence of the key (the needle)                         [NIAH-style tasks]
  text     the string to count / to copy (last "Text:" / "Input:" line)     [count / copy]

  BLT     : official bytelatent entropy model (512 window, xformers) + released threshold, fp as in
            ext_models.build_blt; run in the BLT venv with PYTHONPATH= and BLT_REPO set.
  BPEByte : the checkpoint's own regex_pool.get_levels_mask (pure in the bytes; CPU).

  python patch_probe.py --family blt --ckpt <lc>/ext/blt_weights --data dist512.jsonl ... --out o.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys

KEY_RES = [re.compile(r"What is the access code for (\S+?)\?"),
           re.compile(r"What is the special magic \w+ for (\S+) mentioned")]
TEXT_RES = [re.compile(r"\nText: ([^\n]*)\nNumber of times [^\n]*$"),
            re.compile(r"\nInput: ([^\n]*)\nOutput:$")]


def spans(r):
    """[(name, start_byte, end_byte)] over text = prompt + continuation."""
    p = r["prompt"]
    cont = (" " + r["answers"][0]) if r["mode"] == "gen" else r["options"][r["gold"]]
    if r["mode"] == "gen" and r["task"] in ("hashhop",):
        cont = r["answers"][0]                       # prompt ends with "= '"
    text = p + cont
    pb = len(p.encode())
    out = [("answer", pb + (1 if cont.startswith(" ") else 0), pb + len(cont.encode()))]
    for rx in KEY_RES:
        m = rx.search(p)
        if m:
            k = m.group(1)
            kb = len(k.encode())
            first, q1, last = p.find(k), m.start(1), p.rfind(k)
            for name, at in (("nkey", first), ("qkey1", q1), ("qkey", last)):
                s = len(p[:at].encode())
                out.append((name, s, s + kb))
            break
    for rx in TEXT_RES:
        m = rx.search(p)
        if m:
            a = len(p[:m.start(1)].encode())
            out.append(("text", a, a + len(m.group(1).encode())))
            break
    return text, out


def span_stats(bounds, a, b):
    """bounds: sorted byte offsets where a patch starts. -> (n_boundaries_in_span, longest_run)."""
    inside = [x for x in bounds if a <= x < b]
    cuts = [a] + [x for x in inside if x > a] + [b]
    return len(inside), max(y - x for x, y in zip(cuts, cuts[1:]))


# ------------------------------------------------------------------------------ BLT
class BLTPatches:
    def __init__(self, weights, threshold=1.335442066192627):
        import torch
        sys.path.insert(0, os.environ["BLT_REPO"])
        from bytelatent.transformer import LMTransformer
        from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
        from bytelatent.data.patcher import PatcherArgs, entropy
        from bytelatent.distributed import DistributedArgs
        DistributedArgs().configure_world()
        if not torch.distributed.is_initialized():
            with socket.socket() as sk:
                sk.bind(("127.0.0.1", 0))
                port = sk.getsockname()[1]
            torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}",
                                                 rank=0, world_size=1)
            torch.cuda.set_device(0)
        self.torch, self.entropy = torch, entropy
        self.ent = LMTransformer.from_pretrained(f"{weights}/entropy", local_files_only=True)
        self.ent = self.ent.to("cuda", torch.bfloat16).eval()
        self.tok = BltTokenizer(vocab_size_unit_1=256, bpe_delim=False, add_bos=True, add_eos=False)
        self.patcher = PatcherArgs(threshold=threshold, realtime_patching=False,
                                   patching_device="cuda").build()
        self.patcher.entropy_model = self.ent

    def bounds(self, text, pad_extra=0, fp32=False):
        """Patch-start byte offsets (text byte i == token i+1; BOS is token 0), exactly as the
        eval harness patches a bs-1 row (right-padded with boe to a multiple of 128)."""
        return self.bounds_batch([text], pad_extra, fp32)[0]

    def bounds_batch(self, texts, pad_extra=0, fp32=False):
        """Rows patched together in ONE [bs, Np] entropy forward (the harness path at bs > 1)."""
        torch = self.torch
        ids = [self.tok.encode(t, add_bos=True, add_eos=False) for t in texts]
        Np = ((max(map(len, ids)) + 127) // 128) * 128 + pad_extra
        t = torch.full((len(ids), Np), self.tok.boe_id, dtype=torch.long, device="cuda")
        for i, x in enumerate(ids):
            t[i, :len(x)] = torch.tensor(x, device="cuda")
        model = self.ent
        if fp32:
            if not hasattr(self, "ent32"):
                import copy
                self.ent32 = copy.deepcopy(self.ent).float()
            model = self.ent32
        with torch.inference_mode():
            ent = self.entropy(model(t)).float()
            pl, _ = self.patcher.patch(t, include_next_token=False, entropies=ent)
        out = []
        for r, x in enumerate(ids):
            starts, s = [], 0
            for L in pl[r].tolist():
                if L == 0:
                    break
                starts.append(s)
                s += L
            out.append([y - 1 for y in starts if 1 <= y <= len(x) - 1])   # token -> text byte offset
        return out


# ------------------------------------------------------------------------------ BPEByte / AU-Net
class TrioPatches:
    def __init__(self, ckpt, tok_path, family="aunet"):
        sys.path.insert(0, os.environ.get("LC_SCRIPTS", os.path.dirname(os.path.abspath(__file__))))
        from run_longctx import build_generator
        self.family = family
        self.g, self.tok = build_generator(family, ckpt, tok_path, 8192)

    def bounds(self, text):
        pb = self.tok.encode(text, add_bos=False, add_eos=False)
        if self.family == "subword":                 # token count only (offsets not needed)
            return list(range(len(pb)))
        return [i for i, x in enumerate(self.g.regex_pool.get_levels_mask(pb)) if x > 0]


def _diff(a, b, upto=None):
    """Boundaries present in exactly one of a / b (restricted to offsets < upto)."""
    a, b = set(a), set(b)
    if upto is not None:
        a, b = {x for x in a if x < upto}, {x for x in b if x < upto}
    return len(a ^ b)


def run_windows(P, family, wins, tag, f):
    """Per window: patch count, and boundary flips under perturbations that should not matter:
      prefix  boundaries of text[:n/2] vs the full text's boundaries before n/2 (streaming stability)
      BLT only (entropy model numerics): pad (+1024 trailing boe), batch8 (patched alongside 7 other
      windows of the same domain), fp32 (fp32 entropy model instead of bf16)."""
    for k, w in enumerate(wins):
        text = w["text"]
        b = P.bounds(text)
        half = text.encode()[: w["n_bytes"] // 2].decode(errors="ignore")
        hb = len(half.encode())
        rec = {"id": w["id"], "domain": w["domain"], "tag": tag, "n_bytes": w["n_bytes"],
               "n_patches": len(b)}
        if family != "subword":
            rec["flip_prefix"] = _diff(P.bounds(half), b, upto=hb - 1)
            rec["flip_recompute"] = _diff(P.bounds(text), b)
        if family == "blt":
            rec["flip_pad"] = _diff(P.bounds(text, pad_extra=1024), b)
            dom = [x for x in wins if x["domain"] == w["domain"]]
            j = dom.index(w)
            mates = [dom[(j + d) % len(dom)]["text"] for d in range(1, 8)]
            rec["flip_batch8"] = _diff(P.bounds_batch([text] + mates)[0], b)
            rec["flip_fp32"] = _diff(P.bounds(text, fp32=True), b)
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["blt", "aunet", "subword"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--windows", default=None, help="ood_windows.jsonl: patch counts + determinism")
    ap.add_argument("--tok_path", default=os.environ.get("AUNET_TOK", ""))
    ap.add_argument("--max_rows", type=int, default=None, help="first N rows per (task, cond, length)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = BLTPatches(a.ckpt) if a.family == "blt" else TrioPatches(a.ckpt, a.tok_path, a.family)
    seen = {}
    with open(a.out, "w") as f:
        if a.windows:
            run_windows(P, a.family, [json.loads(l) for l in open(a.windows)], a.tag, f)
        for path in a.data:
            for line in open(path):
                r = json.loads(line)
                k = (r["task"], r["cond"], r["length"])
                seen[k] = seen.get(k, 0) + 1
                if a.max_rows and seen[k] > a.max_rows:
                    continue
                text, sp = spans(r)
                b = P.bounds(text)
                rec = {"id": r["id"], "task": r["task"], "cond": r["cond"], "length": r["length"],
                       "tag": a.tag, "n_bytes": len(text.encode()), "n_patches": len(b)}
                for name, s, e in sp:
                    npch, run = span_stats(b, s, e)
                    rec[name] = {"bytes": e - s, "patches": npch, "longest_run": run}
                f.write(json.dumps(rec) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
