#!/usr/bin/env python3
"""Greedy generation for NUMCoT items (items.jsonl from build_items.py) with one 1B model; resumable.

Families (as rep_probes/score_items.py):
  llama | aunet | bpebyte   paper ckpts via fflm build_generator + EvalHarnessLM.generate_until (lingua generators,
                            KV-cached; BPEByte uses its own boundary-correct generation path)
  blt                       official bytelatent + xformers, eval_suite BLTHarness, batch 1, max_len 4096, released
                            threshold 1.335442066192627
  hnet                      hnet_1stage_XL, eval_suite HNetHarness, batch 8, max_len 8192
All greedy (temperature 0); stop at a newline or at the start of a next question (UNTIL). Budget = item max_bytes bytes; every output is cut to max_bytes bytes, so all models have the same byte budget
(the Transformer is asked for max_bytes tokens, AU-Net/BPEByte for ceil(max_bytes / 6) lm-eval tokens = ~max_bytes bytes).
BLT/H-Net: the harness generation loop (no KV cache) runs to the full budget; here it stops once every row of the
batch has produced the stop string. Greedy decoding is causal, so outputs are identical, only faster.

  python gen_items.py --family bpebyte --items items.jsonl --out out/bpebyte.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch

L = Path(os.environ.get("AUNET_ROOT", "/mnt/ssd2/hyun2/AUNet"))
CKPT = {"llama": "main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated",
        "aunet": "main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated",
        "bpebyte": "main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"}
UNTIL = ["\n", "Question", "question：", "The following"]  # newline, or the start of a next question


class Req:
    def __init__(self, *args):
        self.args = args


def build(family, a):
    if family in CKPT:
        sys.path.insert(0, str(L / "lingua"))
        sys.path.insert(0, str(L / "scripts/fflm"))
        from fflm_probe import build_generator, DEFAULT_TOK
        fam = "subword" if family == "llama" else "aunet"
        gen, _ = build_generator(fam, str(L / CKPT[family]), DEFAULT_TOK, 8192)
        if fam == "subword":
            from apps.main.eval import EvalHarnessLM
        else:
            from apps.aunet.eval import EvalHarnessLM
            # compiled create_block_mask hits a CUDA illegal memory access on long generate_until runs; the eager
            # mask is byte-identical (as reports_afterNAACL/_lib/lmx.py::load)
            import apps.aunet.generate as _g
            _cbm = _g.create_block_mask
            _g.create_block_mask = lambda *x, **kw: _cbm(*x, **{**kw, "_compile": False})
        return EvalHarnessLM(gen)
    sys.path.insert(0, os.environ.get("RUN_EXT_DIR", str(L / "scripts/probes/ext_ci")))
    import run_ext
    if family == "blt":
        lm = run_ext.build_blt_official(a.blt_weights, 4096, a.threshold)
        lm._generate_batch = lambda p, u, m: blt_generate(lm, p, u, m)
    else:
        lm = run_ext.build_hnet_lm(batch_size=8, max_len=8192)
        lm._generate_batch = lambda p, u, m: hnet_generate(lm, p, u, m)
    return lm


def _cut(s, until):
    for u in until:
        j = s.find(u)
        if j >= 0:
            s = s[:j]
    return s


@torch.inference_mode()
def blt_generate(self, prompts, until, max_gen_toks):
    """eval_suite blt_eval.harness.BLTHarness._generate_batch + early stop."""
    a128 = lambda n: ((n + 127) // 128) * 128
    a64 = lambda n: ((n + 63) // 64) * 64
    boe = self.tokenizer.boe_id
    cap = self.max_len - max_gen_toks - 1
    ptoks = [self.tokenizer.encode(p, add_bos=True, add_eos=False)[-cap:] for p in prompts]
    plen = [len(t) for t in ptoks]
    start, end, bs = min(plen), max(plen) + max_gen_toks, len(ptoks)
    toks = torch.full((bs, end), boe, dtype=torch.long, device="cuda")
    real = torch.zeros((bs, end), dtype=torch.bool, device="cuda")
    for i, t in enumerate(ptoks):
        toks[i, :len(t)] = torch.tensor(t, device="cuda"); real[i, :len(t)] = True
    dec = lambda i, hi: self.tokenizer.decode(toks[i, plen[i]:min(hi, plen[i] + max_gen_toks)].tolist())
    for curr in range(start, end):
        Np = a128(curr); cur = toks[:, :curr]
        if Np > curr:
            cur = torch.cat([cur, torch.full((bs, Np - curr), boe, dtype=torch.long, device="cuda")], dim=1)
        pl, _ = self.patcher.patch(cur, include_next_token=False, entropies=self._row_entropies(cur))
        P, Pp = pl.shape[1], a64(pl.shape[1])
        if Pp > P:
            pl = torch.cat([pl, torch.zeros((bs, Pp - P), dtype=pl.dtype, device="cuda")], dim=1)
        nxt = self.model(cur, patch_lengths=pl)[:, curr - 1].argmax(-1)
        toks[:, curr] = torch.where(real[:, curr], toks[:, curr], nxt)
        if all(curr + 1 > plen[i] and any(u in dec(i, curr + 1) for u in until) for i in range(bs)):
            break
    return [_cut(dec(i, end), until) for i in range(bs)]


@torch.inference_mode()
def hnet_generate(self, prompts, until, max_gen_toks):
    """eval_suite hnet_eval.harness.HNetHarness._generate_batch + early stop."""
    from hnet_eval.harness import BOS, DEVICE
    cap = self.max_len - max_gen_toks - 1
    ptoks = [([BOS] + self._enc(p))[-cap:] for p in prompts]
    plen = [len(t) for t in ptoks]
    start, end, bs = min(plen), max(plen) + max_gen_toks, len(ptoks)
    toks = torch.zeros((bs, end), dtype=torch.long, device=DEVICE)
    real = torch.zeros((bs, end), dtype=torch.bool, device=DEVICE)
    for i, t in enumerate(ptoks):
        toks[i, :len(t)] = torch.tensor(t, device=DEVICE); real[i, :len(t)] = True
    full = torch.ones((bs, end), dtype=torch.bool, device=DEVICE)
    dec = lambda i, hi: bytes(b for b in toks[i, plen[i]:min(hi, plen[i] + max_gen_toks)].tolist()
                              if b < 254).decode("utf-8", "ignore")
    for curr in range(start, end):
        nxt = self.model(toks[:, :curr], mask=full[:, :curr]).logits[:, curr - 1].argmax(-1)
        toks[:, curr] = torch.where(real[:, curr], toks[:, curr], nxt)
        if all(curr + 1 > plen[i] and any(u in dec(i, curr + 1) for u in until) for i in range(bs)):
            break
    return [_cut(dec(i, end), until) for i in range(bs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["llama", "aunet", "bpebyte", "blt", "hnet"], required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blt_weights", default=str(L / "ext/blt_weights"))
    ap.add_argument("--threshold", type=float, default=1.335442066192627)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes items i, i+n, ...")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    items = [json.loads(l) for l in open(a.items)]
    if a.limit:
        items = items[:a.limit]
    items = items[si::sn]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["key"] for l in open(a.out)}
    todo = [it for it in items if it["key"] not in done]
    # one generation group per budget; within it, similar prompt lengths together
    todo.sort(key=lambda it: (it["max_bytes"], len(it["prompt"])))
    print(f"{a.family}: {len(todo)} to do ({len(done)} done)", flush=True)
    lm = build(a.family, a)
    t0, n = time.time(), 0
    with open(a.out, "a") as f:
        for b in range(0, len(todo), a.chunk):
            ch = todo[b:b + a.chunk]
            reqs = []
            for it in ch:
                mb = it["max_bytes"]
                # llama: tokens (>= 1 byte each); aunet/bpebyte: EvalHarnessLM turns max_gen_toks t into ~6t bytes
                # (GEN_BYTES_PER_TOKEN); blt/hnet harnesses count bytes
                t = {"llama": mb, "aunet": -(-mb // 6), "bpebyte": -(-mb // 6)}.get(a.family, mb)
                reqs.append(Req(it["prompt"], {"until": UNTIL, "max_gen_toks": t, "temperature": 0.0}))
            outs = lm.generate_until(reqs)
            for it, o in zip(ch, outs):
                o = _cut(o, UNTIL)
                o = o.encode()[:it["max_bytes"]].decode("utf-8", "ignore")
                f.write(json.dumps({"key": it["key"], "output": o}, ensure_ascii=False) + "\n")
            f.flush()
            n += len(ch)
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {el / 60:.1f} min  {el / n:.2f} s/item  eta {(len(todo) - n) * el / n / 3600:.1f} h",
                  flush=True)


if __name__ == "__main__":
    main()
