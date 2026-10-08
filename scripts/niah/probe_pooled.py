#!/usr/bin/env python3
"""Where is the 1-byte loss on S-NIAH-3? Linear probes on AU-Net / BPEByte patch vectors, taken from the eval path
(generator prefill, same level mask as scoring).

Sites (vectors are indexed by patch-start byte; trunk token k = byte-stage hidden at patch k's start byte):
  pool_n   needle side, pooled vector (trans.down output) at the start byte q0 of every patch inside/just after the UUID
  trunk_n  needle side, trunk output at the same tokens
  trunk_a  answer side, trunk output at the start byte s of every answer patch (the vector the decoder reads to emit
           the rest of that patch)
Targets: 16-way hex byte.
  needle sites: byte at q0 - d, d = 1..6 (one probe per d); grouped by (len, offset) of the patch that holds the byte
  answer site:  byte at s + j, j = 1..6 (one probe per j); grouped by (len, offset) and near/far (true distance 512B)
Data: the 1000 S-NIAH-3 items + `--resample` copies with a fresh random UUID (same 8-4-4-4-12 format) put in both
the needle and the answer. Train/test split by ORIGINAL item (80/20), so a test item's copies stay in test.
  phase extract: cd lingua && PYTHONPATH=$PWD .venv/bin/python <this> extract --model aunet --ckpt ... --out X.pt
  phase probe:   ... <this> probe --data X.pt --out X.json
"""
import argparse
import json
import random
import sys
from collections import defaultdict

L = "/mnt/ssd2/hyun2/AUNet"
PAIRS = f"{L}/reports/niah/sniah123_n250_final_pairs.jsonl"
HEX = "0123456789abcdef"


def rand_uuid(rng):
    h = "".join(rng.choice(HEX) for _ in range(32))
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def extract(args):
    import torch
    sys.path.insert(0, f"{L}/scripts/fflm")
    from fflm_probe import build_generator, DEFAULT_TOK
    gen, tok = build_generator("aunet", args.ckpt, DEFAULT_TOK, 16384)
    gen.max_gen_len = 1
    cap = {}
    trans = gen.transitions[0]
    orig_down, orig_trunk = trans.down, gen.trunk.prefill

    def down(x, mask, freq_cis, mask_idx=None, position=None):
        out = orig_down(x, mask, freq_cis, mask_idx, position)
        cap["starts"] = mask_idx[0].detach().cpu()           # token index of each patch start
        cap["pool"] = out[0].detach().float().cpu().half()
        return out

    def trunk(x, lengths):
        out = orig_trunk(x, lengths)
        cap["trunk"] = out[0].detach().float().cpu().half()
        return out

    trans.down, gen.trunk.prefill = down, trunk
    rng = random.Random(1234)
    recs = [r for r in map(json.loads, open(PAIRS)) if r["probe"] == "sniah3"]
    data = defaultdict(list)
    for i, r in enumerate(recs):
        v0 = r["values"][0]
        for rep in range(1 + args.resample):
            v = v0 if rep == 0 else rand_uuid(rng)
            prompt = r["prompt"].replace(v0, v)
            text = prompt + " " + v
            full = text.encode()
            c = full.find(v.encode()); a = len(full) - len(v)
            assert 0 <= c < len(prompt.encode())
            _, _, greedy = gen.generate([text])
            gr = greedy[0]                                    # gr[t]: token t+1 predicted correctly
            st = cap["starts"].tolist()                       # token coords (BOS = 0) -> byte = token - 1
            bst = [s - 1 for s in st if s >= 1]
            starts_b = sorted(set(bst)) + [len(full)]
            sidx = {s - 1: j for j, s in enumerate(st) if s >= 1}

            def patch_of(b):
                s0 = max(x for x in starts_b if x <= b); s1 = min(x for x in starts_b if x > b)
                return s0, s1 - s0

            # needle side: every patch start q0 in (c, c + len(v)]
            for q0 in [s for s in starts_b if c < s <= c + len(v)]:
                if q0 not in sidx:
                    continue
                tg = []
                for d in range(1, 7):
                    b = q0 - d
                    if c <= b < c + len(v) and chr(full[b]) in HEX:
                        p0, n = patch_of(b)
                        tg.append((d, HEX.index(chr(full[b])), n, b - p0))
                if tg:
                    data["n_item"].append(i); data["n_rep"].append(rep)
                    data["n_pool"].append(cap["pool"][sidx[q0]]); data["n_trunk"].append(cap["trunk"][sidx[q0]])
                    data["n_tg"].append(tg)
            # answer side: every patch start s in [a, end)
            pb = [int(x) for x in gr[a:a + len(v)].tolist()]  # gr[t] predicts token t+1 = byte t (BOS at token 0)
            for s in [x for x in starts_b if a <= x < len(full)]:
                if s not in sidx:
                    continue
                tg = []
                for j in range(1, 7):
                    b = s + j
                    if b < len(full) and chr(full[b]) in HEX:
                        p0, n = patch_of(b)
                        tg.append((j, HEX.index(chr(full[b])), n, b - p0, pb[b - a]))
                if tg:
                    data["a_item"].append(i); data["a_rep"].append(rep); data["a_dist"].append(a - c)
                    data["a_trunk"].append(cap["trunk"][sidx[s]]); data["a_tg"].append(tg)
        if i % 100 == 0:
            print(f"[{args.model}] item {i}/{len(recs)}", flush=True)
    for k in ("n_pool", "n_trunk", "a_trunk"):
        data[k] = torch.stack(data[k])
    torch.save(dict(data), args.out)
    print(f"saved {args.out}: needle vecs {len(data['n_item'])}, answer vecs {len(data['a_item'])}", flush=True)


def probe(args):
    import torch
    D = torch.load(args.data)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test_items = set(random.Random(0).sample(range(1000), 200))
    res = {}

    def fit(X, y, tr):
        X = X.float().to(dev); y = torch.tensor(y, device=dev)
        trn = torch.tensor(tr, device=dev)
        mu, sd = X[trn].mean(0), X[trn].std(0) + 1e-4
        X = (X - mu) / sd
        W = torch.nn.Linear(X.shape[1], 16).to(dev)
        opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=args.wd)
        idx = trn.nonzero().squeeze(1)
        for ep in range(args.epochs):
            perm = idx[torch.randperm(len(idx), device=dev)]
            for b in range(0, len(perm), 1024):
                bb = perm[b:b + 1024]
                loss = torch.nn.functional.cross_entropy(W(X[bb]), y[bb])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return (W(X).argmax(-1) == y).cpu().tolist()

    for site in ("n_pool", "n_trunk", "a_trunk"):
        side = site[0]
        items = D[f"{side}_item"]; tgs = D[f"{side}_tg"]; X = D[site]
        out = defaultdict(lambda: [0, 0])
        for k in range(1, 7):
            rows = [(r, t) for r, tg in enumerate(tgs) for t in tg if t[0] == k]
            if not rows:
                continue
            Xi = X[[r for r, _ in rows]]
            y = [t[1] for _, t in rows]
            tr = [items[r] not in test_items for r, _ in rows]
            ok = fit(Xi, y, tr)
            for (r, t), o, is_tr in zip(rows, ok, tr):
                if is_tr:
                    continue
                n, off = t[2], t[3]
                reg = "" if side == "n" else ("|far" if D["a_dist"][r] >= 512 else "|near")
                out[f"k{k}|len{n}|off{off}{reg}"][0] += o; out[f"k{k}|len{n}|off{off}{reg}"][1] += 1
                if side == "a" and off == k:  # model's own correctness, once per byte (byte inside the patch at s)
                    out[f"model|len{n}|off{off}{reg}"][0] += t[4]; out[f"model|len{n}|off{off}{reg}"][1] += 1
            print(f"[{site}] k={k} done", flush=True)
        res[site] = {k: [v[0] / v[1], v[1]] for k, v in sorted(out.items())}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--model", required=True); e.add_argument("--ckpt", required=True)
    e.add_argument("--resample", type=int, default=2); e.add_argument("--out", required=True)
    p = sp.add_parser("probe"); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--wd", type=float, default=1e-2)
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else probe(args)


if __name__ == "__main__":
    main()
