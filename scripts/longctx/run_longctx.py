#!/usr/bin/env python3
"""Run one frozen checkpoint over a long-context task JSONL (from build_tasks.py).

Resumable: rows whose id is already in --out are skipped, so a killed job is
restarted with the same command. Results are appended row-by-row per batch.

gen rows  : greedy decode; the generation is cut to the row's `window` BYTES
            (identical for every family) and scored by
              substr     -- answers[0] occurs in the window
              recall     -- fraction of answers occurring in the window
              exact_line -- first non-empty line of the window == answers[0]
rank rows : log P(option | prompt) summed over the option's tokens/bytes for
            every option; correct iff argmax == gold.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

DEFAULT_TOK = os.environ.get("AUNET_TOK", f"{os.environ.get('AUNET_ROOT', '.')}/tokenizer/llama3/tokenizer.model")


def build_generator(family: str, ckpt: str, tok_path: str, max_tokens: int):
    """(generator, tokenizer) for a consolidated dir; same loaders as fflm_probe.build_generator."""
    if family == "subword":
        from apps.main.generate import (load_consolidated_model_and_tokenizer,
                                        PackedCausalTransformerGenerator,
                                        PackedCausalTransformerGeneratorArgs)
        model, tokenizer, _ = load_consolidated_model_and_tokenizer(ckpt)
        args = PackedCausalTransformerGeneratorArgs(temperature=0.0, max_gen_len=1,
                                                    max_tokens=max_tokens)
        return PackedCausalTransformerGenerator(args, model, tokenizer), tokenizer
    from apps.aunet.generate import (load_consolidated_model_and_tokenizer,
                                     PackedHierarchicalCausalTransformerGenerator,
                                     PackedHierarchicalCausalTransformerGeneratorArgs)
    from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs
    model, tokenizer, regex_pool, _ = load_consolidated_model_and_tokenizer(
        ckpt, model_cls=HierarchicalTransformer, model_args_cls=HierarchicalArgs,
        regex_bpe_tokenizer_path=tok_path, regex_bpe_tokenizer_kind=None)
    args = PackedHierarchicalCausalTransformerGeneratorArgs(temperature=0.0, max_gen_len=1,
                                                            max_tokens=max_tokens)
    return PackedHierarchicalCausalTransformerGenerator(args, model, tokenizer, regex_pool), tokenizer


def cut_window(text: str, window: int) -> str:
    return text.encode()[:window].decode(errors="ignore")


def score_gen(row, gen):
    w = cut_window(gen, row["window"])
    ans = row["answers"]
    if row["score"] == "substr":
        s = float(ans[0] in w)
    elif row["score"] == "recall":
        s = sum(a in w for a in ans) / len(ans)
    elif row["score"] == "exact_line":
        line = next((ln.strip() for ln in w.split("\n") if ln.strip()), "")
        s = float(line == ans[0])
    else:
        raise ValueError(row["score"])
    return s, w


def is_bpe(generator) -> bool:
    strat = getattr(getattr(generator, "regex_pool", None), "strategy", {}) or {}
    return any(str(k).startswith("bpe") for k in strat)


def _bt_chunks(generator, chunk, max_tokens, patch_margin=32):
    """Split rows so each packed bt re-prefill respects the byte budget AND the patch
    transformer's per-chunk seq limit (same packing rule as eval.py _generate_until_bt)."""
    tok = generator.tokenizer
    patch_limit = int(generator.model.trunk.max_seqlen)
    out, cur, cb, cp = [], [], 0, 0
    for r in chunk:
        pb = tok.encode(r["prompt"], add_bos=False, add_eos=False)
        npc = sum(generator.regex_pool.get_levels_mask(pb))
        b_cost, p_cost = len(pb) + 2 * r["window"], npc + r["window"] + patch_margin
        assert p_cost <= patch_limit, f"{r['id']}: {npc} patches exceed the patch window"
        if cur and (cb + b_cost > max_tokens or cp + p_cost > patch_limit):
            out.append(cur)
            cur, cb, cp = [], 0, 0
        cur.append((r, pb))
        cb += b_cost
        cp += p_cost
    if cur:
        out.append(cur)
    return out


_FILL = ("The quick brown fox jumps over the lazy dog. " * 12).encode()[:384]
_MIN_PACKED = 1024


def bt_generate(generator, prompts, max_gen_len):
    """generate_online_bt (greedy) with one change to its forward_fn: when the packed
    re-prefill is shorter than _MIN_PACKED bytes, a filler document is packed alongside
    and its logits dropped. The byte encoder's flex-attention block-mask build hits a CUDA
    illegal memory access on very short packed inputs (a lone ~100-byte ICL prompt);
    documents are attention-isolated, so the filler does not change the real rows."""
    import torch
    from apps.aunet.generate_bt import online_bt_loop
    tok = generator.tokenizer
    fl = list(generator.regex_pool.get_levels_mask(list(_FILL)))[:len(_FILL)]
    fill_tok = [tok.bos_id] + list(_FILL)
    fill_lvl = [1] + [1 if x > 0 else 0 for x in fl] + [0] * (len(_FILL) - len(fl))
    device = generator.device

    def forward_fn(token_rows, level_rows, row_ids):
        n = len(token_rows)
        token_rows, level_rows = list(token_rows), list(level_rows)
        while sum(len(r) for r in token_rows) < _MIN_PACKED:
            token_rows.append(fill_tok)
            level_rows.append(fill_lvl)
        lengths = torch.tensor([len(r) for r in token_rows], dtype=torch.long, device=device)
        tok_t = torch.tensor([t for r in token_rows for t in r], dtype=torch.long, device=device)
        lvl_t = torch.tensor([l for r in level_rows for l in r], dtype=torch.long, device=device)
        logits, _ = generator.prefill(tok_t.unsqueeze(0), lengths, lvl_t.unsqueeze(0))
        return logits[0, lengths.cumsum(0) - 1][:n]

    toks, _, _ = online_bt_loop(forward_fn, generator.regex_pool.make_incremental_bt_parser,
                                prompts, tok.bos_id, max_gen_len, device=device,
                                temperature=1e-6, greedy=True, stops=[], eos=tok.eos_id)
    return toks


def run_gen(generator, rows, family, batch, max_tokens):
    # group by window so one max_gen_len serves the batch; byte families decode
    # bytes, the subword family decodes tokens (>= 1 byte each), so `window`
    # steps always cover `window` bytes for every family
    rows = sorted(rows, key=lambda r: (r["window"], r["prompt_bytes"]))
    # BPEByte: generated bytes need training-consistent patch boundaries, which the
    # cached decode path cannot provide -> the repo's online loop (re-prefill per byte)
    bpe = is_bpe(generator)
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        win = max(r["window"] for r in chunk)
        if bpe:
            gens = {}
            generator.max_gen_len = max(1, (win + 1) // 2)   # setter x2 -> cache pad ~= win
            for sub in _bt_chunks(generator, chunk, max_tokens):
                toks = bt_generate(generator, [pb for _, pb in sub], win)
                for (r, pb), row in zip(sub, toks):
                    gb = [b for b in row[1 + len(pb):] if 0 <= b < 256]
                    gens[r["id"]] = bytes(gb).decode("utf-8", errors="replace")
            gens = [gens[r["id"]] for r in chunk]
        else:
            generator.max_gen_len = win
            gens, _, _ = generator.generate([r["prompt"] for r in chunk])
        out = []
        for r, g in zip(chunk, gens):
            s, w = score_gen(r, g)
            out.append({"score": s, "gen": w})
        yield chunk, out


def run_rank(generator, tokenizer, rows, batch):
    generator.max_gen_len = 1
    pairs = [(r, j) for r in rows for j in range(len(r["options"]))]
    pairs.sort(key=lambda p: p[0]["prompt_bytes"])
    lls = {}
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        _, ll, _ = generator.generate([r["prompt"] + r["options"][j] for r, j in chunk])
        for (r, j), l in zip(chunk, ll):
            p = len(tokenizer.encode(r["prompt"], add_bos=False, add_eos=False))
            lls[(r["id"], j)] = l[p:].sum().item()
    # emit in row order once all options of a row are scored
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        out = []
        for r in chunk:
            v = [lls[(r["id"], j)] for j in range(len(r["options"]))]
            pred = max(range(len(v)), key=v.__getitem__)
            out.append({"score": float(pred == r["gold"]), "pred": pred, "lls": v})
        yield chunk, out


def run_gen_ext(harness, rows, batch, family):
    """BLT / H-Net greedy bytes (no stop strings, `window` bytes). BLT: the eval_suite harness's
    re-forward loop at batch size 1. H-Net: its own inference cache (ext_models.hnet_generate)."""
    import ext_models
    rows = sorted(rows, key=lambda r: (r["window"], r["prompt_bytes"]))
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        if family == "hnet":
            gens = [ext_models.hnet_generate(harness.model, r["prompt"], r["window"]) for r in chunk]
        else:
            gens = harness._generate_batch([r["prompt"] for r in chunk], [], max(r["window"] for r in chunk))
        out = []
        for r, g in zip(chunk, gens):
            s, w = score_gen(r, g)
            out.append({"score": s, "gen": w})
        yield chunk, out


def run_rank_ext(harness, rows, batch):
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        pairs = [(r["prompt"], o) for r in chunk for o in r["options"]]
        lps = [lp for lp, _ in harness.score_pairs(pairs)]
        out, k = [], 0
        for r in chunk:
            v = lps[k:k + len(r["options"])]
            k += len(r["options"])
            pred = max(range(len(v)), key=v.__getitem__)
            out.append({"score": float(pred == r["gold"]), "pred": pred, "lls": v})
        yield chunk, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["subword", "aunet", "blt", "hnet"], required=True,
                    help="blt: --ckpt = weights dir (blt_1b/, entropy/); hnet: --ckpt = model name")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=32768,
                    help="KV-cache budget per packed batch (tokens for subword, bytes for aunet)")
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these task names")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N rows per (task,cond)")
    ap.add_argument("--shard", default=None, help="i/n: keep rows whose line index %% n == i")
    ap.add_argument("--max_bytes", type=int, default=None,
                    help="skip rows whose BOS + prompt + generation/option exceeds this many bytes "
                         "(BLT-1B: its byte-level RoPE table covers 4096 positions)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.max_bytes:
        def need(r):
            tail = r["window"] if r["mode"] == "gen" else max(len(o.encode()) for o in r["options"])
            return 1 + r["prompt_bytes"] + tail
        n0 = len(rows)
        rows = [r for r in rows if need(r) <= args.max_bytes]
        print(f"--max_bytes {args.max_bytes}: kept {len(rows)}/{n0} rows", flush=True)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        rows = rows[i::n]
    if args.only:
        rows = [r for r in rows if r["task"] in args.only]
    if args.limit:
        seen, keep = {}, []
        for r in rows:
            k = (r["task"], r["cond"])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= args.limit:
                keep.append(r)
        rows = keep
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out) if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{args.tag}] {args.data}: {len(rows)} rows, {len(done)} done, {len(todo)} todo",
          flush=True)
    if not todo:
        return

    ext = args.family in ("blt", "hnet")
    if ext:
        import ext_models
        harness = (ext_models.build_blt(args.ckpt) if args.family == "blt"
                   else ext_models.build_hnet(args.ckpt, batch_size=args.batch_size))
    else:
        generator, tokenizer = build_generator(args.family, args.ckpt, args.tok_path, args.max_tokens)
    t0, n = time.time(), 0
    for mode in ("gen", "rank"):
        sub = [r for r in todo if r["mode"] == mode]
        if not sub:
            continue
        if ext:
            it = (run_gen_ext(harness, sub, args.batch_size, args.family) if mode == "gen"
                  else run_rank_ext(harness, sub, args.batch_size))
        else:
            it = (run_gen(generator, sub, args.family, args.batch_size, args.max_tokens) if mode == "gen"
                  else run_rank(generator, tokenizer, sub, args.batch_size))
        for chunk, res in it:
            with open(out, "a") as f:
                for r, o in zip(chunk, res):
                    rec = {k: r[k] for k in ("id", "task", "cond", "length", "pos",
                                             "prompt_bytes", "mode")}
                    rec.update(tag=args.tag, family=args.family, **o)
                    if "answers" in r:
                        rec["answers"] = r["answers"]
                    if "gold" in r:
                        rec["gold"] = r["gold"]
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += len(chunk)
            if n % (args.batch_size * 10) < len(chunk):
                el = time.time() - t0
                print(f"[{args.tag}] {mode} {n}/{len(todo)}  {el / 60:.1f} min", flush=True)
    print(f"[{args.tag}] DONE {n} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
