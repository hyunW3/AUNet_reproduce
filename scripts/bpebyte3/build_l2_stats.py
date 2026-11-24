#!/usr/bin/env python3
"""Distil the `deadend` level-2 segmentation into a ZERO-LAG causal close table.

`superword_rule: deadend` decides where a super-patch ends by testing whether the NEXT level-1
patch can be followed in the superword trie -- so the boundary is only known after reading into
that patch (measured lag up to 8 bytes, vs 0 for the static2 control). That extra lag is a second
variable in the rgbpe2-vs-rgl3 comparison.

This script removes it. It runs the deadend segmentation over a corpus and, at every level-1
boundary, records the super-patch prefix walked so far together with whether deadend closed there.
The resulting P(close | prefix) table lets `superword_rule: stats` decide close/continue from
already-closed patches alone. Same trick `ByteTrie.root_stats_tokenize_boundaries` uses to make
level-1 `bt` leak-free.

  python3 scripts/bpebyte3/build_l2_stats.py \
      --vocab tokenizer/llama3_l2patch/l2_over_greedy_128k.jsonl \
      --out   tokenizer/llama3_l2patch/l2_close_stats_128k.json
"""
import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/lingua")
from apps.aunet.data.regex_cutting import RegexArgs, RegexPool  # noqa: E402

DEFAULT_SRC = "/mnt/ssd2/hyun2/aunet_r10lr/data/dclm_r10lr_2chunk/dclm_baseline_1.0_2shards.chunk.00.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--tokenizer", default="/mnt/ssd2/hyun2/AUNet/tokenizer/llama3/tokenizer.model")
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--fallback-k", type=int, default=2)
    ap.add_argument("--punct-barrier", action="store_true")
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--min-visits", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = RegexArgs(strategy={"bpe_br": "1@1"})
    a.bpe_tokenizer_path = args.tokenizer
    a.bpe_online = True
    a.bpe_online_mode = "greedy"
    a.bpe_online_placement = "root"
    a.bpe_context_prefix = 0
    a.online_level2 = "superword"
    a.superword_vocab_path = args.vocab
    a.superword_rule = "deadend"
    a.superword_fallback_k = args.fallback_k
    a.superword_punct_barrier = args.punct_barrier
    rp = RegexPool(a)
    trie1 = rp.byte_trie
    root2 = rp.superword_byte_trie.root

    # prefix bytes -> [closes, visits]
    tab = defaultdict(lambda: [0, 0])
    got = 0
    ndocs = 0
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            if got >= args.mb * 1024 * 1024:
                break
            try:
                t = json.loads(line).get("text") or ""
            except Exception:
                continue
            if not t:
                continue
            bs = t.encode()
            got += len(bs)
            ndocs += 1
            n = len(bs)
            ends = [e for e in trie1.greedy_tokenize_boundaries(bs) if 0 < e <= n]
            if not ends or ends[-1] != n:
                ends = ends + [n]
            starts = [0] + ends[:-1]
            ntok = len(ends)

            p = 0
            while p < ntok:
                # the deadend teacher; clamp because the static fallback can overrun the tail
                k = min(rp._superword_ktok_at(bs, starts, ends, p), ntok - p)
                # replay the group, recording the causal state at each internal boundary
                node, path = root2, bytearray()
                for j in range(k):
                    q = p + j
                    walk = node
                    for i in range(starts[q], ends[q]):
                        walk = walk.children.get(int(bs[i]))
                        if walk is None:
                            break
                    if walk is None:
                        node, path = root2, bytearray()
                    else:
                        node = walk
                        path += bs[starts[q]:ends[q]]
                    key = bytes(path)
                    tab[key][1] += 1
                    if j == k - 1:                 # deadend closed the group right here
                        tab[key][0] += 1
                p += k

    stats = {k.hex(): v for k, v in tab.items() if v[1] >= args.min_visits}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"stats": stats,
                   "meta": {"src": args.src, "vocab": args.vocab, "mb": args.mb,
                            "fallback_k": args.fallback_k, "punct_barrier": args.punct_barrier,
                            "max_k": args.max_k, "min_visits": args.min_visits,
                            "docs": ndocs}}, f)
    kept = len(stats)
    print(f"{ndocs:,} docs / {got/1e6:.1f} MB -> {len(tab):,} distinct prefixes, "
          f"{kept:,} kept (>= {args.min_visits} visits) -> {args.out}")
    if kept:
        import statistics
        ps = [c / v for c, v in (tab[bytes.fromhex(k)] for k in stats)]
        print(f"  P(close) median {statistics.median(ps):.3f} · "
              f"share >=0.5 {sum(p >= 0.5 for p in ps)/len(ps)*100:.1f}%")


if __name__ == "__main__":
    main()
