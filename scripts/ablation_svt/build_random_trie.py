#!/usr/bin/env python3
"""Random-trie control for the 100M parser ablation: a 128K "vocabulary" of byte n-grams sampled at
UNIFORMLY RANDOM corpus positions (so frequency-weighted, but with no BPE merge statistics and no
word/whitespace alignment), with token lengths drawn from llama3's own byte-length distribution
times `--len_scale`. All 256 single bytes are ranks 0..255, so the online greedy byte-trie is total.

Written as a tiktoken-format .model (base64 token + rank per line) so the existing
`bpe_tokenizer_path` + online greedy root pipeline consumes it unchanged (the trie is built from
decode_single_token_bytes over all ranks; nothing calls BPE encode on the online path).

`--len_scale` is calibrated so greedy-trie bytes/patch on DCLM matches BPEByte root_greedy (4.566).

  python build_random_trie.py --corpus <dclm chunk.jsonl> --out llama3len_random_128k.model --len_scale 1.0
"""
import argparse, base64, itertools, json, random


def greedy_patches(root, b):
    """Mirror of ByteTrie.greedy_tokenize_boundaries (bpe_online_mode=greedy): walk the trie and
    commit EVERY byte read at the dead-end (no backtrack to the last valid token). For llama3 this
    equals longest-match; for a non-prefix-closed random vocab it gives longer patches."""
    cnt = pos = 0
    n = len(b)
    while pos < n:
        node, i = root, pos
        while i < n and b[i] in node:
            node = node[b[i]]
            i += 1
        pos += max(i - pos, 1)
        cnt += 1
    return cnt


def build_trie(tokens):
    root = {}
    for t in tokens:
        nd = root
        for x in t:
            nd = nd.setdefault(x, {})
        nd[-1] = 1
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="DCLM jsonl to sample n-grams from")
    ap.add_argument("--eval_corpus", help="jsonl to measure bytes/patch on (default: --corpus)")
    ap.add_argument("--llama3", default="tokenizer/llama3/tokenizer.model")
    ap.add_argument("--n_vocab", type=int, default=128000)
    ap.add_argument("--len_scale", type=float, default=1.0)
    ap.add_argument("--sample_docs", type=int, default=20000)
    ap.add_argument("--eval_docs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    llama = [base64.b64decode(l.split()[0]) for l in open(args.llama3)]
    lens = [len(t) for t in llama if len(t) > 1]                     # multi-byte length distribution

    docs = [json.loads(l)["text"].encode() for l in itertools.islice(open(args.corpus), args.sample_docs)]
    docs = [d for d in docs if len(d) >= 64]
    cum, tot = [], 0
    for d in docs:
        tot += len(d)
        cum.append(tot)

    import bisect
    vocab = {bytes([b]) for b in range(256)}
    singles = [bytes([b]) for b in range(256)]
    multi = []
    while len(vocab) < args.n_vocab:
        L = max(2, round(rng.choice(lens) * args.len_scale))
        r = rng.randrange(tot)                                         # uniform over corpus bytes
        di = bisect.bisect_right(cum, r)
        d = docs[di]
        off = r - (cum[di] - len(d))
        if off + L > len(d):
            continue
        t = d[off:off + L]
        if t not in vocab:
            vocab.add(t)
            multi.append(t)
    tokens = singles + multi

    ev = docs if not args.eval_corpus else [
        json.loads(l)["text"].encode() for l in itertools.islice(open(args.eval_corpus), args.eval_docs)]
    ev = ev[:args.eval_docs]
    root = build_trie(tokens)
    nb = sum(map(len, ev))
    npatch = sum(greedy_patches(root, b) for b in ev)
    mean_len = sum(map(len, multi)) / len(multi)
    print(f"len_scale={args.len_scale} vocab={len(tokens)} mean_tok_len={mean_len:.2f} "
          f"greedy bytes/patch={nb / npatch:.3f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            for rank, t in enumerate(tokens):
                f.write(f"{base64.b64encode(t).decode()} {rank}\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
