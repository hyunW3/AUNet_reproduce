#!/usr/bin/env python3
"""BPEByte3 level-2 vocabulary: BPE learned over the OnlineBPE PATCH stream.

Why not the existing SuperBPE vocab (`tokenizer/llama3_superbpe/`)? That one merges pairs of
*llama3 tokens*. Our level-1 units are OnlineBPE greedy patches -- trie dead-end spans, i.e. a
vocab PREFIX plus the dead-end byte -- which are NOT llama3 tokens, so those superwords align with
level-1 boundaries only by luck, and `RegexPool._superword_ktok_at` silently falls back to k=1
whenever they do not. Level 2 has to be learned over the actual level-1 stream.

Nesting is constructive: every learned unit is a concatenation of whole level-1 patches, so a
level-2 boundary is always a level-1 boundary (what `get_pool_mask`'s `level_mask > i` requires).
Causality is preserved: level 1 is causal, and level 2 is a function of level-1 output only.
No whitespace anywhere -> applies to Chinese / DNA / code unchanged.

Output is the SAME jsonl schema the existing level-2 path already consumes
(`superword_vocab_path`, field "bytes"), so no model-side code change is needed:

    data.regex.online_level2: superword
    data.regex.superword_vocab_path: <this file>

Usage:
  python3 scripts/bpebyte3/build_l2_vocab.py --mb 32 --merges 8192 \
      --out tokenizer/llama3_l2patch/l2_over_greedy.jsonl
"""
import argparse
import json
import os
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/lingua")

DEFAULT_SRC = "/mnt/ssd2/hyun2/aunet_r10lr/data/dclm_r10lr_2chunk/dclm_baseline_1.0_2shards.chunk.00.jsonl"


def load_trie(tokenizer_path):
    from apps.aunet.data.byte_trie import ByteTrie
    import base64
    trie = ByteTrie()
    n = 0
    with open(tokenizer_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                bs = base64.b64decode(parts[0])
            except Exception:
                continue
            if bs:
                trie.insert(bs)
                n += 1
    for b in range(256):
        trie.insert(bytes([b]))
    return trie, n


def read_docs(src, max_bytes):
    got = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                t = json.loads(line).get("text") or ""
            except Exception:
                continue
            if not t:
                continue
            yield t
            got += len(t.encode())
            if got >= max_bytes:
                return


def patch_stream(trie, texts):
    """Each document -> list of level-1 patch byte-strings, split at newlines.

    Level-2 units are not allowed to cross a line break: a newline patch is emitted as its own
    segment boundary. This keeps segments short (so the BPE trainer's per-word cost stays low) and
    matches the intuition that a super-patch should not straddle a document's structural break.
    """
    for t in texts:
        bs = t.encode()
        n = len(bs)
        ends = [e for e in trie.greedy_tokenize_boundaries(bs) if 0 < e <= n]
        seg, start = [], 0
        for e in ends:
            p = bs[start:e]
            start = e
            if b"\n" in p:
                if seg:
                    yield seg
                seg = []
                continue
            seg.append(p)
        if seg:
            yield seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--tokenizer", default="/mnt/ssd2/hyun2/AUNet/tokenizer/llama3/tokenizer.model")
    ap.add_argument("--mb", type=int, default=32, help="corpus MB to fit on")
    ap.add_argument("--merges", type=int, default=8192, help="number of level-2 merges to learn")
    ap.add_argument("--min-freq", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trie, ntok = load_trie(args.tokenizer)
    print(f"level-1 trie: {ntok} entries")

    segs = list(patch_stream(trie, read_docs(args.src, args.mb * 1024 * 1024)))
    npatch = sum(len(s) for s in segs)
    vocab1 = Counter(p for s in segs for p in s)
    print(f"corpus: {args.mb} MB -> {len(segs):,} segments, {npatch:,} level-1 patches, "
          f"{len(vocab1):,} distinct")

    # --- remap each distinct level-1 patch to a single codepoint so the Rust BPE trainer can run
    # over it. Skip surrogates and anything the tokenizers' NFC-free path would fold.
    def codepoints(k):
        cp, out = 0x4E00, []          # start well past ASCII/control
        while len(out) < k:
            if not (0xD800 <= cp <= 0xDFFF) and unicodedata.category(chr(cp)) != "Cs":
                out.append(cp)
            cp += 1
        return out

    order = [p for p, _ in vocab1.most_common()]
    cps = codepoints(len(order))
    p2c = {p: chr(c) for p, c in zip(order, cps)}
    c2p = {v: k for k, v in p2c.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".remap.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        for s in segs:
            f.write("".join(p2c[p] for p in s))
            f.write("\n")

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, Regex
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\n"), behavior="removed")
    trainer = trainers.BpeTrainer(
        vocab_size=len(order) + args.merges,
        min_frequency=args.min_freq,
        show_progress=True,
        initial_alphabet=[p2c[p] for p in order],
        special_tokens=[],
    )
    tok.train([tmp], trainer)
    os.remove(tmp)

    # --- emit every learned multi-patch unit as its concatenated byte string
    vocab = tok.get_vocab()
    rows = []
    for s, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
        if len(s) < 2:
            continue                      # single level-1 patch: not a level-2 unit
        if any(ch not in c2p for ch in s):
            continue                      # trainer folded something unexpected -> drop
        pieces = [c2p[ch] for ch in s]
        raw = b"".join(pieces)
        rows.append({
            "bytes": list(raw),
            "text": raw.decode("utf-8", errors="replace"),
            "n_l1_patches": len(pieces),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hist = Counter(r["n_l1_patches"] for r in rows)
    print(f"wrote {len(rows):,} level-2 units -> {args.out}")
    print("  units by #level-1 patches: " + ", ".join(f"{k}:{hist[k]}" for k in sorted(hist)[:8]))


if __name__ == "__main__":
    main()
