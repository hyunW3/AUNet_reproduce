#!/usr/bin/env python3
"""How much of a given Unicode script does a BPE vocabulary actually cover?

Motivates the Telugu arm of the compression-stability sweep (plans/plan_compression_stability.md):
Llama-3's 128k vocab contains 4,387 entries with complete CJK characters but ZERO with a
complete Telugu character, so OnlineBPE has no lexical prior to walk on Telugu and the
subword tokenizer falls back to (roughly) bytes.

Two measurements per script:
  * vocab coverage  -- how many vocab entries decode to text containing that script
  * fertility       -- bytes/token on real held-out text, the downstream consequence

Usage:
    python scripts/probes/vocab_script_coverage.py
    python scripts/probes/vocab_script_coverage.py --tokenizer tokenizer/bloom/tokenizer.json
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOK = ROOT / "tokenizer/llama3/tokenizer.model"

# Llama-3 / GPT-4 pretokenizer pattern (tiktoken cl100k-style, as used by lingua.tokenizer)
LLAMA3_PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

SCRIPTS = {
    "Telugu":     (0x0C00, 0x0C7F),
    "Devanagari": (0x0900, 0x097F),
    "Hangul":     (0xAC00, 0xD7AF),
    "CJK":        (0x4E00, 0x9FFF),
    "Cyrillic":   (0x0400, 0x04FF),
    "Arabic":     (0x0600, 0x06FF),
    "Latin":      (0x0041, 0x007A),
}

CORPORA = {
    "English (DCLM)":   ROOT / "data/dclm_baseline_1.0_2shards_shuffled/dclm_baseline_1.0_2shards.val.jsonl",
    "Chinese (wiki-zh)": ROOT / "data/zh_bpb/wiki_zh/wiki.val.jsonl",
    "Telugu (wiki-te)":  ROOT / "data/te_bpb/wiki_te/wiki.val.jsonl",
    "Code (py)":         ROOT / "data/code_bpb/py/code.val.jsonl",
}


def load_vocab_bytes(path: Path) -> list[bytes]:
    """Return every vocab entry as its raw byte string.

    tiktoken `.model` files are "<base64 token> <rank>" per line.
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64 = line.split()[0]
            out.append(base64.b64decode(b64))
    return out


def coverage(toks: list[bytes]) -> dict[str, tuple[int, int]]:
    """Per script: (entries whose bytes could belong to it, entries with a COMPLETE character).

    The distinction matters: a vocab can hold fragments of a multi-byte character's UTF-8
    encoding without ever holding the character itself, which is exactly Telugu in Llama-3.
    """
    lead_bytes = {name: {chr(cp).encode()[0] for cp in (lo, hi)}
                  for name, (lo, hi) in SCRIPTS.items()}
    stats = {name: [0, 0] for name in SCRIPTS}
    for t in toks:
        try:
            s = t.decode("utf-8")
        except UnicodeDecodeError:
            s = None
        for name, (lo, hi) in SCRIPTS.items():
            if s is not None:
                if any(lo <= ord(c) <= hi for c in s):
                    stats[name][0] += 1
                    stats[name][1] += 1
            elif t and t[0] in lead_bytes[name]:
                stats[name][0] += 1
    return {k: (v[0], v[1]) for k, v in stats.items()}


def build_encoder(path: Path):
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    return tiktoken.Encoding(name="vocab", pat_str=LLAMA3_PAT,
                             mergeable_ranks=load_tiktoken_bpe(str(path)),
                             special_tokens={})


def fertility(enc, path: Path, max_docs: int) -> tuple[int, int, int]:
    nb = nt = nd = 0
    with open(path) as f:
        for line in f:
            if nd >= max_docs:
                break
            d = json.loads(line)
            txt = d.get("text") or d.get("content") or ""
            if not txt:
                continue
            nb += len(txt.encode())
            nt += len(enc.encode_ordinary(txt))
            nd += 1
    return nd, nb, nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOK))
    ap.add_argument("--max_docs", type=int, default=300)
    ap.add_argument("--out", default=None, help="write JSON summary here")
    args = ap.parse_args()

    tok_path = Path(args.tokenizer)
    toks = load_vocab_bytes(tok_path)
    print(f"vocab: {tok_path}  ({len(toks)} entries)\n")

    cov = coverage(toks)
    print(f"{'script':<12} {'entries w/ script':>18} {'complete-char entries':>22}")
    print("-" * 56)
    for name in SCRIPTS:
        n_any, n_complete = cov[name]
        print(f"{name:<12} {n_any:>18} {n_complete:>22}")

    enc = build_encoder(tok_path)
    print()
    print(f"{'corpus':<20} {'docs':>6} {'bytes':>10} {'tokens':>10} {'bytes/token':>12}")
    print("-" * 62)
    fert = {}
    for name, path in CORPORA.items():
        if not path.exists():
            print(f"{name:<20} {'MISSING':>6}  ({path})")
            continue
        nd, nb, nt = fertility(enc, path, args.max_docs)
        fert[name] = nb / max(nt, 1)
        print(f"{name:<20} {nd:>6} {nb:>10} {nt:>10} {nb/max(nt,1):>12.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"tokenizer": str(tok_path), "n_entries": len(toks),
                   "coverage": {k: {"any": v[0], "complete": v[1]} for k, v in cov.items()},
                   "bytes_per_token": fert}, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
