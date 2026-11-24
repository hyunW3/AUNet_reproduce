#!/usr/bin/env python3
"""Build fixed held-out windows for the compression-stability sweep.

Reproduces the protocol behind reports/bpb_windows.jsonl (English, 320 windows) and
reports/bpb_windows_zh.jsonl (Chinese, 241 windows): concatenate held-out documents into a
stream, cut non-overlapping <=3840-byte windows, snap each cut back to a UTF-8 character
boundary. Schema matches those files exactly: {"idx", "n_bytes", "text"}.

Domains (plans/plan_compression_stability.md Phase 1):
    en        reused as-is from reports/bpb_windows.jsonl
    zh        reused as-is from reports/bpb_windows_zh.jsonl
    te        Telugu wiki  -- vocab-coverage-adversarial (Llama-3 has 0 complete Telugu chars)
    code      Python + 8-language code
    despace_p{00,25,50,75,100}  English with intra-line spaces dropped w.p. p
    random    uniform random bytes -- the degenerate limit for learned chunkers

Usage:
    python scripts/probes/build_windows.py --all
    python scripts/probes/build_windows.py --domain te
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/windows"
WINDOW_BYTES = 3840
MIN_WINDOWS = 120        # below this a domain is too small to be comparable; warn

# Reuse the paper's despace rule rather than reimplementing it.
sys.path.insert(0, str(ROOT / "lingua"))
from apps.aunet.eval_despace_mc import _despace, _despace_prob  # noqa: E402

CORPORA = {
    "te":   [ROOT / "data/te_bpb/wiki_te/wiki.val.jsonl"],
    "code": [ROOT / "data/code_bpb/py/code.val.jsonl"]
            + [ROOT / f"data/code_multi/{lang}/{lang}.val.jsonl"
               for lang in ("c", "cpp", "go", "java", "js", "py", "ruby", "rust")],
}
EXISTING = {
    "en": ROOT / "reports/bpb_windows.jsonl",
    "zh": ROOT / "reports/bpb_windows_zh.jsonl",
}
DESPACE_PS = [0.0, 0.25, 0.50, 0.75, 1.0]


def read_docs(paths: list[Path]) -> list[str]:
    docs = []
    for p in paths:
        if not p.exists():
            print(f"  ! missing, skipped: {p}", file=sys.stderr)
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("text") or d.get("content") or ""
                if t:
                    docs.append(t)
    return docs


def cut_windows(stream: str, window_bytes: int = WINDOW_BYTES) -> list[str]:
    """Non-overlapping <=window_bytes windows, each ending on a UTF-8 character boundary."""
    raw = stream.encode("utf-8")
    wins, off = [], 0
    while off + window_bytes <= len(raw):
        chunk = raw[off:off + window_bytes]
        # snap back to the last complete character
        while chunk:
            try:
                text = chunk.decode("utf-8")
                break
            except UnicodeDecodeError:
                chunk = chunk[:-1]
        else:
            break
        wins.append(text)
        off += len(chunk)
    return wins


def write(name: str, texts: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with open(path, "w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"idx": i, "n_bytes": len(t.encode()), "text": t},
                               ensure_ascii=False) + "\n")
    total = sum(len(t.encode()) for t in texts)
    flag = "  <-- SMALL" if len(texts) < MIN_WINDOWS else ""
    print(f"  {name:<14} {len(texts):>4} windows  {total/1e6:.2f} MB  -> {path}{flag}")
    return path


def build_text_domain(name: str) -> list[str]:
    docs = read_docs(CORPORA[name])
    return cut_windows("\n".join(docs))


def build_despace(p: float) -> list[str]:
    """Apply the paper's despace rule to the ENGLISH windows, then re-cut.

    Re-cutting matters: removing spaces shortens the text, so keeping the original window
    boundaries would compare unequal byte counts across p.
    """
    en = [json.loads(l)["text"] for l in open(EXISTING["en"])]
    rng = random.Random(777)
    if p >= 1.0:
        despaced = [_despace(t) for t in en]
    else:
        despaced = [_despace_prob(t, p, rng) for t in en]
    return cut_windows("\n".join(despaced))


def build_random(n_windows: int = 240) -> list[str]:
    """Uniform random PRINTABLE ASCII (0x20-0x7E) -- maximally unstructured text.

    Deliberately not arbitrary bytes: the subword arm encodes str, so invalid UTF-8 would
    make the Llama baseline unmeasurable and drop one of the five parsers from the sweep.
    Printable ASCII keeps every arm comparable, stays 1 byte per character (so windows are
    byte-exact), and still carries no learnable structure -- the degenerate case for the
    learned chunkers.
    """
    rng = random.Random(777)
    return ["".join(chr(rng.randrange(0x20, 0x7F)) for _ in range(WINDOW_BYTES))
            for _ in range(n_windows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", action="append", default=None,
                    help="te | code | despace | random (repeatable)")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    todo = args.domain or []
    if args.all or not todo:
        todo = ["te", "code", "despace", "random"]

    print("building windows (protocol: non-overlapping, <=3840B, char-aligned)")
    for name in EXISTING:
        n = sum(1 for _ in open(EXISTING[name]))
        print(f"  {name:<14} {n:>4} windows  (reused, {EXISTING[name]})")

    for d in todo:
        if d in CORPORA:
            write(d, build_text_domain(d))
        elif d == "despace":
            for p in DESPACE_PS:
                write(f"despace_p{int(p*100):03d}", build_despace(p))
        elif d == "random":
            write("random", build_random())
        else:
            print(f"  ! unknown domain: {d}", file=sys.stderr)


if __name__ == "__main__":
    main()
