#!/usr/bin/env python3
"""Stage a small multilingual+code pilot dataset in lingua JSONL format.
Each source -> <root>/<src>/<src>.chunk.{00..NCHUNK-1}.jsonl  with {"text": ...} lines.
Tokenization is ONLINE in the loader, so no preprocessing here — just clean text shards.
Caps per source by #docs AND total bytes (whichever hits first)."""
import os, sys, json
from datasets import load_dataset

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/hwbae/AUNet/data"
MAX_DOCS = int(os.environ.get("MAX_DOCS", "20000"))
MAX_MB   = float(os.environ.get("MAX_MB", "50"))      # per source
NCHUNK   = int(os.environ.get("NCHUNK", "8"))
MIN_CHARS = 200

SOURCES = [
    ("mlpilot_en",   dict(path="wikimedia/wikipedia", name="20231101.en", split="train", streaming=True), "text"),
    ("mlpilot_fi",   dict(path="wikimedia/wikipedia", name="20231101.fi", split="train", streaming=True), "text"),
    ("mlpilot_zh",   dict(path="wikimedia/wikipedia", name="20231101.zh", split="train", streaming=True), "text"),
    ("mlpilot_code", dict(path="codeparrot/codeparrot-clean-valid", split="train", streaming=True), "content"),
]

for src, kw, field in SOURCES:
    d = os.path.join(ROOT, src)
    os.makedirs(d, exist_ok=True)
    fhs = [open(os.path.join(d, f"{src}.chunk.{i:02d}.jsonl"), "w", encoding="utf-8") for i in range(NCHUNK)]
    ds = load_dataset(**kw)
    n = 0; nbytes = 0; cap = MAX_MB * 1024 * 1024
    for ex in ds:
        t = ex.get(field) or ex.get("text") or ex.get("content")
        if not t or len(t) < MIN_CHARS:
            continue
        line = json.dumps({"text": t}, ensure_ascii=False)
        fhs[n % NCHUNK].write(line + "\n")
        n += 1; nbytes += len(line.encode("utf-8"))
        if n >= MAX_DOCS or nbytes >= cap:
            break
    for fh in fhs: fh.close()
    print(f"{src}: {n} docs, {nbytes/1e6:.1f} MB -> {d}", flush=True)

print("DONE", flush=True)
