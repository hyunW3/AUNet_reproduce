"""Shared bookkeeping for the per-benchmark patch-length measurements.

Convention (same as compression_stability.py / tab:zh_stats): a parser yields patch START byte
indices over the raw text (BOS excluded); patch k spans [start_k, start_{k+1}), so patches tile
the text exactly and bytes/patch = total_bytes / total_patches.

Every model script writes reports/patch_stats/hist_<model>.json with one entry per
"<group>/<bench>": item count, bytes, patches, and the full patch-length histogram — enough to
derive mean/std/percentiles without storing per-item lengths.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/mnt/ssd2/hyun2/AUNet")
OUT_DIR = ROOT / "reports/patch_stats"
TEXTS = OUT_DIR / "texts.jsonl"


def load_texts(groups=None):
    rows = [json.loads(l) for l in open(TEXTS)]
    return [r for r in rows if groups is None or r["group"] in groups]


def lengths_from_starts(starts, n_bytes):
    starts = sorted(set(s for s in starts if 0 <= s < n_bytes) | {0})
    return [e - s for s, e in zip(starts, starts[1:] + [n_bytes])]


class Hist:
    def __init__(self):
        self.d = defaultdict(lambda: {"n_items": 0, "total_bytes": 0, "n_patches": 0, "hist": Counter()})

    def add(self, row, lens):
        nb = len(row["text"].encode("utf-8"))
        if sum(lens) != nb or min(lens) < 1:
            raise AssertionError(f"{row['group']}/{row['bench']}/{row['key']}: patches do not tile "
                                 f"the text ({sum(lens)} vs {nb} bytes, min {min(lens)})")
        e = self.d[f"{row['group']}/{row['bench']}"]
        e["n_items"] += 1
        e["total_bytes"] += nb
        e["n_patches"] += len(lens)
        e["hist"].update(lens)

    def save(self, model, meta=None):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = {"model": model, "meta": meta or {},
               "benches": {k: {**v, "hist": {str(L): c for L, c in sorted(v["hist"].items())}}
                           for k, v in sorted(self.d.items())}}
        p = OUT_DIR / f"hist_{model}.json"
        p.write_text(json.dumps(out, indent=1))
        return p

    def bpp(self, key):
        e = self.d[key]
        return e["total_bytes"] / e["n_patches"]


# Gate: each implementation must reproduce the paper-time measurement on the DCLM windows
# (reports/bpb_windows.jsonl) before any benchmark number is trusted.
GATE_BPP = {
    "llama": 4.55,      # compression_stability.py en (tiktoken)
    "aunet": 4.89,      # compression_stability.py en (RegexPool word1)
    "bpebyte": 4.58,    # compression_stability.py en (OnlineBPE greedy/root)
    "blt": 1228802 / 1067826,   # reports/latency_realtext/blt_windows_patchstats.json (bytelatent)
    "hnet": 1228802 / 290654,   # reports/latency_realtext/hnet_windows_stats.json
}
GATE_REL_TOL = 0.01


def check_gate(model, h):
    got, ref = h.bpp("dclm/dclm"), GATE_BPP[model]
    ok = abs(got / ref - 1) <= GATE_REL_TOL
    print(f"[gate] {model}: DCLM bytes/patch {got:.4f} vs paper-time {ref:.4f} "
          f"({100 * (got / ref - 1):+.2f}%) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok
