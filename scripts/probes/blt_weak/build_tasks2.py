#!/usr/bin/env python3
"""Second batch of BLT-weakness probes (CANDIDATES.md #3, #4, #5) in run_longctx.py format.

  echo    (#3) MC prompt priming. Options untouched; only the context changes:
            clean | echo_all ("Options: (A) .. (B) .." before the question) | echo_gold ("Hint: <gold>")
            | echo_wrong ("Hint: <a wrong choice>") | repeat_q (question stated twice)
  insert  (#5) semantics-preserving byte insertions into the question text (cue + options intact):
            zwsp (U+200B inside 25% of words) | shy (U+00AD likewise) | nbsp (50% of spaces -> U+00A0)
            | emoji (an emoji after 15% of words) | homoglyph (15% of a/e/o/c/p/x/y/i -> Cyrillic)
  ood     (#4) 3000-byte windows from 12 domains, scored as rank rows with prompt "" and the window as
            the single option -> the runner's summed log-prob gives NLL -> BPB. The same windows go to
            ood_windows.jsonl for patch counts / determinism / latency.

MC items come from export_mc.py (lm-eval doc_to_text/choice/target, first 500 docs per task).

  python scripts/probes/blt_weak/build_tasks2.py --mc reports/blt_weak/mc_items.jsonl --out_dir reports/blt_weak/data
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import string
import sys
import zlib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "niah"))
from niah_data import _essay_pool  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = Path("/mnt/ssd2/hyun2/AUNet/data")


def _seed(*parts):
    return zlib.crc32(repr(parts).encode())


def _nb(s):
    return len(s.encode())


def rank_row(rid, task, cond, length, pos, prompt, options, gold):
    return {"id": rid, "task": task, "cond": cond, "length": length, "pos": pos, "mode": "rank",
            "prompt": prompt, "options": options, "gold": gold, "prompt_bytes": _nb(prompt)}


def split_q(it):
    """(question_text, suffix) so perturbations touch the question only: ARC/PIQA end in
    "\\nAnswer:"; HellaSwag's context is a sentence prefix whose LAST word is kept intact."""
    c = it["context"]
    if c.endswith("\nAnswer:"):
        return c[: -len("\nAnswer:")], "\nAnswer:"
    head, _, last = c.rpartition(" ")
    return head, " " + last


# ------------------------------------------------------------------------------ #3 echo / priming
def echo_rows(items):
    rows = []
    for it in items:
        rng = random.Random(_seed("echo", it["task"], it["idx"]))
        ch, g = it["choices"], it["gold"]
        ctx = it["context"]
        q, suf = split_q(it)
        wrong = rng.choice([c for i, c in enumerate(ch) if i != g])
        letters = "ABCDEFGH"
        variants = {
            "clean": ctx,
            "echo_all": "Options: " + " ".join(f"({letters[i]}) {c}" for i, c in enumerate(ch)) + "\n" + ctx,
            "echo_gold": f"Hint: {ch[g]}\n" + ctx,
            "echo_wrong": f"Hint: {wrong}\n" + ctx,
            "repeat_q": (f"{q}\n{q}{suf}" if suf == "\nAnswer:" else f"{ctx}\n\n{ctx}"),
        }
        for v, p in variants.items():
            rows.append(rank_row(f"echo-{v}-{it['task']}-{it['idx']}", "echo", v, it["task"], it["idx"],
                                 p, [" " + c for c in ch], g))
    return rows


# ------------------------------------------------------------------------------ #5 insertions
_EMOJI = ["🙂", "✨", "🔥", "👍", "🌟", "😀", "🚀", "🎉"]
_HOMO = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "y": "у", "i": "і"}


def _in_word(text, ch, p, rng):
    out = []
    for w in text.split(" "):
        if len(w) >= 2 and rng.random() < p:
            k = rng.randrange(1, len(w))
            w = w[:k] + ch + w[k:]
        out.append(w)
    return " ".join(out)


def perturb(text, kind, rng):
    if kind == "zwsp":
        return _in_word(text, "​", 0.25, rng)
    if kind == "shy":
        return _in_word(text, "­", 0.25, rng)
    if kind == "nbsp":
        return "".join(" " if c == " " and rng.random() < 0.5 else c for c in text)
    if kind == "emoji":
        return " ".join(w + (rng.choice(_EMOJI) if w and rng.random() < 0.15 else "") for w in text.split(" "))
    if kind == "homoglyph":
        return "".join(_HOMO[c] if c in _HOMO and rng.random() < 0.15 else c for c in text)
    raise ValueError(kind)


def insert_rows(items):
    rows = []
    for it in items:
        q, suf = split_q(it)
        pre = "Question: " if q.startswith("Question: ") else ""
        body = q[len(pre):]
        for kind in ("zwsp", "shy", "nbsp", "emoji", "homoglyph"):
            rng = random.Random(_seed("ins", kind, it["task"], it["idx"]))
            p = pre + perturb(body, kind, rng) + suf
            rows.append(rank_row(f"ins-{kind}-{it['task']}-{it['idx']}", "insert", kind, it["task"], it["idx"],
                                 p, [" " + c for c in it["choices"]], it["gold"]))
    return rows


# ------------------------------------------------------------------------------ #4 OOD windows
def _cut(s, n):
    return s.encode()[:n].decode(errors="ignore")


def _jsonl_pool(path, cap=2_000_000):
    out, tot = [], 0
    for line in open(path):
        t = json.loads(line).get("text", "")
        if len(t) > 200:
            out.append(t)
            tot += len(t)
            if tot > cap:
                break
    return "\n\n".join(out)


def ood_windows(n, W=3000):
    pools = {"en": _essay_pool(), "code": _jsonl_pool(DATA / "code_bpb/py/code.val.jsonl"),
             "zh": _jsonl_pool(DATA / "zh_bpb/wiki_zh/wiki.val.jsonl"),
             "te": _jsonl_pool(DATA / "te_bpb/wiki_te/wiki.val.jsonl")}
    out = []
    for dom in ("en", "code", "zh", "te", "random_ascii", "hex", "base64", "uuid", "json_log", "csv",
                "boilerplate", "repeat_line"):
        rng = random.Random(_seed("ood", dom))
        for i in range(n):
            if dom in pools:
                pool = pools[dom]
                s = rng.randrange(0, len(pool) - 3 * W)
                t = pool[s:s + 3 * W]
            elif dom == "random_ascii":
                t = "".join(rng.choice(string.printable[:94] + " ") for _ in range(W))
            elif dom == "hex":
                t = "".join(rng.choice("0123456789abcdef") for _ in range(W))
            elif dom == "base64":
                t = base64.b64encode(bytes(rng.randrange(256) for _ in range(W))).decode()
            elif dom == "uuid":
                h = "0123456789abcdef"
                t = "\n".join("-".join("".join(rng.choice(h) for _ in range(k)) for k in (8, 4, 4, 4, 12))
                              for _ in range(W // 37 + 2))
            elif dom == "json_log":
                lv = ["INFO", "WARN", "DEBUG", "ERROR"]
                t = "\n".join(json.dumps({"ts": f"2026-10-02T{rng.randrange(24):02d}:{rng.randrange(60):02d}:"
                                                f"{rng.randrange(60):02d}.{rng.randrange(1000):03d}Z",
                                          "level": rng.choice(lv), "req": "%08x" % rng.getrandbits(32),
                                          "latency_ms": rng.randrange(1, 900), "path": rng.choice(
                                              ["/api/v1/users", "/api/v1/items", "/health", "/login"])})
                              for _ in range(W // 90 + 2))
            elif dom == "csv":
                t = "id,price,qty,lat,lon\n" + "\n".join(
                    f"{k},{rng.uniform(1, 999):.2f},{rng.randrange(1, 99)},{rng.uniform(-90, 90):.5f},"
                    f"{rng.uniform(-180, 180):.5f}" for k in range(W // 30 + 2))
            elif dom == "boilerplate":
                pool = pools["en"]
                s = rng.randrange(0, len(pool) - 400)
                para = pool[s:s + 300].strip() + "\n"
                t = para * (W // len(para) + 2)
            elif dom == "repeat_line":
                line = f"[{rng.randrange(10**6):06d}] connection reset by peer, retrying in 5s\n"
                t = line * (W // len(line) + 2)
            out.append({"id": f"ood-{dom}-{i}", "domain": dom, "text": _cut(t, W)})
    for w in out:
        w["n_bytes"] = _nb(w["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_ood", type=int, default=40)
    a = ap.parse_args()
    od = Path(a.out_dir)
    items = [json.loads(l) for l in open(a.mc)]
    wins = ood_windows(a.n_ood)
    with open(od / "ood_windows.jsonl", "w") as f:
        for w in wins:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    ood = [rank_row(w["id"], "ood", w["domain"], w["n_bytes"], None, "", [w["text"]], 0) for w in wins]
    for name, rows in (("echo", echo_rows(items)), ("insert", insert_rows(items)), ("ood", ood)):
        assert len({r["id"] for r in rows}) == len(rows)
        big = [r["id"] for r in rows if 1 + r["prompt_bytes"] + max(_nb(o) for o in r["options"]) > 4096]
        assert not big, (name, big[:3])
        with open(od / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:7s} {len(rows):6d} rows  " + " ".join(f"{k}={v}" for k, v in sorted(Counter(r['cond'] for r in rows).items())))


if __name__ == "__main__":
    main()
