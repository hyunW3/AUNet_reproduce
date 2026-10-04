#!/usr/bin/env python3
"""Build the long-context probe suite as prompt JSONL (model-free, deterministic).

Four task families, in the order they are run:
  needle_types : S-NIAH on a DCLM essay haystack with 10 needle value types
                 (num7 / num20 / uuid / hex32 / alnum12 / ident / nonce / word /
                 han4 / hangul4) -- which strings survive a tokenizer / patcher.
  kv_litm      : Lost-in-the-Middle key-value retrieval (Liu et al. 2023): a JSON
                 object of k UUID->UUID pairs, gold pair at a swept position.
  ruler_agg    : RULER Variable Tracking (length sweep), Common Words Extraction,
                 Frequent Words Extraction (Hsieh et al. 2024), each with RULER's
                 one-shot in-context example.
  icl          : many-shot in-context classification (SST-2, TREC-coarse, AG News
                 by label log-likelihood; TREC-fine by generation), shots doubled
                 until the prompt no longer fits the byte budget.

Every row is one of two modes:
  gen  : greedy-generate after `prompt`; score the first `window` bytes of the
         generation against `answers` (substring / recall / exact-first-line).
  rank : score log P(option | prompt) for every option; argmax vs `gold`.
Scoring is the same for all three families (no teacher-forced format cut), so a
subword model is not penalized for choosing its own separator.

Prompts are capped at MAX_PROMPT_BYTES so the byte models (8192-byte encoder
window) never truncate them; the subword model (4096 tokens) never does either.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "niah"))
from niah_data import _essay_pool, NOISE, KEYS, _MAGIC_WORDS, _CONS, _VOW  # noqa: E402

MAX_PROMPT_BYTES = 7400


def nbytes(s: str) -> int:
    return len(s.encode())


# --------------------------------------------------------------------------- #
# 1. needle value types
# --------------------------------------------------------------------------- #
_HEX = "0123456789abcdef"
_B62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_IDENT_PARTS = ["parse", "fetch", "build", "render", "resolve", "flush", "merge", "encode",
                "Widget", "Cache", "Buffer", "Token", "Node", "Queue", "Handler", "Index",
                "Router", "Stream", "Config", "Batch", "Shard", "Vector", "Frame", "Socket"]
# frequent CJK characters / Hangul syllables: in-vocabulary for every tokenizer,
# 3 UTF-8 bytes each, so a 4-char value is 12 bytes
_HAN = list("的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长")
_HANGUL = list("가나다라마바사아자차카타파하고노도로모보소오조호구누두루무부수우주후기니디리미비시이지히강남동산성정민준서연지영수현우진")
VALUE_TYPES = ["num7", "num20", "uuid", "hex32", "alnum12", "ident", "nonce", "word",
               "han4", "hangul4"]


def _nonce(rng, avoid):
    for _ in range(100):
        w = "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(rng.randint(3, 4)))
        w += rng.choice(_CONS)
        if w not in avoid:
            return w
    return w


def make_value(vt: str, rng: random.Random, avoid: str) -> str:
    if vt == "num7":
        return "".join(rng.choice("0123456789") for _ in range(7))
    if vt == "num20":
        return rng.choice("123456789") + "".join(rng.choice("0123456789") for _ in range(19))
    if vt == "uuid":
        return "-".join("".join(rng.choice(_HEX) for _ in range(n)) for n in (8, 4, 4, 4, 12))
    if vt == "hex32":
        return "".join(rng.choice(_HEX) for _ in range(32))
    if vt == "alnum12":
        return "".join(rng.choice(_B62) for _ in range(12))
    if vt == "ident":
        parts = rng.sample(_IDENT_PARTS[:8], 1) + rng.sample(_IDENT_PARTS[8:], 2)
        return "".join(parts) + str(rng.randint(10, 99))
    if vt == "nonce":
        return _nonce(rng, avoid.lower())
    if vt == "word":
        cands = [w for w in _MAGIC_WORDS if w not in avoid.lower()]
        return rng.choice(cands) if cands else _nonce(rng, avoid.lower())
    if vt == "han4":
        return "".join(rng.choice(_HAN) for _ in range(4))
    if vt == "hangul4":
        return "".join(rng.choice(_HANGUL) for _ in range(4))
    raise ValueError(vt)


NT_INSTR = ("Some special magic values are hidden within the following text. Make sure to "
            "memorize them. I will quiz you about the values afterwards.")


def build_needle_types(lengths, depths, n_per_cell, seed, byte_budget=False):
    pool = _essay_pool()
    rows = []
    for vt in VALUE_TYPES:
        rng = random.Random(f"{seed}-nt-{vt}")
        for tb in lengths:
            for d in depths:
                for i in range(n_per_cell):
                    key = rng.choice(KEYS)
                    start = rng.randint(0, len(pool) - tb - 1)
                    body = pool[start:start + tb]
                    if byte_budget:      # haystack of <= tb BYTES (multi-byte text cannot overshoot)
                        body = body.encode()[:tb].decode("utf-8", "ignore")
                    value = make_value(vt, rng, body + NT_INSTR + " ".join(KEYS))
                    needle = f"One of the special magic values for {key} is {value}. "
                    cut = int(len(body) * d)
                    while 0 < cut < len(body) and body[cut] != " ":
                        cut += 1
                    cut = min(cut + 1, len(body))   # needle starts after the space
                    hay = body[:cut] + needle + body[cut:]
                    q = (f"What is the special magic value for {key} mentioned in the provided "
                         f"text? The special magic value for {key} is")
                    prompt = f"{NT_INSTR}\n\n{hay}\n\n{q}"
                    rows.append({"id": f"nt-{vt}-{tb}-{d}-{i}", "task": "needle_types",
                                 "cond": vt, "length": tb, "pos": d, "mode": "gen",
                                 "score": "substr", "prompt": prompt, "answers": [value],
                                 "window": nbytes(value) + 24})
    return rows


# --------------------------------------------------------------------------- #
# 2. Lost-in-the-Middle KV retrieval
# --------------------------------------------------------------------------- #
def _uuid(rng):
    return "-".join("".join(rng.choice(_HEX) for _ in range(n)) for n in (8, 4, 4, 4, 12))


def build_kv(ks, positions, n_per_cell, seed):
    rows = []
    rng = random.Random(f"{seed}-kv")
    for k in ks:
        for pf in positions:
            gi = round((k - 1) * pf)
            for i in range(n_per_cell):
                kv = [(_uuid(rng), _uuid(rng)) for _ in range(k)]
                lines = ",\n ".join(f'"{a}": "{b}"' for a, b in kv)
                qk, qv = kv[gi]
                prompt = ("Extract the value corresponding to the specified key in the JSON "
                          "object below.\n\nJSON data:\n{" + lines + "}\n\n"
                          f'Key: "{qk}"\nCorresponding value:')
                rows.append({"id": f"kv-{k}-{pf}-{i}", "task": "kv_litm", "cond": f"k{k}",
                             "length": k, "pos": pf, "mode": "gen", "score": "substr",
                             "prompt": prompt, "answers": [qv], "window": nbytes(qv) + 24})
    return rows


# --------------------------------------------------------------------------- #
# 3. RULER aggregation-style tasks: VT / CWE / FWE
# --------------------------------------------------------------------------- #
_UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _vt_instance(rng, target_bytes, num_chains, num_hops):
    """RULER VT: chains 'VAR A = 12345', 'VAR B = VAR A', ... inside noise text."""
    chains, names_used = [], set()
    for _ in range(num_chains):
        val = str(rng.randint(10000, 99999))
        names = []
        while len(names) < num_hops + 1:
            nm = "".join(rng.choice(_UP) for _ in range(5))
            if nm not in names_used:
                names_used.add(nm)
                names.append(nm)
        stmts = [f"VAR {names[0]} = {val}."] + [
            f"VAR {names[j]} = VAR {names[j - 1]}." for j in range(1, len(names))]
        chains.append((val, names, stmts))
    unit = [s.strip() + "." for s in NOISE.strip().rstrip(".").split(". ")]
    sents = []
    while sum(len(s) + 1 for s in sents) < target_bytes:
        sents.extend(unit)
    # insert the statements in order (per chain) at non-decreasing random slots
    all_stmts = []
    for _, _, st in chains:
        slots = sorted(rng.choices(range(len(sents) + 1), k=len(st)))
        all_stmts.extend(zip(slots, [rng.random() for _ in st], st))
    out, by_slot = [], collections.defaultdict(list)
    for slot, r, st in all_stmts:
        by_slot[slot].append(st)
    for j in range(len(sents) + 1):
        out.extend(by_slot.get(j, []))
        if j < len(sents):
            out.append(sents[j])
    ctx = " ".join(out)
    val, names, _ = chains[0]
    return ctx, val, names


VT_TMPL = ("Memorize and track the chain(s) of variable assignment hidden in the following "
           "text.\n\n{ctx}\nQuestion: Find all variables that are assigned the value {val} in "
           "the text above. Answer: According to the chain(s) of variable assignment in the text "
           "above, {n} variables are assigned the value {val}, they are:")


def build_vt(lengths, n, seed, num_chains, num_hops):
    rows = []
    rng = random.Random(f"{seed}-vt-{num_chains}-{num_hops}")
    for tb in lengths:
        for i in range(n):
            ectx, eval_, enames = _vt_instance(rng, 300, 1, num_hops)
            example = VT_TMPL.format(ctx=ectx, val=eval_, n=len(enames)) + " " + " ".join(enames) + "\n\n"
            budget = tb - nbytes(example) - 400
            ctx, val, names = _vt_instance(rng, max(200, budget), num_chains, num_hops)
            prompt = example + VT_TMPL.format(ctx=ctx, val=val, n=len(names))
            rows.append({"id": f"vt-c{num_chains}h{num_hops}-{tb}-{i}", "task": "vt",
                         "cond": f"c{num_chains}h{num_hops}", "length": tb, "pos": None,
                         "mode": "gen", "score": "recall", "prompt": prompt, "answers": names,
                         "window": 6 * len(names) + 24})
    return rows


_STOP = set("""the and that this with from have were they their there which what when where would
could should about after before other than then them these those into over also been more most
some such only very just your will shall each because while being both through during without
again further here once under until above below between same itself myself yourself""".split())


def common_vocab(min_rank=150, max_rank=3000):
    pool = _essay_pool()
    cnt = collections.Counter(w for w in re.findall(r"\b[a-z]{4,9}\b", pool))
    vocab = [w for w, _ in cnt.most_common(max_rank)[min_rank:] if w not in _STOP]
    return vocab


CWE_TMPL = ("Below is a numbered list of words. In these words, some appear more often than "
            "others. Memorize the ones that appear most often.\n{ctx}\nQuestion: What are the 10 "
            "most common words in the above list? Answer: The top 10 words that appear most "
            "often in the list are:")


def _cwe_instance(rng, vocab, n_words, num_cw=10, freq_ucw=2):
    words = rng.sample(vocab, num_cw + n_words)
    cw, ucw_pool = words[:num_cw], words[num_cw:]
    freq_cw = max(freq_ucw + 2, math.ceil(0.4 * n_words / num_cw))
    seq = [w for w in cw for _ in range(freq_cw)]
    k = 0
    while len(seq) < n_words:
        seq.extend([ucw_pool[k]] * freq_ucw)
        k += 1
    rng.shuffle(seq)
    ctx = " ".join(f"{j + 1}. {w}" for j, w in enumerate(seq))
    return ctx, cw


def build_cwe(lengths, n, seed, vocab):
    rows = []
    rng = random.Random(f"{seed}-cwe")
    for tb in lengths:
        for i in range(n):
            ectx, ecw = _cwe_instance(rng, vocab, 50)
            example = (CWE_TMPL.format(ctx=ectx) + " " +
                       " ".join(f"{j + 1}. {w}" for j, w in enumerate(ecw)) + "\n\n")
            # ~11.5 B per numbered entry; fit the list to the remaining budget
            n_words = max(40, int((tb - nbytes(example) - 350) / 11.5))
            ctx, cw = _cwe_instance(rng, vocab, n_words)
            prompt = example + CWE_TMPL.format(ctx=ctx)
            rows.append({"id": f"cwe-{tb}-{i}", "task": "cwe", "cond": "cwe", "length": tb,
                         "pos": None, "mode": "gen", "score": "recall", "prompt": prompt,
                         "answers": cw, "window": 14 * len(cw) + 24})
    return rows


FWE_TMPL = ("Read the following coded text and track the frequency of each coded word. Find the "
            "three most frequently appeared coded words. {ctx}\nQuestion: Do not provide any "
            "explanation. Please ignore the dots '....'. What are the three most frequently "
            "appeared words in the above coded text? Answer: According to the coded text above, "
            "the three most frequently appeared words are:")


def _fwe_instance(rng, n_words, alpha=2.0, vocab_size=500):
    vocab = set()
    while len(vocab) < vocab_size:
        vocab.add("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 6))))
    vocab = list(vocab)
    rng.shuffle(vocab)
    wts = [1.0 / (r + 1) ** alpha for r in range(vocab_size)]
    seq = rng.choices(vocab, weights=wts, k=n_words)
    cnt = collections.Counter(seq).most_common()
    # require a strict top-3 (no tie at the 3rd/4th boundary) so the answer is unique
    if len(cnt) > 3 and cnt[2][1] == cnt[3][1]:
        return None
    top3 = [w for w, _ in cnt[:3]]
    ctx = " ".join(w if rng.random() > 0.1 else w + " ...." for w in seq)
    return ctx, top3


def build_fwe(lengths, n, seed):
    rows = []
    rng = random.Random(f"{seed}-fwe")
    for tb in lengths:
        i = 0
        while i < n:
            ex = _fwe_instance(rng, 60)
            if ex is None:
                continue
            example = FWE_TMPL.format(ctx=ex[0]) + " " + " ".join(ex[1]) + "\n\n"
            n_words = max(40, int((tb - nbytes(example) - 400) / 6.2))
            inst = _fwe_instance(rng, n_words)
            if inst is None:
                continue
            ctx, top3 = inst
            prompt = example + FWE_TMPL.format(ctx=ctx)
            rows.append({"id": f"fwe-{tb}-{i}", "task": "fwe", "cond": "fwe", "length": tb,
                         "pos": None, "mode": "gen", "score": "recall", "prompt": prompt,
                         "answers": top3, "window": 8 * 3 + 24})
            i += 1
    return rows


# --------------------------------------------------------------------------- #
# 4. many-shot ICL
# --------------------------------------------------------------------------- #
def _load_icl():
    from datasets import load_dataset
    out = {}
    d = load_dataset("stanfordnlp/sst2")
    out["sst2"] = dict(
        train=[(r["sentence"].strip(), r["label"]) for r in d["train"]],
        test=[(r["sentence"].strip(), r["label"]) for r in d["validation"]],
        labels=["negative", "positive"], q="Review", a="Sentiment", mode="rank")
    d = load_dataset("CogComp/trec", revision="refs/convert/parquet")
    coarse = ["abbreviation", "entity", "description", "human", "location", "number"]
    out["trec_coarse"] = dict(
        train=[(r["text"], r["coarse_label"]) for r in d["train"]],
        test=[(r["text"], r["coarse_label"]) for r in d["test"]],
        labels=coarse, q="Question", a="Type", mode="rank")
    fine = d["train"].features["fine_label"].names
    out["trec_fine"] = dict(
        train=[(r["text"], r["fine_label"]) for r in d["train"]],
        test=[(r["text"], r["fine_label"]) for r in d["test"]],
        labels=fine, q="Question", a="Type", mode="gen")
    d = load_dataset("fancyzhx/ag_news")
    out["agnews"] = dict(
        train=[(r["text"].strip(), r["label"]) for r in d["train"]],
        test=[(r["text"].strip(), r["label"]) for r in d["test"]],
        labels=["world", "sports", "business", "technology"], q="Article", a="Topic",
        mode="rank")
    return out


def build_icl(shots, n_test, seed, max_demo_bytes=400):
    rows = []
    for name, ds in _load_icl().items():
        rng = random.Random(f"{seed}-icl-{name}")
        labs = ds["labels"]
        by_lab = collections.defaultdict(list)
        for x, y in ds["train"]:
            if nbytes(x) <= max_demo_bytes:
                by_lab[y].append(x)
        present = sorted(by_lab)
        test = rng.sample(ds["test"], min(n_test, len(ds["test"])))
        for ti, (x, y) in enumerate(test):
            # nested demo list: concatenated class-balanced shuffled blocks, so every
            # prefix whose length is a multiple of #classes is exactly balanced and the
            # k-shot set is a prefix of the 2k-shot set
            demos = []
            while len(demos) < max(shots):
                block = [(rng.choice(by_lab[c]), c) for c in present]
                rng.shuffle(block)
                demos.extend(block)
            q = f"{ds['q']}: {x}\n{ds['a']}:"
            for k in shots:
                body = "".join(f"{ds['q']}: {dx}\n{ds['a']}: {labs[dy]}\n\n" for dx, dy in demos[:k])
                prompt = body + q
                if nbytes(prompt) + 32 > MAX_PROMPT_BYTES:
                    break
                row = {"id": f"icl-{name}-{ti}-{k}", "task": "icl", "cond": name, "length": k,
                       "pos": None, "prompt": prompt, "gold": y}
                if ds["mode"] == "rank":
                    row.update(mode="rank", options=[" " + l for l in labs])
                else:
                    row.update(mode="gen", score="exact_line", answers=[labs[y]],
                               window=max(nbytes(l) for l in labs) + 8)
                rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# 4K-fit cells: the ~4 KB condition of every length-swept task, shrunk so that
# BOS + prompt + generation window <= 4096 bytes (BLT-1B's byte RoPE table)
# --------------------------------------------------------------------------- #
FIT_BYTES = 4096


def _fits(rows):
    return max(1 + nbytes(r["prompt"]) + r["window"] for r in rows) <= FIT_BYTES


def _largest_fit(make, start, step):
    """Largest target (stepping down from `start`) whose rows ALL fit; returns (target, rows)."""
    t = start
    while t > 0:
        rows = make(t)
        if _fits(rows):
            return t, rows
        t -= step
    raise RuntimeError("nothing fits")


def _tag_fit(rows, target):
    for r in rows:
        r["id"] = "f4k-" + r["id"]
        r["fit4k"] = True
        r["target_bytes"] = target
        if r["task"] != "kv_litm":
            r["length"] = "4K-fit"
    return rows


def build_fit4k(seed, vocab):
    out = {}
    parts = [
        ("needle_types", lambda t: build_needle_types([t], [0.1, 0.3, 0.5, 0.7, 0.9], 8, seed,
                                                       byte_budget=True), 4096, 32),
        ("kv_litm", lambda k: build_kv([k], [0.0, 0.25, 0.5, 0.75, 1.0], 30, seed), 60, 1),
        ("vt1", lambda t: build_vt([t], 50, seed, 1, 4), 4096, 32),
        ("vt2", lambda t: build_vt([t], 50, seed, 2, 4), 4096, 32),
        ("cwe", lambda t: build_cwe([t], 50, seed, vocab), 4096, 32),
        ("fwe", lambda t: build_fwe([t], 50, seed), 4096, 32),
    ]
    for name, make, start, step in parts:
        t, rows = _largest_fit(make, start, step)
        print(f"fit4k {name}: target {t} -> max {max(1 + nbytes(r['prompt']) + r['window'] for r in rows)} B "
              f"incl. BOS+window, {len(rows)} rows")
        out[name] = _tag_fit(rows, t)
    return (out["needle_types"], out["kv_litm"],
            out["vt1"] + out["vt2"] + out["cwe"] + out["fwe"])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(
        os.environ.get("AUNET_ROOT", "."), "data", "longctx"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="+",
                    default=["needle_types", "kv_litm", "ruler_agg", "icl"])
    ap.add_argument("--patch_counts", nargs="*", default=[],
                    help="count_patches.py outputs; rows whose prompt + generation exceed ANY "
                         "byte model's patch window are dropped for every family")
    args = ap.parse_args()
    counts = [json.load(open(p)) for p in args.patch_counts]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = args.seed
    built = {}
    if "needle_types" in args.tasks:
        built["needle_types"] = build_needle_types([1024, 2048, 4096, 6144],
                                                   [0.1, 0.3, 0.5, 0.7, 0.9], 8, s)
    if "kv_litm" in args.tasks:
        built["kv_litm"] = build_kv([5, 10, 25, 50, 60], [0.0, 0.25, 0.5, 0.75, 1.0], 30, s)
    if "ruler_agg" in args.tasks:
        vocab = common_vocab()
        built["ruler_agg"] = (build_vt([1024, 2048, 4096, 6144], 50, s, 1, 4)
                              + build_vt([1024, 2048, 4096, 6144], 50, s, 2, 4)
                              + build_cwe([2048, 4096, 6144], 50, s, vocab)
                              + build_fwe([2048, 4096, 6144], 50, s))
    if "icl" in args.tasks:
        built["icl"] = build_icl([1, 2, 4, 8, 16, 32, 64, 128], 300, s)
    if "fit4k" in args.tasks:
        nt, kv, ra = build_fit4k(s, common_vocab())
        built["fit4k"] = nt + kv + ra
    for name, rows in built.items():
        if counts:
            def fits(r):
                need = r.get("window", 1) + 32          # generation + BOS/edge margin
                return all(c[r["id"]] + need <= c["_patch_limit"] for c in counts)
            n0 = len(rows)
            dropped = collections.Counter((r["task"], r["cond"], r["length"]) for r in rows if not fits(r))
            rows = [r for r in rows if fits(r)]
            if dropped:
                print(f"{name}: dropped {n0 - len(rows)} rows over the patch window {dict(dropped)}")
        too_long = [r["id"] for r in rows if nbytes(r["prompt"]) + r.get("window", 32) > 8000]
        assert not too_long, f"{name}: {len(too_long)} prompts exceed the byte window, e.g. {too_long[:3]}"
        for r in rows:
            r["prompt_bytes"] = nbytes(r["prompt"])
        p = out / f"{name}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = collections.Counter((r["task"], r["cond"]) for r in rows)
        mx = max(r["prompt_bytes"] for r in rows)
        print(f"{name}: {len(rows)} rows -> {p}  (max prompt {mx} B)  {dict(by)}")


if __name__ == "__main__":
    main()
