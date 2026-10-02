"""Format / whitespace / punctuation robustness for MC loglikelihood scoring.

Same items and scorer as the paper robustness protocol (eval_despace_mc / eval_brittle_mc):
HellaSwag / ARC-E / ARC-C / PIQA / BoolQ, first 2000 docs per task (ARC-C 1172 and PIQA 1838
are used in full), context = lm-eval doc_to_text, continuation = " <option>", acc = argmax of
the raw total logprob, acc_norm = per-byte normalised. Only the QUESTION BLOCK of the context is
perturbed: a final cue line ending in ':' ("Answer:") and every option string stay byte-identical,
so a delta is attributable to the prompt change alone. HellaSwag has no cue line ("label: ctx"
continues straight into the option), so its whole context is perturbed and trailing padding is
not added (it would sit at the context/option cut, i.e. a PBP effect, not a padding one).

Variant families (rate p in {0.5, 1.0}; each eligible site is hit independently with prob p, one
uniform draw per site from a per-item seed, so the p=0.5 sites are a subset of the p=1.0 sites):

  whitespace  space_run    word gap " " -> 2-4 spaces                        (BrittleBench op)
              pad_space    4 spaces before/after the block + gap -> 5 spaces (BrittleBench padding,
                           made frequent by also padding word gaps)
              pad_newline  "\\n\\n" before/after the block + gap -> "\\n\\n"
              punct_space  " " inserted before , . ; : ? !                   (BrittleBench op)
  removal     punct_drop   delete prose punctuation  . , ; : ? ! ' " and curly quotes / ellipsis
              symbol_drop  delete every other non-alphanumeric, non-space char ( ) [ ] - / & % $ ...
  ReCode-     line_split   word gap -> "\\n"                                  (ReCode LineSplit)
  style       tab          word gap -> "\\t"                                  (ReCode Tab-Indent)
              nl_insert    empty line inserted at line breaks and sentence ends (ReCode NewlineInsert)
  NL-Augmenter (verbatim transformation code + default parameters + fixed per-call seeds,
              GEM-benchmark/NL-Augmenter): whitespace_perturbation, underscore_trick,
              butter_fingers_perturbation, change_char_case, swap_characters, leet_letters
  FormatSpread (Sclar et al., ICLR 2024; grammar_definition.py value lists): K prompt formats
              sampled with random.Random(42) over (descriptor casing x descriptor separator x
              field joiner). The field structure is the task's own: ARC/PIQA "Question: q\\nAnswer:",
              BoolQ "passage\\nQuestion: q?\\nAnswer:", HellaSwag "label: ctx". Option enumeration
              formats do not apply (options are scored, not listed).

Identical (context, continuation) pairs are scored once and shared across variants (a variant that
leaves an item unchanged reuses the clean score exactly), and every unique score is appended to a
jsonl cache so a killed run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
SEED = 1234
PROBS = (0.5, 1.0)

# ----------------------------------------------------------------------------- per-site ops
PROSE_PUNCT = set(".,;:?!'\"") | set("‘’“”…")
_SPACE_PUNCT = set(",.;:?!")          # BrittleBench punctuation-spaces set


def _is_symbol(ch: str) -> bool:
    return not ch.isalnum() and not ch.isspace() and ch not in PROSE_PUNCT


def _per_space(t: str, p: float, rng: random.Random, repl: Callable[[random.Random], str]) -> str:
    out = []
    for ch in t:
        if ch == " ":
            u, r = rng.random(), repl(rng)          # always draw both -> sites nest across p
            out.append(r if u < p else ch)
        else:
            out.append(ch)
    return "".join(out)


def space_run(t, p, rng):
    return _per_space(t, p, rng, lambda r: " " * r.randint(2, 4))


def gap_pad_space(t, p, rng):
    return _per_space(t, p, rng, lambda r: " " * 5)


def gap_pad_newline(t, p, rng):
    return _per_space(t, p, rng, lambda r: "\n\n")


def line_split(t, p, rng):
    return _per_space(t, p, rng, lambda r: "\n")


def tab(t, p, rng):
    return _per_space(t, p, rng, lambda r: "\t")


def punct_space(t, p, rng):
    out = []
    for ch in t:
        if ch in _SPACE_PUNCT:
            if rng.random() < p:
                out.append(" ")
        out.append(ch)
    return "".join(out)


def _drop(t, p, rng, pred):
    out = []
    for ch in t:
        if pred(ch):
            if rng.random() < p:
                continue
        out.append(ch)
    return "".join(out)


def punct_drop(t, p, rng):
    return _drop(t, p, rng, lambda c: c in PROSE_PUNCT)


def symbol_drop(t, p, rng):
    return _drop(t, p, rng, _is_symbol)


_SENT_END = re.compile(r"(?<=[.?!]) (?=\S)")


def nl_insert(t, p, rng):
    """ReCode NewlineInsertion analogue: lines of prose are sentences, so the insertion sites are the
    existing line breaks plus the gaps after sentence-final . ? ! ; each gets an empty line w.p. p."""
    out, last = [], 0
    sites = sorted([(m.start(), " ") for m in _SENT_END.finditer(t)] +
                   [(i, "\n") for i, ch in enumerate(t) if ch == "\n"])
    for pos, kind in sites:
        out.append(t[last:pos])
        u = rng.random()
        out.append("\n\n" if u < p else kind)
        last = pos + 1
    out.append(t[last:])
    return "".join(out)


# ----------------------------------------------------------------------------- NL-Augmenter
# Transformation bodies copied from GEM-benchmark/NL-Augmenter (nlaugmenter/transformations/*/
# transformation.py) with each class's default generate() parameters and seed.
def nla_whitespace(text, seed=0, remove_prob=0.1, add_prob=0.05):
    random.seed(seed)
    out = []
    for char in text:
        rn = random.random()
        if char.isspace() and rn < remove_prob:
            continue
        out.append(char)
        if (not char.isspace()) and rn < add_prob:
            out.append(" ")
    return "".join(out)


def nla_underscore(text, new_symbol="_", prob=0.05, seed=42):
    random.seed(seed)
    return "".join([letter if letter != " " or random.random() > prob else new_symbol for letter in text])


_KEY_APPROX = {
    "q": "qwasedzx", "w": "wqesadrfcx", "e": "ewrsfdqazxcvgt", "r": "retdgfwsxcvgt",
    "t": "tryfhgedcvbnju", "y": "ytugjhrfvbnji", "u": "uyihkjtgbnmlo", "i": "iuojlkyhnmlp",
    "o": "oipklujm", "p": "plo['ik", "a": "aqszwxwdce", "s": "swxadrfv", "d": "decsfaqgbv",
    "f": "fdgrvwsxyhn", "g": "gtbfhedcyjn", "h": "hyngjfrvkim", "j": "jhknugtblom", "k": "kjlinyhn",
    "l": "lokmpujn", "z": "zaxsvde", "x": "xzcsdbvfrewq", "c": "cxvdfzswergb", "v": "vcfbgxdertyn",
    "b": "bvnghcftyun", "n": "nbmhjvgtuik", "m": "mnkjloik", " ": " ",
}


def nla_butter_fingers(text, prob=0.05, seed=0):
    random.seed(seed)
    prob_of_typo = int(prob * 100)
    butter_text = ""
    for letter in text:
        lcletter = letter.lower()
        if lcletter not in _KEY_APPROX:
            new_letter = lcletter
        elif random.choice(range(0, 100)) <= prob_of_typo:
            new_letter = random.choice(_KEY_APPROX[lcletter])
        else:
            new_letter = lcletter
        if not lcletter == letter:
            new_letter = new_letter.upper()
        butter_text += new_letter
    return butter_text


def nla_change_char_case(text, prob=0.1, seed=0):
    random.seed(seed)
    result = []
    for c in text:
        if c.isupper() and random.random() < prob:
            result.append(c.lower())
        elif c.islower() and random.random() < prob:
            result.append(c.upper())
        else:
            result.append(c)
    return "".join(result)


def nla_swap_characters(text, prob=0.05, seed=0):
    import numpy as np
    np.random.seed((seed + sum([ord(c) for c in text])) % (2 ** 32))
    num_pairs = len(text) - 1
    if num_pairs < 1:
        return text
    idx = np.argwhere(np.random.rand(num_pairs) < prob).reshape(-1)
    np.random.shuffle(idx)
    text = list(text)
    for i in idx:
        text[i], text[i + 1] = text[i + 1], text[i]
    return "".join(text)


_LEET = {"!": "1", "7": "1", "A": "4", "B": "8", "C": "0", "D": "0", "E": "3", "G": "6", "I": "1",
         "J": "9", "L": "7", "N": "11", "O": "0", "S": "5", "T": "7", "X": "8", "Z": "2", "b": "6",
         "e": "3", "g": "9", "h": "4", "j": "7", "m": "3", "o": "0", "w": "3", "y": "4", "|": "1",
         "Θ": "0", "ε": "3", "ω": "3", "∈": "3", "∩∩": "3"}


def nla_leet(sentence, seed=0, max_leet=0.5):
    random.seed(seed)
    k = int(max_leet * len(sentence))
    cands = [(i, _LEET[ch]) for i, ch in enumerate(sentence) if ch in _LEET]
    if not cands:                                   # upstream raises IndexError here
        return sentence
    s = list(sentence)
    for i, leet in random.choices(cands, k=k):
        s[i] = str(leet)
    return "".join(s)


NLA = {"nla_whitespace": nla_whitespace, "nla_underscore": nla_underscore,
       "nla_butter_fingers": nla_butter_fingers, "nla_change_char_case": nla_change_char_case,
       "nla_swap_characters": nla_swap_characters, "nla_leet": nla_leet}

# ----------------------------------------------------------------------------- FormatSpread
# Value lists from msclar/formatspread grammar_definition.py. sep '' is excluded: upstream notes it
# is only meant for enumerations ("there is already formatting there").
FS_SEPARATORS = ['::: ', ':: ', ': ', ' \n\t', '\n    ', ' : ', ' - ', ' ', '\n ', '\n\t', ':', '::', '- ', '\t']
FS_SPACES = ['', ' ', '\n', ' \n', ' -- ', '  ', '; \n', ' || ', ' <sep> ', ' -- ', ', ', ' \n ', ' , ', '\n ', '. ', ' ,  ']
FS_CASING = {"id": lambda x: x, "title": lambda x: x.title(), "upper": lambda x: x.upper(), "lower": lambda x: x.lower()}
FS_DEFAULT = ("id", ": ", "\n")


def formatspread_formats(k=10, seed=42):
    rng = random.Random(seed)
    seen, out = {FS_DEFAULT}, []
    while len(out) < k:
        f = (rng.choice(list(FS_CASING)), rng.choice(FS_SEPARATORS), rng.choice(FS_SPACES))
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


_QA = re.compile(r"^(?:(?P<passage>.*)\n)?Question: (?P<q>.*)\nAnswer:$", re.S)


def render_format(task: str, ctx: str, fmt) -> str:
    case, sep, space = fmt
    D = FS_CASING[case]
    if task == "hellaswag":
        label, body = ctx.split(": ", 1) if ": " in ctx else ("", ctx)
        return (D(label) + sep + body) if label else body
    m = _QA.match(ctx)
    if not m:
        raise ValueError(f"unrecognised {task} context: {ctx[:80]!r}")
    parts = ([m.group("passage")] if m.group("passage") is not None else []) + [D("Question") + sep + m.group("q")]
    return space.join(parts) + space + D("Answer") + sep.rstrip(" \t")


# ----------------------------------------------------------------------------- variant table
OPS = {"space_run": space_run, "pad_space": gap_pad_space, "pad_newline": gap_pad_newline,
       "punct_space": punct_space, "punct_drop": punct_drop, "symbol_drop": symbol_drop,
       "line_split": line_split, "tab": tab, "nl_insert": nl_insert}
BLOCK_PAD = {"pad_space": "    ", "pad_newline": "\n\n"}
FAMILY = {"space_run": "whitespace", "pad_space": "whitespace", "pad_newline": "whitespace",
          "punct_space": "whitespace", "punct_drop": "removal", "symbol_drop": "removal",
          "line_split": "recode", "tab": "recode", "nl_insert": "recode"}


def all_variants(fs_k=10):
    v = ["clean"]
    for op in OPS:
        for p in PROBS:
            v.append(f"{op}_p{int(round(p * 100))}")
    v += list(NLA)
    v += [f"fs{i:02d}" for i in range(fs_k)]
    return v


def variant_group(g: str, fs_k=10):
    """Two roughly equal-cost halves for sharding slow models (BLT bs1); both carry 'clean'."""
    allv = all_variants(fs_k)
    a = ["clean"] + [v for v in allv if v != "clean" and family_of(v) in ("whitespace", "removal")
                     or v.startswith(("line_split", "tab_"))]
    if g == "A":
        return a
    return ["clean"] + [v for v in allv if v not in a]


def family_of(v: str) -> str:
    if v == "clean":
        return "clean"
    if v.startswith("nla_"):
        return "nlaugmenter"
    if v.startswith("fs"):
        return "formatspread"
    return FAMILY[v.rsplit("_p", 1)[0]]


def _split_cue(ctx: str):
    lines = ctx.split("\n")
    if len(lines) > 1 and lines[-1].rstrip().endswith(":"):
        return "\n".join(lines[:-1]), lines[-1]
    return ctx, None


def perturb(task: str, ctx: str, variant: str, idx: int, fs_formats) -> str:
    if variant == "clean":
        return ctx
    if variant.startswith("fs"):
        return render_format(task, ctx, fs_formats[int(variant[2:])])
    block, cue = _split_cue(ctx)
    if variant in NLA:
        block = NLA[variant](block)
    else:
        op, ptag = variant.rsplit("_p", 1)
        rng = random.Random(SEED + idx)
        block = OPS[op](block, int(ptag) / 100, rng)
        if op in BLOCK_PAD:
            pad = BLOCK_PAD[op]
            block = pad + block + (pad if cue is not None else "")
        if op == "nl_insert" and cue is not None and rng.random() < int(ptag) / 100:
            return block + "\n\n" + cue        # the question/cue line break is a site too
    return block + "\n" + cue if cue is not None else block


# ----------------------------------------------------------------------------- items + scoring
def load_items(task: str, limit: int, items_dir: str | None):
    """Items from the task's lm-eval docs (eval_pbp_mc._items_from_task), frozen to json so every
    host and model scores byte-identical prompts."""
    path = os.path.join(items_dir, f"{task}_{limit}.json") if items_dir else None
    if path and os.path.exists(path):
        return json.load(open(path))
    full = os.path.join(items_dir, f"{task}_2000.json") if items_dir else None
    if full and limit < 2000 and os.path.exists(full):
        return json.load(open(full))[:limit]           # smoke runs: prefix of the frozen items
    from apps.aunet.eval_pbp_mc import _items_from_task
    items = [{"context": it["context"], "choices": it["choices"], "gold": it["gold"]}
             for it in _items_from_task(task, limit)]
    if path:
        os.makedirs(items_dir, exist_ok=True)
        json.dump(items, open(path, "w"))
    return items


class _Req:
    __slots__ = ("args",)

    def __init__(self, c, k):
        self.args = (c, k)


def _key(c, k):
    return hashlib.sha1((c + "\x00" + k).encode("utf-8")).hexdigest()


def run_format_mc(loglikelihood, tasks=TASKS, variants=None, limit=2000, items_dir=None, cache=None,
                  chunk=4096, fs_k=10, log=print) -> Dict[str, dict]:
    variants = variants or all_variants(fs_k)
    fs_formats = formatspread_formats(fs_k)
    store = {}
    if cache and os.path.exists(cache):
        for line in open(cache):
            try:
                r = json.loads(line)
                store[r["k"]] = r["ll"]
            except Exception:
                pass                                   # torn last line of a killed run
        log(f"[format_mc] cache {cache}: {len(store)} scores")
    fc = open(cache, "a") if cache else None
    results: Dict[str, dict] = {"_meta": {"variants": variants, "tasks": list(tasks), "limit": limit,
                                          "seed": SEED, "probs": list(PROBS),
                                          "formatspread": {f"fs{i:02d}": list(f) for i, f in enumerate(fs_formats)}}}
    for task in tasks:
        items = load_items(task, limit, items_dir)
        layout, pairs = [], []
        changed = defaultdict(list)
        for ii, it in enumerate(items):
            base = it["context"].rstrip(" ")
            for v in variants:
                ctx = perturb(task, base, v, ii, fs_formats)
                changed[v].append(int(ctx != base))
                ks, lens = [], []
                for opt in it["choices"]:
                    o = " " + (opt[1:] if opt.startswith(" ") else opt)
                    ks.append(_key(ctx, o))
                    lens.append(max(1, len(o.encode("utf-8"))))
                    pairs.append((ctx, o))
                layout.append((ii, v, ks, lens))
        todo, seen = [], set()
        for c, k in pairs:
            h = _key(c, k)
            if h not in store and h not in seen:
                seen.add(h)
                todo.append((h, c, k))
        log(f"[format_mc] {task}: {len(items)} items x {len(variants)} variants = {len(pairs)} pairs, "
            f"{len(set(_key(c, k) for c, k in pairs))} unique, {len(todo)} to score")
        t0 = time.time()
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            out = loglikelihood([_Req(c, k) for _, c, k in part])
            for (h, _, _), (ll, _g) in zip(part, out):
                store[h] = float(ll)
                if fc:
                    fc.write(json.dumps({"k": h, "ll": float(ll)}) + "\n")
            if fc:
                fc.flush()
            done = s + len(part)
            el = time.time() - t0
            log(f"[format_mc] {task}: {done}/{len(todo)} scored, {el:.0f}s, eta {el / done * (len(todo) - done):.0f}s")
        bits = defaultdict(list)
        bits_n = defaultdict(list)
        for ii, v, ks, lens in layout:
            lls = [store[h] for h in ks]
            gold = items[ii]["gold"]
            bits[v].append(int(max(range(len(lls)), key=lambda j: lls[j]) == gold))
            bits_n[v].append(int(max(range(len(lls)), key=lambda j: lls[j] / lens[j]) == gold))
        for v in variants:
            r = {"acc": sum(bits[v]) / len(bits[v]), "acc_norm": sum(bits_n[v]) / len(bits_n[v]),
                 "n": len(bits[v]), "changed": sum(changed[v]) / len(changed[v]),
                 "bits": {"acc": bits[v], "acc_norm": bits_n[v], "changed": changed[v]}}
            results[f"fmt_{task}_{v}"] = r
        cl = results[f"fmt_{task}_clean"]
        for v in variants:
            if v != "clean":
                r = results[f"fmt_{task}_{v}"]
                r["delta_acc"] = r["acc"] - cl["acc"]
                r["delta_acc_norm"] = r["acc_norm"] - cl["acc_norm"]
    if fc:
        fc.close()
    present = [t for t in tasks if f"fmt_{t}_clean" in results]
    for v in variants:
        rows = [results[f"fmt_{t}_{v}"] for t in present]
        agg = {m: sum(r[m] for r in rows) / len(rows) for m in ("acc", "acc_norm")}
        if v != "clean":
            agg.update({m: sum(r[m] for r in rows) / len(rows) for m in ("delta_acc", "delta_acc_norm")})
        results[f"fmt_avg_{v}"] = agg
    return results
