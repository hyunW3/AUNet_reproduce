"""Multilingual LAMBADA: translation-noise diagnostics + paired-bootstrap CIs for the 1B direct eval.

(A) data only — per version (gt = EleutherAI Google-Translate `lambada_openai`, sl = StableLM-2
    re-translation `lambada_multilingual_stablelm`) and language:
      in_ctx    target word occurs verbatim in the passage (LAMBADA was built so that it usually does: en 81%)
      punct     target carries punctuation / non-word chars, so an exact-match model must also emit them
      base_*    trivial baselines: random passage word, most frequent >3-char passage word
      =en       translated target identical to the English target (untranslated names)
      gt=sl     both translations end in the same word
(B) models — acc with 95% bootstrap CI, the paired byte−llama gap with CI, and the gap split by
    in_ctx / not in_ctx (is the byte advantage just copying from context?).

Usage: python lambada_analysis.py [runs_dir] [out_md]
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

R = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/ssd2/hyun2/AUNet/runs/multilingual")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "reports/lambada_multilingual_analysis.md")
# scorer policy (lingua main): BPEByte with vocab_norm (fixed tree), AU-Net official (no vocab_norm); Llama unaffected
ROOT = {"llama": (R / "1B_direct_vnfix", "llama"), "rg": (R / "1B_direct_vnfix", "rg"),
        "aunet_word": (R / "1B_direct", "aunet_word"), "aunet_word_fix": (R / "1B_direct_vnfix", "aunet_word")}
VERSIONS = {
    "gt": ("EleutherAI/lambada_openai", ["en", "de", "es", "fr", "it"], "lambada", "lambada_openai_mt_{}"),
    "sl": ("EleutherAI/lambada_multilingual_stablelm", ["en", "de", "es", "fr", "it", "nl", "pt"], "lambada_sl",
           "lambada_openai_mt_stablelm_{}"),
}
MODELS = [("llama", "Llama"), ("rg", "BPEByte-rg"), ("aunet_word", "AU-Net"), ("aunet_word_fix", "AU-Net (fix)")]
B = 10000
GAPS = [("rg", "llama"), ("aunet_word", "llama"), ("aunet_word", "rg"), ("aunet_word_fix", "llama"), ("aunet_word_fix", "rg")]
GAP_LABELS = ["rg − Llama", "AU-Net − Llama", "AU-Net − rg", "AU-Net(fix) − Llama", "AU-Net(fix) − rg"]
rng = np.random.default_rng(0)
random.seed(0)


def split(text):
    ctx, tgt = text.rsplit(" ", 1)  # lm-eval: context = all but last space-separated token, target = " " + it
    return ctx, tgt


def data_stats(docs):
    n = len(docs)
    in_ctx, punct, rand_w, freq_w = [], 0, 0.0, 0
    for text in docs:
        ctx, tgt = split(text)
        words = re.findall(r"\w+", ctx)
        in_ctx.append(tgt in words)
        punct += bool(re.search(r"[^\w]", tgt))
        if words:
            rand_w += sum(random.choice(words) == tgt for _ in range(20)) / 20
            cand = [w for w in words if len(w) > 3] or words
            freq_w += Counter(cand).most_common(1)[0][0] == tgt
    return {"n": n, "in_ctx": 100 * np.mean(in_ctx), "punct": 100 * punct / n,
            "base_rand": 100 * rand_w / n, "base_freq": 100 * freq_w / n}, np.array(in_ctx)


def ci(x):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return 100 * x.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def load_acc(model, suite, task):
    root, mdir = ROOT[model]
    f = root / mdir / suite / "results.json"
    if not f.exists():
        return None
    s = json.load(open(f)).get("samples", {}).get(task)
    if not s:
        return None
    s = sorted(s, key=lambda r: int(r["doc_id"]))
    return np.array([float(r["acc"]) for r in s])


L = ["# Multilingual LAMBADA — translation noise and paired-bootstrap CIs (1B direct eval)", "",
     "Companion to `reports/multilingual_1B_direct.md`. **gt** = EleutherAI `lambada_openai` non-English splits "
     "(Google Translate); **sl** = StableLM-2 re-translation (`lambada_multilingual_stablelm`, made because the gt "
     "version was judged too noisy by native speakers). Exact-match greedy accuracy, zero-shot; "
     f"95% CIs from {B:,} bootstrap resamples over documents; byte−Llama gaps are paired on the same documents.", ""]

texts, inctx = {}, {}
L += ["## A. Translation-noise diagnostics (data only)", "",
      "| version | lang | n | target in context % | target has punct % | rand-word base % | most-freq base % | =en target % | gt=sl target % |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for v, (path, langs, _, _) in VERSIONS.items():
    for lang in langs:
        texts[v, lang] = load_dataset(path, lang, split="test")["text"]
for v, (path, langs, _, _) in VERSIONS.items():
    for lang in langs:
        st, inctx[v, lang] = data_stats(texts[v, lang])
        en = [split(t)[1] for t in texts[v, "en"]]
        tg = [split(t)[1] for t in texts[v, lang]]
        same_en = "–" if lang == "en" else f"{100 * np.mean([a == b for a, b in zip(en, tg)]):.1f}"
        other = "sl" if v == "gt" else "gt"
        if (other, lang) in texts and len(texts[other, lang]) == len(tg):
            og = [split(t)[1] for t in texts[other, lang]]
            gs = f"{100 * np.mean([a == b for a, b in zip(og, tg)]):.1f}"
        else:
            gs = "–"
        L.append(f"| {v} | {lang} | {st['n']} | {st['in_ctx']:.1f} | {st['punct']:.1f} | {st['base_rand']:.1f} | "
                 f"{st['base_freq']:.1f} | {same_en} | {gs} |")
L.append("")

for v, (path, langs, suite, tmpl) in VERSIONS.items():
    acc = {(m, lang): load_acc(m, suite, tmpl.format(lang)) for m, _ in MODELS for lang in langs}
    if not any(a is not None for a in acc.values()):
        L += [f"## B-{v}. Model accuracy — not run yet (`{suite}`)", ""]
        continue
    L += [f"## B-{v}. Model accuracy with 95% CI (`{suite}`)", "",
          "| lang | " + " | ".join(n for _, n in MODELS) + " | " + " | ".join(GAP_LABELS) + " |",
          "|---|" + "---:|" * (len(MODELS) + len(GAPS))]
    for lang in langs:
        cells = []
        for m, _ in MODELS:
            a = acc[m, lang]
            cells.append("–" if a is None else "{:.1f} [{:.1f}, {:.1f}]".format(*ci(a)))
        for m, o in GAPS:
            a, b = acc[m, lang], acc[o, lang]
            cells.append("–" if a is None or b is None or len(a) != len(b) else "{:+.1f} [{:+.1f}, {:+.1f}]".format(*ci(a - b)))
        L.append(f"| {lang} | " + " | ".join(cells) + " |")
    L += ["", f"**Clean targets only** (target is a bare word, no punctuation; `{suite}`) — removes the exact-match "
          "punctuation artifact (on punctuated targets the subword model collapses, e.g. gt-es 3.9% vs ≈20% for bytes)", "",
          "| lang | n clean | " + " | ".join(n for _, n in MODELS) + " | " + " | ".join(GAP_LABELS) + " |",
          "|---|---:|" + "---:|" * (len(MODELS) + len(GAPS))]
    for lang in langs:
        if any(acc[m, lang] is None for m, _ in MODELS):
            continue
        clean = np.array([not re.search(r"[^\w]", split(t)[1]) for t in texts[v, lang]])
        cells = ["{:.1f} [{:.1f}, {:.1f}]".format(*ci(acc[m, lang][clean])) for m, _ in MODELS]
        cells += ["{:+.1f} [{:+.1f}, {:+.1f}]".format(*ci(acc[m, lang][clean] - acc[o, lang][clean]))
                  for m, o in GAPS]
        L.append(f"| {lang} | {int(clean.sum())} | " + " | ".join(cells) + " |")
    L += ["", f"**Gap split by whether the target occurs in the passage** (`{suite}`; acc Llama / rg / AU-Net / AU-Net(fix), then rg−Llama)", "",
          "| lang | in-ctx n | in-ctx acc | in-ctx rg−Llama | not-in-ctx n | not-in-ctx acc | not-in-ctx rg−Llama |",
          "|---|---:|---|---:|---:|---|---:|"]
    for lang in langs:
        if any(acc[m, lang] is None for m, _ in MODELS):
            continue
        mask = inctx[v, lang]
        row = [lang]
        for sel in (mask, ~mask):
            accs = " / ".join(f"{100 * acc[m, lang][sel].mean():.1f}" for m, _ in MODELS)
            d = acc["rg", lang][sel] - acc["llama", lang][sel]
            row += [str(int(sel.sum())), accs, "{:+.1f} [{:+.1f}, {:+.1f}]".format(*ci(d))]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

L += ["## Takeaways for the paper", "",
      "- **Both translations carry exact-match artifacts.** gt-es attaches punctuation to 59.7% of targets "
      "(`W.` / `W?`); sl has `W..` endings in es/nl (~18–20%), `W.\"` in it, and 456 fr targets with no word "
      "characters at all. On punctuated targets the subword model collapses (gt-es 3.9% vs ≈20% bytes), so "
      "**clean-target accuracy is the metric to report**; full-set accuracy goes to the appendix.",
      "- **Translation also breaks LAMBADA's design property** that the target recurs in the passage "
      "(en 81% → gt-es 31%, gt-de 46%; sl 48–76%). Report this rate next to the scores.",
      "- **Scorer policy** (lingua `apps/aunet/SCORING_POLICY.md`, main 6237a7b): BPEByte-rg = "
      "`generate_bpebyte.BPEByteGenerator`, vocab_norm applied; AU-Net = official `generate.py`, no vocab_norm; "
      "AU-Net (fix) = same checkpoint with vocab_norm (sensitivity only). Llama is unaffected.",
      "- **The byte>subword gap survives the cleaner sl translation and the clean-target filter** (sl, clean): "
      "BPEByte-rg − Llama is +2.1 (de), +4.1 (nl), +6.2 to +11.0 (Romance es/it/pt/fr), CIs excluding 0; "
      "English +0.3 (n.s.). AU-Net (official) − Llama is larger, +1.6 to +16.3, with the same family ordering "
      "(Germanic small, Romance large), matching the direction of the AU-Net paper's multilingual-MMLU gains.",
      "- **BPEByte vs AU-Net is decided by the scorer, not the architecture, on this task.** Under the policy "
      "(rg fixed, AU-Net official) AU-Net leads by +1.7 (en) and +3.3 to +8.3 (Romance). Like-for-like (both "
      "with vocab_norm) the two are tied in en/es/fr/it (|Δ| ≤ 0.9, CIs covering 0), rg leads in de (−1.3) and "
      "nl (−4.5), and AU-Net leads only in pt (+3.0). The official no-norm head scores *higher* on exact-match "
      "LAMBADA than the normed one (same AU-Net checkpoint: +1.7 to +5.3), so any rg-vs-AU-Net LAMBADA claim "
      "must state this asymmetry.",
      "- **Mechanism caveat**: the gap lives on targets that occur in the passage (rg − Llama +4.9 to +10.3 in "
      "the non-English sl languages); when the word must be produced without a copy source all models score "
      "2–9% and the gap is ≈0 (de −0.8, nl −0.4). Frame it as *reproducing context words in an unfamiliar "
      "orthography*, not as better cross-lingual semantic prediction. (sl-fr / sl-it not-in-ctx gaps are "
      "inflated by the punctuation-only targets; the clean-target table is the reference.)", ""]
DST.write_text("\n".join(L) + "\n")
print(f"wrote {DST}")
