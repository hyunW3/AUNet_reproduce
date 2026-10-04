"""Aggregate runs/multilingual/1B_direct/<model>/<suite>/results.json into reports/multilingual_1B_direct.md.

Per suite: one row per language, one column per model, plus the non-English mean and the
above-chance transfer ratio  (mean_nonen - chance) / (en - chance)  — how much of the model's
English margin survives when the same task is posed in another language.
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/ssd2/hyun2/AUNet/runs/multilingual/1B_direct")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "reports/multilingual_1B_direct.md")
MODELS = [("llama", "Llama (subword)"), ("rg", "BPEByte-rg"), ("aunet_word", "AU-Net word"), ("aunet_char", "AU-Net char")]
# suite -> (metric, chance, english-anchor task, task -> lang)
SUITES = {
    "xstorycloze": ("acc", 50, "xstorycloze_en"),
    "xcopa": ("acc", 50, "copa"),
    "xwinograd": ("acc", 50, "xwinograd_en"),
    "pawsx": ("acc", 50, "paws_en"),
    "xnli": ("acc", 33.3, "xnli_en"),
    "lambada": ("acc", 0, "lambada_openai_mt_en"),
    "arc": ("acc_norm", 25, "arc_challenge"),
    "hellaswag": ("acc_norm", 25, "hellaswag"),
    "nospace": ("acc", None, None),
}
ANCHOR_LANG = {"copa": "en", "arc_challenge": "en", "hellaswag": "en"}


def lang(task):
    return ANCHOR_LANG.get(task, task.rsplit("_", 1)[-1])


def load(model, suite, metric):
    f = OUT / model / suite / "results.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    groups = set(r.get("groups", {})) | set(r.get("group_subtasks", {}))
    out = {}
    for t, v in r["results"].items():
        if t in groups or f"{metric},none" not in v:
            continue
        out[t] = 100 * v[f"{metric},none"]
    return out


def fmt(x):
    return "–" if x is None else f"{x:.1f}"


lines = ["# Direct multilingual eval — 1B (English-trained, no further training)", "",
         "Zero-shot loglikelihood eval of the DCLM-trained 1B checkpoints on non-English tasks — the "
         "multilingual counterpart of `reports/zh_cloze_1B.md` (same configs, only the task list changes). "
         "**transfer** = (non-en mean − chance) / (en − chance): the share of the English above-chance margin "
         "that survives in other languages. AU-Net char = the same `aunet2_1.3B` re-pooled per codepoint at "
         "eval (no-space scripts only).", ""]
summary = []
for suite, (metric, chance, anchor) in SUITES.items():
    res = {m: load(m, suite, metric) for m, _ in MODELS}
    if suite == "nospace" and res["aunet_char"]:
        # word-vs-char on the same tasks: pull every model's scores for these tasks from its own suites
        for m, _ in MODELS:
            if m == "aunet_char":
                continue
            pooled = {}
            for s, (met, _, _) in SUITES.items():
                if s != "nospace":
                    pooled.update(load(m, s, met) or {})
            res[m] = {t: pooled[t] for t in res["aunet_char"] if t in pooled} or None
    cols = [(m, n) for m, n in MODELS if res[m]]
    if not cols:
        continue
    tasks = sorted({t for m, _ in cols for t in res[m]}, key=lambda t: (t != anchor, lang(t)))
    lines += [f"## {suite}  (`{metric}`" + (f", chance {chance}" if chance is not None else "") + ")", "",
              "| task | lang | " + " | ".join(n for _, n in cols) + " |",
              "|---|---|" + "---:|" * len(cols)]
    for t in tasks:
        lines.append(f"| {t} | {lang(t)} | " + " | ".join(fmt(res[m].get(t)) for m, _ in cols) + " |")
    if anchor:
        nonen = {m: [v for t, v in res[m].items() if t != anchor] for m, _ in cols}
        mean = {m: sum(v) / len(v) if v else None for m, v in nonen.items()}
        en = {m: res[m].get(anchor) for m, _ in cols}
        tr = {m: (mean[m] - chance) / (en[m] - chance) if en[m] and mean[m] is not None and en[m] > chance else None
              for m, _ in cols}
        lines.append("| **non-en mean** | | " + " | ".join(f"**{fmt(mean[m])}**" for m, _ in cols) + " |")
        lines.append("| **transfer** | | " + " | ".join("–" if tr[m] is None else f"{tr[m]:.2f}" for m, _ in cols) + " |")
        summary.append((suite, metric, chance, {m: (en[m], mean[m], tr[m]) for m, _ in cols}))
    lines.append("")

if summary:
    lines += ["## Summary (en → non-en mean · transfer)", "",
              "| suite | chance | " + " | ".join(n for m, n in MODELS if m != "aunet_char") + " |",
              "|---|---:|" + "---:|" * (len(MODELS) - 1)]
    for suite, metric, chance, d in summary:
        cells = []
        for m, _ in MODELS:
            if m == "aunet_char":
                continue
            en, mean, tr = d.get(m, (None, None, None))
            cells.append("–" if mean is None else f"{fmt(en)} → {fmt(mean)} · {'–' if tr is None else f'{tr:.2f}'}")
        lines.append(f"| {suite} | {chance} | " + " | ".join(cells) + " |")
    lines.append("")
lines += ["## Notes", "",
          "- **Same picture as Chinese**: every family is strong in English and near chance elsewhere; "
          "between-family gaps on the classification suites are within noise. Direct transfer from an "
          "English-only 1B does not separate the architectures, except on LAMBADA.",
          "- **LAMBADA is the one clear signal**: byte models beat the subword model on es/fr/it "
          "(non-en mean ≈26–27 vs 18.9), i.e. last-word prediction in Latin-script languages transfers "
          "better when the model is not bound to an English subword vocab.",
          "- **PAWS-X is label bias, not ability**: the test sets are 44–45% positive in every language, so "
          "always-'No' scores ≈55 and always-'Yes' ≈45. Llama's ≈55 and the byte models' 44–46 on ja/ko/zh "
          "are those two degenerate answers.",
          "- **AU-Net char vs word** (nospace table): the eval-time per-codepoint re-pool does not help an "
          "English-trained checkpoint on CJK/Thai/Burmese either (same caveat as `zh_cloze_1B.md`: train/eval "
          "boundary mismatch). char and word agree within ±1.5 pt on every task.",
          "- **Llama's small CJK edge**: xwinograd_zh 60.7 vs 48.6–53.0 (n=504, ~±2.2 pt s.e.) and xstorycloze_zh "
          "53.3 vs ≈47 — plausibly the llama3 vocab's CJK tokens; elsewhere the families are tied.",
          "- Anchors: xcopa's English anchor is SuperGLUE COPA **validation** (100 items, noisy); xnli uses "
          "lm-eval's default validation split (2490/lang); hellaswag_* capped at 2000 docs/lang. Llama is the "
          "`llama_1.8B_paper` step-60000 checkpoint (same as the Chinese run).", ""]
lines.append(f"_Sources:`{OUT}/<model>/<suite>/results.json`; launcher `scripts/multilingual/run_1b_direct.sh`._")
DST.write_text("\n".join(lines) + "\n")
print(f"wrote {DST}")
