"""Aggregate the 1B direct multilingual runs into reports/multilingual_1B_direct.md.

Per suite: one row per language, one column per model, plus the non-English mean and the
above-chance transfer ratio  (mean_nonen - chance) / (en - chance)  — how much of the model's
English margin survives when the same task is posed in another language.

Scorer policy (lingua main, 2026-10-05): BPEByte is scored WITH the final vocab_norm, AU-Net with the
official generator (without it); Llama goes through apps.main and is unaffected. So the primary columns
read rg from the fixed tree and AU-Net from the official tree; "AU-Net word (fix)" is a sensitivity column.
"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/ssd2/hyun2/AUNet/runs/multilingual")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "reports/multilingual_1B_direct.md")
OFFICIAL, FIXED = R / "1B_direct", R / "1B_direct_vnfix"
# key, label, result tree, model dir
MODELS = [("llama", "Llama (subword)", FIXED, "llama"),
          ("rg", "BPEByte-rg", FIXED, "rg"),
          ("aunet_word", "AU-Net word", OFFICIAL, "aunet_word"),
          ("aunet_word_fix", "AU-Net word (fix)", FIXED, "aunet_word"),
          ("aunet_char", "AU-Net char", OFFICIAL, "aunet_char")]
# suite -> (metric, chance, english-anchor task)
SUITES = {
    "xstorycloze": ("acc", 50, "xstorycloze_en"),
    "xcopa": ("acc", 50, "copa"),
    "xwinograd": ("acc", 50, "xwinograd_en"),
    "pawsx": ("acc", 50, "paws_en"),
    "xnli": ("acc", 33.3, "xnli_en"),
    "lambada": ("acc", 0, "lambada_openai_mt_en"),
    "lambada_sl": ("acc", 0, "lambada_openai_mt_stablelm_en"),
    "arc": ("acc_norm", 25, "arc_challenge"),
    "hellaswag": ("acc_norm", 25, "hellaswag"),
    "gmmlu": ("acc_norm", 25, "gmmlu_cloze_en"),
    "nospace": ("acc", None, None),
}
ANCHOR_LANG = {"copa": "en", "arc_challenge": "en", "hellaswag": "en"}
SUMMARY_MODELS = ["llama", "rg", "aunet_word", "aunet_word_fix"]


def lang(task):
    return ANCHOR_LANG.get(task, task.rsplit("_", 1)[-1])


def load(key, suite, metric):
    _, _, root, mdir = next(m for m in MODELS if m[0] == key)
    f = root / mdir / suite / "results.json"
    if not f.exists():
        return None
    r = json.load(open(f))
    groups = set(r.get("groups", {})) | set(r.get("group_subtasks", {}))
    return {t: 100 * v[f"{metric},none"] for t, v in r["results"].items()
            if t not in groups and f"{metric},none" in v} or None


def fmt(x):
    return "–" if x is None else f"{x:.1f}"


lines = ["# Direct multilingual eval — 1B (English-trained, no further training)", "",
         "Zero-shot loglikelihood eval of the DCLM-trained 1B checkpoints on non-English tasks — the "
         "multilingual counterpart of `reports/zh_cloze_1B.md` (same configs, only the task list changes). "
         "**transfer** = (non-en mean − chance) / (en − chance): the share of the English above-chance margin "
         "that survives in other languages.", "",
         "**Scorer policy** (lingua main): BPEByte-rg is scored with the final `vocab_norm` (`1B_direct_vnfix`), "
         "AU-Net with the official generator, i.e. without it (`1B_direct`); Llama is unaffected. "
         "*AU-Net word (fix)* re-scores the same AU-Net checkpoint with `vocab_norm` as a sensitivity check. "
         "AU-Net char = `aunet2_1.3B` re-pooled per codepoint at eval (no-space scripts only).", ""]
summary = []
for suite, (metric, chance, anchor) in SUITES.items():
    keys = [m[0] for m in MODELS]
    res = {k: load(k, suite, metric) for k in keys}
    if suite == "nospace" and res["aunet_char"]:
        for k in keys:
            if k == "aunet_char":
                continue
            pooled = {}
            for s, (met, _, _) in SUITES.items():
                if s != "nospace":
                    pooled.update(load(k, s, met) or {})
            res[k] = {t: pooled[t] for t in res["aunet_char"] if t in pooled} or None
    cols = [(k, n) for k, n, _, _ in MODELS if res[k]]
    if not cols:
        continue
    tasks = sorted({t for k, _ in cols for t in res[k]}, key=lambda t: (t != anchor, lang(t)))
    lines += [f"## {suite}  (`{metric}`" + (f", chance {chance}" if chance is not None else "") + ")", "",
              "| task | lang | " + " | ".join(n for _, n in cols) + " |",
              "|---|---|" + "---:|" * len(cols)]
    for t in tasks:
        lines.append(f"| {t} | {lang(t)} | " + " | ".join(fmt(res[k].get(t)) for k, _ in cols) + " |")
    if anchor:
        nonen = {k: [v for t, v in res[k].items() if t != anchor] for k, _ in cols}
        mean = {k: sum(v) / len(v) if v else None for k, v in nonen.items()}
        en = {k: res[k].get(anchor) for k, _ in cols}
        tr = {k: (mean[k] - chance) / (en[k] - chance) if en[k] and mean[k] is not None and en[k] > chance else None
              for k, _ in cols}
        lines.append("| **non-en mean** | | " + " | ".join(f"**{fmt(mean[k])}**" for k, _ in cols) + " |")
        lines.append("| **transfer** | | " + " | ".join("–" if tr[k] is None else f"{tr[k]:.2f}" for k, _ in cols) + " |")
        summary.append((suite, chance, {k: (en[k], mean[k], tr[k]) for k, _ in cols}))
    lines.append("")

if summary:
    labels = {k: n for k, n, _, _ in MODELS}
    lines += ["## Summary (en → non-en mean · transfer)", "",
              "| suite | chance | " + " | ".join(labels[k] for k in SUMMARY_MODELS) + " |",
              "|---|---:|" + "---:|" * len(SUMMARY_MODELS)]
    for suite, chance, d in summary:
        cells = []
        for k in SUMMARY_MODELS:
            en, mean, tr = d.get(k, (None, None, None))
            cells.append("–" if mean is None else f"{fmt(en)} → {fmt(mean)} · {'–' if tr is None else f'{tr:.2f}'}")
        lines.append(f"| {suite} | {chance} | " + " | ".join(cells) + " |")
    lines.append("")
lines += ["## Notes", "",
          "- **Classification suites do not separate the families**: all four columns sit near chance outside "
          "English on xstorycloze/xcopa/xwinograd/xnli/arc/hellaswag, whichever scorer is used.",
          "- **Global-MMLU does not reproduce the AU-Net paper's multilingual-MMLU gains at this scale**: English "
          "34.1–34.8 and non-English mean 25.3–25.4 (chance 25) for every model, per-language spread within "
          "±2 pt. The paper's 1B (370B tokens) sat at 30–41% outside English, which is where its Romance ≈+4 / "
          "Germanic ≈+3 gaps live; our checkpoints have no above-chance margin to transfer. (The paper does not "
          "name its multilingual MMLU; Global-MMLU covers 24 of its 27 languages.)",
          "- **LAMBADA is the one suite with a byte>subword signal**; see `reports/lambada_multilingual_analysis.md` "
          "for the translation-noise controls, paired CIs and the scorer asymmetry between BPEByte and AU-Net.",
          "- **PAWS-X is label bias, not ability**: the test sets are 44–45% positive in every language, so "
          "always-'No' scores ≈55 and always-'Yes' ≈45.",
          "- Anchors: xcopa's English anchor is SuperGLUE COPA **validation** (100 items, noisy); xnli uses "
          "lm-eval's default validation split (2490/lang); hellaswag_* capped at 2000 docs/lang. lambada = "
          "EleutherAI Google-Translate splits, lambada_sl = StableLM-2 re-translation (see "
          "`reports/lambada_multilingual_analysis.md`). gmmlu = Global-MMLU in cloze form (as our English "
          "`mmlu_text`), 2044 parallel items, the 24 AU-Net Table-3 languages Global-MMLU has (fi/hu/th missing). "
          "Llama is the `llama_1.8B_paper` step-60000 checkpoint (same as the Chinese run).", ""]
lines.append(f"_Sources: `{OFFICIAL}` and `{FIXED}` `/<model>/<suite>/results.json`; launcher "
             "`scripts/multilingual/run_1b_direct.sh`._")
DST.write_text("\n".join(lines) + "\n")
print(f"wrote {DST}")
