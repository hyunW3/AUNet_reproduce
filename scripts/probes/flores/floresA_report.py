"""Aggregate the AU-Net-language-set FLORES-200 runs: BLEU with 95% CIs, group means, and
paired model-vs-model bootstrap. Corpus BLEU is not a mean over sentences, so every interval
here is a bootstrap over sentences (or over languages, for a group mean) -- never n*stderr.

  python scripts/probes/flores/floresA_report.py [--n_boot 1000]
"""
import json, argparse, random, statistics as st
from pathlib import Path
import sacrebleu

R = Path("/mnt/ssd2/hyun2/AUNet/runs")
# Per-model subdir: the byte models were rerun with the two generator fixes
# (AUNET_FIX_VOCAB_NORM=1 AUNET_INC_PARSE=1) into eval_floresA_fix; their pre-fix eval_floresA
# output is withdrawn (see reports_afterAAAIsub/flores/aunet_language_set.md). Llama is unaffected
# -- apps/main/generate.py calls model.forward -- so it keeps its original run.
RUNS = {"Llama": "llama_1.8B_paper", "AU-Net2": "aunet2_1.3B",
        "BPEByte": "bpebyte_br_greedy_root_1.3B"}
# Scoring path (lingua b395f11, 2026-10-07): EVERY byte model is scored with the official
# PackedHierarchicalCausalTransformerGenerator -- `generate_bpebyte.py` and its vocab_norm wrapper
# were dropped, so no byte model gets the extra RMSNorm. Both byte models therefore read their
# pre-fix run; `eval_floresA_fix` (BPEByte with vocab_norm) is kept on disk as a sensitivity check
# only. Llama is unaffected either way -- apps/main calls model.forward.
SUBDIR = {"Llama": "eval_floresA", "AU-Net2": "eval_floresA",
          "BPEByte": "eval_floresA"}
# NOTE: AUNet lists these 23 languages flat, with no group headers; the Germanic /
# Romance split below is ours. "Esperanto" is one constructed language, not a family --
# it is kept as its own row for completeness, never reported as a group mean.
GROUP = {
 "Germanic": ["deu", "nld", "afr", "fao", "isl", "lim", "ltz"],
 "Romance":  ["ita", "fur", "lij", "lmo", "srd", "scn", "vec", "spa", "ast", "cat",
              "oci", "por", "glg", "pap", "kea"],
 "Esperanto": ["epo"],
}
NAME = {"deu":"German","nld":"Dutch","afr":"Afrikaans","fao":"Faroese","isl":"Icelandic",
 "lim":"Limburgish","ltz":"Luxembourgish","ita":"Italian","fur":"Friulian","lij":"Ligurian",
 "lmo":"Lombard","srd":"Sardinian","scn":"Sicilian","vec":"Venetian","spa":"Spanish",
 "ast":"Asturian","cat":"Catalan","oci":"Occitan","por":"Portuguese","glg":"Galician",
 "pap":"Papiamento","kea":"Kabuverdianu","epo":"Esperanto"}
LANGS = [l for g in GROUP.values() for l in g]


def load(run, sub):
    """{lang: (hyps, refs)} from the logged samples."""
    d = json.load(open(R / run / sub / "results.json"))
    out = {}
    for lang in LANGS:
        key = f"floresA_{lang}-en"
        if key not in d.get("samples", {}):
            continue
        smp = d["samples"][key]
        out[lang] = ([s["bleu"][1] for s in smp], [s["bleu"][0] for s in smp])
    return out


def bleu(hyps, refs):
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def boot_ci(hyps, refs, n, rng):
    """95% CI of corpus BLEU by resampling sentences."""
    k = len(hyps); xs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        xs.append(bleu([hyps[i] for i in idx], [refs[i] for i in idx]))
    xs.sort()
    return xs[int(.025 * n)], xs[int(.975 * n)], st.pstdev(xs)


def paired(a, b, refs, n, rng):
    """Paired bootstrap on BLEU(a) - BLEU(b) over the same resampled sentences."""
    k = len(refs); d = bleu(a, refs) - bleu(b, refs); c = 0
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        rr = [refs[i] for i in idx]
        dd = bleu([a[i] for i in idx], rr) - bleu([b[i] for i in idx], rr)
        if (dd <= 0) if d > 0 else (dd >= 0):
            c += 1
    return d, (c + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--sub", default="eval_floresA")
    a = ap.parse_args()
    rng = random.Random(0)
    data = {t: load(r, SUBDIR[t]) for t, r in RUNS.items()}
    have = [l for l in LANGS if all(l in data[t] for t in RUNS)]
    n_sent = len(next(iter(data["Llama"].values()))[0])
    print(f"# FLORES-200 X->en, AU-Net Table 3 language set — devtest, 5-shot, "
          f"n={n_sent}/lang, {len(have)} languages, {a.n_boot} bootstrap resamples\n")

    res = {}
    print("| lang | | " + " | ".join(f"{t} BLEU [95% CI]" for t in RUNS) + " |")
    print("|---|---|" + "---|" * len(RUNS))
    for g, members in GROUP.items():
        for l in [x for x in members if x in have]:
            row, cells = {}, []
            for t in RUNS:
                h, r = data[t][l]
                b = bleu(h, r); lo, hi, sd = boot_ci(h, r, a.n_boot, rng)
                row[t] = dict(bleu=b, lo=lo, hi=hi, sd=sd)
                cells.append(f"{b:.2f} [{lo:.1f}, {hi:.1f}]")
            res[l] = row
            print(f"| {NAME[l]} | {g[:3]} | " + " | ".join(cells) + " |")

    print(f"\n## Group means (unweighted over languages; CI = bootstrap over languages)\n")
    print("| group | n | " + " | ".join(RUNS) + " |")
    print("|---|---|" + "---|" * len(RUNS))
    for g, members in GROUP.items():
        ms = [x for x in members if x in have]
        if not ms: continue
        cells = []
        for t in RUNS:
            v = [res[l][t]["bleu"] for l in ms]
            if len(v) > 1:
                bs = sorted(st.mean([v[rng.randrange(len(v))] for _ in v])
                            for _ in range(a.n_boot))
                cells.append(f"{st.mean(v):.2f} [{bs[int(.025*a.n_boot)]:.1f}, "
                             f"{bs[int(.975*a.n_boot)]:.1f}]")
            else:
                cells.append(f"{v[0]:.2f}")
        print(f"| {g} | {len(ms)} | " + " | ".join(cells) + " |")
    allv = {t: [res[l][t]["bleu"] for l in have] for t in RUNS}
    print(f"| **all** | {len(have)} | " + " | ".join(f"**{st.mean(allv[t]):.2f}**" for t in RUNS) + " |")

    print(f"\n## Paired bootstrap, per language (BLEU difference on the same resampled sentences)\n")
    print("| lang | BPEByte − AU-Net2 | p | Llama − BPEByte | p |")
    print("|---|---|---|---|---|")
    sig = {"BPE-AU": [], "Llama-BPE": []}
    for l in have:
        refs = data["Llama"][l][1]
        d1, p1 = paired(data["BPEByte"][l][0], data["AU-Net2"][l][0], refs, a.n_boot, rng)
        d2, p2 = paired(data["Llama"][l][0], data["BPEByte"][l][0], refs, a.n_boot, rng)
        sig["BPE-AU"].append((l, d1, p1)); sig["Llama-BPE"].append((l, d2, p2))
        m1 = "**" if p1 < .05 else ""; m2 = "**" if p2 < .05 else ""
        print(f"| {NAME[l]} | {m1}{d1:+.2f}{m1} | {m1}{p1:.3f}{m1} | {m2}{d2:+.2f}{m2} | {m2}{p2:.3f}{m2} |")
    print()
    for k, rows in sig.items():
        w = sum(1 for _, d, _ in rows if d > 0); s = sum(1 for _, _, p in rows if p < .05)
        pos = sum(1 for _, d, p in rows if p < .05 and d > 0)
        print(f"{k}: mean {st.mean([d for _, d, _ in rows]):+.2f} BLEU · "
              f"positive in {w}/{len(rows)} · significant in {s} ({pos} favouring the first)")
    json.dump({"n_sent": n_sent, "per_lang": res,
               "paired": {k: [[l, d, p] for l, d, p in v] for k, v in sig.items()}},
              open("/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/flores/floresA.json", "w"), indent=1)


if __name__ == "__main__":
    main()
