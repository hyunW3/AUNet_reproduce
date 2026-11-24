"""Paired item-level bootstrap CI for the FLORES X->En column of the main table.

FLORES-200 is PARALLEL: item i is a translation of the same underlying sentence in every language.
So the correct item bootstrap resamples the 200 sentence indices ONCE per replicate and applies that
same index set to all 23 languages and all models -- matching the main table's "paired item-level
bootstrap" convention (B=2000) and keeping every model and language on identical items.

Resampling languages instead would measure "would another language sample change the ranking", which
is a different (and much wider) question; it is reported separately as the language-level spread.

  python scripts/probes/flores/floresA_ci.py [--B 2000]
"""
import argparse, json, random, statistics as st
import sacrebleu

R = "/mnt/ssd2/hyun2/AUNet/runs"
GROUP = {"Germanic": ["deu", "nld", "afr", "fao", "isl", "lim", "ltz"],
         "Romance / Italic": ["ita", "fur", "lij", "lmo", "srd", "scn", "vec", "spa", "ast", "cat",
                              "oci", "por", "glg", "pap", "kea"],
         "Esperanto": ["epo"]}
LANGS = [l for m in GROUP.values() for l in m]
# lingua b395f11: every byte model is scored with the official generator (no vocab_norm).
MODELS = {"Transformer": ("llama_1.8B_paper", "eval_floresA"),
          "AUNet": ("aunet2_1.3B", "eval_floresA"),
          "BPEByte": ("bpebyte_br_greedy_root_1.3B", "eval_floresA")}
EXT = {"H-Net": "hnet_floresA.json", "BLT t=1.34": "blt_rel_s*.json", "BLT t=1.61": "blt_t161_s*.json"}


def load_matched():
    out = {}
    for m, (run, sub) in MODELS.items():
        d = json.load(open(f"{R}/{run}/{sub}/results.json"))["samples"]
        out[m] = {l: ([x["bleu"][1] for x in d[f"floresA_{l}-en"]],
                      [x["bleu"][0] for x in d[f"floresA_{l}-en"]]) for l in LANGS}
    return out


def mean_bleu(data, m, idx):
    """Unweighted mean over languages of corpus BLEU on one resampled item set."""
    vals = []
    for l in LANGS:
        h, r = data[m][l]
        vals.append(sacrebleu.corpus_bleu([h[i] for i in idx], [[r[i] for i in idx]]).score)
    return st.mean(vals)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    data = load_matched()
    n = len(next(iter(data["Transformer"].values()))[0])
    names = list(MODELS)
    point = {m: mean_bleu(data, m, range(n)) for m in names}
    rng = random.Random(1234)
    draws = {m: [] for m in names}
    diffs = {("BPEByte", "AUNet"): [], ("BPEByte", "Transformer"): [], ("AUNet", "Transformer"): []}
    for b in range(a.B):
        idx = [rng.randrange(n) for _ in range(n)]          # one index set, shared by all models
        rep = {m: mean_bleu(data, m, idx) for m in names}
        for m in names:
            draws[m].append(rep[m])
        for (x, y) in diffs:
            diffs[(x, y)].append(rep[x] - rep[y])
    lo, hi = int(.025 * a.B), int(.975 * a.B)
    print(f"# FLORES X->En, paired item-level bootstrap (B={a.B}, {n} items shared across "
          f"{len(LANGS)} parallel languages)\n")
    print("| model | BLEU | 95% CI | half-width |")
    print("|---|---|---|---|")
    for m in names:
        s = sorted(draws[m])
        print(f"| {m} | {point[m]:.2f} | [{s[lo]:.2f}, {s[hi]:.2f}] | $\\pm${(s[hi]-s[lo])/2:.1f} |")
    print("\n| contrast | mean diff | 95% CI | p (two-sided) |")
    print("|---|---|---|---|")
    for (x, y), d in diffs.items():
        s = sorted(d); pt = point[x] - point[y]
        p = 2 * min(sum(1 for v in d if v <= 0), sum(1 for v in d if v >= 0)) / a.B
        print(f"| {x} - {y} | {pt:+.2f} | [{s[lo]:+.2f}, {s[hi]:+.2f}] | {min(1.0, p):.4f} |")
    # language-level spread, reported separately -- a different question
    per = {m: [sacrebleu.corpus_bleu(*[[x] for x in ([data[m][l][0]], [data[m][l][1]])]).score
               if False else sacrebleu.corpus_bleu(data[m][l][0], [data[m][l][1]]).score
               for l in LANGS] for m in names}
    print("\nlanguage-level spread (sd over the 23 per-language BLEUs, NOT a CI on the mean):")
    for m in names:
        print(f"  {m:12s} sd {st.pstdev(per[m]):5.2f}  min {min(per[m]):5.2f}  max {max(per[m]):5.2f}")
    json.dump({"point": point, "ci": {m: [sorted(draws[m])[lo], sorted(draws[m])[hi]] for m in names},
               "B": a.B, "n_items": n},
              open("/mnt/ssd2/hyun2/AUNet/reports_NAACL/flores_multilingual/results/xen_ci.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
