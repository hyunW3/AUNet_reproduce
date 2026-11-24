"""BLT's entropy threshold on FLORES-200 generation: does theta change what it translates?

Why this axis rather than the multiple-choice one. The existing threshold ablation
(table_appendix/blt_threshold_*.tex) scores theta on likelihood-based tasks, where the effect is
small and not consistent -- about -0.5 to -0.9 pp on the 5-task mean, driven almost entirely by
HellaSwag (-1.56), with WinoGrande moving the other way. Likelihood scoring makes ONE pass over a
fully known string and only has to rank the options correctly, so a shifted patch boundary rarely
flips the ranking.

Free generation is different in three ways that make it the better test:
  1. patch decisions ACCUMULATE: ~130 decoded bytes means ~130 chances for the entropy rule to
     place a boundary differently, and each one conditions everything after it;
  2. BLEU reads the surface string, so a degraded continuation costs score immediately, where MC
     accuracy is unchanged as long as the gold option still wins;
  3. 23 languages give 23 PAIRED comparisons on identical source sentences, so the verdict is a
     sign test over languages rather than one aggregate number.

Usage:
  python scripts/probes/flores/blt_theta_flores.py --rel <dir-or-files> --alt <dir-or-files>
"""
import argparse, glob, json, os, random, statistics as st
from math import comb

NAME = {"deu":"German","nld":"Dutch","afr":"Afrikaans","fao":"Faroese","isl":"Icelandic",
        "lim":"Limburgish","ltz":"Luxembourgish","ita":"Italian","fur":"Friulian","lij":"Ligurian",
        "lmo":"Lombard","srd":"Sardinian","scn":"Sicilian","vec":"Venetian","spa":"Spanish",
        "ast":"Asturian","cat":"Catalan","oci":"Occitan","por":"Portuguese","glg":"Galician",
        "pap":"Papiamento","kea":"Kabuverdianu","epo":"Esperanto"}
GROUP = {"Germanic": ["deu","nld","afr","fao","isl","lim","ltz"],
         "Romance / Italic": ["ita","fur","lij","lmo","srd","scn","vec","spa","ast","cat","oci",
                              "por","glg","pap","kea"],
         "Esperanto": ["epo"]}
LANGS = [l for m in GROUP.values() for l in m]


def collect(paths):
    """Merge `per_lang` across shard files; a shard never overwrites a language already seen."""
    out, thetas = {}, set()
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else glob.glob(p)
    for f in files:
        if f.endswith("_samples.json"):
            continue
        d = json.load(open(f))
        if "per_lang" not in d:
            continue
        thetas.add(round(float(d.get("threshold", float("nan"))), 4))
        for l, v in d["per_lang"].items():
            out.setdefault(l, v)
    return out, thetas


def sign_p(k, n):
    k = max(k, n - k)
    return 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", nargs="+", required=True, help="released-threshold shard dirs/files")
    ap.add_argument("--alt", nargs="+", required=True, help="alternative-threshold shard dirs/files")
    a = ap.parse_args()
    rel, th_r = collect(a.rel)
    alt, th_a = collect(a.alt)
    have = [l for l in LANGS if l in rel and l in alt]
    print(f"# BLT entropy threshold on FLORES-200 X->en  "
          f"(theta {sorted(th_r)} vs {sorted(th_a)}; {len(have)}/23 languages)\n")
    print("| language | | theta_rel BLEU | theta_alt BLEU | delta |")
    print("|---|---|---|---|---|")
    d = {}
    for g, ms in GROUP.items():
        for l in [x for x in ms if x in have]:
            r, v = rel[l]["bleu"], alt[l]["bleu"]
            d[l] = v - r
            print(f"| {NAME[l]} | {g[:3]} | {r:.2f} | {v:.2f} | {v - r:+.2f} |")
    print()
    for g, ms in GROUP.items():
        k = [x for x in ms if x in have]
        if not k:
            continue
        print(f"| *{g}* (n={len(k)}) | | {st.mean(rel[l]['bleu'] for l in k):.2f} | "
              f"{st.mean(alt[l]['bleu'] for l in k):.2f} | {st.mean(d[l] for l in k):+.2f} |")
    mr, ma = st.mean(rel[l]["bleu"] for l in have), st.mean(alt[l]["bleu"] for l in have)
    print(f"| **All** (n={len(have)}) | | **{mr:.2f}** | **{ma:.2f}** | **{ma - mr:+.2f}** |")
    worse = sum(1 for l in have if d[l] < 0)
    print(f"\nlower at the alternative threshold in {worse}/{len(have)} languages "
          f"(sign test p = {sign_p(worse, len(have)):.5f}); "
          f"mean {st.mean(d.values()):+.2f} BLEU, "
          f"worst {min(d.values()):+.2f} ({NAME[min(d, key=d.get)]}), "
          f"best {max(d.values()):+.2f} ({NAME[max(d, key=d.get)]})")
    json.dump({"per_lang_delta": d, "theta_rel": sorted(th_r), "theta_alt": sorted(th_a)},
              open("/mnt/ssd2/hyun2/AUNet/reports_NAACL/flores_multilingual/results/ext/"
                   "blt_theta_delta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
