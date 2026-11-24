"""Aggregate the en->X runs on the AU-Net language set.

Two differences from floresA_report.py (the X->en side):

 1. BPEByte is sharded over four dump dirs, so its samples are merged before scoring.
 2. The decisive baseline is **copying the English source**, not the shuffled-reference floor.
    All three models are trained on an English DCLM mix, so en->X asks them to generate a language
    they have barely seen; for Romance targets, cognate overlap gives an untranslated English copy
    a non-trivial BLEU. A model at or below source-copy has not translated anything, and the
    shuffled-reference floor (unrelated target-language text) is the weaker, second check.

  python scripts/probes/flores/floresAX_report.py [--n_boot 1000]
"""
import json, argparse, random, statistics as st
from pathlib import Path
import sacrebleu

R = Path("/mnt/ssd2/hyun2/AUNet/runs")
# BPEByte's en->X was sharded over snu20's four GPUs (8/8/7 languages); its samples are merged.
# Llama's en->X predates the generator fixes but is unaffected by them (apps/main/generate.py calls
# model.forward); AU-Net2 and BPEByte were rerun with AUNET_FIX_VOCAB_NORM=1 AUNET_INC_PARSE=1.
SHARDS = ["eval_floresAX_s1", "eval_floresAX_s2", "eval_floresAX_s3", "eval_floresAX_s4"]
RUNS = {"Llama": ("llama_1.8B_paper", ["eval_floresAX"]),
        "AU-Net2": ("aunet2_1.3B", ["eval_floresAX"]),
        "BPEByte": ("bpebyte_br_greedy_root_1.3B", SHARDS)}
# NOTE: AUNet lists these 23 languages flat, with no group headers; the Germanic /
# Romance split below is ours. "Esperanto" is one constructed language, not a family --
# it is kept as its own row for completeness, never reported as a group mean.
GROUP = {"Germanic": ["deu", "nld", "afr", "fao", "isl", "lim", "ltz"],
         "Romance": ["ita", "fur", "lij", "lmo", "srd", "scn", "vec", "spa", "ast", "cat",
                     "oci", "por", "glg", "pap", "kea"],
         "Esperanto": ["epo"]}
NAME = {"deu":"German","nld":"Dutch","afr":"Afrikaans","fao":"Faroese","isl":"Icelandic",
 "lim":"Limburgish","ltz":"Luxembourgish","ita":"Italian","fur":"Friulian","lij":"Ligurian",
 "lmo":"Lombard","srd":"Sardinian","scn":"Sicilian","vec":"Venetian","spa":"Spanish",
 "ast":"Asturian","cat":"Catalan","oci":"Occitan","por":"Portuguese","glg":"Galician",
 "pap":"Papiamento","kea":"Kabuverdianu","epo":"Esperanto"}
LANGS = [l for g in GROUP.values() for l in g]


def load(run, subs):
    """{lang: (hyps, refs, srcs)} merged across shards; a later shard never overwrites an earlier one."""
    out = {}
    for sub in subs:
        f = R / run / sub / "results.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        for lang in LANGS:
            key = f"floresAX_en-{lang}"
            if key not in d.get("samples", {}):
                continue
            smp = d["samples"][key]
            if lang in out:
                continue
            out[lang] = ([s["bleu"][1] for s in smp], [s["bleu"][0] for s in smp],
                         [s["doc"]["sentence_eng_Latn"] for s in smp])
    return out


def bleu(h, r):
    return sacrebleu.corpus_bleu(h, [r]).score


def chrf(h, r):
    return sacrebleu.corpus_chrf(h, [r]).score


def paired(a, b, refs, n, rng):
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
    a = ap.parse_args(); rng = random.Random(0)
    data = {t: load(r, s) for t, (r, s) in RUNS.items()}
    have = [l for l in LANGS if all(l in data[t] for t in RUNS)]
    missing = [l for l in LANGS if l not in have]
    n_sent = len(next(iter(data["Llama"].values()))[0])
    print(f"# FLORES-200 en->X, AU-Net language set — devtest, 5-shot, n={n_sent}/lang, "
          f"{len(have)}/23 languages" + (f" (missing: {missing})" if missing else "") + "\n")

    res = {}
    print("| lang | | Llama | AU-Net2 | BPEByte | **copy-EN src** | shuffled-ref floor | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for g, members in GROUP.items():
        for l in [x for x in members if x in have]:
            hyp_ll, refs, srcs = data["Llama"][l]
            cp = bleu(srcs, refs)                       # untranslated English copy
            sh = refs[:]; random.Random(0).shuffle(sh)
            fl = bleu(sh, refs)                         # unrelated target-language text
            row = {}
            for t in RUNS:
                h, r, _ = data[t][l]
                row[t] = dict(bleu=bleu(h, r), chrf=chrf(h, r))
            best = max(row[t]["bleu"] for t in RUNS)
            v = "**TRANSLATES**" if best > cp * 1.25 else (
                "marginal" if best > cp else "**AT/BELOW COPY**")
            res[l] = dict(models=row, copy_src=cp, floor=fl, verdict=v)
            print(f"| {NAME[l]} | {g[:3]} | " + " | ".join(f"{row[t]['bleu']:.2f}" for t in RUNS)
                  + f" | **{cp:.2f}** | {fl:.2f} | {v} |")

    print(f"\n## Group means (BLEU)\n")
    print("| group | n | " + " | ".join(RUNS) + " | copy-EN src |")
    print("|---|---|" + "---|" * (len(RUNS) + 1))
    for g, members in GROUP.items():
        ms = [x for x in members if x in have]
        if not ms: continue
        print(f"| {g} | {len(ms)} | "
              + " | ".join(f"{st.mean([res[l]['models'][t]['bleu'] for l in ms]):.2f}" for t in RUNS)
              + f" | {st.mean([res[l]['copy_src'] for l in ms]):.2f} |")
    print(f"| **all** | {len(have)} | "
          + " | ".join(f"**{st.mean([res[l]['models'][t]['bleu'] for l in have]):.2f}**" for t in RUNS)
          + f" | **{st.mean([res[l]['copy_src'] for l in have]):.2f}** |")

    print(f"\n## Paired bootstrap (BLEU)\n")
    print("| lang | BPEByte − AU-Net2 | p | BPEByte − Llama | p |")
    print("|---|---|---|---|---|")
    d1s, d2s = [], []
    for l in have:
        refs = data["Llama"][l][1]
        d1, p1 = paired(data["BPEByte"][l][0], data["AU-Net2"][l][0], refs, a.n_boot, rng)
        d2, p2 = paired(data["BPEByte"][l][0], data["Llama"][l][0], refs, a.n_boot, rng)
        d1s.append((d1, p1)); d2s.append((d2, p2))
        m1 = "**" if p1 < .05 else ""; m2 = "**" if p2 < .05 else ""
        print(f"| {NAME[l]} | {m1}{d1:+.2f}{m1} | {m1}{p1:.3f}{m1} | {m2}{d2:+.2f}{m2} | {m2}{p2:.3f}{m2} |")
    from math import comb
    sgn = lambda k, n: 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    for lab, ds in (("BPEByte−AU-Net2", d1s), ("BPEByte−Llama", d2s)):
        w = sum(1 for d, _ in ds if d > 0); s = sum(1 for _, p in ds if p < .05)
        print(f"\n{lab}: mean {st.mean([d for d, _ in ds]):+.2f} BLEU · positive in {w}/{len(ds)} "
              f"(sign p={sgn(max(w, len(ds)-w), len(ds)):.5f}) · significant in {s}")
    json.dump(res, open("/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/flores/floresAX.json", "w"), indent=1)


if __name__ == "__main__":
    main()
