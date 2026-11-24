"""COMET-22 on the AU-Net-paper FLORES language set, with the per-language floor.

Most of these 23 languages are low-resource varieties absent from the English DCLM training mix,
so the question that decides whether the BLEU table means anything is the same one as in
reports_afterAAAIsub/flores/comet_evaluation.md: how far above the shuffled-reference floor does
each language sit? Several are close relatives of a high-resource language (Ligurian/Lombard/
Venetian ~ Italian, Asturian/Galician ~ Spanish/Portuguese, Occitan ~ Catalan, Limburgish ~ Dutch,
Faroese ~ Icelandic), so lexical transfer may keep them off the floor where Finnish could not.

Needs the isolated COMET venv (see comet_evaluation.md §7), not the project venv.
"""
import json, random, statistics as st
from comet import download_model, load_from_checkpoint

R = "/mnt/ssd2/hyun2/AUNet/runs"
RUNS = {"Llama": "llama_1.8B_paper", "AU-Net2": "aunet2_1.3B",
        "BPEByte": "bpebyte_br_greedy_root_1.3B"}
CODE = {"deu":"deu_Latn","nld":"nld_Latn","afr":"afr_Latn","fao":"fao_Latn","isl":"isl_Latn",
 "lim":"lim_Latn","ltz":"ltz_Latn","ita":"ita_Latn","fur":"fur_Latn","lij":"lij_Latn",
 "lmo":"lmo_Latn","srd":"srd_Latn","scn":"scn_Latn","vec":"vec_Latn","spa":"spa_Latn",
 "ast":"ast_Latn","cat":"cat_Latn","oci":"oci_Latn","por":"por_Latn","glg":"glg_Latn",
 "pap":"pap_Latn","kea":"kea_Latn","epo":"epo_Latn"}
GROUP = {"Germanic": ["deu","nld","afr","fao","isl","lim","ltz"],
         "Romance": ["ita","fur","lij","lmo","srd","scn","vec","spa","ast","cat","oci","por",
                     "glg","pap","kea"],
         "Constructed": ["epo"]}

m = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da")); m.eval()
def seg(d): return m.predict(d, batch_size=32, gpus=1, progress_bar=False).scores

def boot(a, b, n=10000, seed=1):
    rng = random.Random(seed); k = len(a); d = st.mean(a) - st.mean(b); c = 0
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        dd = sum(a[i] - b[i] for i in idx) / k
        if (dd <= 0) if d > 0 else (dd >= 0): c += 1
    return d, (c + 1) / (n + 1)

S = {}
for g, members in GROUP.items():
    for lg in members:
        col = "sentence_" + CODE[lg]; S[lg] = {}
        for tag, run in RUNS.items():
            smp = json.load(open(f"{R}/{run}/eval_floresA/results.json"))["samples"][f"floresA_{lg}-en"]
            src = [x["doc"][col] for x in smp]
            ref = [x["bleu"][0] for x in smp]; hyp = [x["bleu"][1] for x in smp]
            S[lg][tag] = seg([{"src": a, "mt": b, "ref": c} for a, b, c in zip(src, hyp, ref)])
            if tag == "Llama":
                rng = random.Random(0); sh = ref[:]; rng.shuffle(sh)
                S[lg]["_floor"] = seg([{"src": a, "mt": b, "ref": c} for a, b, c in zip(src, sh, ref)])
                S[lg]["_copysrc"] = seg([{"src": a, "mt": a, "ref": c} for a, c in zip(src, ref)])
        p = S[lg]; M = lambda t: st.mean(p[t])
        d1, q1 = boot(p["BPEByte"], p["AU-Net2"])
        print(f"  {lg} ({g[:3]})  Llama {M('Llama'):.4f}  AU {M('AU-Net2'):.4f}  BPE {M('BPEByte'):.4f}"
              f" | floor {M('_floor'):.4f}  headroom {max(M('Llama'),M('AU-Net2'),M('BPEByte'))-M('_floor'):+.3f}"
              f" | BPE-AU {d1:+.4f} p={q1:.3f}", flush=True)
json.dump(S, open("/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/flores/comet_floresA_segments.json", "w"))
print("WROTE comet_floresA_segments.json")
