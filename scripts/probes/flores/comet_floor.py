"""Does COMET discriminate where chrF cannot? Calibrate it against the same floor baselines.

chrF gives 'unrelated fluent English' 17.92 against a true-random 11.73, so the models that score
21.8-23.9 on fi/hu/tr/vi are only 4-6 points above 'ignored the source'. COMET is a learned
adequacy metric, so it may separate that region. If it does, the agglutinative test becomes
answerable with these models after all.
"""
import json, random, string, sys
from comet import download_model, load_from_checkpoint
R="/mnt/ssd2/hyun2/AUNet/runs"
d=json.load(open(f"{R}/llama_1.8B_paper/eval_flores_dev/results.json"))
S=d["samples"]["flores_dev_de-en"]
src=[s["doc"]["sentence_deu_Latn"] for s in S]
refs=[s["bleu"][0] for s in S]
hyps=[s["bleu"][1] for s in S]
rng=random.Random(0)
sh=refs[:]; rng.shuffle(sh)
words=[w for r in refs for w in r.split()]
randw=[" ".join(rng.choice(words) for _ in range(len(r.split()))) for r in refs]
randc=["".join(rng.choice(string.ascii_lowercase+" ") for _ in range(len(r))) for r in refs]
const=["The government said that the new system will be available next year." for _ in refs]
p=download_model("Unbabel/wmt22-comet-da")
m=load_from_checkpoint(p); m.eval()
print("\nde->en, 200 FLORES dev sentences — COMET-22 vs the chrF floor ladder\n")
for name, h, chrf in [("model: Llama 1.8B", hyps, 46.05),
                      ("unrelated real English (shuffled refs)", sh, 17.92),
                      ("bag of English words, random order", randw, 19.06),
                      ("one constant English sentence", const, 12.64),
                      ("random lowercase characters", randc, 11.73),
                      ("the German source, copied", src, 28.44),
                      ("the reference itself", refs, 100.0)]:
    data=[{"src":a,"mt":b,"ref":c} for a,b,c in zip(src,h,refs)]
    out=m.predict(data, batch_size=32, gpus=1, progress_bar=False)
    print(f"  {name:42s} COMET {out.system_score:.4f}   (chrF {chrf:5.2f})")
