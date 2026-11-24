import json, random
from comet import download_model, load_from_checkpoint
R="/mnt/ssd2/hyun2/AUNet/runs"
COL={"de":"sentence_deu_Latn","fr":"sentence_fra_Latn","es":"sentence_spa_Latn","it":"sentence_ita_Latn",
     "nl":"sentence_nld_Latn","ko":"sentence_kor_Hang","sv":"sentence_swe_Latn","fi":"sentence_fin_Latn",
     "tr":"sentence_tur_Latn","hu":"sentence_hun_Latn","ru":"sentence_rus_Cyrl","vi":"sentence_vie_Latn"}
SUB={**{l:"eval_flores_dev" for l in ("de","fr","es","it","nl","ko")},
     **{l:"eval_flores2" for l in ("sv","fi","tr","hu","ru","vi")}}
RUNS={"Llama":"llama_1.8B_paper","AU-Net2":"aunet2_1.3B","BPEByte":"bpebyte_br_greedy_root_1.3B"}
m=load_from_checkpoint(download_model("Unbabel/wmt22-comet-da")); m.eval()
def score(data): return m.predict(data, batch_size=32, gpus=1, progress_bar=False).system_score
out={}
for lg,col in COL.items():
    per={}
    for tag,run in RUNS.items():
        S=json.load(open(f"{R}/{run}/{SUB[lg]}/results.json"))["samples"][f"flores_dev_{lg}-en"]
        src=[s["doc"][col] for s in S]; ref=[s["bleu"][0] for s in S]; hyp=[s["bleu"][1] for s in S]
        per[tag]=score([{"src":a,"mt":b,"ref":c} for a,b,c in zip(src,hyp,ref)])
        if tag=="Llama":
            rng=random.Random(0); sh=ref[:]; rng.shuffle(sh)
            per["_floor"]=score([{"src":a,"mt":b,"ref":c} for a,b,c in zip(src,sh,ref)])
            per["_copysrc"]=score([{"src":a,"mt":a,"ref":c} for a,c in zip(src,ref)])
    out[lg]=per
    print(f"  {lg:3s} Llama {per['Llama']:.4f}  AU-Net2 {per['AU-Net2']:.4f}  BPEByte {per['BPEByte']:.4f}"
          f"   | floor {per['_floor']:.4f}  copy-src {per['_copysrc']:.4f}"
          f"   ΔBPE-AU {per['BPEByte']-per['AU-Net2']:+.4f}", flush=True)
json.dump(out, open("/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/flores/comet.json","w"), indent=2)
print("WROTE comet.json")
