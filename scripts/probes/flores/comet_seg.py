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
def seg(d): return m.predict(d, batch_size=32, gpus=1, progress_bar=False).scores
S={}
for lg,col in COL.items():
    S[lg]={}
    for tag,run in RUNS.items():
        s=json.load(open(f"{R}/{run}/{SUB[lg]}/results.json"))["samples"][f"flores_dev_{lg}-en"]
        src=[x["doc"][col] for x in s]; ref=[x["bleu"][0] for x in s]; hyp=[x["bleu"][1] for x in s]
        S[lg][tag]=seg([{"src":a,"mt":b,"ref":c} for a,b,c in zip(src,hyp,ref)])
        if tag=="Llama":
            rng=random.Random(0); sh=ref[:]; rng.shuffle(sh)
            S[lg]["_floor"]=seg([{"src":a,"mt":b,"ref":c} for a,b,c in zip(src,sh,ref)])
    print("done",lg,flush=True)
json.dump(S, open("/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/flores/comet_segments.json","w"))

def boot(a,b,n=10000):
    rng=random.Random(1); k=len(a); d=sum(a)/k-sum(b)/k; c=0
    for _ in range(n):
        idx=[rng.randrange(k) for _ in range(k)]
        dd=sum(a[i]-b[i] for i in idx)/k
        if (dd<=0) if d>0 else (dd>=0): c+=1
    return d, (c+1)/(n+1)
print(f"\n{'lg':3s} {'BPE-AU':>16s} {'Llama-BPE':>16s} {'best-floor':>16s}")
for lg in COL:
    p=S[lg]
    d1,p1=boot(p["BPEByte"],p["AU-Net2"]); d2,p2=boot(p["Llama"],p["BPEByte"])
    bt=max(("Llama","AU-Net2","BPEByte"), key=lambda t:sum(p[t])); d3,p3=boot(p[bt],p["_floor"])
    print(f"{lg:3s} {d1:+.4f} p={p1:<6.3f} {d2:+.4f} p={p2:<6.3f} {d3:+.4f} p={p3:<6.4f} ({bt})")
