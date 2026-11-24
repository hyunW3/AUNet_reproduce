import json, random, statistics as st
from comet import download_model, load_from_checkpoint
R="/mnt/ssd2/hyun2/AUNet/runs"
RUNS={"Llama":"llama_1.8B_paper","AU-Net2":"aunet2_1.3B","BPEByte":"bpebyte_br_greedy_root_1.3B"}
TASK={"wmt16-de-en":"de","wmt14-fr-en":"fr"}
m=load_from_checkpoint(download_model("Unbabel/wmt22-comet-da")); m.eval()
def seg(d): return m.predict(d, batch_size=32, gpus=1, progress_bar=False).scores
S={}
for task,sl in TASK.items():
    S[task]={}
    for tag,run in RUNS.items():
        smp=json.load(open(f"{R}/{run}/eval_wmt/results.json"))["samples"][task]
        src=[x["doc"]["translation"][sl] for x in smp]
        ref=[x["bleu"][0] for x in smp]; hyp=[x["bleu"][1] for x in smp]
        S[task][tag]=seg([{"src":a,"mt":b,"ref":c} for a,b,c in zip(src,hyp,ref)])
        if tag=="Llama":
            rng=random.Random(0); sh=ref[:]; rng.shuffle(sh)
            S[task]["_floor"]=seg([{"src":a,"mt":b,"ref":c} for a,b,c in zip(src,sh,ref)])
            S[task]["_copysrc"]=seg([{"src":a,"mt":a,"ref":c} for a,c in zip(src,ref)])
    print("done",task,flush=True)
json.dump(S, open("/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/flores/comet_wmt_segments.json","w"))
def boot(a,b,n=10000,seed=1):
    rng=random.Random(seed); k=len(a); d=st.mean(a)-st.mean(b); c=0
    for _ in range(n):
        idx=[rng.randrange(k) for _ in range(k)]
        dd=sum(a[i]-b[i] for i in idx)/k
        if (dd<=0) if d>0 else (dd>=0): c+=1
    return d,(c+1)/(n+1)
print()
for task in TASK:
    p=S[task]; M=lambda t: st.mean(p[t])
    print(f"{task}  Llama {M('Llama'):.4f}  AU-Net2 {M('AU-Net2'):.4f}  BPEByte {M('BPEByte'):.4f}"
          f"  floor {M('_floor'):.4f}  copy-src {M('_copysrc'):.4f}")
    for a,b in [("BPEByte","AU-Net2"),("Llama","BPEByte"),("Llama","AU-Net2")]:
        d,q=boot(p[a],p[b]); print(f"    {a:8s}-{b:8s} {d:+.4f} p={q:.3f}")
    bt=max(("Llama","AU-Net2","BPEByte"),key=M); d,q=boot(p[bt],p["_floor"])
    print(f"    best({bt}) - floor {d:+.4f} p={q:.4f}")
