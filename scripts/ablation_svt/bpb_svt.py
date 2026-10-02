import json, math
H = "/home/hwbae/AUNet/runs/poc/portable_aunetlaw"
S = "/home/hwbae/AUNet_svt/runs"
runs = {
    "baseline lb_rg_100M": f"{H}/lb_rg_100M",
    "stride4p57": f"{S}/stride4p57",
    "rg_llama3_V32k": f"{S}/rg_llama3_V32k",
    "rg_gpt2": f"{S}/rg_gpt2",
    "rg_qwen2": f"{S}/rg_qwen2",
    "aunet lb_aunet_100M": f"{H}/lb_aunet_100M",
}
for n, d in runs.items():
    L = [json.loads(l) for l in open(d + "/metrics.jsonl")]
    L = [x for x in L if "loss/out" in x]
    last = L[-20:]
    bpb = sum(x["loss/out"] for x in last) / len(last) / math.log(2)
    it = sorted(x["speed/curr_iter_time"] for x in L)[len(L) // 2]
    keys = [k for k in L[-1] if any(s in k for s in ("patch", "ratio", "compress", "level", "bpp"))]
    extra = {k: round(L[-1][k], 3) for k in keys}
    step = L[-1]["global_step"]
    print(f"{n:22s} steps={step} BPB={bpb:.4f} median_iter={it:.3f}s {extra}")
