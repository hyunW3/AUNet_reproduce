import json
D = "/mnt/ssd2/hyun2/AUNet/runs/robustness_boolq_both"
F = {"llama": [f"{D}/trio_nt_llama/results.json"], "aunet": [f"{D}/trio_nt_aunet/results.json"],
     "bpebyte": [f"{D}/trio_nt_bpebyte/results.json"],
     "blt": [f"{D}/snu20/blt1335_noiseboth_boolq.json", f"{D}/snu20/blt1335_typoboth_boolq.json"],
     "hnet": [f"{D}/snu20/hnet_noiseboth_boolq.json", f"{D}/snu20/hnet_typoboth_boolq.json"]}
S = ["antspeak", "drop", "randomcase", "repeat", "uppercase"]
T = [f"{o}_{l}" for o in ("delete", "swap", "key", "insert") for l in ("char", "word")]
def a(r):
    return 100 * r.get("acc,none", r.get("acc"))
print("model    clean | noise-both: " + " ".join(f"{s[:6]:>6s}" for s in S) + " | typo-both: " + " ".join(f"{t[:4]+t[-1]:>5s}" for t in T))
for m, fs in F.items():
    r = {}
    for f in fs:
        r.update(json.load(open(f))["results"])
    print(f"{m:8s} {a(r['boolq']):5.1f} |             " + " ".join(f"{a(r[f'boolq_noise_{s}_both']):6.1f}" for s in S)
          + " |            " + " ".join(f"{a(r[f'boolq_typoboth_{t}']):5.1f}" for t in T))
