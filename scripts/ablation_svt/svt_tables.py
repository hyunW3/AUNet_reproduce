import json, os
S = "/home/hwbae/AUNet_svt"
OLD = "/home/hwbae/AUNet/runs/poc/portable_aunetlaw/eval_law100M_full"
ALL = ["hellaswag", "arc_easy", "arc_challenge", "boolq", "piqa", "winogrande"]
FOCUS = ["hellaswag", "arc_easy", "piqa"]


def get(r, t):
    v = r.get(t, {})
    return v.get("acc_norm,none", v.get("acc,none"))


def row(name, path):
    if not os.path.exists(path):
        return
    r = json.load(open(path))["results"]
    vals = [get(r, t) for t in ALL]
    cells = "  ".join(f"{v * 100:5.1f}" if v is not None else "   NA" for v in vals)
    f = [get(r, t) for t in FOCUS]
    a = [v for v in vals if v is not None]
    print(f"{name:26s} {cells}   | {sum(f) / len(f) * 100:6.2f}  {sum(a) / len(a) * 100:6.2f}")


print(f"{'model':26s} " + "  ".join(t[:5] for t in ALL) + "   | HS/AE/PI  all-6")
row("lb_rg_100M (July eval)", f"{OLD}/lb_rg_100M/results.json")
row("lb_aunet_100M (July eval)", f"{OLD}/lb_aunet_100M/results.json")
for a in ["lb_rg_100M", "stride4p57", "rg_llama3_V32k", "rg_gpt2", "rg_qwen2", "rg_snapshot_repro", "rg_randtrie_mcr"]:
    row(a, f"{S}/eval/{a}/results.json")
print()
for l in open(f"{S}/heldout/bpb_results.jsonl"):
    print(l.strip())
