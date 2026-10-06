import json
D = "/mnt/ssd2/hyun2/AUNet/runs/robustness_boolq_both"
it = json.load(open("/mnt/ssd2/hyun2/AUNet/reports/format_robustness/items/boolq_2000.json"))
gold = [x["gold"] for x in it]
print("choices", it[0]["choices"], "gold=yes rate", round(sum(g == 1 for g in gold) / len(gold), 3))
F = {"llama": f"{D}/trio_leet_llama/results.json", "aunet": f"{D}/trio_leet_aunet/results.json",
     "bpebyte": f"{D}/trio_leet_bpebyte/results.json", "blt": f"{D}/snu20/blt1335_leetboth_boolq.json",
     "blt1609": f"{D}/snu20/blt1609_leetboth_boolq.json", "hnet": f"{D}/snu20/hnet_leetboth_boolq.json"}
for m, f in F.items():
    r = json.load(open(f))["results"]
    out = []
    for v in ("fmt_boolq_clean", "fmt_boolq_nla_leet_both"):
        b = r[v]["bits"]["acc"]
        n = len(b)
        pyes = sum((g == 1) == bool(c) for g, c in zip(gold[:n], b)) / n
        out.append(f"{v[10:]}: acc={100 * sum(b) / n:.1f} pred_yes={100 * pyes:.1f}%")
    print(f"{m:8s}", " | ".join(out))
