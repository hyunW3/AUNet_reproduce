"""Why BLT loses 24 points on BoolQ under Leet (context-only, format_mc nla_leet, seed 0) while the other models lose 0-9.
Per-item correctness bits of the main Leet runs (reports/format_robustness, FMT_SRC of paper_robustness_tables.py) on the
frozen 2,000 BoolQ items: (1) predicted-"yes" rate clean vs Leet; (2) accuracy change by context length quartile;
(3) patch statistics of the scored strings per task under Leet (reports/patch_stats_conditions/leet) vs clean
(reports/patch_stats downstream/*): bytes per item, patches per item, mean patch length, single-byte share.
  python blt_leet_boolq_analysis.py -> runs/robustness_boolq_ctx/blt_leet_boolq_analysis.json"""
import glob, json, os, statistics as st, sys

L = "/mnt/ssd2/hyun2/AUNet"
sys.path.insert(0, f"{L}/scripts/probes")
import paper_robustness_tables as P  # noqa: E402

ITEMS = json.load(open(f"{L}/reports/format_robustness/items/boolq_2000.json"))
GOLD = [x["gold"] for x in ITEMS]
LEN = [len(x["context"].encode()) for x in ITEMS]
SRC = dict(P.FMT_SRC, blt1609=["leet_extra/blt1609_boolq.json"])
NAMES = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte", "blt": "BLT 1.34", "blt1609": "BLT 1.61",
         "hnet": "H-Net"}
TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")


def bits(m):
    r = {}
    for p in SRC[m]:
        for f in glob.glob(os.path.join(P.FMT, p)):
            r.update(json.load(open(f))["results"])
    return r["fmt_boolq_clean"]["bits"]["acc"], r["fmt_boolq_nla_leet"]["bits"]["acc"]


def pred_yes(b):
    return 100 * st.mean((g == 1) == bool(c) for g, c in zip(GOLD, b))


def patch_row(h):
    n, nb, np_ = h["n_items"], h["total_bytes"], h["n_patches"]
    return dict(bytes_per_item=nb / n, patches_per_item=np_ / n, mean_len=nb / np_,
                single_pct=100 * h["hist"].get("1", 0) / np_)


def main():
    out = {"pred_yes": {}, "by_len": {}, "patches": {}}
    qs = sorted(LEN)
    cuts = [qs[len(qs) * k // 4] for k in (1, 2, 3)]
    q = [sum(x >= c for c in cuts) for x in LEN]
    print("context-length quartile cuts (bytes):", cuts)
    print(f"{'model':12s} clean  leet  | pred-yes clean leet | delta by length quartile Q1..Q4")
    for m in NAMES:
        c, p = bits(m)
        out["pred_yes"][m] = (pred_yes(c), pred_yes(p))
        dq = []
        for k in range(4):
            ix = [i for i in range(len(q)) if q[i] == k]
            dq.append(100 * (st.mean(p[i] for i in ix) - st.mean(c[i] for i in ix)))
        out["by_len"][m] = dq
        print(f"{NAMES[m]:12s} {100 * st.mean(c):5.1f} {100 * st.mean(p):5.1f} | {pred_yes(c):6.1f} {pred_yes(p):6.1f}   |",
              " ".join(f"{d:+6.1f}" for d in dq))
    print("\npatch statistics of the scored strings (clean -> Leet)")
    for m in ("blt", "bpebyte"):
        hc = json.load(open(f"{L}/reports/patch_stats/hist_{m}.json"))["benches"]
        hl = json.load(open(f"{L}/reports/patch_stats_conditions/leet/hist_{m}.json"))["benches"]
        for t in TASKS:
            a, b = patch_row(hc[f"downstream/{t}"]), patch_row(hl[f"leet/{t}"])
            out["patches"][f"{m}/{t}"] = {"clean": a, "leet": b}
            print(f"  {m:8s} {t:14s} bytes/item {a['bytes_per_item']:6.0f}->{b['bytes_per_item']:6.0f}  "
                  f"patches/item {a['patches_per_item']:6.1f}->{b['patches_per_item']:6.1f}  "
                  f"mean {a['mean_len']:4.2f}->{b['mean_len']:4.2f}  single {a['single_pct']:4.1f}%->{b['single_pct']:4.1f}%")
    json.dump(out, open(f"{L}/runs/robustness_boolq_ctx/blt_leet_boolq_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
