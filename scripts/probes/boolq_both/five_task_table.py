"""Main-table robustness with BoolQ under the both-region protocol (review before the CI pass).
Per model: BoolQ delta per axis (Noise/Typo/Leet from runs/robustness_boolq_both, Despace = despace_mc_boolq_despaceall,
PBP = pbp_mc_boolq), and |mean delta| over the four tasks (current table) vs five tasks.
  python five_task_table.py [--bq_dir runs/robustness_boolq_both]
Needs paper_robustness_tables.py (AUNet scripts/probes) for the four-task cells."""
import argparse, glob, json, os, statistics as st, sys

L = "/mnt/ssd2/hyun2/AUNet"
sys.path.insert(0, f"{L}/scripts/probes")
import paper_robustness_tables as P  # noqa: E402

AX = ("pbp", "noise", "typo", "despace", "leet")
EXT = {"blt": "blt1335", "blt1609": "blt1609", "hnet": "hnet"}


def acc(r):
    return P._get(r, "acc")


def ld(f):
    return json.load(open(f))["results"] if os.path.exists(f) else {}


def bq_new(d, m):
    """BoolQ clean/perturbed for the new both-region runs: noise, typo, leet (None where missing)."""
    if m in P.MATCHED:
        nt, lt = ld(f"{d}/trio_nt_{m}/results.json"), ld(f"{d}/trio_leet_{m}/results.json")
        nr = tr = nt
    else:
        p = f"{d}/snu20/{EXT[m]}"   # BLT / H-Net ran on snu20 (run_boolq_both_snu20.sh), copied back here
        nr, tr = ld(f"{p}_noiseboth_boolq.json"), ld(f"{p}_typoboth_boolq.json")
        lt = ld(f"{p}_leetboth_boolq.json")
    out = {}
    nk = [f"boolq_noise_{s}_both" for s in P.NOISE]
    if "boolq" in nr and all(k in nr for k in nk):
        out["noise"] = (acc(nr["boolq"]), st.mean(acc(nr[k]) for k in nk))
    tk = [f"boolq_typoboth_{s}" for s in (f"{o}_{l}" for o in P.TYPO for l in P.TYPO_LVL)]
    if "boolq" in tr and all(k in tr for k in tk):
        out["typo"] = (acc(tr["boolq"]), st.mean(acc(tr[k]) for k in tk))
    c, p_ = lt.get("fmt_boolq_clean"), lt.get("fmt_boolq_nla_leet_both")
    if c and p_:
        out["leet"] = (100 * st.mean(c["bits"]["acc"]), 100 * st.mean(p_["bits"]["acc"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bq_dir", default=f"{L}/runs/robustness_boolq_both")
    a = ap.parse_args()
    R = P.load_results()
    rows = {}
    for m in P.MODELS:
        new = bq_new(a.bq_dir, m)
        dr, pr = R[m]["despace"], R[m]["pbp"]
        new["despace"] = (acc(dr["despace_mc_boolq_clean"]), acc(dr["despace_mc_boolq_despaceall"]))
        new["pbp"] = (acc(pr["pbp_mc_boolq_canonical"]), acc(pr["pbp_mc_boolq_space"]))
        d4 = {x: P.axis_delta(R[m], x) for x in AX}
        bq = {x: (new[x][1] - new[x][0]) if x in new else None for x in AX}
        d5 = {x: (4 * d4[x] + bq[x]) / 5 if bq[x] is not None else None for x in AX}
        rows[m] = dict(clean={x: new[x][0] for x in new}, bq=bq, d4=d4, d5=d5)
    f = lambda v, s="": "  --  " if v is None else f"{v:{s}6.2f}"
    print("BoolQ clean acc per run (within-run baseline)")
    for m, r in rows.items():
        print(f"  {m:8s}", "  ".join(f"{x}={f(r['clean'].get(x))}" for x in AX))
    print("\nBoolQ delta (signed, pts)")
    for m, r in rows.items():
        print(f"  {m:8s}", "  ".join(f"{x}={f(r['bq'][x], '+')}" for x in AX))
    print("\n|mean delta|: 4 tasks -> 5 tasks")
    for m, r in rows.items():
        cells = [f"{x}={abs(r['d4'][x]):5.2f}->{f(abs(r['d5'][x]) if r['d5'][x] is not None else None)}" for x in AX]
        av4 = st.mean(abs(r["d4"][x]) for x in AX)
        av5 = st.mean(abs(r["d5"][x]) for x in AX) if all(r["d5"][x] is not None for x in AX) else None
        print(f"  {m:8s}", "  ".join(cells), f"  Avg={av4:5.2f}->{f(av5)}")
    json.dump(rows, open(f"{a.bq_dir}/five_task_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
