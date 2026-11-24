"""Per-task 0-shot 95% CIs (paired item bootstrap, B=2000), reusing ci_main_table.py loaders/bootstrap."""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes")
import ci_main_table as C

out = {}
for prefix, tag in (("", ""), ("t1609_", "_t1609")):
    C.BLT_PREFIX = prefix
    models = C.MODELS if not prefix else ("blt",)
    pm = {m: d for m in models if (d := C.load_downstream(m, 0)) is not None and all(t in d for t in C.DS5)}
    A = C.to_arrays(pm, C.DS5)
    for t in C.DS5:
        n = len(next(iter(A[t].values())))
        rng = np.random.default_rng(0)
        fn = lambda idx, t=t: {m: 100 * A[t][m][idx[t]].mean() for m in pm}
        s = C.summarize(*C.bootstrap(fn, {t: n}, 2000, rng))
        for m, v in s.items():
            out.setdefault(m + tag, {})[t] = {"value": round(v["value"], 1), "hw": round(v["half_width"], 1), "n": n}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for m, d in out.items():
    print(f"{m:12s}", "  ".join(f"{t[:6]} {v['value']:.1f}±{v['hw']:.1f} (n={v['n']})" for t, v in d.items()))
