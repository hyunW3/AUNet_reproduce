import glob, os, time
try:
    import orjson as J
    loads=J.loads
except ImportError:
    import json as J
    loads=J.loads
from multiprocessing import Pool
D="data/dclm_baseline_1.0_2shards_shuffled"
chunks=sorted(glob.glob(f"{D}/*.chunk.*.jsonl"))
def work(path):
    tb=nd=0; raw=os.path.getsize(path)
    with open(path,"rb") as f:
        for line in f:
            try: o=loads(line)
            except Exception: continue
            t=o.get("text")
            if t is not None:
                tb+=len(t.encode("utf-8")); nd+=1
    return (os.path.basename(path), tb, nd, raw)
if __name__=="__main__":
    t0=time.time()
    with Pool(16) as p:
        res=p.map(work, chunks)
    TB=sum(r[1] for r in res); ND=sum(r[2] for r in res); RAW=sum(r[3] for r in res)
    print(f"chunks={len(res)}  elapsed={time.time()-t0:.0f}s")
    print(f"raw_bytes      = {RAW:,}  ({RAW/1e9:.2f} GB)")
    print(f"text_bytes     = {TB:,}  ({TB/1e9:.2f} GB)")
    print(f"n_docs         = {ND:,}")
    print(f"text/raw frac  = {TB/RAW:.4f}")
    print(f"usable tokens (text bytes)      = {TB:,}  = {TB/1e9:.1f}B")
    print(f"usable tokens (+bos+eos, 2/doc) = {TB+2*ND:,}  = {(TB+2*ND)/1e9:.1f}B")
