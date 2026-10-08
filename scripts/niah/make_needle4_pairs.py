"""(1) needle vs answer parse of the S-NIAH-3 UUID; (2) build tok4 / rand4 needle variants and show their parses."""
import json, sys, random, collections
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/lingua"); sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes")
from compression_stability import build_parser, TOK
from lingua.tokenizer import TikTokenTokenizer
P = {m: build_parser(m) for m in ("aunet", "bpebyte")}
pairs = [json.loads(l) for l in open("/mnt/ssd2/hyun2/AUNet/reports/niah/sniah123_n250_final_pairs.jsonl")]
recs = [p for p in pairs if p["probe"] == "sniah3"]

def segs(m, text, lo, hi):
    full = text.encode(); st = sorted(set(P[m](text))) + [len(full)]
    out = []
    for s0, s1 in zip(st, st[1:]):
        if s1 > lo and s0 < hi:
            out.append(full[max(s0, lo):min(s1, hi)].decode(errors="replace") + ("" if s0 >= lo and s1 <= hi else "~"))
    return out

def needle_vs_answer(rs, tag):
    for m in P:
        same = 0; ex = None
        for p in rs:
            v = p["values"][0]; text = p["prompt"] + " " + v; full = text.encode()
            c = full.find(v.encode()); a = len(full) - len(v)
            sn, sa = segs(m, text, c, c + len(v)), segs(m, text, a, a + len(v))
            same += sn == sa
            if sn != sa and ex is None: ex = (sn, sa)
        print(f"[{tag}] {m}: needle parse == answer parse in {same}/{len(rs)}" + (f"; e.g. needle {'|'.join(ex[0])} vs answer {'|'.join(ex[1])}" if ex else ""))

needle_vs_answer(recs, "uuid")
tk = TikTokenTokenizer(TOK).tkt_model
vocab4 = sorted({b.decode() for b in (tk.decode_single_token_bytes(i) for i in range(tk.n_vocab))
                 if len(b) == 4 and b.isalpha() and b.islower() and b.isascii()})
print("4-letter lowercase vocab tokens:", len(vocab4), vocab4[:10])
rng = random.Random(7)
def make(kind):
    if kind == "tok4": return "-".join(rng.choice(vocab4) for _ in range(7))
    return "-".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(4)) for _ in range(7))
for kind in ("tok4", "rand4"):
    out = []
    for p in recs:
        v0 = p["values"][0]; v = make(kind)
        out.append({**p, "prompt": p["prompt"].replace(v0, v), "values": [v]})
    json.dump(None, open("/dev/null", "w"))
    with open(f"/mnt/ssd2/hyun2/AUNet/reports/niah/force_byte/pairs_{kind}.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    for m in P:
        bw = collections.Counter()
        for r in out:
            v = r["values"][0]; text = r["prompt"] + " " + v; full = text.encode(); a = len(full) - len(v)
            st = sorted(set(P[m](text))) + [len(full)]
            for i in range(a, len(full)):
                s0 = max(x for x in st if x <= i); s1 = min(x for x in st if x > i); bw[s1 - s0] += 1
        n = sum(bw.values())
        r0 = out[0]; t0 = r0["prompt"] + " " + r0["values"][0]; e0 = len(t0.encode())
        print(f"[{kind}] {m}: e.g. {'|'.join(segs(m, t0, e0 - len(r0['values'][0]), e0))}  bytes by patch len: "
              + " ".join(f"{k}:{100*c/n:.0f}%" for k, c in sorted(bw.items())))
    needle_vs_answer(out, kind)
