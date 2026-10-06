import json, re
T = "/home/hyunw3/.claude/jobs/003c5e28/tmp/"
P = json.load(open(T + "lambada_prompts.json"))
W = lambda s: re.findall(r"[A-Za-z0-9'’]+", s)
for reg in ("llama_1.3b", "bpebyte_1.3b"):
    g = json.load(open(f"{T}gen_{reg}.json"))
    c = {"same as target (scoring nuance)": 0, "word from the scored passage": 0, "word only in the demonstrations": 0,
         "word in neither (new word)": 0, "empty/other": 0}
    cap = {"pred capitalized (name-like)": 0, "target capitalized": 0}
    for n, i in enumerate(g["ids"]):
        tgt = P["0"][i]["target"].strip()
        ctx0 = P["0"][i]["ctx"]
        demo = P["3"][i]["ctx"][: len(P["3"][i]["ctx"]) - len(ctx0)]
        w = W(g["gen"]["3"][n])
        if not w:
            c["empty/other"] += 1; continue
        w = w[0]
        cap["pred capitalized (name-like)"] += w[0].isupper()
        cap["target capitalized"] += tgt[:1].isupper()
        if w == tgt.strip('"“'):
            c["same as target (scoring nuance)"] += 1
        elif w in W(ctx0):
            c["word from the scored passage"] += 1
        elif w in W(demo):
            c["word only in the demonstrations"] += 1
        else:
            c["word in neither (new word)"] += 1
    N = len(g["ids"])
    print(f"\n{reg}: {N} items right 0-shot, wrong 3-shot; first predicted word at 3-shot:")
    for k, v in {**c, **cap}.items():
        print(f"  {k:36s} {v:4d} ({v/N:5.1%})")
