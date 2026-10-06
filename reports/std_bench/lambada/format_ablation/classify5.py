"""Where does a wrong GPT-3-format 5-shot prediction come from?"""
import json, re
T = "/home/hyunw3/.claude/jobs/003c5e28/tmp/"
P5 = json.load(open(T + "gpt3_prompts5.json"))["5"]
W = lambda s: re.findall(r"[A-Za-z0-9'’]+", s)
for name in ("blt", "hnet"):
    g = json.load(open(f"{T}gen5_{name}.json"))
    cnt = {"= target": 0, "an earlier demo's answer": 0, "word in the scored passage": 0, "word in a demo passage only": 0,
           "new word": 0, "empty": 0}
    for i, out in zip(g["ids"], g["gen"]):
        ctx, tgt = P5[i]["ctx"], P5[i]["target"].strip()
        blocks = ctx.split("\n\n")
        # demos end with "____. -> answer"; the scored passage is everything after the last demo answer
        cut = ctx.rfind(" ->", 0, len(ctx) - 3)
        demo_part = ctx[:ctx.find("\n", cut) if cut >= 0 else 0]
        scored = ctx[len(demo_part):]
        answers = re.findall(r"____\. -> ([^\n]+)", demo_part)
        w = W(out)
        if not w:
            cnt["empty"] += 1; continue
        w = w[0]
        if w == tgt:
            cnt["= target"] += 1
        elif w in [a.strip() for a in answers]:
            cnt["an earlier demo's answer"] += 1
        elif w in W(scored):
            cnt["word in the scored passage"] += 1
        elif w in W(demo_part):
            cnt["word in a demo passage only"] += 1
        else:
            cnt["new word"] += 1
    n = len(g["ids"])
    print(f"\n{name}: {n} items (BPEByte right, {name} wrong, GPT-3 format 5-shot)")
    for k, v in cnt.items():
        print(f"  {k:30s} {v:4d} ({v/n:5.1%})")
