#!/usr/bin/env python3
"""Referent choice on LAMBADA name items: score every capitalized word of the scored passage as the answer.

Items: the target is capitalized and occurs earlier in the passage (a character name to copy), first N such items
in doc order. Candidates: distinct capitalized words of the scored passage that occur mid-sentence (names), plus the target.
Prompts (5-shot, exact std_bench prompts): std = lambada_prompts.json["5"]; cloze = gpt3_prompts5.json["5"]
("____. ->"); answer = the cloze prompts with " ____. ->" replaced by " ____. Answer:" (= task lambada_gpt3_answer).
For each (item, format) the continuation " <cand>" is scored with the model's loglikelihood; we keep the summed
log-prob and is_greedy of every candidate.

  runpy.sh cand_score.py --model llama_1.3b ; runext.sh blt cand_score.py --model blt_1b ; ...
  -> <tmp>/cand_<model>.json
"""
import argparse, json, re, sys
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/.claude/worktrees/lambada-fewshot-avg6/scripts/probes/std_bench")

T = "/home/hyunw3/.claude/jobs/003c5e28/tmp"
W = re.compile(r"[A-Z][a-z]+")
STOP = set("""I A An The It Its He She We You They Me Him Her Us Them My Your His Our Their This That These Those
What Who Whom Whose Which Why How When Where There Here Then Now Yes No Not Oh Ok Okay Well So But And Or If
Mr Mrs Ms Miss Dr Sir Lord Lady Mom Dad Mother Father God Do Did Does Is Are Was Were Be Been Have Has Had
Can Could Will Would Should Shall May Might Must Let Just Please Thanks Thank Hey Hi Hello Sorry Maybe Even
All Any Some Every Each One Two Three First Last Next Still Yet Too Also Only Very Really Of In On At To For
With From By As About After Before Because Since While Until Though Although""".split())


def names(text):
    """Capitalized words that occur at least once mid-sentence (proper nouns, not sentence-initial words)."""
    out = set()
    for m in W.finditer(text):
        pre = text[:m.start()].rstrip(" \t\"'“‘(")
        if pre and pre[-1] not in ".!?\n:;—-" and not pre.endswith("..."):
            if m.group() not in STOP:
                out.add(m.group())
    return out


def items(n):
    P0 = json.load(open(f"{T}/lambada_prompts.json"))["0"]
    S5 = json.load(open(f"{T}/lambada_prompts.json"))["5"]
    G5 = json.load(open(f"{T}/gpt3_prompts5.json"))["5"]
    S3 = json.load(open(f"{T}/lambada_prompts.json"))["3"]
    G3 = json.load(open(f"{T}/gpt3_prompts.json"))["3"]
    out = []
    for i, x in enumerate(P0):
        tgt = x["target"].strip()
        passage = x["ctx"]
        if not W.fullmatch(tgt) or not re.search(r"\b" + re.escape(tgt) + r"\b", passage):
            continue
        cands = sorted(set(names(passage)) | {tgt})
        if len(cands) < 2:
            continue
        g = G5[i]["ctx"]
        assert g.endswith(" ____. ->") and g.count(" ____. ->") == 6
        assert G3[i]["ctx"].endswith(passage + " ____. ->") and S3[i]["ctx"].endswith(passage)
        out.append({"i": i, "tgt": tgt, "cands": cands,
                    "ctx": {"std": S5[i]["ctx"], "cloze": g, "answer": g.replace(" ____. ->", " ____. Answer:"),
                            "std0": passage, "std3": S3[i]["ctx"], "cloze0": passage + " ____. ->", "cloze3": G3[i]["ctx"]}})
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--formats", default="std,cloze,answer", help="subset of std,cloze,answer,std0,std3,cloze0,cloze3")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import models
    from lm_eval.api.instance import Instance
    X = items(a.n)
    keep = a.formats.split(",")
    for it in X:
        it["ctx"] = {f: it["ctx"][f] for f in keep}
    spec = models.resolve(a.model, None, None, None)
    lm, _ = models.build(spec)
    reqs, keys = [], []
    for k, it in enumerate(X):
        for f, ctx in it["ctx"].items():
            for c in it["cands"]:
                reqs.append(Instance(request_type="loglikelihood", doc={}, arguments=(ctx, " " + c), idx=0))
                keys.append((k, f, c))
    print(f"{a.model}: {len(X)} items, {len(reqs)} requests", flush=True)
    res = lm.loglikelihood(reqs)
    out = [{"i": it["i"], "tgt": it["tgt"], "cands": it["cands"], "ll": {f: {} for f in it["ctx"]}, "greedy": {f: {} for f in it["ctx"]}}
           for it in X]
    for (k, f, c), (ll, g) in zip(keys, res):
        out[k]["ll"][f][c] = float(ll)
        out[k]["greedy"][f][c] = bool(g)
    json.dump(out, open(f"{T}/cand_{a.model}{a.tag}.json", "w"))
    print("wrote", f"{T}/cand_{a.model}{a.tag}.json")


if __name__ == "__main__":
    main()
