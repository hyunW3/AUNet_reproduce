#!/usr/bin/env python3
"""NUMCoT (Xu et al., ACL Findings 2024, arXiv 2406.02864) English items for base LMs.

Source: github.com/CAS-SIAT-ConsistencyAI/NUMCoT @ 437a60b6 (2024-06-06), the ChatGPT result CSVs, which hold every
question, gold answer and the exact chat prompt of each condition:
  exp1 numeral conversion  num2word / word2num x easy(400) medium(400) hard_fraction(200) hard_decimal(200)
  exp2 units               easy / medium / hard (200 each)
  exp3 MWPs                AddSub 395, SingleOp 562, MultiArith 600
Conditions as in the paper: zeroshot, cot_zeroshot, fewshot, cot_fewshot, and (units only) numcot = few-shot CoT with
unit-conversion knowledge. Chinese splits and SuanJing are left out (the models are English-only).

The chat prompts are flattened for base models: system message, blank line, then each shot as
"<user text, right-stripped> <assistant text>", shots separated by a blank line, and the final user text right-stripped
(so a prompt never ends in a space; the model writes " <answer>"). Every model scores byte-identical prompts.

Generation budget (bytes), fixed per split (never per item): without CoT max(64, 1.5 x longest gold + 32) rounded up
to 16; with CoT min(1024, 1.5 x longest CoT demonstration answer of the split's few-shot CoT prompt) rounded up to 64
(zero-shot CoT uses the same budget). Generation stops at a newline or at the start of a next question (gen_items.py).
fits4k = prompt + budget <= 4095 bytes: BLT's 4096-byte context; items without it are left-truncated for BLT only.

  python reports_afterNAACL/numcot/scripts/build_items.py <NUMCoT repo dir>
"""
import ast, csv, json, re, sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(10**9)
REPO = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "items.jsonl"
D = REPO / "datasets/data"
FILES = {
    ("num2word", "easy"): "experience_first/num2word_chatGPT_easy_en_20240125152903_400.csv",
    ("num2word", "medium"): "experience_first/num2word_chatGPT_medium_en_20240125083941_400.csv",
    ("num2word", "hard_fraction"): "experience_first/num2word_chatGPT_hard_fraction_en_20240126104828_200.csv",
    ("num2word", "hard_decimal"): "experience_first/num2word_chatGPT_hard_decimal_en_20240126102450_200.csv",
    ("word2num", "easy"): "experience_first/word2num_chatGPT_easy_en_20240125155113_400.csv",
    ("word2num", "medium"): "experience_first/word2num_chatGPT_medium_en_20240125085607_400.csv",
    ("word2num", "hard_fraction"): "experience_first/word2num_chatGPT_hard_fraction_en_20240126110704_200.csv",
    ("word2num", "hard_decimal"): "experience_first/word2num_chatGPT_hard_decimal_en_20240126103542_200.csv",
    ("units", "easy"): "experience_second/chatGPT_easy_en_20240129095408_200.csv",
    ("units", "medium"): "experience_second/chatGPT_medium_en_20240129095623_200.csv",
    ("units", "hard"): "experience_second/chatGPT_hard_en_20240129095856_200.csv",
    ("mwp", "AddSub"): "experience_third/chatGPT_AddSub_en_20240130192341_395.csv",
    ("mwp", "SingleOp"): "experience_third/chatGPT_SingleOp_en_20240130192904_562.csv",
    ("mwp", "MultiArith"): "experience_third/chatGPT_MultiArith_en_20240130190900_600.csv",
}
CONDS = ("zeroshot", "cot_zeroshot", "fewshot", "cot_fewshot", "numcot")
CTX = 4095


MSG = re.compile(r"\{'role': '(system|user|assistant)', 'content': (['\"])(.*?)\2\}(?=, \{'role'|\]$)", re.S)


def parse_chat(cell):
    """Python-repr chat list. Some cells hold unescaped quotes (e.g. "Let's" inside '...'), which literal_eval
    rejects; those are split on the message boundaries instead."""
    try:
        return ast.literal_eval(cell)
    except SyntaxError:
        msgs = [{"role": m.group(1), "content": m.group(3)} for m in MSG.finditer(cell.strip())]
        assert msgs and cell.strip().endswith("}]"), cell[:200]
        return msgs


def flatten(msgs):
    head, shots, cur = [], [], None
    for m in msgs:
        if m["role"] == "system":
            head.append(m["content"].strip())
        elif m["role"] == "user":
            cur = m["content"].rstrip()
        else:
            shots.append(cur + " " + m["content"].strip()); cur = None
    assert cur is not None and msgs[-1]["role"] == "user"
    return "\n\n".join(head + shots + [cur])


def up(x, m):
    return -(-int(x) // m) * m


def budgets(rows):
    gold = max(len(r["answer"].encode()) for r in rows)
    demos = [m["content"] for m in parse_chat(rows[0]["chatGPT_cot_fewshot_quesion"]) if m["role"] == "assistant"]
    return max(64, up(1.5 * gold + 32, 16)), min(1024, up(1.5 * max(len(d.encode()) for d in demos), 64))


n = Counter()
with open(OUT, "w") as f:
    for (task, split), rel in FILES.items():
        rows = list(csv.DictReader(open(D / rel, encoding="utf-8-sig")))
        b_plain, b_cot = budgets(rows)
        print(task, split, "budget", b_plain, b_cot)
        for i, r in enumerate(rows):
            for c in CONDS:
                col = f"chatGPT_{c}_quesion"
                if col not in r or not r[col].strip():
                    continue
                prompt = flatten(parse_chat(r[col]))
                budget = b_cot if "cot" in c else b_plain
                f.write(json.dumps({"key": f"{task}/{split}/{i}/{c}", "task": task, "split": split, "idx": i,
                                    "cond": c, "question": r["number"], "gold": r["answer"], "prompt": prompt,
                                    "max_bytes": budget,
                                    "fits4k": len(prompt.encode()) + budget <= CTX}, ensure_ascii=False) + "\n")
                n[(task, c)] += 1
for k, v in sorted(n.items()):
    print(k, v)
print("total", sum(n.values()), "->", OUT)
