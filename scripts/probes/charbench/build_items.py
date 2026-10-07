#!/usr/bin/env python3
"""Build CharBench + Strawberry-problem items and their lm-eval task YAMLs.

Two external character-level suites, re-cast for 1.3B base (non-instruct) models:

* CharBench (Uzan & Pinter, AAAI 2026; HF omriuz/CharBench, 4 tasks x 43,750 rows):
  count_character_frequency / count_unique_chars / find_first_occurrence /
  find_last_occurrence on MiniPile words of length 4-10. The paper is zero-shot with an
  instruction ("Answer this question only with the final number ..."); base models do not
  follow that, so every prompt here bakes in K in-context demos of the same task (CUTE style),
  after the paper's instruction line. We take N_CB rows per task, stratified uniformly over
  word length, and draw each item's demos from rows outside the evaluated set.
    - gen   : generate_until "\n", first integer of the output, exact match.
    - cloze : multiple_choice over " 0" .. " len(word)" (every possible answer), acc.

* Strawberry problem (Cosma et al., EMNLP 2025; github cosmaadrian/strawberry-problem):
  the 20 test tasks of scripts/generate_dataset_wikipedia.py (reverse / remove / replace /
  rewrite-every-k / swap-every-k, letter and word level) on Wikipedia sentences <= 64 chars,
  produced with the repo's own task functions (scripts/tasks.py) and sentence filter. The paper
  trains small models on these with task tokens; for pretrained models we write each item as
  K demos of the same task (each demo with its own random parameter) + the probe:
      Task: <description>\nInput: <input>\nOutput: <answer>
    - gen   : generate_until "\n", exact match after stripping outer whitespace (case-sensitive).
    - cloze : multiple_choice, gold vs 3 distractors: the same task under another parameter on
              the same input, the unchanged input, and the gold with one adjacent-char swap.

  python build_items.py --charbench_csv benchmark.csv --strawberry_repo <clone> \
      --wiki_articles articles.jsonl --out reports/charbench/items
"""
import argparse, collections, csv, importlib.util, json, os, random

CB_TASKS = ["count_character_frequency", "count_unique_chars", "find_first_occurrence",
            "find_last_occurrence"]
CB_SHORT = {"count_character_frequency": "freq", "count_unique_chars": "unique",
            "find_first_occurrence": "first", "find_last_occurrence": "last"}
CB_INSTR = ("Answer this question only with the final number, without any other text. "
            "Lowercase and uppercase letters are considered different characters.")


def build_charbench(csv_path, n_per_task, k_shot, seed):
    rows = list(csv.DictReader(open(csv_path)))
    rng = random.Random(seed)
    out = {}
    for task in CB_TASKS:
        by_len = collections.defaultdict(list)
        for r in rows:
            if r["task"] == task:
                by_len[len(r["word"])].append(r)
        lens = sorted(by_len)
        per_len = [n_per_task // len(lens) + (i < n_per_task % len(lens)) for i in range(len(lens))]
        evald, pool = [], []
        for L, n in zip(lens, per_len):
            rs = by_len[L][:]
            rng.shuffle(rs)
            evald += rs[:n]
            pool += rs[n:n + 200]          # demo pool, disjoint from the evaluated rows
        items = []
        for i, r in enumerate(evald):
            demos = [d for d in rng.sample(pool, k_shot + 4) if d["word"] != r["word"]][:k_shot]
            ctx = CB_INSTR + "\n\n" + "".join(f"Question: {d['question']}\nAnswer: {d['answer']}\n\n"
                                              for d in demos)
            items.append({"id": i, "prompt": ctx + f"Question: {r['question']}\nAnswer:",
                          "answer": str(int(r["answer"])), "word": r["word"],
                          "character": r["character"] or "", "word_len": len(r["word"]),
                          "choices": [str(j) for j in range(len(r["word"]) + 1)],
                          "gold": int(r["answer"]), "question_idx": int(r["question_idx"])})
        out[CB_SHORT[task]] = items
    return out


def load_strawberry_tasks(repo):
    spec = importlib.util.spec_from_file_location("sp_tasks", os.path.join(repo, "scripts", "tasks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The 20 test tasks, in the order of scripts/generate_dataset_wikipedia.py:TASKS.
SP_TASKS = ["remove_letter", "remove_letter_every_k", "remove_word", "remove_word_every_k",
            "replace_letters", "replace_words", "reverse_from_clean", "reverse_from_clean_word",
            "reverse_from_dirty", "reverse_from_dirty_word", "reverse_the_words_clean",
            "reverse_the_words_dirty", "rewrite_uppercase_every_k_letter",
            "rewrite_uppercase_every_k_words", "rewrite_with_every_k_letter",
            "rewrite_with_every_k_words", "swap_every_k_letters_clean", "swap_every_k_letters_dirty",
            "swap_every_k_words_clean", "swap_every_k_words_dirty"]


def wiki_sentences(articles_path, nltk_dir):
    """scripts/generate_dataset_wikipedia.py:sentence_generator, verbatim filter."""
    import nltk
    if nltk_dir:
        nltk.data.path.insert(0, nltk_dir)
    from nltk.tokenize import sent_tokenize
    for line in open(articles_path):
        try:
            text = json.loads(line)["text"]
        except json.JSONDecodeError:
            continue
        for s in sent_tokenize(text):
            s = s.strip()
            if "\n" in s or " " not in s or len(s) < 8:
                continue
            if sum(1 for c in s if c.isalpha()) / len(s) < 0.7:
                continue
            if len(s) > 64:
                j = random.randint(0, len(s) - 64 - 1)
                s = s[j:j + 64]
            yield " ".join(s.split())


def _one_swap(s, rng):
    idx = [i for i in range(len(s) - 1) if s[i] != s[i + 1]]
    if not idx:
        return None
    i = rng.choice(idx)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def build_strawberry(repo, articles, nltk_dir, n_per_task, k_shot, seed):
    mod = load_strawberry_tasks(repo)
    random.seed(seed)                      # the task functions draw from the global RNG
    rng = random.Random(seed + 1)
    sents = wiki_sentences(articles, nltk_dir)
    fmt = lambda o: f"Task: {o['task_description']}\nInput: {o['input']}\nOutput:"
    out = {}
    for name in SP_TASKS:
        fn = getattr(mod, "task_" + name)
        items = []
        while len(items) < n_per_task:
            batch = []
            while len(batch) < k_shot + 1:   # demos first, then the probe
                s = next(sents)
                try:
                    o = fn(s)
                except (IndexError, ValueError):   # e.g. replace_words on a 1-unique-word sentence
                    continue
                o["answer"] = str(o["answer"])
                if o["answer"] == o["input"] and len(batch) == k_shot:
                    continue                 # probe whose answer equals its input is trivial
                batch.append((s, o))
            demos, (s, probe) = batch[:k_shot], batch[k_shot]
            gold = probe["answer"]
            # distractors: same task, other parameter, same input; identity; one adjacent swap
            alts = []
            for _ in range(30):
                try:
                    o2 = fn(s)
                except (IndexError, ValueError):
                    continue
                if o2["input"] == probe["input"] and str(o2["answer"]) not in (gold, *alts):
                    alts.append(str(o2["answer"]))
                    break
            cands = [gold] + alts + [probe["input"], _one_swap(gold, rng)]
            choices = []
            for c in cands:
                if c is not None and c not in choices:
                    choices.append(c)
            while len(choices) < 4:          # pad with further single swaps
                c = _one_swap(rng.choice(choices), rng)
                if c is not None and c not in choices:
                    choices.append(c)
            choices = choices[:4]
            rng.shuffle(choices)
            ctx = "".join(fmt(d) + f" {d['answer']}\n\n" for _, d in demos)
            items.append({"id": len(items), "prompt": ctx + fmt(probe), "answer": gold,
                          "input": probe["input"], "task_description": probe["task_description"],
                          "choices": choices, "gold": choices.index(gold)})
        out[name] = items
    return out


GEN_TMPL = """# {suite} {split} ({mode}). Generated by scripts/probes/charbench/build_items.py.
task: {task}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
num_fewshot: 0
doc_to_text: "{{{{prompt}}}}"
doc_to_target: "{{{{answer{trim}}}}}"
target_delimiter: " "
output_type: generate_until
generation_kwargs:
  until: ["\\n"]
  do_sample: false
  temperature: 0.0
  max_gen_toks: {max_gen}
filter_list:
  - name: {filter_name}
    filter:
      - function: regex
        regex_pattern: '{regex}'
        fallback: "[invalid]"
      - function: take_first
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

MC_TMPL = """# {suite} {split} (cloze). Generated by scripts/probes/charbench/build_items.py.
task: {task}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
num_fewshot: 0
output_type: multiple_choice
doc_to_text: "{{{{prompt}}}}"
doc_to_choice: "{{{{choices}}}}"
doc_to_target: gold
target_delimiter: " "
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""


def write_yamls(out_dir, suites):
    """One gen + one cloze task per split. Re-run with --yaml_only after moving out_dir."""
    tdir = os.path.join(out_dir, "tasks")
    os.makedirs(tdir, exist_ok=True)
    names = collections.defaultdict(list)
    for suite, splits in suites.items():
        for split in splits:
            data = os.path.abspath(os.path.join(out_dir, "data", f"{suite}_{split}.jsonl"))
            if suite == "charbench":
                gen = dict(max_gen=8, filter_name="first-int", regex=r"(-?\d+)", trim="")
            else:   # outer-whitespace strip only; case and punctuation are part of the task
                gen = dict(max_gen=128, filter_name="strip", regex=r"^\s*(.*?)\s*$", trim="|trim")
            for mode in ("gen", "cloze"):
                task = f"{suite}_{split}_{mode}"
                body = (GEN_TMPL.format(suite=suite, split=split, mode=mode, task=task, data=data, **gen)
                        if mode == "gen" else MC_TMPL.format(suite=suite, split=split, task=task, data=data))
                open(os.path.join(tdir, task + ".yaml"), "w").write(body)
                names[f"{suite}_{mode}"].append(task)
    json.dump(names, open(os.path.join(out_dir, "task_lists.json"), "w"), indent=1)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charbench_csv")
    ap.add_argument("--strawberry_repo")
    ap.add_argument("--wiki_articles")
    ap.add_argument("--nltk_dir", default=None)
    ap.add_argument("--n_charbench", type=int, default=1000, help="items per CharBench task")
    ap.add_argument("--n_strawberry", type=int, default=200, help="items per Strawberry task")
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yaml_only", action="store_true", help="only rewrite tasks/ for the existing data/")
    a = ap.parse_args()
    ddir = os.path.join(a.out, "data")
    if a.yaml_only:
        suites = collections.defaultdict(list)
        for f in sorted(os.listdir(ddir)):
            suite, split = f[:-len(".jsonl")].split("_", 1)
            suites[suite].append(split)
        print(write_yamls(a.out, suites))
        return
    os.makedirs(ddir, exist_ok=True)
    suites = {"charbench": build_charbench(a.charbench_csv, a.n_charbench, a.k_shot, a.seed),
              "strawberry": build_strawberry(a.strawberry_repo, a.wiki_articles, a.nltk_dir,
                                             a.n_strawberry, a.k_shot, a.seed)}
    for suite, splits in suites.items():
        for split, items in splits.items():
            with open(os.path.join(ddir, f"{suite}_{split}.jsonl"), "w") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")
            print(suite, split, len(items))
    json.dump(vars(a), open(os.path.join(a.out, "build_args.json"), "w"), indent=1)
    write_yamls(a.out, {s: list(v) for s, v in suites.items()})


if __name__ == "__main__":
    main()
