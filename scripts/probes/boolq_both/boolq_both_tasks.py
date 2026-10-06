"""BoolQ on the "both" region (context AND answer options) for Noise and Typo, as an overlay on lingua's
apps.aunet.eval_noise / eval_typo_ds (no lingua change).

The four other tasks apply each perturbation to the context and to every answer option. BoolQ's options are the
labels "no"/"yes" (scored as continuations " no" / " yes"); they are perturbed like any other option, so the label
surface changes while the gold index stays the same. The stock yaml's ``doc_to_choice`` is a static list, so the
perturbed labels are stored per doc and read back by a callable ``doc_to_choice``.

Seed fields (``eval_noise._seed``): passage 0, question 1, option j 2 + j.
  - Noise: the context equals the existing ``boolq_noise_<s>_prompt`` variant byte for byte.
  - Typo: the passage equals the existing ``boolq_typo_<s>`` variant (passage-only) byte for byte; the question is
    now perturbed too (field 1), as for Noise and as the context of the other tasks.
Word-mode typos only edit words of >= 4 characters (eval_typo.MIN_WORD_LEN), so they leave "yes"/"no" unchanged.

Sentinels: ``boolq_noiseboth`` -> ``boolq_noise_<strategy>_both`` (5 tasks);
           ``boolq_typoboth``  -> ``boolq_typoboth_<op>_<char|word>`` (8 tasks).
``install()`` wraps the expanders (and their re-exports in apps.main.eval / apps.aunet.eval) so any entry point that
calls them understands the sentinels. With BOOLQ_BOTH_NOISE_SENTINEL=1, run_ext's ``boolq_noise`` sentinel (axis
noise) also maps to the both variants.
"""
import copy, os, random
import sys

CHOICES = ("no", "yes")
CHOICE_FIELD = "_both_choices"
NOISE_SENTINEL = "boolq_noiseboth"
TYPO_SENTINEL = "boolq_typoboth"
# Context only (passage + question, labels untouched): the region BoolQ uses in the five-task average, since its
# options are the labels themselves (perturbing them measures a label-string preference, not robustness).
TYPO_CTX_SENTINEL = "boolq_typoctx"


def _doc_to_choice(doc):
    return list(doc[CHOICE_FIELD])


def _make_pd(base_pd, fn, scode, base_seed, seed_fn, options=True):
    def process_docs(dataset):
        if base_pd is not None:
            dataset = base_pd(dataset)

        def _p(doc, idx):
            r = lambda f: random.Random(seed_fn(base_seed, idx, scode, f))
            return {"passage": fn(doc["passage"], r(0)), "question": fn(doc["question"], r(1)),
                    CHOICE_FIELD: [fn(c, r(2 + j)) if options else c for j, c in enumerate(CHOICES)]}

        return dataset.map(_p, with_indices=True)

    return process_docs


def _build(names_fns, base_seed, options=True):
    from lm_eval.tasks import TaskManager
    from lm_eval.api.task import ConfigurableTask
    from apps.aunet.eval_noise import _seed
    task = TaskManager().load("boolq")["tasks"]["boolq"]
    base_dict = task.config.to_dict()
    base_pd = task.config.process_docs
    callables = {k: getattr(task.config, k) for k in base_dict if callable(getattr(task.config, k, None))}
    out = []
    for name, fn, scode in names_fns:
        cfg = copy.deepcopy(base_dict)
        cfg.update(callables)
        cfg["task"] = name
        cfg["process_docs"] = _make_pd(base_pd, fn, scode, base_seed, _seed, options)
        cfg["doc_to_choice"] = _doc_to_choice
        out.append(ConfigurableTask(config=cfg))
    return out


def build_noise_both(base_seed=0):
    from apps.aunet import eval_noise as N
    return _build([(f"boolq_noise_{s}_both", N._NOISE_FNS[s], N._STRATEGY_CODE[s]) for s in N.STRATEGIES], base_seed)


def build_typo_both(base_seed=0):
    from apps.aunet import eval_typo as T
    return _build([(f"boolq_typoboth_{s}", T._NOISE_FNS[s], T._STRATEGY_CODE[s]) for s in T.STRATEGIES], base_seed)


def build_typo_ctx(base_seed=0):
    from apps.aunet import eval_typo as T
    return _build([(f"boolq_typoctx_{s}", T._NOISE_FNS[s], T._STRATEGY_CODE[s]) for s in T.STRATEGIES], base_seed,
                  options=False)


def _wrap(orig, sentinels):
    def expand(tasks, base_seed=0, *a, **k):
        if not tasks:
            return orig(tasks, base_seed, *a, **k)
        out = []
        for t in tasks:
            if isinstance(t, str) and t in sentinels:
                out.extend(sentinels[t](base_seed))
            else:
                out.append(t)
        return orig(out, base_seed, *a, **k)
    expand._boolq_both = True
    return expand


def install():
    from apps.aunet import eval_noise, eval_typo_ds
    if getattr(eval_noise.expand_noise_tasks, "_boolq_both", False):
        return
    ns = {NOISE_SENTINEL: build_noise_both}
    if os.environ.get("BOOLQ_BOTH_NOISE_SENTINEL") == "1":
        ns["boolq_noise"] = build_noise_both
    eval_noise.expand_noise_tasks = _wrap(eval_noise.expand_noise_tasks, ns)
    ts = {TYPO_SENTINEL: build_typo_both, TYPO_CTX_SENTINEL: build_typo_ctx}
    if os.environ.get("BOOLQ_TYPO_CTX") == "1":   # run_ext's typoboth axis -> context-only variants
        ts[TYPO_SENTINEL] = build_typo_ctx
    eval_typo_ds.expand_typo_ds_tasks = _wrap(eval_typo_ds.expand_typo_ds_tasks, ts)
    # Entry points imported later (runpy'd `-m apps.*.eval`, run_ext's in-function import) pick up the wrapped
    # attributes; modules already imported keep their own references, so rebind those.
    for mod in ("apps.main.eval", "apps.aunet.eval"):
        m = sys.modules.get(mod)
        if m is None:
            continue
        if hasattr(m, "expand_noise_tasks"):
            m.expand_noise_tasks = eval_noise.expand_noise_tasks
        if hasattr(m, "expand_typo_ds_tasks"):
            m.expand_typo_ds_tasks = eval_typo_ds.expand_typo_ds_tasks
