#!/usr/bin/env python3
"""Per-item re-evaluation of the external references (BLT-1B, H-Net 1-stage XL) for bootstrap CIs.

Same prompts, tasks, and perturbations as the matched trio (AU-Net repo code), with every
per-item correctness bit saved so the paired item bootstrap of robustness_bootstrap_ci.py applies.

BLT runs through the HF transformers port (itazap/blt-1b-hf); the bytelatent checkout the paper
used is gone. Its entropy patcher is evaluated WITH the 512-byte sliding window the released
entropy model was trained with (facebook/blt-entropy: attn_bias_type=local_block_causal,
sliding_window=512); --window 0 reproduces the paper-time behaviour (plain causal attention,
~1 byte/patch past position 512). Patch lengths are computed here and passed to the model, so
the model never re-patches internally.

Perturbations use base_seed 1234 = the trio's harness.numpy_random_seed, so every model scores
byte-identical perturbed prompts (the Aug-29 external runs used base_seed 0).

  python run_ext.py --family blt --axis downstream --num_fewshot 0 --tasks hellaswag arc_easy ... --out X.json
  python run_ext.py --family blt --axis noise --tasks hellaswag arc_easy arc_challenge piqa boolq --out X.json
  python run_ext.py --family blt --axis sniah --out X.json
Axes: downstream | noise | typo | typoboth | despace | sniah.   Needs AUNET_LINGUA (apps.aunet eval code)
and AUNET_TASKS (lm-eval YAMLs) on the path; see ext_ci/README.md.
"""
import argparse, glob, json, math, os, statistics as st, sys, time
from collections import defaultdict

import torch
from lm_eval.api.model import LM

HERE = os.path.dirname(os.path.abspath(__file__))
LINGUA = os.environ.get("AUNET_LINGUA", "/mnt/ssd2/hyun2/AUNet/lingua")
TASKS_DIR = os.environ.get("AUNET_TASKS", f"{LINGUA}/eval_tasks")
PAIRS = os.environ.get("SNIAH_PAIRS", "/mnt/ssd2/hyun2/AUNet/reports/niah/sniah123_n250_le4096_pairs.jsonl")
PERTURB_SEED = 1234
METRIC = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa": "acc_norm",
          "winogrande": "acc", "boolq": "acc", "mmlu_text": "acc",
          "lambada_openai": "acc"}  # acc = greedy exact match of the whole last word (tokenizer-neutral)


# =============================================================================== BLT (HF)
class BLTHF(LM):
    """lm-eval LM over HF BltForCausalLM with externally computed (windowed) entropy patches."""
    BOS, OFFSET, PAD = 1, 4, 0

    def __init__(self, window=512, batch_tokens=32768, max_len=4096, model_id="itazap/blt-1b-hf"):
        super().__init__()
        from transformers import BltConfig, BltForCausalLM
        from transformers.models.blt.modeling_blt import BltPatcher
        from huggingface_hub import snapshot_download
        snap = snapshot_download(model_id)
        self.model = BltForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).cuda().eval()
        cfg = BltConfig.from_pretrained(snap)
        self.threshold = float(cfg.patching_threshold)
        # fp32 copy of the entropy model: reproduces the paper-time DCLM patch count to 0.1%
        self.patcher = BltPatcher(cfg.patcher_config)
        self.patcher.load_state_dict(self.model.model.patcher.state_dict())
        self.patcher = self.patcher.to("cuda", torch.float32).eval()
        self.window, self.batch_tokens, self.max_len = window, batch_tokens, max_len
        self._rank, self._world_size = 0, 1

    # -- patching ---------------------------------------------------------------------------
    @torch.inference_mode()
    def _entropies(self, toks):
        B, L = toks.shape
        mask = None
        if self.window and L > self.window:
            i = torch.arange(L, device="cuda")
            ok = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)
            mask = torch.zeros((1, 1, L, L), device="cuda").masked_fill(~ok, float("-inf")).expand(B, 1, L, L)
        ent, _, _ = self.patcher(input_ids=toks, attention_mask=mask)
        return ent.float()

    def _patch_lengths(self, ent, lens, L):
        """bytelatent/HF rule per row: tokens 0 and 1 start patches; token p+1 starts one when the
        entropy at p (p >= 1) exceeds the threshold. Padding after a row's n real tokens forms its
        own patch, so it never pools with real bytes. Rows are zero-padded to the max patch count."""
        rows = []
        for b, n in enumerate(lens):
            p = torch.nonzero(ent[b, 1:n - 1] > self.threshold).flatten() + 2       # starts in [2, n-1]
            starts = torch.cat([torch.tensor([0, 1], device=ent.device), p])
            starts = starts[starts < n]
            ends = torch.cat([starts[1:], torch.tensor([n], device=ent.device)])
            pl = (ends - starts).tolist()
            if n < L:
                pl.append(L - n)
            rows.append(pl)
        P = max(len(r) for r in rows)
        return torch.tensor([r + [0] * (P - len(r)) for r in rows], device="cuda", dtype=torch.long)

    # -- scoring ----------------------------------------------------------------------------
    def _enc(self, s):
        return [b + self.OFFSET for b in s.encode("utf-8")]

    @torch.inference_mode()
    def _score_batch(self, batch):
        enc = []
        for ctx, cont in batch:
            ctx_ids = [self.BOS] + self._enc(ctx)
            full = [self.BOS] + self._enc(ctx + cont)
            if len(full) > self.max_len:                      # left-truncate like eval_suite/blt_eval
                drop = len(full) - self.max_len
                full = full[drop:]
                ctx_ids = ctx_ids[drop:] if len(ctx_ids) > drop else ctx_ids[:1]
            enc.append((ctx_ids, full))
        L = max(len(f) for _, f in enc)
        toks = torch.full((len(enc), L), self.PAD, dtype=torch.long, device="cuda")
        for i, (_, f) in enumerate(enc):
            toks[i, :len(f)] = torch.tensor(f, device="cuda")
        pl = self._patch_lengths(self._entropies(toks), [len(f) for _, f in enc], L)
        logp = torch.log_softmax(self.model(input_ids=toks, patch_lengths=pl).logits.float(), dim=-1)
        out = []
        for i, (ctx_ids, full) in enumerate(enc):
            c0, c1 = len(ctx_ids), len(full)
            tgt = torch.tensor(full[c0:c1], device="cuda")
            lp = logp[i, torch.arange(c0 - 1, c1 - 1, device="cuda")]
            out.append((float(lp.gather(-1, tgt.unsqueeze(-1)).sum()), bool((lp.argmax(-1) == tgt).all())))
        return out

    def score_pairs(self, pairs):
        order = sorted(range(len(pairs)), key=lambda i: -len((pairs[i][0] + pairs[i][1]).encode()))
        res, i = [None] * len(pairs), 0
        while i < len(order):
            L = len((pairs[order[i]][0] + pairs[order[i]][1]).encode()) + 1
            n = max(1, min(len(order) - i, self.batch_tokens // max(L, 1)))
            idx = order[i:i + n]
            for k, s in zip(idx, self._score_batch([pairs[k] for k in idx])):
                res[k] = s
            i += n
        return res

    def loglikelihood(self, requests):
        return self.score_pairs([r.args for r in requests])

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        raise NotImplementedError


# =============================================================================== BLT (official bytelatent)
def build_blt_official(weights, max_len, threshold=1.335442066192627):
    """Official facebookresearch/blt code with the released facebook/blt-1b + facebook/blt-entropy
    weights, driven by the paper-time lm-eval harness (eval_suite/blt_eval/harness.py) at batch size 1.

    With attn_impl=xformers (the released configs) every sliding window is applied natively: the
    entropy model's local_block_causal window 512 and the local encoder/decoder windows. Batch size 1
    because the harness pads each batch to a common 128-byte / 64-patch shape, which makes scores
    depend on batch composition (reports/blt_batch_invariance, 6 tasks x 500 items, 0-shot): the bf16
    entropy model flips patch boundaries near the threshold when the batch shape changes, so 1-3% of
    items flip per task and the mean moves -0.27..-0.40pt at bs4/8/16 (paired sign test p=0.10-0.27).
    The pre-Aug-29 harness, whose calculate_entropies flattened the batch into one stream, lost
    1.3-2.0pt. bs>1 also hits an intermittent CUDA illegal memory access in the compiled
    create_block_mask (dynamic shapes). bs=1 is bit-for-bit deterministic across runs.
    Run in the bytelatent venv with PYTHONPATH=<blt repo> and BLT_SUPPRESS_ATTN_ERROR unset or 1
    (only the sdpa path reads it; the xformers path used here never does)."""
    from bytelatent.model.blt import ByteLatentTransformer
    from bytelatent.transformer import LMTransformer
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    from bytelatent.data.patcher import PatcherArgs
    from bytelatent.distributed import DistributedArgs, setup_torch_distributed
    # bytelatent picks MASTER_PORT from random.Random(SLURM_JOB_ID or -1) outside torchrun, i.e. always
    # 28805, so two concurrent jobs collide (EADDRINUSE). Initialise the single-process group ourselves
    # on a free port; setup_torch_distributed then sees it initialised and is skipped.
    import socket
    da = DistributedArgs(); da.configure_world()
    if not torch.distributed.is_initialized():
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
        torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}",
                                             rank=0, world_size=1)
        torch.cuda.set_device(0)
    model = ByteLatentTransformer.from_pretrained(f"{weights}/blt_1b", local_files_only=True)
    model = model.to("cuda", torch.bfloat16).eval()
    ent = LMTransformer.from_pretrained(f"{weights}/entropy", local_files_only=True).to("cuda", torch.bfloat16).eval()
    tok = BltTokenizer(vocab_size_unit_1=256, bpe_delim=False, add_bos=True, add_eos=False)
    patcher = PatcherArgs(threshold=threshold, realtime_patching=False, patching_device="cuda").build()
    patcher.entropy_model = ent
    sys.path.insert(0, os.environ.get("EVAL_SUITE", "/mnt/ssd2/hyun2/eval_suite"))
    from blt_eval.harness import BLTHarness
    return BLTHarness(model, tok, patcher, batch_size=1, max_len=max_len)


# =============================================================================== H-Net
def build_hnet_lm(batch_size, max_len, name="hnet_1stage_XL"):
    sys.path.insert(0, os.environ.get("EVAL_SUITE", "/mnt/ssd2/hyun2/eval_suite"))
    sys.path.insert(0, os.environ.get("HNET_REPO", "/mnt/ssd2/hyun2/hnet"))
    from hnet_eval.harness import HNetHarness
    from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
    from hnet.models.mixer_seq import HNetForCausalLM
    from huggingface_hub import hf_hub_download
    wp = hf_hub_download(f"cartesia-ai/{name}", filename=f"{name}.pt")
    cfg = json.load(open(f"{os.environ.get('HNET_REPO', '/mnt/ssd2/hyun2/hnet')}/configs/{name}.json"))
    cfg["ssm_cfg"], cfg["attn_cfg"] = SSMConfig(**cfg["ssm_cfg"]), AttnConfig(**cfg["attn_cfg"])
    model = HNetForCausalLM(HNetConfig(**cfg), device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(torch.load(wp, map_location="cuda", weights_only=True), strict=False)
    return HNetHarness(model.eval(), batch_size, max_len)


# =============================================================================== early-stop greedy decode
# The eval_suite harnesses (blt_eval / hnet_eval `_generate_batch`) always decode max_gen_toks bytes
# and cut at the first stop string afterwards. These are the same loops (same padding, same forward,
# same argmax) that break once every row has emitted a stop string, so the output is identical and
# only the wasted steps are skipped. Needed for CUTE at max_gen_toks=128 with BLT at batch size 1.
def _hit(s, until):
    return any(u in s for u in until)


def _cut(s, until):
    for u in until:
        j = s.find(u)
        if j >= 0:
            s = s[:j]
    return s


@torch.inference_mode()
def _blt_generate_early(h, prompts, until, max_gen_toks):
    a128 = lambda n: ((n + 127) // 128) * 128
    a64 = lambda n: ((n + 63) // 64) * 64
    boe, cap, outs = h.tokenizer.boe_id, h.max_len - max_gen_toks - 1, []
    for p in prompts:  # batch size 1: BLT scores depend on batch composition (build_blt_official)
        t = h.tokenizer.encode(p, add_bos=True, add_eos=False)[-cap:]
        n = len(t)
        toks = torch.full((1, n + max_gen_toks), boe, dtype=torch.long, device="cuda")
        toks[0, :n] = torch.tensor(t, device="cuda")
        for curr in range(n, n + max_gen_toks):
            Np = a128(curr); cur = toks[:, :curr]
            if Np > curr:
                cur = torch.cat([cur, torch.full((1, Np - curr), boe, dtype=torch.long, device="cuda")], dim=1)
            pl, _ = h.patcher.patch(cur, include_next_token=False, entropies=h._row_entropies(cur))
            P, Pp = pl.shape[1], a64(pl.shape[1])
            if Pp > P:
                pl = torch.cat([pl, torch.zeros((1, Pp - P), dtype=pl.dtype, device="cuda")], dim=1)
            toks[0, curr] = h.model(cur, patch_lengths=pl)[0, curr - 1].argmax(-1)
            if _hit(h.tokenizer.decode(toks[0, n:curr + 1].tolist()), until):
                break
        outs.append(_cut(h.tokenizer.decode(toks[0, n:curr + 1].tolist()), until))
    return outs


@torch.inference_mode()
def _hnet_generate_early(h, prompts, until, max_gen_toks):
    import hnet_eval.harness as hh
    dec = lambda ids: bytes(b for b in ids if b < 254).decode("utf-8", "ignore")
    cap = h.max_len - max_gen_toks - 1
    ptoks = [([hh.BOS] + h._enc(p))[-cap:] for p in prompts]
    plen = [len(t) for t in ptoks]
    start, end, bs = min(plen), max(plen) + max_gen_toks, len(ptoks)
    toks = torch.zeros((bs, end), dtype=torch.long, device=hh.DEVICE)
    real = torch.zeros((bs, end), dtype=torch.bool, device=hh.DEVICE)
    for i, t in enumerate(ptoks):
        toks[i, :len(t)] = torch.tensor(t, device=hh.DEVICE); real[i, :len(t)] = True
    full = torch.ones((bs, end), dtype=torch.bool, device=hh.DEVICE)
    for curr in range(start, end):
        nxt = h.model(toks[:, :curr], mask=full[:, :curr]).logits[:, curr - 1].argmax(-1)
        toks[:, curr] = torch.where(real[:, curr], toks[:, curr], nxt)
        rows = toks[:, :curr + 1].tolist()
        if all(curr >= n and _hit(dec(r[n:n + max_gen_toks]), until) for r, n in zip(rows, plen)):
            break
    rows = toks.tolist()
    return [_cut(dec(r[n:min(n + max_gen_toks, curr + 1)]), until) for r, n in zip(rows, plen)]


def attach_early_stop(lm, family):
    """Replace lm.generate_until with the early-stop decode, keeping the harness grouping/order."""
    gen = _hnet_generate_early if family == "hnet" else _blt_generate_early

    def generate_until(requests):
        groups = defaultdict(list)
        for i, req in enumerate(requests):
            gk = req.args[1] or {}
            u = gk.get("until", ["\n"]) or ["\n"]
            u = [u] if isinstance(u, str) else list(u)
            groups[(tuple(u), int(gk.get("max_gen_toks", 32)))].append(i)
        res = [None] * len(requests)
        for (u, mgt), idxs in groups.items():
            idxs.sort(key=lambda i: len(requests[i].args[0]))
            for b in range(0, len(idxs), lm.batch_size):
                chunk = idxs[b:b + lm.batch_size]
                for i, o in zip(chunk, gen(lm, [requests[i].args[0] for i in chunk], list(u), mgt)):
                    res[i] = o
        return res
    lm.generate_until = generate_until


# =============================================================================== lm-eval shims
def _lm_eval_compat():
    """The AU-Net perturbation helpers call TaskManager.load (removed in lm-eval 0.4.10/0.4.11) and
    lm-eval's pretty-printer KeyErrors on dynamically built noise/typo tasks; same shims as
    eval_suite/common/compat.py."""
    from lm_eval.tasks import TaskManager
    if not hasattr(TaskManager, "load"):
        TaskManager.load = lambda self, name: {"tasks": self.load_task_or_group([name])}


def safe_task_manager(include_path=None):
    from lm_eval.tasks import TaskManager

    class _DefIdx(dict):
        def __missing__(self, k):
            return {"yaml_path": "-", "type": "task"}
    tm = TaskManager(include_path=include_path) if include_path else TaskManager()
    try:
        tm._task_index = _DefIdx(tm._task_index)
    except AttributeError:
        pass
    return tm


# =============================================================================== axes
def compact_samples(samples):
    """lm-eval samples -> {task: {doc_id: [...], acc: [...], acc_norm: [...]}} sorted by doc_id."""
    out = {}
    for task, rows in samples.items():
        rows = sorted(rows, key=lambda r: int(r["doc_id"]))
        out[task] = {"doc_id": [int(r["doc_id"]) for r in rows]}
        for m in ("acc", "acc_norm"):
            if m in rows[0]:
                out[task][m] = [float(r[m]) for r in rows]
    return out


def run_lm_eval(lm, tasks, num_fewshot, limit):
    from lm_eval import simple_evaluate
    r = simple_evaluate(lm, tasks=tasks, num_fewshot=num_fewshot, task_manager=safe_task_manager(TASKS_DIR),
                        limit=limit, log_samples=True, bootstrap_iters=0, fewshot_random_seed=PERTURB_SEED,
                        random_seed=0, numpy_random_seed=PERTURB_SEED, torch_random_seed=PERTURB_SEED)
    res = {k: v for k, v in r["results"].items()}
    return res, compact_samples(r["samples"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["blt", "blt_official", "hnet"], required=True)
    ap.add_argument("--axis", choices=["downstream", "noise", "typo", "typoboth", "despace", "sniah", "bpb", "pbp", "cute"], required=True)
    ap.add_argument("--tasks", nargs="*", default=["hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq"])
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--window", type=int, default=512, help="BLT entropy-model sliding window (0 = none)")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--batch_tokens", type=int, default=32768)
    ap.add_argument("--threshold", type=float, default=1.335442066192627,
                    help="BLT entropy threshold (blt_official); 1.335442066192627 = released")
    ap.add_argument("--blt_weights", default=os.path.expanduser("~/aunet_ext/blt_weights"))
    ap.add_argument("--windows", default=None,
                    help="axis=bpb: jsonl of {text, n_bytes} windows to score")
    ap.add_argument("--hnet_model", default="hnet_1stage_XL",
                    help="hnet_1stage_XL | hnet_2stage_XL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["DESPACE_TASKS"] = ",".join(a.tasks)
    os.environ["PBP_MC_TASKS"] = ",".join(a.tasks)   # read at import time by eval_pbp_mc
    os.environ.update(DESPACE_ANSWER="1", DESPACE_FULL="1", DESPACE_BITS="1", DESPACE_PROBS="")
    _lm_eval_compat()
    sys.path.insert(0, LINGUA)
    sys.modules.pop("apps", None)
    from apps.aunet.eval_noise import expand_noise_tasks, summarize_noise
    from apps.aunet.eval_typo import expand_typo_tasks, summarize_typo
    from apps.aunet.eval_typo_ds import expand_typo_ds_tasks, summarize_typo_ds
    from apps.aunet.eval_despace_mc import run_despace_mc
    from apps.aunet.eval_pbp_mc import run_pbp_mc
    from apps.aunet.eval_cute import expand_cute_tasks, summarize_cute

    t0 = time.time()
    if a.family == "blt":
        lm = BLTHF(window=a.window, batch_tokens=a.batch_tokens, max_len=a.max_len)
        meta = {"model": "itazap/blt-1b-hf", "threshold": lm.threshold, "window": a.window}
    elif a.family == "blt_official":
        lm = build_blt_official(a.blt_weights, a.max_len, a.threshold)
        meta = {"model": "facebook/blt-1b (official bytelatent, xformers)", "threshold": a.threshold,
                "window": "native (entropy 512 + local encoder/decoder)", "batch_size": 1}
    else:
        lm = build_hnet_lm(batch_size=8, max_len=a.max_len, name=a.hnet_model)
        meta = {"model": f"cartesia-ai/{a.hnet_model}"}
    out = {"family": a.family, "axis": a.axis, "tasks": a.tasks, "num_fewshot": a.num_fewshot,
           "limit": a.limit, "max_len": a.max_len, "perturb_seed": PERTURB_SEED, **meta}

    if a.axis == "downstream":
        res, samples = run_lm_eval(lm, a.tasks, a.num_fewshot, a.limit)
        out["results"], out["samples"] = res, samples
    elif a.axis in ("noise", "typo", "typoboth"):
        if a.axis == "noise":
            tl = expand_noise_tasks([x for t in a.tasks for x in (t, f"{t}_noise")], base_seed=PERTURB_SEED)
        elif a.axis == "typoboth":   # typo on the context AND the answer options (hellaswag/arc/piqa)
            tl = expand_typo_ds_tasks([x for t in a.tasks for x in (t, f"{t}_typoboth")], base_seed=PERTURB_SEED)
        else:
            tl = expand_typo_tasks([x for t in a.tasks for x in (t, f"{t}_typo")], base_seed=PERTURB_SEED)
            tl = expand_typo_ds_tasks(tl, base_seed=PERTURB_SEED)
        res, samples = run_lm_eval(lm, tl, 0, a.limit)
        res.update(summarize_noise(res) if a.axis == "noise" else {**summarize_typo(res), **summarize_typo_ds(res)})
        out["results"], out["samples"] = res, samples
    elif a.axis == "bpb":
        # Byte-normalised BPB, teacher-forced with ctx="" -- the same protocol as
        # eval_suite/common/axes.py::bpb and scripts/probes/bpb_windows_local.py, so the number is
        # comparable to the matched models. Byte models sum over byte tokens, but every model is
        # divided by the SAME utf-8 byte count of the windows.
        if not a.windows:
            raise SystemExit("--axis bpb needs --windows")
        wins = [json.loads(l) for l in open(a.windows)]
        got = lm.score_pairs([("", w["text"]) for w in wins])
        total_lp = sum(lp for lp, _g in got)
        total_bytes = sum(w["n_bytes"] for w in wins)
        out["windows"] = a.windows
        out["results"] = {"n_windows": len(wins), "total_bytes": total_bytes,
                          "total_nll_nats": -total_lp,
                          "bpb": -total_lp / (math.log(2) * total_bytes)}
        print(f"[{a.family}] {os.path.basename(a.windows)}: windows={len(wins)} "
              f"bytes={total_bytes} BPB={out['results']['bpb']:.4f}", flush=True)
    elif a.axis == "cute":
        # 14 CUTE subtasks (eval_tasks/gen_mc/cute_*.yaml), generate_until + exact_match on the quoted
        # span; per-item raw/filtered responses kept so truncation and extraction can be audited.
        from lm_eval import simple_evaluate
        if not os.environ.get("CUTE_NO_EARLY_STOP"):   # =1: canonical harness decode (equivalence check)
            attach_early_stop(lm, a.family)
        tl = [t for t in a.tasks if t.startswith("cute_")] or expand_cute_tasks(["cute"])  # shard by subtask
        out["tasks"] = tl
        r = simple_evaluate(lm, tasks=tl, num_fewshot=0,
                            task_manager=safe_task_manager(f"{TASKS_DIR}/gen_mc"), limit=a.limit,
                            log_samples=True, bootstrap_iters=0, random_seed=0,
                            numpy_random_seed=PERTURB_SEED, torch_random_seed=PERTURB_SEED)
        res = dict(r["results"])
        res.update(summarize_cute(res))
        out["results"] = res
        out["samples"] = {t: [{"doc_id": int(x["doc_id"]), "target": x["target"], "resp": x["resps"][0][0],
                               "filtered": x["filtered_resps"][0], "exact_match": float(x["exact_match"])}
                              for x in sorted(rows, key=lambda x: int(x["doc_id"]))]
                          for t, rows in r["samples"].items()}
    elif a.axis == "despace":
        out["results"] = run_despace_mc(lm.loglikelihood, limit=a.limit or 2000)
    elif a.axis == "pbp":
        # Prompt-boundary shift (eval_suite/robust_axes_ext.py --axis pbp protocol): the canonical and the
        # space-committed prompt/answer cut, scored with acc; byte models see identical bytes in both.
        out["results"] = run_pbp_mc(lm.loglikelihood, limit=a.limit or 2000)
    else:  # sniah
        recs = [json.loads(l) for l in open(PAIRS)]
        got = lm.score_pairs([(r["prompt"], " " + r["values"][0]) for r in recs])
        per = defaultdict(list)
        for r, (_lp, g) in zip(recs, got):
            per[r["cell"]].append(int(g))
        out["per_item"] = {c: v for c, v in per.items()}
        out["per_cell"] = {c: st.mean(v) for c, v in per.items()}
        tasks = sorted({c.split("/")[0] for c in per})
        out["per_task"] = {t: st.mean(out["per_cell"][c] for c in per if c.startswith(t + "/")) for t in tasks}
        out["mean"] = st.mean(out["per_task"].values())
    out["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), default=str)
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()
