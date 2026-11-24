#!/usr/bin/env python3
"""FLORES-200 X->en for the released Cartesia H-Net checkpoints.

Same protocol as the AU-Net trio and the BLT run: 5-shot, greedy, limit=200, stop at "\\n",
288 bytes of generation budget (= the byte-model budget ceil(96*6/2) used by apps.aunet.eval).

H-Net is byte-level (vocab 256, BOS=254) and its forward takes an explicit boolean `mask`, so
unlike BLT it batches mixed-length rows correctly; batching is therefore kept, with early exit
once every row in the batch has emitted a stop string.

Model code: goombalab/hnet (current layout `hnet.models.*`). Weights: cartesia-ai/<name> on the
HF hub. Run with an interpreter that has mamba-ssm + causal-conv1d + flash-attn.

  python hnet_flores.py --model hnet_2stage_XL --split dev --tasks_dir ... --out out/hnet.json
"""
import argparse, json, os, sys, time, urllib.request
from collections import defaultdict

import torch
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.tasks import TaskManager

BOS = 254
DEVICE, DTYPE = "cuda", torch.bfloat16
LANGS = ["de", "fr", "es", "it", "nl", "ko"]
CONFIG_URL = "https://raw.githubusercontent.com/goombalab/hnet/main/configs/{name}.json"


def build_hnet(name, repo, config_path=None):
    sys.path.insert(0, repo)
    from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
    from hnet.models.mixer_seq import HNetForCausalLM
    from huggingface_hub import hf_hub_download
    wp = hf_hub_download(f"cartesia-ai/{name}", filename=f"{name}.pt")
    if config_path and os.path.exists(config_path):
        cfg = json.load(open(config_path))
    else:
        cfg = json.loads(urllib.request.urlopen(CONFIG_URL.format(name=name)).read().decode())
    cfg["ssm_cfg"] = SSMConfig(**cfg.get("ssm_cfg", {}))
    cfg["attn_cfg"] = AttnConfig(**cfg.get("attn_cfg", {}))
    model = HNetForCausalLM(HNetConfig(**cfg), device=DEVICE, dtype=DTYPE)
    from omegaconf import ListConfig
    with torch.serialization.safe_globals([ListConfig]):
        sd = torch.load(wp, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[hnet] {name}: {n/1e9:.2f}B params")
    return model


class HNetLM(LM):
    def __init__(self, model, batch_size=8, max_len=8192, gen_bytes=288):
        super().__init__()
        self.model, self.batch_size, self.max_len, self.gen_bytes = model, batch_size, max_len, gen_bytes
        self._device, self._rank, self._world_size = DEVICE, 0, 1
        self.n_trunc = self.n_gen = 0

    @staticmethod
    def _enc(s):
        return list(s.encode("utf-8"))

    @torch.inference_mode()
    def _generate_batch(self, prompts, until, max_new):
        cap = self.max_len - max_new - 1
        ptoks = [([BOS] + self._enc(p))[-cap:] for p in prompts]
        plen = [len(t) for t in ptoks]
        start, end, bs = min(plen), max(plen) + max_new, len(ptoks)
        toks = torch.zeros((bs, end), dtype=torch.long, device=DEVICE)
        real = torch.zeros((bs, end), dtype=torch.bool, device=DEVICE)
        for i, t in enumerate(ptoks):
            toks[i, :len(t)] = torch.tensor(t, device=DEVICE)
            real[i, :len(t)] = True
        full = torch.ones((bs, end), dtype=torch.bool, device=DEVICE)
        done = [False] * bs
        for curr in range(start, end):
            if all(done):
                break
            nxt = self.model(toks[:, :curr], mask=full[:, :curr]).logits[:, curr - 1].argmax(-1)
            toks[:, curr] = torch.where(real[:, curr], toks[:, curr], nxt)
            if (curr - start) % 8 == 7:
                for i in range(bs):
                    if done[i] or curr < plen[i]:
                        continue
                    s = bytes(b for b in toks[i, plen[i]:curr + 1].tolist() if b < 254).decode("utf-8", "ignore")
                    if any(u in s for u in until):
                        done[i] = True
        outs = []
        for i, t in enumerate(ptoks):
            s = bytes(b for b in toks[i, len(t):len(t) + max_new].tolist() if b < 254).decode("utf-8", "ignore")
            hit = False
            for u in until:
                j = s.find(u)
                if j >= 0:
                    s, hit = s[:j], True
            self.n_gen += 1
            self.n_trunc += (not hit)
            outs.append(s)
        return outs

    def generate_until(self, requests):
        groups = defaultdict(list)
        for i, req in enumerate(requests):
            gk = req.args[1] or {}
            u = gk.get("until", ["\n"]) or ["\n"]
            u = [u] if isinstance(u, str) else list(u)
            groups[tuple(u)].append(i)
        res = [None] * len(requests)
        t0, n = time.time(), 0
        for u, idxs in groups.items():
            idxs.sort(key=lambda i: len(requests[i].args[0]))
            for b in range(0, len(idxs), self.batch_size):
                chunk = idxs[b:b + self.batch_size]
                for i, o in zip(chunk, self._generate_batch(
                        [requests[i].args[0] for i in chunk], list(u), self.gen_bytes)):
                    res[i] = o
                n += len(chunk)
                el = time.time() - t0
                print(f"  [gen] {n}/{len(requests)}  {el/60:.1f} min ({el/n:.2f} s/sample)", flush=True)
        return res

    @torch.inference_mode()
    def loglikelihood(self, requests):
        out = []
        for r in requests:
            ctx, cont = r.args
            c = [BOS] + self._enc(ctx)
            f = ([BOS] + self._enc(ctx + cont))[-self.max_len:]
            x = torch.tensor([f], device=DEVICE)
            m = torch.ones_like(x, dtype=torch.bool)
            lp = torch.log_softmax(self.model(x, mask=m).logits[0].float(), -1)
            tgt = torch.tensor(f[len(c):], device=DEVICE)
            sl = lp[len(c) - 1:len(f) - 1]
            out.append((float(sl.gather(-1, tgt.unsqueeze(-1)).sum()), bool((sl.argmax(-1) == tgt).all())))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hnet_2stage_XL")
    ap.add_argument("--repo", default="/mnt/ssd2/hyun2/hnet")
    ap.add_argument("--config_path", default=None)
    ap.add_argument("--split", choices=["dev", "devtest"], required=True)
    ap.add_argument("--task_prefix", default=None)   # e.g. floresA_ for the AU-Net 23-language set
    ap.add_argument("--tasks_dir", required=True)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--gen_bytes", type=int, default=288)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    pre = args.task_prefix or ("flores_dev_" if args.split == "dev" else "flores_")
    tasks = [f"{pre}{l}-en" for l in langs]

    model = build_hnet(args.model, args.repo, args.config_path)
    lm = HNetLM(model, args.batch_size, args.max_len, args.gen_bytes)
    tm = TaskManager(include_path=args.tasks_dir)
    res = simple_evaluate(lm, tasks=tasks, num_fewshot=args.num_fewshot, task_manager=tm,
                          limit=args.limit, log_samples=True, bootstrap_iters=1000,
                          gen_kwargs="max_gen_toks=%d" % args.gen_bytes)
    r = res["results"]
    per = {l: {"bleu": r[f"{pre}{l}-en"]["bleu,none"],
               "bleu_stderr": r[f"{pre}{l}-en"]["bleu_stderr,none"],
               "chrf": r[f"{pre}{l}-en"]["chrf,none"]} for l in langs}
    b = [per[l]["bleu"] for l in langs]; c = [per[l]["chrf"] for l in langs]
    bl = [per[l]["bleu"] for l in langs if l != "ko"]; cl = [per[l]["chrf"] for l in langs if l != "ko"]
    row = {"model": args.model, "family": "hnet", "split": args.split, "n": args.limit,
           "num_fewshot": args.num_fewshot, "decoding": "greedy", "gen_bytes": args.gen_bytes,
           "truncated_generations": f"{lm.n_trunc}/{lm.n_gen}", "langs": langs, "per_lang": per,
           "avg_bleu": round(sum(b)/len(b), 3), "avg_chrf": round(sum(c)/len(c), 3),
           "avg_latin_bleu": round(sum(bl)/len(bl), 3) if bl else None,
           "avg_latin_chrf": round(sum(cl)/len(cl), 3) if cl else None}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(row, open(args.out, "w"), indent=2)
    json.dump(res.get("samples", {}), open(args.out.replace(".json", "_samples.json"), "w"), default=str)
    print(json.dumps(row, indent=2)); print("WROTE", args.out)


if __name__ == "__main__":
    main()
