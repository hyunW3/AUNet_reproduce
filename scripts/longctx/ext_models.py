"""External byte-level references for the long-context suite: official BLT-1B and Cartesia H-Net.

Both are driven through the eval_suite harnesses (the same ones the paper's external rows use):
  generation -> harness._generate_batch(prompts, until=[], max_gen_toks=window)  (greedy, full
                re-forward per byte, no cache)
  ranking    -> harness.score_pairs([(prompt, option), ...])                     (sum log-prob)

BLT follows scripts/probes/ext_ci/run_ext.py::build_blt_official: official facebookresearch/blt code
with xformers (entropy-model 512 window + local encoder/decoder windows applied natively), released
threshold 1.335442066192627, batch size 1 (the harness pads batches to 128 B / 64 patches, which
makes scores depend on batch-mates). Env: EVAL_SUITE, BLT_REPO, HNET_REPO.
"""
from __future__ import annotations

import json
import os
import socket
import sys

import torch

EVAL_SUITE = os.environ.get("EVAL_SUITE", "/mnt/ssd2/hyun2/eval_suite")


def build_blt(weights: str, max_len: int = 8192, threshold: float = 1.335442066192627):
    sys.path.insert(0, os.environ["BLT_REPO"])
    from bytelatent.model.blt import ByteLatentTransformer
    from bytelatent.transformer import LMTransformer
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    from bytelatent.data.patcher import PatcherArgs
    from bytelatent.distributed import DistributedArgs
    # bytelatent always picks port 28805 outside torchrun -> init our own group on a free port
    DistributedArgs().configure_world()
    if not torch.distributed.is_initialized():
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}",
                                             rank=0, world_size=1)
        torch.cuda.set_device(0)
    model = ByteLatentTransformer.from_pretrained(f"{weights}/blt_1b", local_files_only=True)
    model = model.to("cuda", torch.bfloat16).eval()
    ent = LMTransformer.from_pretrained(f"{weights}/entropy", local_files_only=True)
    ent = ent.to("cuda", torch.bfloat16).eval()
    tok = BltTokenizer(vocab_size_unit_1=256, bpe_delim=False, add_bos=True, add_eos=False)
    patcher = PatcherArgs(threshold=threshold, realtime_patching=False, patching_device="cuda").build()
    patcher.entropy_model = ent
    sys.path.insert(0, EVAL_SUITE)
    from blt_eval.harness import BLTHarness
    return BLTHarness(model, tok, patcher, batch_size=1, max_len=max_len)


@torch.inference_mode()
def hnet_generate(model, prompt: str, n: int, bos: int = 254) -> str:
    """Greedy n bytes with H-Net's own inference cache (prefill + model.step, batch size 1) --
    the official decode path; the harness's _generate_batch re-runs the full forward per byte."""
    ids = torch.tensor([[bos] + list(prompt.encode("utf-8"))], dtype=torch.long, device="cuda")
    cache = model.allocate_inference_cache(1, ids.shape[1] + n + 8, dtype=torch.bfloat16)
    out = model(ids, mask=torch.ones_like(ids, dtype=torch.bool), inference_params=cache,
                num_last_tokens=1)
    nxt = out.logits[:, -1].argmax(-1)
    gen = [int(nxt)]
    for _ in range(n - 1):
        out = model.step(nxt.view(1, 1), cache)
        nxt = out.logits[:, -1].argmax(-1)
        gen.append(int(nxt))
    return bytes(b for b in gen if b < 254).decode("utf-8", "ignore")


def build_hnet(name: str = "hnet_1stage_XL", max_len: int = 8192, batch_size: int = 8):
    repo = os.environ["HNET_REPO"]
    sys.path.insert(0, repo)
    sys.path.insert(0, EVAL_SUITE)
    from hnet_eval.harness import HNetHarness, AttnConfig, HNetConfig, SSMConfig, HNetForCausalLM
    from huggingface_hub import hf_hub_download
    wp = hf_hub_download(f"cartesia-ai/{name}", filename=f"{name}.pt")
    cfg_path = next(p for p in (f"{repo}/configs/{name}.json", f"{repo}/configs_local/{name}.json")
                    if os.path.exists(p))
    cfg = json.load(open(cfg_path))
    cfg["ssm_cfg"], cfg["attn_cfg"] = SSMConfig(**cfg["ssm_cfg"]), AttnConfig(**cfg["attn_cfg"])
    model = HNetForCausalLM(HNetConfig(**cfg), device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(torch.load(wp, map_location="cuda", weights_only=True), strict=False)
    return HNetHarness(model.eval(), batch_size, max_len)
