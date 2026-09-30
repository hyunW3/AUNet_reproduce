#!/usr/bin/env python3
"""H-Net (cartesia-ai/hnet_1stage_XL) chunk lengths from its dynamic-chunking router.

Model load mirrors eval_suite/hnet_eval/harness.py::build_hnet, except the config is read from
the local goombalab checkout (/mnt/ssd2/hyun2/hnet/configs) instead of GitHub, so it runs offline.
The forward returns bpred_output[0].boundary_mask [B, L]: True where a chunk starts. Position 0
is BOS (254); position t >= 1 is byte t-1, and the BOS chunk is excluded like every model's BOS.

  CUDA_VISIBLE_DEVICES=2 /home/hyunwoong/miniconda3/envs/spacebyte/bin/python \
      scripts/probes/patch_stats/measure_hnet.py [--gate_only]
"""
import argparse, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_common import Hist, check_gate, lengths_from_starts, load_texts  # noqa: E402

HNET_REPO = "/mnt/ssd2/hyun2/hnet"
sys.path.insert(0, HNET_REPO)
from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig  # noqa: E402
from hnet.models.mixer_seq import HNetForCausalLM  # noqa: E402

BOS = 254
NAME = "hnet_1stage_XL"


def build(name=NAME):
    from huggingface_hub import hf_hub_download
    wp = hf_hub_download(f"cartesia-ai/{name}", filename=f"{name}.pt")
    cfg = json.load(open(f"{HNET_REPO}/configs/{name}.json"))
    cfg["ssm_cfg"] = SSMConfig(**cfg.get("ssm_cfg", {}))
    cfg["attn_cfg"] = AttnConfig(**cfg.get("attn_cfg", {}))
    model = HNetForCausalLM(HNetConfig(**cfg), device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(torch.load(wp, map_location="cuda", weights_only=True), strict=False)
    return model.eval()


@torch.inference_mode()
def boundary_masks(model, batch_ids):
    L = max(len(x) for x in batch_ids)
    toks = torch.zeros((len(batch_ids), L), dtype=torch.long, device="cuda")
    mask = torch.zeros((len(batch_ids), L), dtype=torch.bool, device="cuda")
    for i, x in enumerate(batch_ids):
        toks[i, :len(x)] = torch.tensor(x, device="cuda")
        mask[i, :len(x)] = True
    out = model(toks, mask=mask)
    return out.bpred_output[0].boundary_mask.bool().cpu()   # outermost (only) stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--tok_budget", type=int, default=32768)
    a = ap.parse_args()
    model = build()
    rows = load_texts(["dclm"] if a.gate_only else None)
    gate_rows = [r for r in rows if r["group"] == "dclm"]
    rest = sorted((r for r in rows if r["group"] != "dclm"), key=lambda r: len(r["text"].encode()))
    h, t0 = Hist(), time.time()
    for phase, rs in (("gate", gate_rows), ("bench", rest)):
        i = 0
        while i < len(rs):
            j, L = i, 0
            while j < len(rs):
                L2 = max(L, len(rs[j]["text"].encode()) + 1)
                if (j - i + 1) * L2 > a.tok_budget and j > i:
                    break
                L, j = L2, j + 1
            batch = rs[i:j]
            ids = [[BOS] + list(r["text"].encode()) for r in batch]
            bm = boundary_masks(model, ids)
            for k, r in enumerate(batch):
                nb = len(ids[k]) - 1
                starts = (torch.nonzero(bm[k, 1:nb + 1]).flatten()).tolist()   # position t -> byte t-1
                h.add(r, lengths_from_starts(starts, nb))
            i = j
            if phase == "bench" and i % 5000 < len(batch):
                print(f"  {i}/{len(rs)}  {time.time() - t0:.0f}s", flush=True)
        if phase == "gate":
            if not check_gate("hnet", h):
                raise SystemExit("gate failed — does not reproduce the paper-time H-Net measurement")
            if a.gate_only:
                return
    print("wrote", h.save("hnet", {"model": NAME, "dtype": "bfloat16"}))


if __name__ == "__main__":
    main()
