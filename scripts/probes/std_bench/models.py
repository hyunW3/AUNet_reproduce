"""Model registry + builders for std_bench.py.

Each entry names a family; the family fixes the interpreter/environment (run_std_bench.sh) and the
lm-eval harness that scores it:
  blt           official bytelatent BLT-1B, eval_suite BLTHarness, batch size 1 (run_ext.build_blt_official)
  hnet          cartesia H-Net, eval_suite HNetHarness (run_ext.build_hnet_lm)
  lingua_main   lingua LMTransformer checkpoint, apps.main.eval.EvalHarnessLM
  lingua_aunet  lingua HierarchicalTransformer (AU-Net / BPEByte) checkpoint, apps.aunet.eval.EvalHarnessLM

The lingua entries use the generator settings of the paper's matched evals
(runs/robustness_paper1p3b/<arm>/config.yaml): bf16, temperature 0, max_tokens 4096 (Transformer)
or 16384 (hierarchical), every other EvalArgs option at its default.

This module imports nothing heavy at top level, so `python3 models.py family <name>` works from any
interpreter (run_std_bench.sh uses it to pick the environment).
"""
import os
import sys

AUNET = os.environ.get("AUNET_ROOT", "/mnt/ssd2/hyun2/AUNet")  # data / checkpoints / weights
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # this code
CK = f"{AUNET}/main/main/1.3B"

MODELS = {
    "blt_1b": {"family": "blt", "weights": f"{AUNET}/runs/ext_ci_snu55/blt_weights",
               "threshold": 1.335442066192627},
    "hnet_1stage_XL": {"family": "hnet", "name": "hnet_1stage_XL", "batch_size": 8},
    "hnet_2stage_XL": {"family": "hnet", "name": "hnet_2stage_XL", "batch_size": 8},
    "llama_1.3b": {"family": "lingua_main", "max_tokens": 4096,
                   "ckpt": f"{CK}/llama_1.8B_paper/checkpoints/0000060000/consolidated"},
    "aunet_1.3b": {"family": "lingua_aunet", "max_tokens": 16384,
                   "ckpt": f"{CK}/aunet2_1.3B/checkpoints/0000180000/consolidated"},
    "bpebyte_1.3b": {"family": "lingua_aunet", "max_tokens": 16384,
                     "ckpt": f"{CK}/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"},
}
FAMILIES = ("blt", "hnet", "lingua_main", "lingua_aunet")


def resolve(model, ckpt=None, family=None, max_tokens=None):
    """Registry entry, or an ad-hoc lingua checkpoint (model name = anything, --ckpt + --family)."""
    if ckpt:
        if family not in ("lingua_main", "lingua_aunet"):
            raise SystemExit("--ckpt needs --family lingua_main|lingua_aunet")
        return {"family": family, "ckpt": ckpt,
                "max_tokens": max_tokens or (4096 if family == "lingua_main" else 16384)}
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}; known: {', '.join(MODELS)} (or pass --ckpt/--family)")
    spec = dict(MODELS[model])
    if max_tokens:
        spec["max_tokens"] = max_tokens
    return spec


def init_single_process_group():
    """One-process NCCL group on a free port (several jobs on one host must not share a port)."""
    import socket
    import torch
    if torch.distributed.is_initialized():
        return
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
    torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}",
                                         rank=0, world_size=1)
    torch.cuda.set_device(0)


def build(spec, max_len=4096):
    """-> (lm, weight_files). lm follows the lm-eval LM interface."""
    fam = spec["family"]
    sys.path.insert(0, f"{REPO}/scripts/probes/ext_ci")
    import run_ext
    if fam == "blt":
        init_single_process_group()
        lm = run_ext.build_blt_official(spec["weights"], max_len, spec["threshold"])
        assert lm.batch_size == 1
        return lm, [f"{spec['weights']}/{d}/model.safetensors" for d in ("blt_1b", "entropy")]
    if fam == "hnet":
        from huggingface_hub import hf_hub_download
        lm = run_ext.build_hnet_lm(batch_size=spec["batch_size"], max_len=max_len, name=spec["name"])
        return lm, [hf_hub_download(f"cartesia-ai/{spec['name']}", filename=f"{spec['name']}.pt")]

    init_single_process_group()
    if fam == "lingua_main":
        import apps.main.generate as gen_mod
        _build_tok = gen_mod.build_tokenizer

        def build_tokenizer(name, path):
            # Checkpoints carry the training host's tokenizer path; fall back like the hierarchical
            # loader does: $TOKENIZER_PATH, then <AUNET_ROOT>/tokenizer/<parent>/<file>.
            if path and not os.path.exists(path):
                alt = os.environ.get("TOKENIZER_PATH") or os.path.join(
                    AUNET, "tokenizer", os.path.basename(os.path.dirname(path)), os.path.basename(path))
                print(f"[std_bench] tokenizer {path} not found -> {alt}", flush=True)
                path = alt
            return _build_tok(name, path)
        gen_mod.build_tokenizer = build_tokenizer
        from apps.main.eval import EvalHarnessLM
        from apps.main.generate import (PackedCausalTransformerGenerator as Gen,
                                        PackedCausalTransformerGeneratorArgs as GenArgs,
                                        load_consolidated_model_and_tokenizer)
        from apps.main.transformer import LMTransformer, LMTransformerArgs
        model, tok, _ = load_consolidated_model_and_tokenizer(spec["ckpt"], model_cls=LMTransformer,
                                                              model_args_cls=LMTransformerArgs)
        extra = ()
    else:
        from apps.aunet.eval import EvalHarnessLM
        from apps.aunet.generate import (PackedHierarchicalCausalTransformerGenerator as Gen,
                                         PackedHierarchicalCausalTransformerGeneratorArgs as GenArgs,
                                         load_consolidated_model_and_tokenizer)
        from apps.aunet.hierarchical import HierarchicalArgs, HierarchicalTransformer
        model, tok, regex_pool, _ = load_consolidated_model_and_tokenizer(
            spec["ckpt"], model_cls=HierarchicalTransformer, model_args_cls=HierarchicalArgs)
        extra = (regex_pool,)
    gen = Gen(GenArgs(temperature=0.0, max_gen_len=512, max_tokens=spec["max_tokens"], until=[], dtype="bf16"),
              model.eval(), tok, *extra)
    ck = spec["ckpt"]
    return EvalHarnessLM(gen), [f"{ck}/params.json"] + sorted(
        f"{ck}/{f}" for f in os.listdir(ck) if f.endswith(".pth"))


if __name__ == "__main__":  # python3 models.py family <name>   |   python3 models.py list
    if sys.argv[1] == "list":
        for k, v in MODELS.items():
            print(f"{k:16s} {v['family']:13s} {v.get('ckpt') or v.get('weights') or v.get('name')}")
    elif sys.argv[1] == "family":
        print(MODELS[sys.argv[2]]["family"] if sys.argv[2] in MODELS else "")
