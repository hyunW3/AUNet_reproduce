"""Generate the SVT ablation configs from the RESOLVED config of the leaderboard baseline
(lb_rg_100M) so every hyperparameter is identical; only name/dump_dir/checkpoint/regex change."""
import copy, os
import yaml

H = "/home/hwbae/AUNet"
S = "/home/hwbae/AUNet_svt"
base = yaml.safe_load(open(f"{H}/runs/poc/portable_aunetlaw/lb_rg_100M/config.yaml"))

ARMS = {
    # fixed stride matched to BPEByte root_greedy's measured 4.566 B/patch on DCLM
    "stride4p57": {"strategy": {"stride": "4.57@1"}, "bpe_tokenizer_path": None, "bpe_online": False},
    # llama3 truncated to its first 32,768 merge ranks (nested sub-vocab)
    "rg_llama3_V32k": {"bpe_tokenizer_path": f"{S}/tokenizer_extra/llama3_V32k.model"},
    # other tokenizer sources, same online greedy root BPEByte recipe
    "rg_gpt2": {"bpe_tokenizer_path": f"{S}/tokenizer_extra/gpt2/tokenizer.json"},
    "rg_qwen2": {"bpe_tokenizer_path": f"{H}/tokenizer/qwen2/tokenizer.json"},
    # equivalence check of the snapshot code vs the baseline (short smoke only)
    "smoke_rg_repro": {},
}
os.makedirs(f"{S}/configs", exist_ok=True)
for arm, regex in ARMS.items():
    c = copy.deepcopy(base)
    c["name"] = f"svt_{arm}_100M"
    c["dump_dir"] = f"{S}/runs/{arm}"
    c["checkpoint"]["path"] = f"{S}/runs/{arm}/checkpoints"
    c["data"]["regex"].update(regex)
    with open(f"{S}/configs/{arm}.yaml", "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)
    print(arm, c["data"]["regex"]["strategy"], c["data"]["regex"]["bpe_tokenizer_path"],
          c["data"]["regex"]["bpe_online"])
