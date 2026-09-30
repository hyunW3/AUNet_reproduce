"""CPU check for the SVT ablation arms: runs the REAL training tokenize() generator and reports
bytes/patch per arm, plus unit checks of the fixed-stride strategy."""
import itertools, json, sys
import numpy as np
sys.path.insert(0, ".")
from apps.aunet.data.regex_cutting import RegexArgs, RegexPool
from apps.aunet.data.data import tokenize

H = "/home/hwbae/AUNet"
S = "/home/hwbae/AUNet_svt"
DOC = f"{H}/data/dclm_baseline_1.0_2shards_shuffled/dclm_baseline_1.0_2shards.chunk.14.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

# ---- unit checks: stride ----
pool = RegexPool(RegexArgs(strategy={"stride": "4.5@1"}))
off, lvl = pool.str_offset("abcdefghijklmnopqrstuvwxyz")
lens = np.diff([-1] + list(off))
print("stride4.5 ascii ends", off, "lens", lens.tolist())
assert lens[:4].tolist() == [5, 4, 5, 4], lens
pool4 = RegexPool(RegexArgs(strategy={"stride": "4@1"}))
off4, _ = pool4.str_offset("x" * 17)
assert off4 == [3, 7, 11, 15, 16], off4
offz, _ = pool4.str_offset("가나다라마")          # 3-byte chars: byte ends 3,7,11 -> chars 1,2,3 (+4)
print("stride4 hangul", offz)
assert offz == [1, 2, 3, 4], offz
print("unit checks OK")

LLAMA = f"{H}/tokenizer/llama3/tokenizer.model"
def online(path):
    return dict(strategy={"bpe_br": "1@1"}, bpe_tokenizer_path=path, bpe_online=True,
                bpe_online_mode="greedy", bpe_online_placement="root", bpe_context_prefix=0)
ARMS = {
    "rg_llama3_128k(baseline)": online(LLAMA),
    "stride4.5": dict(strategy={"stride": "4.5@1"}),
    "rg_llama3_V32k": online(f"{S}/tokenizer_extra/llama3_V32k.model"),
    "rg_gpt2": online(f"{S}/tokenizer_extra/gpt2/tokenizer.json"),
    "rg_qwen2": online(f"{H}/tokenizer/qwen2/tokenizer.json"),
}
docs = [json.loads(l) for l in itertools.islice(open(DOC), N)]
for name, kw in ARMS.items():
    it = ((d, None) for d in docs)
    nb = npatch = 0
    for tn, _ in tokenize(it, add_bos=True, add_eos=True, tokenizer_type="bytes", regex=RegexArgs(**kw)):
        mask = tn[1]
        nb += mask.shape[0] - 2             # exclude bos/eos
        npatch += int((mask > 0).sum()) - 1  # exclude forced bos mark
    print(f"{name:28s} bytes/patch = {nb / npatch:.3f}   (patches per 8192 B ~ {8192 * npatch / nb:.0f})")
