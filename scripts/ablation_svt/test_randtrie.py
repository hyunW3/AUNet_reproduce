"""CPU check: the random-trie vocab loads through the real tiktoken loader and the training
tokenize() generator, and its online-greedy bytes/patch matches BPEByte root_greedy."""
import itertools, json, sys
sys.path.insert(0, ".")
from apps.aunet.data.regex_cutting import RegexArgs
from apps.aunet.data.data import tokenize

H = "/home/hwbae/AUNet"
S = "/home/hwbae/AUNet_svt"
DOC = f"{H}/data/dclm_baseline_1.0_2shards_shuffled/dclm_baseline_1.0_2shards.chunk.14.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def online(path):
    return dict(strategy={"bpe_br": "1@1"}, bpe_tokenizer_path=path, bpe_online=True,
                bpe_online_mode="greedy", bpe_online_placement="root", bpe_context_prefix=0)


ARMS = {
    "rg_llama3_128k(baseline)": online(f"{H}/tokenizer/llama3/tokenizer.model"),
    "rg_randtrie_mcr": online(f"{S}/tokenizer_extra/randtrie_V44500_s0.model"),
}
docs = [json.loads(l) for l in itertools.islice(open(DOC), N)]
for name, kw in ARMS.items():
    nb = npatch = 0
    for tn, _ in tokenize(((d, None) for d in docs), add_bos=True, add_eos=True,
                          tokenizer_type="bytes", regex=RegexArgs(**kw)):
        mask = tn[1]
        nb += mask.shape[0] - 2
        npatch += int((mask > 0).sum()) - 1
    print(f"{name:28s} bytes/patch = {nb / npatch:.3f}")
# show how the two parsers cut one sentence
from apps.aunet.data.regex_cutting import RegexPool
txt = "The quick brown fox jumps over the lazy dog, and then it runs away."
for name, kw in ARMS.items():
    pool = RegexPool(RegexArgs(**kw))
    m = pool.online_levels_mask(txt, add_bos=0, add_eos=0)
    b = txt.encode()
    cuts, prev = [], 0
    for i, v in enumerate(m):
        if v > 0 and i > 0:
            cuts.append(b[prev:i].decode(errors="replace")); prev = i
    cuts.append(b[prev:].decode(errors="replace"))
    print(f"{name:28s} " + "|".join(cuts))
