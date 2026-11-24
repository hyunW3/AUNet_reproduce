#!/bin/bash
# Download the 4 swap tokenizers (tokenizer.json only) into $TOK_DIR/<name>/.
# Each is a DIFFERENT tokenizer FAMILY — the byte-trie builder auto-detects the
# recovery scheme from tokenizer.json (see lingua/lingua/tokenizer.py
# HFOffsetTokenizer.iter_token_bytes):
#   qwen2      byte-level BPE  (GPT-2 bytes<->unicode inverse)   Chinese-dense, 151k
#   bloom      byte-level BPE                                    Telugu-covering, 250k
#   starcoder2 byte-level BPE                                    code-aware, 49k
#   gemma      SentencePiece metaspace (▁ + <0xNN>)              English same-granularity, 256k
#
# NOTE bloom: bigscience/bloom and bigscience/tokenizer ship BYTE-IDENTICAL
# tokenizer.json (verified this session) — either repo works.
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"
activate_env

declare -A REPO=(
  [qwen2]=Qwen/Qwen2-7B
  [bloom]=bigscience/bloom
  [starcoder2]=bigcode/starcoder2-15b
  [gemma]=google/gemma-7b
)

for name in "${!REPO[@]}"; do
  out="$TOK_DIR/$name"; mkdir -p "$out"
  if [[ -f "$out/tokenizer.json" ]]; then echo "[skip] $name already present"; continue; fi
  echo "[dl] $name <- ${REPO[$name]}"
  python - "$name" "${REPO[$name]}" "$out" <<'PY'
import sys
from huggingface_hub import hf_hub_download
name, repo, out = sys.argv[1:4]
p = hf_hub_download(repo_id=repo, filename="tokenizer.json", local_dir=out)
print(f"  -> {p}")
PY
done

echo "=== sha256 (bloom repos should match) ==="
for name in "${!REPO[@]}"; do sha256sum "$TOK_DIR/$name/tokenizer.json"; done
