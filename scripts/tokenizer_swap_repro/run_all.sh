#!/bin/bash
# One-shot driver: reproduces every arm of PoC/tokenizer_swap_all_results.md in order.
# Long-running (full sets) — set LIMIT=50 for a smoke test, or comment out sections.
# Each step is independent; re-run individually if one fails.
here="$(cd "$(dirname "$0")" && pwd)"; . "$here/00_env.sh"

echo "########## 0. tokenizers + data ##########"
"$here/01_download_tokenizers.sh"
"$here/02_stage_data.sh"

echo "########## 1. BPB swaps (§1-3) ##########"
"$here/10_bpb_swap.sh" zh         llama3
"$here/10_bpb_swap.sh" zh         qwen2
"$here/10_bpb_swap.sh" te         llama3
"$here/10_bpb_swap.sh" te         bloom
"$here/10_bpb_swap.sh" code_multi llama3
"$here/10_bpb_swap.sh" code_multi starcoder2

echo "########## 2. add-not-swap (§2 note) ##########"
"$here/30_augment_not_swap.sh"

echo "########## 3. subword reference (iso-byte) ##########"
"$here/40_subword_ref.sh" bpb code_multi

echo "########## 4. downstream / cloze (§3-5) ##########"
"$here/20_downstream_swap.sh" llama3 hellaswag,arc_easy,arc_challenge,piqa   # §4 English baseline
"$here/20_downstream_swap.sh" gemma  hellaswag,arc_easy,arc_challenge,piqa   # §4 English swap
"$here/20_downstream_swap.sh" llama3 cmmlu_cloze,ceval-cloze,belebele_cloze  # §1/§5 Chinese
"$here/20_downstream_swap.sh" qwen2  cmmlu_cloze,ceval-cloze,belebele_cloze
"$here/20_downstream_swap.sh" llama3 xstorycloze_te,belebele_tel             # §2 Telugu
"$here/20_downstream_swap.sh" bloom  xstorycloze_te,belebele_tel
"$here/20_downstream_swap.sh" llama3 mmlu_cloze,humaneval_mc                 # §3 code MC
"$here/20_downstream_swap.sh" starcoder2 mmlu_cloze,humaneval_mc

echo "########## done — collect metrics under $RUN/repro_* ##########"
