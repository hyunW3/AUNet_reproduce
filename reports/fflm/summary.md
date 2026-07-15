# FFLM in-context state-tracking probe

Read-token accuracy of frozen 1.3B checkpoints on flip-flop sequences (Liu et al. 2023, arXiv:2306.00946). No training — in-context only.

- **greedy_acc**: model's argmax next token == correct bit (the paper's strict metric; <100% = a *reasoning error*). Chance = 0.50.
- **binary_acc**: P(correct bit) > P(wrong bit) — pure 0/1 discrimination.
- **margin**: mean logprob(correct) − logprob(wrong).

Config: T=512 symbols, shots=0, max_reads/seq=16.

| model | step | regime | greedy_acc | binary_acc | margin | n_reads |
|---|---|---|---|---|---|---|
| subword (Llama) | 0000060000 | dense | 0.904 | 0.897 | +1.50 | 3200 |
| subword (Llama) | 0000060000 | indist | 0.735 | 0.715 | +0.73 | 3195 |
| subword (Llama) | 0000060000 | sparse | 0.772 | 0.765 | +1.09 | 7126 |
| AUNet | 0000180000 | dense | 0.908 | 0.905 | +1.58 | 3200 |
| AUNet | 0000180000 | indist | 0.679 | 0.665 | +0.65 | 3195 |
| AUNet | 0000180000 | sparse | 0.695 | 0.692 | +1.15 | 7126 |
| BPEByte rg | 0000180000 | dense | 0.924 | 0.920 | +1.43 | 3200 |
| BPEByte rg | 0000180000 | indist | 0.728 | 0.720 | +0.74 | 3195 |
| BPEByte rg | 0000180000 | sparse | 0.796 | 0.792 | +1.09 | 7126 |
