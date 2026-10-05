# FFLM in-context state-tracking probe

Read-token accuracy of frozen 1.3B checkpoints on flip-flop sequences (Liu et al. 2023, arXiv:2306.00946). No training — in-context only.

- **greedy_acc**: model's argmax next token == correct bit (the paper's strict metric; <100% = a *reasoning error*). Chance = 0.50.
- **binary_acc**: P(correct bit) > P(wrong bit) — pure 0/1 discrimination.
- **margin**: mean logprob(correct) − logprob(wrong).

Config: T=128 symbols, shots=0, max_reads/seq=8.

| model | step | regime | greedy_acc | binary_acc | margin | n_reads |
|---|---|---|---|---|---|---|
| subword (Llama BPE) | 0000060000 | dense | 0.823 | 0.811 | +1.05 | 800 |
| subword (Llama BPE) | 0000060000 | indist | 0.783 | 0.768 | +0.84 | 642 |
| subword (Llama BPE) | 0000060000 | sparse | 0.808 | 0.802 | +1.02 | 997 |
| AU-Net (word pooling) | 0000180000 | dense | 0.853 | 0.850 | +1.42 | 800 |
| AU-Net (word pooling) | 0000180000 | indist | 0.724 | 0.715 | +0.89 | 642 |
| AU-Net (word pooling) | 0000180000 | sparse | 0.666 | 0.666 | +0.95 | 997 |
| byte (greedy-root) | 0000180000 | dense | 0.865 | 0.859 | +1.18 | 800 |
| byte (greedy-root) | 0000180000 | indist | 0.768 | 0.765 | +0.84 | 642 |
| byte (greedy-root) | 0000180000 | sparse | 0.869 | 0.866 | +0.94 | 997 |
