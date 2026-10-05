# S-NIAH-1 single-needle retrieval (RULER)

Haystack/value: repeated-noise haystack + 7-digit number. Exact-match retrieval (teacher-forced greedy == model would emit the value verbatim), iso-byte context length.

| model | 512B | 1024B | 2048B | 4096B | 6144B |
|---|---|---|---|---|---|
| subword (Llama) | 0.85 | 0.85 | 1.00 | 1.00 | 0.95 |
| AUNet | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
