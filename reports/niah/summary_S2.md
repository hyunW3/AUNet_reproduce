# S-NIAH-2 single-needle retrieval (RULER)

Haystack/value: natural-essay (DCLM) haystack + 7-digit number. Exact-match retrieval (teacher-forced greedy == model would emit the value verbatim), iso-byte context length.

| model | 512B | 1024B | 2048B | 4096B | 6144B |
|---|---|---|---|---|---|
| subword (Llama) | 0.90 | 0.70 | 0.65 | 0.65 | 0.60 |
| AUNet | 1.00 | 1.00 | 0.95 | 1.00 | 0.95 |
| BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
