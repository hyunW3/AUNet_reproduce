# S-NIAH-3 single-needle retrieval (RULER)

Haystack/value: natural-essay (DCLM) haystack + UUID. Exact-match retrieval (teacher-forced greedy == model would emit the value verbatim), iso-byte context length.

| model | 512B | 1024B | 2048B | 4096B | 6144B |
|---|---|---|---|---|---|
| subword (Llama) | 0.15 | 0.15 | 0.05 | 0.25 | 0.20 |
| AUNet | 0.85 | 0.60 | 0.45 | 0.30 | 0.40 |
| BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
