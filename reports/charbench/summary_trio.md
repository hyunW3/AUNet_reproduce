
### CharBench — gen (exact match)

| task | Llama | AU-Net2 | BPEByte |
|---|--:|--:|--:|
| freq (baseline 75.2) | 74.4 | 74.6 | 73.7 |
| unique (baseline 20.8) | 14.3 | 7.3 | 12.8 |
| first (baseline 19.3) | 18.8 | 18.3 | 18.3 |
| last (baseline 15.1) | 13.7 | 11.9 | 11.5 |
| **avg** (baseline 32.6) | 30.3 | 28.0 | 29.1 |

### CharBench — cloze (acc over 0..len(word))

| task | Llama | AU-Net2 | BPEByte |
|---|--:|--:|--:|
| freq (baseline 13.4) | 74.0 | 74.5 | 73.6 |
| unique (baseline 13.4) | 24.2 | 10.1 | 19.0 |
| first (baseline 13.4) | 18.0 | 18.4 | 18.4 |
| last (baseline 13.4) | 15.3 | 12.0 | 11.8 |
| **avg** (baseline 13.4) | 32.9 | 28.8 | 30.7 |

### Strawberry — gen (exact match)

| task | Llama | AU-Net2 | BPEByte |
|---|--:|--:|--:|
| remove_letter | 0.5 | 0.0 | 0.0 |
| remove_letter_every_k | 0.0 | 0.0 | 0.0 |
| remove_word | 0.0 | 0.0 | 0.5 |
| remove_word_every_k | 0.0 | 0.0 | 0.0 |
| replace_letters | 0.0 | 0.0 | 0.0 |
| replace_words | 0.5 | 2.0 | 2.0 |
| reverse_from_clean | 0.0 | 0.0 | 0.0 |
| reverse_from_clean_word | 0.0 | 0.0 | 0.0 |
| reverse_from_dirty | 0.0 | 0.0 | 0.0 |
| reverse_from_dirty_word | 0.0 | 0.0 | 0.0 |
| reverse_the_words_clean | 0.5 | 0.0 | 0.0 |
| reverse_the_words_dirty | 1.0 | 0.5 | 0.0 |
| rewrite_uppercase_every_k_letter | 0.0 | 0.0 | 0.0 |
| rewrite_uppercase_every_k_words | 3.0 | 7.0 | 6.5 |
| rewrite_with_every_k_letter | 0.0 | 0.0 | 0.0 |
| rewrite_with_every_k_words | 0.0 | 0.0 | 0.0 |
| swap_every_k_letters_clean | 0.0 | 0.0 | 0.0 |
| swap_every_k_letters_dirty | 0.5 | 0.5 | 0.5 |
| swap_every_k_words_clean | 0.0 | 0.0 | 0.0 |
| swap_every_k_words_dirty | 0.0 | 0.0 | 0.0 |
| **avg** | 0.3 | 0.5 | 0.5 |

### Strawberry — cloze (acc, 4 choices (chance 25))

| task | Llama | AU-Net2 | BPEByte |
|---|--:|--:|--:|
| remove_letter | 0.5 | 0.5 | 0.5 |
| remove_letter_every_k | 0.0 | 0.0 | 0.0 |
| remove_word | 9.5 | 9.5 | 10.0 |
| remove_word_every_k | 1.0 | 1.0 | 1.0 |
| replace_letters | 0.5 | 0.5 | 0.0 |
| replace_words | 0.5 | 2.5 | 1.5 |
| reverse_from_clean | 0.0 | 0.0 | 0.0 |
| reverse_from_clean_word | 0.0 | 0.0 | 0.0 |
| reverse_from_dirty | 0.0 | 0.0 | 0.0 |
| reverse_from_dirty_word | 1.0 | 0.5 | 0.5 |
| reverse_the_words_clean | 0.5 | 0.0 | 0.0 |
| reverse_the_words_dirty | 3.0 | 6.5 | 4.0 |
| rewrite_uppercase_every_k_letter | 0.0 | 0.0 | 0.0 |
| rewrite_uppercase_every_k_words | 34.0 | 37.5 | 40.0 |
| rewrite_with_every_k_letter | 0.0 | 0.0 | 0.0 |
| rewrite_with_every_k_words | 2.0 | 1.5 | 1.5 |
| swap_every_k_letters_clean | 0.0 | 0.0 | 0.0 |
| swap_every_k_letters_dirty | 0.5 | 1.0 | 0.5 |
| swap_every_k_words_clean | 0.0 | 0.0 | 0.0 |
| swap_every_k_words_dirty | 0.0 | 0.0 | 0.0 |
| **avg** | 2.6 | 3.0 | 3.0 |

### Strawberry — copy diagnostics (mean over 20 tasks)

| task | Llama | AU-Net2 | BPEByte |
|---|--:|--:|--:|
| cloze no-copy acc (3 choices, chance 33.3) | 56.2 | 56.4 | 56.4 |
| gen: output == input (copy rate) | 28.6 | 33.4 | 35.8 |

### CharBench gen by word length

| task | Llama | AU-Net2 | BPEByte |
|---|--:|--:|--:|
| freq len=4 | 85.3 | 85.3 | 83.2 |
| freq len=5 | 81.8 | 81.8 | 79.7 |
| freq len=6 | 76.9 | 75.5 | 76.2 |
| freq len=7 | 74.8 | 76.2 | 74.1 |
| freq len=8 | 73.4 | 74.8 | 74.1 |
| freq len=9 | 69.9 | 70.6 | 69.9 |
| freq len=10 | 58.5 | 57.7 | 58.5 |
| unique len=4 | 18.9 | 2.1 | 15.4 |
| unique len=5 | 23.1 | 14.0 | 12.6 |
| unique len=6 | 19.6 | 9.1 | 23.1 |
| unique len=7 | 16.8 | 8.4 | 17.5 |
| unique len=8 | 8.4 | 8.4 | 9.8 |
| unique len=9 | 9.1 | 7.0 | 5.6 |
| unique len=10 | 4.2 | 2.1 | 5.6 |
| first len=4 | 28.0 | 21.0 | 24.5 |
| first len=5 | 22.4 | 27.3 | 29.4 |
| first len=6 | 23.1 | 18.9 | 18.9 |
| first len=7 | 15.4 | 11.2 | 11.2 |
| first len=8 | 16.8 | 21.0 | 16.8 |
| first len=9 | 14.0 | 13.3 | 13.3 |
| first len=10 | 12.0 | 15.5 | 14.1 |
| last len=4 | 25.2 | 19.6 | 14.0 |
| last len=5 | 22.4 | 21.0 | 19.6 |
| last len=6 | 12.6 | 13.3 | 14.0 |
| last len=7 | 11.2 | 14.0 | 14.0 |
| last len=8 | 9.1 | 9.1 | 10.5 |
| last len=9 | 7.7 | 3.5 | 4.9 |
| last len=10 | 7.7 | 2.8 | 3.5 |
