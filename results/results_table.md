# Stage 5–6: Retrieval Metrics Table

| Stratification Subset | Condition | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| **No-mismatch** | 0: Random | 0.0199 | 0.0026 | 0.0064 |
| **No-mismatch** | 1: Dense LSA | 0.7960 | 0.3003 | 0.4186 |
| **No-mismatch** | 2: BM25 only | 0.7811 | 0.3103 | 0.4229 |
| **No-mismatch** | 3: BM25 + Dense LSA (Baseline) | 0.7811 | 0.3155 | 0.4268 |
| **No-mismatch** | 4: Entity-linked only | 0.7761 | 0.3018 | 0.4145 |
| **No-mismatch** | 5: Full System (Fixed Weight) | 0.7811 | 0.2951 | 0.4114 |
| **No-mismatch** | 9: Gated Entity RAG (Mismatch Gated) | 0.7612 | 0.3039 | 0.4132 |
| **No-mismatch** | 6: Neural Dense (all-MiniLM-L6-v2) | 0.7960 | 0.3213 | 0.4343 |
| **No-mismatch** | 7: BM25 + Neural Dense (Neural Baseline) | 0.8010 | 0.3093 | 0.4268 |
| **No-mismatch** | 8: Full Neural System (Fixed Weight) | 0.8159 | 0.3058 | 0.4269 |
| **No-mismatch** | 10: Gated Neural System (Mismatch Gated) | 0.8010 | 0.2996 | 0.4188 |
| **Partial-mismatch** | 0: Random | 0.0043 | 0.0006 | 0.0014 |
| **Partial-mismatch** | 1: Dense LSA | 0.8462 | 0.3692 | 0.4841 |
| **Partial-mismatch** | 2: BM25 only | 0.8034 | 0.3788 | 0.4810 |
| **Partial-mismatch** | 3: BM25 + Dense LSA (Baseline) | 0.8547 | 0.4182 | 0.5234 |
| **Partial-mismatch** | 4: Entity-linked only | 0.8376 | 0.3606 | 0.4757 |
| **Partial-mismatch** | 5: Full System (Fixed Weight) | 0.8590 | 0.4170 | 0.5237 |
| **Partial-mismatch** | 9: Gated Entity RAG (Mismatch Gated) | 0.8547 | 0.3916 | 0.5031 |
| **Partial-mismatch** | 6: Neural Dense (all-MiniLM-L6-v2) | 0.7778 | 0.3169 | 0.4268 |
| **Partial-mismatch** | 7: BM25 + Neural Dense (Neural Baseline) | 0.8291 | 0.3742 | 0.4828 |
| **Partial-mismatch** | 8: Full Neural System (Fixed Weight) | 0.8632 | 0.3806 | 0.4962 |
| **Partial-mismatch** | 10: Gated Neural System (Mismatch Gated) | 0.8419 | 0.3809 | 0.4916 |
| **Full alias-mismatch (Held-Out Test)** | 0: Random | 0.0312 | 0.0156 | 0.0197 |
| **Full alias-mismatch (Held-Out Test)** | 1: Dense LSA | 0.5312 | 0.2954 | 0.3515 |
| **Full alias-mismatch (Held-Out Test)** | 2: BM25 only | 0.2500 | 0.0429 | 0.0893 |
| **Full alias-mismatch (Held-Out Test)** | 3: BM25 + Dense LSA (Baseline) | 0.4688 | 0.1189 | 0.2014 |
| **Full alias-mismatch (Held-Out Test)** | 4: Entity-linked only | 0.7188 | 0.2678 | 0.3742 |
| **Full alias-mismatch (Held-Out Test)** | 5: Full System (Fixed Weight) | 0.4375 | 0.1493 | 0.2183 |
| **Full alias-mismatch (Held-Out Test)** | 9: Gated Entity RAG (Mismatch Gated) | 0.5312 | 0.1883 | 0.2692 |
| **Full alias-mismatch (Held-Out Test)** | 6: Neural Dense (all-MiniLM-L6-v2) | 0.2500 | 0.0409 | 0.0875 |
| **Full alias-mismatch (Held-Out Test)** | 7: BM25 + Neural Dense (Neural Baseline) | 0.2812 | 0.0478 | 0.1001 |
| **Full alias-mismatch (Held-Out Test)** | 8: Full Neural System (Fixed Weight) | 0.3125 | 0.0660 | 0.1224 |
| **Full alias-mismatch (Held-Out Test)** | 10: Gated Neural System (Mismatch Gated) | 0.6250 | 0.1456 | 0.2570 |
| **Full alias-mismatch (All)** | 0: Random | 0.0154 | 0.0077 | 0.0097 |
| **Full alias-mismatch (All)** | 1: Dense LSA | 0.5231 | 0.2694 | 0.3288 |
| **Full alias-mismatch (All)** | 2: BM25 only | 0.2154 | 0.0327 | 0.0730 |
| **Full alias-mismatch (All)** | 3: BM25 + Dense LSA (Baseline) | 0.4308 | 0.1118 | 0.1867 |
| **Full alias-mismatch (All)** | 4: Entity-linked only | 0.7385 | 0.2765 | 0.3850 |
| **Full alias-mismatch (All)** | 5: Full System (Fixed Weight) | 0.4615 | 0.1308 | 0.2083 |
| **Full alias-mismatch (All)** | 9: Gated Entity RAG (Mismatch Gated) | 0.5846 | 0.1932 | 0.2849 |
| **Full alias-mismatch (All)** | 6: Neural Dense (all-MiniLM-L6-v2) | 0.3077 | 0.0572 | 0.1138 |
| **Full alias-mismatch (All)** | 7: BM25 + Neural Dense (Neural Baseline) | 0.3231 | 0.0562 | 0.1155 |
| **Full alias-mismatch (All)** | 8: Full Neural System (Fixed Weight) | 0.3231 | 0.0676 | 0.1259 |
| **Full alias-mismatch (All)** | 10: Gated Neural System (Mismatch Gated) | 0.6000 | 0.1478 | 0.2532 |
| **Overall (N=500)** | 0: Random | 0.0120 | 0.0023 | 0.0045 |
| **Overall (N=500)** | 1: Dense LSA | 0.7840 | 0.3285 | 0.4376 |
| **Overall (N=500)** | 2: BM25 only | 0.7180 | 0.3063 | 0.4046 |
| **Overall (N=500)** | 3: BM25 + Dense LSA (Baseline) | 0.7700 | 0.3371 | 0.4408 |
| **Overall (N=500)** | 4: Entity-linked only | 0.8000 | 0.3260 | 0.4393 |
| **Overall (N=500)** | 5: Full System (Fixed Weight) | 0.7760 | 0.3308 | 0.4376 |
| **Overall (N=500)** | 9: Gated Entity RAG (Mismatch Gated) | 0.7820 | 0.3305 | 0.4386 |
| **Overall (N=500)** | 6: Neural Dense (all-MiniLM-L6-v2) | 0.7240 | 0.2849 | 0.3891 |
| **Overall (N=500)** | 7: BM25 + Neural Dense (Neural Baseline) | 0.7520 | 0.3068 | 0.4125 |
| **Overall (N=500)** | 8: Full Neural System (Fixed Weight) | 0.7740 | 0.3098 | 0.4202 |
| **Overall (N=500)** | 10: Gated Neural System (Mismatch Gated) | 0.7940 | 0.3179 | 0.4313 |
