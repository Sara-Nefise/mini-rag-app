# Live eval analysis — `20260511T103047Z_62ebd1e7`

- Evaluable samples: **50**
- Retrieval profile: **baseline**

> Only one question_type in this run ('Inference Question'). Stratification by type matches overall. Use Turkuaz CSV / larger slice / jsonl with mixed types for cross-type comparison.

## Metrics by question type

| question_type | n | single@10 | both@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|---:|---:|
| Inference Question | 50 | 0.76 | 0.3 | 0.5314 | 0.4535 |

## Metrics by question length (words)

| length_bucket | n | single@10 | both@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|---:|---:|
| medium_41-79_words | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| short_<=40_words | 49 | 0.7755 | 0.3061 | 0.5422 | 0.4627 |

## Hardest types (lowest both@10, first = hardest)

1. `Inference Question` — both@10=0.3, n=50
