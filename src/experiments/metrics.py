"""Retrieval metrics aligned with Turkuaz-RAG (single-doc vs both-doc recall)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def recall_single_both(
    ranked_doc_ids: Sequence[int],
    gold_pair: Tuple[int, int],
    k: int,
) -> Tuple[bool, bool]:
    """Return (single_news_recall@k, both_news_recall@k)."""
    top = set(ranked_doc_ids[:k])
    g0, g1 = gold_pair
    single = g0 in top or g1 in top
    both = g0 in top and g1 in top
    return single, both


def mrr_first_relevant(ranked_doc_ids: Sequence[int], gold_pair: Tuple[int, int]) -> float:
    """Reciprocal rank of the first retrieved gold document."""
    gold = set(gold_pair)
    for i, did in enumerate(ranked_doc_ids):
        if did in gold:
            return 1.0 / (i + 1)
    return 0.0


def aggregate_means(rows: Iterable[dict], ks: Sequence[int]) -> dict:
    """Average boolean recalls across samples for keys like both@10."""
    buckets = {f"single@{k}": [] for k in ks}
    buckets.update({f"both@{k}": [] for k in ks})
    mrrs: List[float] = []

    for row in rows:
        mrrs.append(row.get("mrr", 0.0))
        for k in ks:
            buckets[f"single@{k}"].append(1.0 if row.get(f"single@{k}") else 0.0)
            buckets[f"both@{k}"].append(1.0 if row.get(f"both@{k}") else 0.0)

    out = {key: (sum(vals) / len(vals) if vals else 0.0) for key, vals in buckets.items()}
    out["mrr"] = sum(mrrs) / len(mrrs) if mrrs else 0.0
    return out
