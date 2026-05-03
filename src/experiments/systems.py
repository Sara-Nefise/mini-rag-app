"""Retrieval system variants for Turkuaz-RAG comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.bm25 import BM25Index
from experiments.retrieval_backend import DenseEmbeddingIndex, embed_query

RANK_DEPTH = 80
RRF_K = 60


def reciprocal_rank_fusion(rank_lists: Sequence[Sequence[int]], k_const: int = RRF_K) -> List[int]:
    scores: Dict[int, float] = defaultdict(float)
    for ranks in rank_lists:
        for r, doc_id in enumerate(ranks):
            scores[int(doc_id)] += 1.0 / (k_const + r + 1)
    return sorted(scores.keys(), key=lambda x: (-scores[x], x))


def mmr_high_diversity(
    query_vec: Sequence[float],
    cand_ranked_ids: Sequence[int],
    doc_matrix: np.ndarray,
    limit: int,
    lambda_param: float = 0.65,
) -> List[int]:
    """Maximal marginal relevance over embedding cosine similarities."""
    q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
    q = q / max(np.linalg.norm(q), 1e-12)
    X = _normalize_rows(np.asarray(doc_matrix, dtype=np.float64))

    relevances: Dict[int, float] = {}
    for did in cand_ranked_ids:
        relevances[did] = float(X[did] @ q)

    selected: List[int] = []
    candidates = list(cand_ranked_ids)

    while candidates and len(selected) < limit:
        best_id = None
        best_score = -1e9
        for did in candidates:
            rel = relevances.get(did, 0.0)
            if selected:
                div = max(float(X[did] @ X[sid]) for sid in selected)
            else:
                div = 0.0
            score = lambda_param * rel - (1.0 - lambda_param) * div
            if score > best_score:
                best_score = score
                best_id = did
        if best_id is None:
            break
        selected.append(best_id)
        candidates.remove(best_id)
    return selected


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


class RetrievalSystems:
    """Factory for ranked doc-id lists given a benchmark query string."""

    def __init__(
        self,
        embedding_client,
        dense_index: DenseEmbeddingIndex,
        doc_matrix: np.ndarray,
        bm25: BM25Index,
        doc_texts: List[str],
    ):
        self.embedding_client = embedding_client
        self.dense_index = dense_index
        self.doc_matrix = doc_matrix
        self.bm25 = bm25
        self.doc_texts = doc_texts

    def _dense_rank(self, query: str, depth: int = RANK_DEPTH) -> List[int]:
        qv = embed_query(self.embedding_client, query)
        ids, _ = self.dense_index.search(qv, min(depth, len(self.doc_texts)))
        return ids

    def _bm25_rank(self, query: str, depth: int = RANK_DEPTH) -> List[int]:
        scores = self.bm25.scores(query)
        order = np.argsort(-np.asarray(scores))[:depth]
        return order.astype(int).tolist()

    def run(self, name: str, query: str) -> List[int]:
        n = len(self.doc_texts)
        depth = min(RANK_DEPTH, n)

        if name == "dense_top1":
            r = self._dense_rank(query, depth=1)
            return r

        if name == "dense_topk":
            return self._dense_rank(query, depth=depth)

        if name == "hybrid_rrf":
            d_rank = self._dense_rank(query, depth=depth)
            b_rank = self._bm25_rank(query, depth=depth)
            return reciprocal_rank_fusion([d_rank, b_rank])[:depth]

        if name == "dense_bm25_rerank":
            cand = self._dense_rank(query, depth=min(depth, max(20, n)))
            scores = self.bm25.scores(query)
            cand_sorted = sorted(cand, key=lambda i: -scores[i])
            return cand_sorted[:depth]

        if name == "fusion_mmr":
            qv = embed_query(self.embedding_client, query)
            cand = self._dense_rank(query, depth=min(40, n))
            picked = mmr_high_diversity(qv, cand, self.doc_matrix, limit=min(10, n))
            # pad with remaining dense order for metric truncation
            tail = [i for i in cand if i not in picked]
            return picked + tail[: max(0, depth - len(picked))]

        if name == "pseudo_agent_multiquery":
            # Two queries without an LLM: full question + tail substring (weak decomposition proxy).
            q1 = query.strip()
            tail_start = max(len(q1) // 2, 1)
            q2 = q1[tail_start:].strip() or q1
            r1 = self._dense_rank(q1, depth=depth)
            r2 = self._dense_rank(q2, depth=depth)
            return reciprocal_rank_fusion([r1, r2])[:depth]

        raise ValueError(f"Unknown system: {name}")


def default_system_names() -> List[str]:
    return [
        "dense_top1",
        "dense_topk",
        "hybrid_rrf",
        "dense_bm25_rerank",
        "fusion_mmr",
        "pseudo_agent_multiquery",
    ]
