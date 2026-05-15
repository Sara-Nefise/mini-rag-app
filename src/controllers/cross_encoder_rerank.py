"""Cross-encoder reranking (sentence-transformers) over short candidate lists."""

from __future__ import annotations

from threading import Lock
from typing import List, Optional, Sequence, Tuple

from models.db_schemes import RetrievedDocument

_lock = Lock()
_models: dict[tuple[str, Optional[str]], object] = {}


def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars]


def _get_model(model_id: str, device: Optional[str]) -> object:
    key = (model_id, device)
    with _lock:
        if key not in _models:
            from sentence_transformers import CrossEncoder

            _models[key] = CrossEncoder(model_id, device=device or None)
        return _models[key]


def cross_encoder_raw_scores_parallel(
    query: str,
    docs: Sequence[RetrievedDocument],
    candidate_indices: Sequence[int],
    model_id: str,
    device: Optional[str],
    max_input_chars: int,
    batch_size: int = 16,
) -> List[float]:
    """Raw CE scores in the same order as ``candidate_indices``."""
    if not candidate_indices:
        return []
    model = _get_model(model_id, device)
    pairs = [
        (query, _truncate(docs[int(i)].text, max_input_chars)) for i in candidate_indices
    ]
    raw = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(x) for x in raw]


def cross_encoder_rank(
    query: str,
    docs: Sequence[RetrievedDocument],
    candidate_indices: Sequence[int],
    model_id: str,
    device: Optional[str],
    max_input_chars: int,
    batch_size: int = 16,
) -> Tuple[List[int], List[float]]:
    """
    Returns (indices ordered by descending cross-encoder score, raw CE scores aligned to that order).
    """
    if not candidate_indices:
        return [], []
    raw_list = cross_encoder_raw_scores_parallel(
        query=query,
        docs=docs,
        candidate_indices=candidate_indices,
        model_id=model_id,
        device=device,
        max_input_chars=max_input_chars,
        batch_size=batch_size,
    )
    order = sorted(range(len(candidate_indices)), key=lambda j: -raw_list[j])
    ranked_idx = [int(candidate_indices[j]) for j in order]
    ranked_scores = [raw_list[j] for j in order]
    return ranked_idx, ranked_scores


def cross_encoder_rerank_topk(
    query: str,
    docs: Sequence[RetrievedDocument],
    candidate_indices: Sequence[int],
    limit: int,
    model_id: str,
    device: Optional[str],
    max_input_chars: int,
    batch_size: int = 16,
) -> Tuple[List[int], List[float]]:
    """Top `limit` doc indices by cross-encoder, with scores min-max normalized to [0, 1]."""
    ranked_idx, ranked_scores = cross_encoder_rank(
        query=query,
        docs=docs,
        candidate_indices=candidate_indices,
        model_id=model_id,
        device=device,
        max_input_chars=max_input_chars,
        batch_size=batch_size,
    )
    if not ranked_scores:
        return ranked_idx[:limit], []
    top_idx = ranked_idx[:limit]
    top_raw = ranked_scores[:limit]
    lo, hi = min(top_raw), max(top_raw)
    span = hi - lo
    if span <= 1e-12:
        norm = [1.0] * len(top_raw)
    else:
        norm = [(s - lo) / span for s in top_raw]
    return top_idx, norm
