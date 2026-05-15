from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from models.db_schemes import RetrievedDocument

logger = logging.getLogger("uvicorn.error")

RRF_K = 60


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "for",
    "to",
    "of",
    "in",
    "on",
    "is",
    "are",
    "was",
    "were",
    "with",
    "by",
    "from",
    "that",
    "this",
    "bu",
    "ve",
    "ile",
    "ya",
    "veya",
    "da",
    "de",
    "bir",
}


@dataclass(frozen=True)
class RetrievalProfileConfig:
    name: str
    adaptive_k: bool
    hybrid: bool
    calibration: bool
    coverage: bool
    overfetch_factor: int
    max_overfetch: int
    use_cross_encoder: bool = False


def get_retrieval_profile(name: str | None) -> RetrievalProfileConfig:
    """Map API ``retrieval_profile`` string to pipeline flags.

    Notes:
    - ``hybrid_xe`` = dense + RRF + CE with **no** inner dense calibration (compare vs ``hybrid_calibrated_xe``).
    - ``coverage`` / ``full`` use extra RRF lists from sub-intents + greedy coverage rerank.
    """
    key = (name or "baseline").strip().lower()
    if key == "baseline":
        return RetrievalProfileConfig(
            name="baseline",
            adaptive_k=False,
            hybrid=False,
            calibration=False,
            coverage=False,
            overfetch_factor=1,
            max_overfetch=30,
            use_cross_encoder=False,
        )
    if key == "hybrid":
        return RetrievalProfileConfig(
            name="hybrid",
            adaptive_k=False,
            hybrid=True,
            calibration=False,
            coverage=False,
            overfetch_factor=3,
            max_overfetch=90,
            use_cross_encoder=False,
        )
    if key == "hybrid_calibrated":
        return RetrievalProfileConfig(
            name="hybrid_calibrated",
            adaptive_k=False,
            hybrid=True,
            calibration=True,
            coverage=False,
            overfetch_factor=3,
            max_overfetch=90,
            use_cross_encoder=False,
        )
    if key == "coverage":
        return RetrievalProfileConfig(
            name="coverage",
            adaptive_k=False,
            hybrid=True,
            calibration=True,
            coverage=True,
            overfetch_factor=5,
            max_overfetch=150,
            use_cross_encoder=False,
        )
    if key == "full":
        return RetrievalProfileConfig(
            name="full",
            adaptive_k=True,
            hybrid=True,
            calibration=True,
            coverage=True,
            overfetch_factor=6,
            max_overfetch=165,
            use_cross_encoder=False,
        )
    if key == "hybrid_xe":
        return RetrievalProfileConfig(
            name="hybrid_xe",
            adaptive_k=False,
            hybrid=True,
            calibration=False,
            coverage=False,
            overfetch_factor=4,
            max_overfetch=120,
            use_cross_encoder=True,
        )
    if key == "hybrid_calibrated_xe":
        return RetrievalProfileConfig(
            name="hybrid_calibrated_xe",
            adaptive_k=False,
            hybrid=True,
            calibration=True,
            coverage=False,
            overfetch_factor=4,
            max_overfetch=120,
            use_cross_encoder=True,
        )
    if key == "full_xe":
        return RetrievalProfileConfig(
            name="full_xe",
            adaptive_k=True,
            hybrid=True,
            calibration=True,
            coverage=False,
            overfetch_factor=7,
            max_overfetch=175,
            use_cross_encoder=True,
        )
    return get_retrieval_profile("baseline")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


def _content_tokens(text: str) -> Set[str]:
    return {t for t in _tokenize(text) if t not in _STOPWORDS}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _score_calibration(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores))
    std = math.sqrt(max(var, 1e-9))
    return [_sigmoid((s - mean) / std) for s in scores]


def _idf_map(corpus_tokens: Sequence[Set[str]]) -> Dict[str, float]:
    n_docs = max(1, len(corpus_tokens))
    df: Dict[str, int] = {}
    for toks in corpus_tokens:
        for tok in toks:
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0 for tok, freq in df.items()}


def _lexical_score(query_tokens: Set[str], doc_tokens: Set[str], idf: Dict[str, float]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = query_tokens & doc_tokens
    if not overlap:
        return 0.0
    denom = sum(idf.get(t, 1.0) for t in query_tokens) or 1.0
    return sum(idf.get(t, 1.0) for t in overlap) / denom


def _lex_order_for_subset(
    subset: Set[str],
    doc_tokens: Sequence[Set[str]],
    idf: Dict[str, float],
    n: int,
    raw_dense: Sequence[float],
) -> List[int]:
    """Lexical ranking using only ``subset`` tokens (sub-query proxy for multi-evidence)."""
    if not subset or n <= 0:
        return sorted(range(n), key=lambda i: -float(raw_dense[i]))
    lex_s = [_lexical_score(subset, dt, idf) for dt in doc_tokens]
    return sorted(range(n), key=lambda i: (-float(lex_s[i]), -float(raw_dense[i])))


def _dedupe_doc_key(doc: RetrievedDocument) -> str:
    md = doc.metadata or {}
    mid = md.get("mlsum_id")
    if mid is not None and str(mid).strip() != "":
        return f"mlsum:{str(mid).strip()}"
    aid = md.get("asset_id", md.get("article_id"))
    if aid is not None and str(aid).strip() != "":
        return f"asset:{str(aid).strip()}"
    if doc.chunk_id is not None:
        return f"chunk:{doc.chunk_id}"
    return f"row:{id(doc)}"


def _rrf_scores(n: int, rank_lists: Sequence[Sequence[int]], k_const: int = RRF_K) -> List[float]:
    scores = [0.0] * n
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            if 0 <= i < n:
                scores[i] += 1.0 / (k_const + r + 1)
    return scores


def _normalize_01(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 1e-12:
        return [1.0] * len(values)
    return [(float(v) - lo) / span for v in values]


def _jaccard_tokens(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _build_candidate_pool(
    order: Sequence[int],
    docs: Sequence[RetrievedDocument],
    pool_size: int,
) -> List[int]:
    """Prefer one strong chunk per article (mlsum_id), then fill to pool_size."""
    if pool_size <= 0:
        return []
    seen_key: Set[str] = set()
    out: List[int] = []
    for i in order:
        if len(out) >= pool_size:
            break
        k = _dedupe_doc_key(docs[i])
        if k in seen_key:
            continue
        seen_key.add(k)
        out.append(int(i))
    for i in order:
        if len(out) >= pool_size:
            break
        ii = int(i)
        if ii not in out:
            out.append(ii)
    return out[:pool_size]


def _mmr_select(
    pool: Sequence[int],
    relevance: Sequence[float],
    doc_tokens: Sequence[Set[str]],
    limit: int,
    lambda_param: float = 0.88,
) -> List[int]:
    if not pool:
        return []
    if len(pool) <= limit:
        return list(pool[:limit])
    selected: List[int] = []
    cand = set(int(i) for i in pool)
    while cand and len(selected) < limit:
        best_i: int | None = None
        best_s = -1e9
        for i in cand:
            rel_i = float(relevance[i]) if i < len(relevance) else 0.0
            if selected:
                div = max(_jaccard_tokens(doc_tokens[i], doc_tokens[j]) for j in selected)
            else:
                div = 0.0
            score = lambda_param * rel_i - (1.0 - lambda_param) * div
            if score > best_s:
                best_s = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        cand.remove(best_i)
    return selected


def _documents_with_scores(
    docs: Sequence[RetrievedDocument],
    indices: Sequence[int],
    scores: Sequence[float],
    *,
    scores_parallel_to_indices: bool = False,
) -> List[RetrievedDocument]:
    out: List[RetrievedDocument] = []
    for pos, i in enumerate(indices):
        d = docs[i]
        if scores_parallel_to_indices:
            sc = float(scores[pos]) if pos < len(scores) else float(d.score)
        else:
            sc = float(scores[i]) if i < len(scores) else float(d.score)
        try:
            out.append(d.model_copy(update={"score": sc}))
        except AttributeError:
            out.append(
                RetrievedDocument(
                    text=d.text,
                    score=sc,
                    metadata=d.metadata,
                    chunk_id=d.chunk_id,
                )
            )
    return out


def _split_sub_intents(query: str, max_parts: int = 3) -> List[Set[str]]:
    q = (query or "").strip()
    if not q:
        return []
    chunks = [p.strip() for p in re.split(r"[?؟!.,؛;]|(?:\s+(?:and|or|ve|veya|ama|ile|ya)\s+)", q, flags=re.IGNORECASE) if p.strip()]
    if len(chunks) < 2 and len(q) > 60:
        mid = max(1, len(q) // 2)
        chunks = [q[:mid], q[mid:]]
    intents: List[Set[str]] = []
    for ch in chunks[:max_parts]:
        toks = _content_tokens(ch)
        if toks:
            intents.append(toks)
    if not intents:
        all_toks = _content_tokens(q)
        if all_toks:
            intents.append(all_toks)
    return intents


def adaptive_fetch_limit(query: str, requested_limit: int, profile: RetrievalProfileConfig) -> int:
    base = max(1, requested_limit)
    if not profile.adaptive_k:
        return min(profile.max_overfetch, base * profile.overfetch_factor)
    q = (query or "").strip().lower()
    complexity = 0
    complexity += 1 if len(q.split()) >= 8 else 0
    complexity += 1 if len(q.split()) >= 14 else 0
    complexity += 1 if any(tok in q for tok in (" and ", " ve ", " veya ", " or ", " ile ", " then ", " sonra ")) else 0
    factor = min(6, profile.overfetch_factor + complexity)
    return min(profile.max_overfetch, max(base, base * factor))


def rerank_documents(
    query: str,
    docs: Sequence[RetrievedDocument],
    limit: int,
    profile: RetrievalProfileConfig,
    fusion_ce_weight: Optional[float] = None,
) -> List[RetrievedDocument]:
    if not docs:
        return []
    if profile.name == "baseline":
        return list(docs)[:limit]

    from helpers.config import get_settings

    settings = get_settings()
    n = len(docs)
    query_tokens = _content_tokens(query)
    doc_tokens = [_content_tokens(d.text) for d in docs]
    idf = _idf_map(doc_tokens)
    raw_dense = [float(d.score) for d in docs]
    dense = _score_calibration(raw_dense) if profile.calibration else raw_dense
    lex = [_lexical_score(query_tokens, toks, idf) for toks in doc_tokens]

    dense_order = sorted(range(n), key=lambda i: (-float(dense[i]), -raw_dense[i]))
    lex_order = sorted(range(n), key=lambda i: (-float(lex[i]), -raw_dense[i]))
    rank_lists: List[List[int]] = [dense_order, lex_order]
    if profile.coverage:
        for intent in _split_sub_intents(query)[:2]:
            if len(intent) >= 2:
                rank_lists.append(_lex_order_for_subset(intent, doc_tokens, idf, n, raw_dense))
    rrf_vec = _rrf_scores(n, rank_lists, k_const=RRF_K)
    rrf_norm = _normalize_01(rrf_vec)
    dense_norm = _normalize_01(raw_dense)
    rel = [0.62 * r + 0.38 * d for r, d in zip(rrf_norm, dense_norm)]

    if not profile.coverage:
        order_rrf = sorted(range(n), key=lambda i: (-rrf_vec[i], -raw_dense[i]))
        pool_cap = min(max(settings.RERANKER_POOL_SIZE, limit * 4, 25), n)
        pool = _build_candidate_pool(order_rrf, docs, pool_cap)
        if profile.use_cross_encoder and not settings.RERANKER_ENABLED:
            logger.info(
                "Profile %s uses cross-encoder when RERANKER_ENABLED=true; falling back to MMR.",
                profile.name,
            )
        ce_on = profile.use_cross_encoder and settings.RERANKER_ENABLED
        if ce_on:
            try:
                from controllers.cross_encoder_rerank import cross_encoder_raw_scores_parallel

                raw_ce = cross_encoder_raw_scores_parallel(
                    query=query,
                    docs=docs,
                    candidate_indices=pool,
                    model_id=settings.RERANKER_MODEL_ID,
                    device=(settings.RERANKER_DEVICE or "cpu").strip() or "cpu",
                    max_input_chars=settings.RERANKER_MAX_INPUT_CHARS,
                    batch_size=settings.RERANKER_BATCH_SIZE,
                )
                ce_n = _normalize_01(raw_ce)
                w_cfg = float(settings.RERANKER_FUSION_CE_WEIGHT)
                if fusion_ce_weight is not None:
                    w_cfg = float(fusion_ce_weight)
                w = max(0.0, min(1.0, w_cfg))
                fused: List[tuple[float, int]] = []
                for pos, idx in enumerate(pool):
                    idx = int(idx)
                    score = w * ce_n[pos] + (1.0 - w) * float(rel[idx])
                    fused.append((score, idx))
                fused.sort(key=lambda t: (-t[0], -raw_dense[t[1]]))
                picked = [idx for _, idx in fused[:limit]]
                norms = [s for s, _ in fused[:limit]]
                return _documents_with_scores(
                    docs, picked, norms, scores_parallel_to_indices=True
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cross-encoder rerank failed; using MMR fallback: %s", exc)
        picked = _mmr_select(pool, rel, doc_tokens, limit, lambda_param=0.88)
        return _documents_with_scores(docs, picked, rel)

    base_scores = list(rel)
    intents = _split_sub_intents(query)
    # Only reward "coverage" for tokens tied to the question (not random article vocabulary).
    surface: Set[str] = set(query_tokens)
    for intent in intents:
        surface |= intent
    if not surface:
        surface = set(query_tokens)
    selected: List[int] = []
    covered_tokens: Set[str] = set()
    available = set(range(len(docs)))
    while available and len(selected) < limit:
        best_idx = None
        best_score = -1e9
        for i in available:
            toks = doc_tokens[i]
            # New question-relevant lemmas this doc would add vs already selected.
            relevant_new = (toks & surface) - covered_tokens
            novelty = len(relevant_new) / max(1, len(surface))
            intent_gain = 0.0
            for intent in intents:
                if not intent:
                    continue
                intent_surface = intent & surface
                if not intent_surface:
                    continue
                before = len(intent_surface & covered_tokens)
                after = len(intent_surface & (covered_tokens | toks))
                intent_gain += max(0, after - before) / max(1, len(intent_surface))
            redundancy = 0.0
            if selected:
                redundancy = max(
                    len(toks & doc_tokens[j]) / max(1, len(toks | doc_tokens[j]))
                    for j in selected
                )
            # Sub-intent coverage bonuses (tuned for both-hit).
            score = base_scores[i] + 0.15 * novelty + 0.22 * intent_gain - 0.10 * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
        covered_tokens |= doc_tokens[best_idx]
        available.remove(best_idx)
    return _documents_with_scores(docs, selected[:limit], rel)
