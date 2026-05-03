"""Dense retrieval over a precomputed embedding matrix (cosine similarity)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from stores.llm.LLMEnums import DocumentTypeEnum


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


class DenseEmbeddingIndex:
    def __init__(self, doc_matrix: np.ndarray):
        self.doc_matrix = _normalize_rows(np.asarray(doc_matrix, dtype=np.float64))
        self.dim = self.doc_matrix.shape[1]

    def search(self, query_vec: Sequence[float], limit: int) -> Tuple[List[int], List[float]]:
        q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
        q = q / max(np.linalg.norm(q), 1e-12)
        sims = self.doc_matrix @ q
        if limit >= len(sims):
            order = np.argsort(-sims)
        else:
            # partial sort
            ind = np.argpartition(-sims, limit - 1)[:limit]
            order = ind[np.argsort(-sims[ind])]
        order = order[:limit]
        scores = sims[order].tolist()
        return order.astype(int).tolist(), scores


def _truncate_embedding_input(
    text: str,
    max_chars: int,
    max_tokens: int | None = None,
) -> str:
    """Truncate for embedding APIs (often max 8192 tokens); prefers tiktoken when installed."""
    t = (text or "").strip()
    if not t:
        return t
    if max_tokens is not None and max_tokens > 0:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            ids = enc.encode(t)
            if len(ids) > max_tokens:
                t = enc.decode(ids[:max_tokens])
        except ImportError:
            # no tiktoken: conservative char cap vs token limit (~2 chars/token worst case)
            cap = min(max_chars, max_tokens * 2)
            if len(t) > cap:
                t = t[:cap]
        except Exception:
            # first-time tiktoken can fetch BPE from network; proxy/offline -> approximate cap
            cap = min(max_chars, max_tokens * 2)
            if len(t) > cap:
                t = t[:cap]
    if len(t) > max_chars:
        t = t[:max_chars]
    return t


def embed_documents_batched(
    embedding_client,
    texts: List[str],
    batch_size: int | None = None,
    max_chars: int | None = None,
) -> np.ndarray:
    """Embed document texts using the app embedding client (batched)."""
    from helpers.config import get_settings

    settings = get_settings()
    bs = batch_size if batch_size is not None else int(settings.EMBEDDING_BATCH_SIZE)
    mc = max_chars if max_chars is not None else int(settings.EMBEDDING_INPUT_MAX_CHARS)
    mt = int(settings.EMBEDDING_MAX_INPUT_TOKENS)

    out: List[List[float]] = []
    for i in range(0, len(texts), bs):
        batch_raw = texts[i : i + bs]
        batch = [_truncate_embedding_input(t, mc, max_tokens=mt) for t in batch_raw]
        vecs = embedding_client.embed_text(
            text=batch,
            document_type=DocumentTypeEnum.DOCUMENT.value,
        )
        if not vecs or len(vecs) != len(batch):
            raise RuntimeError("Embedding provider returned an unexpected batch result")
        out.extend(vecs)
    return np.asarray(out, dtype=np.float64)


def embed_query(embedding_client, text: str) -> List[float]:
    from helpers.config import get_settings

    settings = get_settings()
    mc = int(settings.EMBEDDING_INPUT_MAX_CHARS)
    mt = int(settings.EMBEDDING_MAX_INPUT_TOKENS)
    text = _truncate_embedding_input(text, mc, max_tokens=mt)
    vecs = embedding_client.embed_text(
        text=text,
        document_type=DocumentTypeEnum.QUERY.value,
    )
    if not vecs or len(vecs) == 0:
        raise RuntimeError("Failed to embed query")
    return vecs[0]


def make_embedding_client_from_settings():
    """Construct the same embedding client as FastAPI startup."""
    from helpers.config import get_settings
    from stores.llm.LLMProviderFactory import LLMProviderFactory

    settings = get_settings()
    factory = LLMProviderFactory(settings)
    client = factory.create(provider=settings.EMBEDDING_BACKEND, embedding=True)
    if client is None:
        raise RuntimeError(
            "Could not create embedding client. Set EMBEDDING_BACKEND to OPENAI, COHERE, or "
            f"SENTENCE_TRANSFORMERS (got {settings.EMBEDDING_BACKEND!r})."
        )
    client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )
    return client
