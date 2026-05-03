"""Deterministic pseudo-embeddings for offline pipeline tests (no API keys)."""

from __future__ import annotations

import hashlib
from typing import List, Union

import numpy as np


class MockEmbeddingClient:
    """Mimics embedding_client.embed_text shape; vectors are L2-normalized."""

    def __init__(self, dim: int):
        self.embedding_size = dim

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        if isinstance(text, str):
            text = [text]
        out: List[List[float]] = []
        for t in text:
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.embedding_size)
            v = v / max(np.linalg.norm(v), 1e-12)
            out.append(v.astype(np.float64).tolist())
        return out
