"""Minimal Okapi BM25 over whitespace tokenization (language-agnostic baseline)."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence


_WS = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    return [t for t in _WS.split(text.lower().strip()) if t]


class BM25Index:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = [tokenize(d) for d in documents]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.N = len(documents)
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0.0

        df: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        for toks in self.doc_tokens:
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1

        self.idf: Dict[str, float] = {}
        for term, dfi in df.items():
            # BM25 idf variant (positive)
            self.idf[term] = math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))

    def scores(self, query: str) -> List[float]:
        q_terms = tokenize(query)
        scores = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_lens[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            tf_map = self.term_freqs[i]
            s = 0.0
            for qt in q_terms:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                idf = self.idf.get(qt, 0.0)
                s += idf * (tf * (self.k1 + 1)) / (tf + denom_norm)
            scores[i] = s
        return scores
