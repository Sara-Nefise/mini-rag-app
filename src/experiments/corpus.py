"""Closed retrieval pool from benchmark samples (deduplicated article texts)."""

from __future__ import annotations

from typing import Dict, List, Tuple

from experiments.types import TurkuazSample


def build_doc_pool(samples: List[TurkuazSample]) -> Tuple[List[str], Dict[int, Tuple[int, int]]]:
    """
    Map unique context strings to contiguous doc_ids [0..N-1].

    Returns:
        doc_texts: list indexed by doc_id
        gold_by_sample_id: sample_id -> (doc_id_a, doc_id_b)
    """
    text_to_id: Dict[str, int] = {}
    doc_texts: List[str] = []

    def _get_id(text: str) -> int:
        if text not in text_to_id:
            text_to_id[text] = len(doc_texts)
            doc_texts.append(text)
        return text_to_id[text]

    gold_by_sample_id: Dict[int, Tuple[int, int]] = {}
    for s in samples:
        if len(s.contexts) != 2:
            raise ValueError(f"Sample {s.sample_id} must have exactly 2 contexts")
        a = _get_id(s.contexts[0])
        b = _get_id(s.contexts[1])
        gold_by_sample_id[s.sample_id] = (a, b)

    return doc_texts, gold_by_sample_id
