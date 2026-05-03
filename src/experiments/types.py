from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TurkuazSample:
    """One benchmark row aligned with eneSadi/turkuaz-rag."""

    sample_id: int
    question: str
    answer: str
    contexts: Tuple[str, str]
    question_type: str
    #: MLSUM Turkish train row ids as strings (from CSV); used for Scenario 2 full-corpus eval.
    gold_mlsum_ids: Optional[Tuple[str, str]] = None
