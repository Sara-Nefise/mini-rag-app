"""Load MLSUM-derived manifest and map Turkuaz gold MLSUM ids to corpus indices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from experiments.types import TurkuazSample


def _gold_id_set(samples: List[TurkuazSample]) -> Set[str]:
    out: Set[str] = set()
    for s in samples:
        if not s.gold_mlsum_ids:
            continue
        a, b = s.gold_mlsum_ids
        out.add(str(a).strip())
        out.add(str(b).strip())
    return out


def load_mlsum_manifest_subset(
    manifest_path: Path,
    samples: List[TurkuazSample],
    extra_noise_docs: int,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Stream manifest once: keep every row whose id is gold for the loaded samples,
    plus up to ``extra_noise_docs`` other distinct ids as retrieval distractors.
    Stops early once all gold ids are present and the noise budget is filled.

    Use for quick runs without embedding the full MLSUM corpus.
    """
    manifest_path = Path(manifest_path).resolve()
    needed = _gold_id_set(samples)
    if not needed:
        raise ValueError("No gold MLSUM ids in samples; use --corpus-mode closed.")

    doc_texts: List[str] = []
    id_to_idx: Dict[str, int] = {}
    noise_added = 0

    def try_add(kid: str, text: str) -> bool:
        if kid in id_to_idx:
            return False
        id_to_idx[kid] = len(doc_texts)
        doc_texts.append(text)
        return True

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            kid = str(row["id"]).strip()
            text = str(row["text"])

            if kid in needed:
                try_add(kid, text)
            elif noise_added < extra_noise_docs:
                if try_add(kid, text):
                    noise_added += 1

            gold_ok = needed.issubset(id_to_idx.keys())
            if gold_ok and noise_added >= extra_noise_docs:
                break

    return doc_texts, id_to_idx


def load_mlsum_manifest(
    manifest_path: Path,
    max_docs: Optional[int] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Load JSONL manifest lines: {"id": "<string>", "text": "<article>"}.

    Returns doc_texts (order = corpus index) and id -> corpus_index.
    """
    manifest_path = Path(manifest_path).resolve()
    doc_texts: List[str] = []
    id_to_idx: Dict[str, int] = {}

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if max_docs is not None and len(doc_texts) >= max_docs:
                break
            row = json.loads(line)
            kid = str(row["id"]).strip()
            text = str(row["text"])
            if kid in id_to_idx:
                raise ValueError(f"Duplicate manifest id {kid!r} in {manifest_path}")
            id_to_idx[kid] = len(doc_texts)
            doc_texts.append(text)

    return doc_texts, id_to_idx


def build_mlsum_gold_map(
    samples: List[TurkuazSample],
    id_to_idx: Dict[str, int],
) -> Dict[int, Tuple[int, int]]:
    """Map each benchmark sample to two corpus indices using Turkuaz gold MLSUM ids."""
    gold_by_sample_id: Dict[int, Tuple[int, int]] = {}
    missing: List[str] = []

    for s in samples:
        if not s.gold_mlsum_ids:
            raise ValueError(
                f"Sample {s.sample_id} has no 1st_news_id/2nd_news_id — "
                "reload Turkuaz CSV or use --corpus-mode closed."
            )
        ga, gb = s.gold_mlsum_ids
        if ga not in id_to_idx:
            missing.append(f"sample {s.sample_id}: 1st_news_id={ga!r} not in MLSUM manifest")
        if gb not in id_to_idx:
            missing.append(f"sample {s.sample_id}: 2nd_news_id={gb!r} not in MLSUM manifest")
        if ga in id_to_idx and gb in id_to_idx:
            gold_by_sample_id[s.sample_id] = (id_to_idx[ga], id_to_idx[gb])

    if missing:
        raise KeyError(
            "Gold article ids missing from MLSUM manifest (wrong manifest/split or "
            "--mlsum-max-docs too small). First errors:\n"
            + "\n".join(missing[:12])
        )

    if len(gold_by_sample_id) != len(samples):
        raise RuntimeError("Internal error: incomplete gold map.")

    return gold_by_sample_id
