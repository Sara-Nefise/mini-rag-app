"""Load Turkuaz-RAG from Hugging Face or a local JSONL file."""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.types import TurkuazSample


def _split_contexts_field(raw: Any) -> Tuple[str, str]:
    if isinstance(raw, list):
        if len(raw) >= 2:
            return str(raw[0]).strip(), str(raw[1]).strip()
        if len(raw) == 1:
            t = str(raw[0]).strip()
            return t, t
        raise ValueError("Empty contexts list")
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        return raw.strip(), raw.strip()
    raise TypeError(f"Unsupported contexts type: {type(raw)}")


def _question_type(row: Dict[str, Any]) -> str:
    for key in ("question_type", "type", "category", "Question Type"):
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    return "unknown"


def _context_pair_from_row(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Hugging Face ships `eneSadi/turkuaz-rag` as CSV with `1st_news` / `2nd_news`.
    Some local JSONL files may use a single `contexts` field instead.
    """
    if "contexts" in row and row["contexts"] is not None:
        return _split_contexts_field(row["contexts"])
    if "1st_news" in row and "2nd_news" in row:
        return str(row["1st_news"]).strip(), str(row["2nd_news"]).strip()
    for a, b in (("first_news", "second_news"), ("news_1", "news_2"), ("context_1", "context_2")):
        if a in row and b in row:
            return str(row[a]).strip(), str(row[b]).strip()
    raise KeyError(
        "Could not find news contexts. Expected 'contexts' or '1st_news'+'2nd_news' "
        f"(got columns: {list(row.keys())})"
    )


def row_to_sample(idx: int, row: Dict[str, Any]) -> TurkuazSample:
    q = str(row["question"]).strip()
    a = str(row.get("answer", "")).strip()
    ca, cb = _context_pair_from_row(row)
    qt = _question_type(row)
    gid_a = row.get("1st_news_id")
    gid_b = row.get("2nd_news_id")
    gold_ids: Optional[Tuple[str, str]] = None
    if gid_a is not None and gid_b is not None:
        gold_ids = (str(gid_a).strip(), str(gid_b).strip())
    return TurkuazSample(
        sample_id=idx,
        question=q,
        answer=a,
        contexts=(ca, cb),
        question_type=qt,
        gold_mlsum_ids=gold_ids,
    )


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[TurkuazSample]:
    samples: List[TurkuazSample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append(row_to_sample(i, row))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def _normalize_csv_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Strip BOM/spaces from CSV headers (stdlib csv keeps exact keys)."""
    return {(k or "").strip().lstrip("\ufeff"): v for k, v in raw.items()}


def load_samples_from_zip_csv(zip_path: Path, limit: Optional[int] = None) -> List[TurkuazSample]:
    """Read Turkuaz-RAG rows from a zip that contains one main CSV file."""
    zip_path = Path(zip_path)
    samples: List[TurkuazSample] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV inside {zip_path}")
        member = csv_members[0]
        with zf.open(member) as raw_f:
            text_f = io.TextIOWrapper(raw_f, encoding="utf-8", newline="")
            reader = csv.DictReader(text_f)
            for raw_row in reader:
                if limit is not None and len(samples) >= limit:
                    break
                row = _normalize_csv_row(raw_row)
                samples.append(row_to_sample(len(samples), row))
    return samples


def load_samples_from_csv_file(csv_path: Path, limit: Optional[int] = None) -> List[TurkuazSample]:
    """Read Turkuaz-RAG rows from a plain CSV file on disk."""
    csv_path = Path(csv_path)
    samples: List[TurkuazSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            if limit is not None and len(samples) >= limit:
                break
            row = _normalize_csv_row(raw_row)
            samples.append(row_to_sample(len(samples), row))
    return samples


def load_local_directory(root: Path, limit: Optional[int] = None) -> List[TurkuazSample]:
    """
    Load from a folder previously populated by `download_dataset.py` (snapshot)
    or containing `turkuaz_rag_huggingface.csv.zip` / extracted CSV.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    preferred = sorted(root.rglob("turkuaz_rag_huggingface.csv.zip"))
    if preferred:
        return load_samples_from_zip_csv(preferred[0], limit)

    zips = sorted([p for p in root.rglob("*.csv.zip") if p.is_file()])
    if zips:
        return load_samples_from_zip_csv(zips[0], limit)

    turkuaz_csv = [p for p in root.rglob("*.csv") if p.is_file() and "turkuaz" in p.name.lower()]
    if turkuaz_csv:
        return load_samples_from_csv_file(sorted(turkuaz_csv)[0], limit)

    any_csv = sorted([p for p in root.rglob("*.csv") if p.is_file()])
    if any_csv:
        return load_samples_from_csv_file(any_csv[0], limit)

    raise FileNotFoundError(
        f"No Turkuaz CSV or csv.zip found under {root}. Run: python -m experiments.download_dataset"
    )


def _load_huggingface_hub_zip_csv(limit: Optional[int] = None) -> List[TurkuazSample]:
    """
    Load the official CSV release without `datasets` / pandas (stdlib only).

    Use when numpy/pandas/datasets fail on bleeding-edge Python builds (e.g. 3.14).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Install `huggingface_hub` (pip install huggingface_hub) "
            "or fix your numpy/pandas stack and use `datasets`."
        ) from e

    repo_id = "eneSadi/turkuaz-rag"
    zip_names = (
        "turkuaz_rag_huggingface.csv.zip",
        "data/turkuaz_rag_huggingface.csv.zip",
    )
    zip_path = None
    last_err: Optional[Exception] = None
    for fname in zip_names:
        try:
            zip_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname)
            break
        except Exception as e:
            last_err = e
            zip_path = None
    if zip_path is None:
        raise RuntimeError(
            f"Could not download dataset zip from {repo_id}; last error: {last_err}"
        )

    return load_samples_from_zip_csv(Path(zip_path), limit)


def _load_huggingface_datasets(limit: Optional[int] = None) -> List[TurkuazSample]:
    from datasets import load_dataset

    ds = load_dataset("eneSadi/turkuaz-rag", split="train")
    n = len(ds) if limit is None else min(limit, len(ds))
    samples: List[TurkuazSample] = []
    for i in range(n):
        row = ds[i]
        row_dict = {k: row[k] for k in row.keys()}
        samples.append(row_to_sample(i, row_dict))
    return samples


def load_huggingface(limit: Optional[int] = None) -> List[TurkuazSample]:
    try:
        return _load_huggingface_datasets(limit)
    except Exception as e:
        print(
            f"Warning: `datasets` loader failed ({e!s}); "
            "falling back to Hugging Face Hub CSV zip (no pandas).",
            file=sys.stderr,
            flush=True,
        )
        return _load_huggingface_hub_zip_csv(limit)
