"""Print question_type counts from local Turkuaz CSV (same loader as live eval)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/turkuaz-rag",
        help="Path under project root with Turkuaz CSV/zip (same as eval --data-dir).",
    )
    args = p.parse_args()
    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        data_path = _repo_root() / data_path
    if not data_path.is_dir():
        print(f"Not a directory: {data_path}", file=sys.stderr)
        return 1

    from experiments.turkuaz_loader import load_local_directory

    samples = load_local_directory(data_path, limit=None)
    c = Counter(s.question_type for s in samples)
    print(f"Total rows: {len(samples)}")
    print("question_type -> count (sorted by count desc)")
    for qt, n in c.most_common():
        print(f"  {qt!r}: {n}")
    if len(c) == 1 and "unknown" in c:
        print(
            "\nWarning: all 'unknown' — check CSV column name. "
            "Loader accepts: question_type, Question_Type, type, category, ...",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
