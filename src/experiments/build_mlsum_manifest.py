"""
Build a JSONL manifest for Turkish MLSUM (`reciTAL/mlsum`, config `tu`).

Each line: {"id": "<row_index>", "text": "<article body>"}

Turkuaz-RAG CSV columns `1st_news_id` / `2nd_news_id` refer to **row indices**
into this manifest when built from `split=train` in dataset iteration order
(the default Hugging Face `datasets` order).

Requires a working `datasets` + numpy stack (use Python 3.11 or 3.12 if 3.14 fails).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build MLSUM Turkish manifest JSONL for Scenario 2 RAG.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: <project>/data/mlsum-tu/manifest.jsonl)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="MSLUM split (default: train — matches Turkuaz paper corpus scope)",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for debugging only.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use HF IterableDataset (can fail: HTTP range requests not supported on some hosts).",
    )
    args = parser.parse_args(argv)

    # Non-interactive; allow reciTAL/mlsum custom builder (datasets 2.x)
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

    out = args.out if args.out is not None else _repo_root() / "data" / "mlsum-tu" / "manifest.jsonl"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(
            "\nCould not import `datasets` (often broken on Python 3.14 + bad numpy stack).\n\n"
            "Fix (pick one):\n"
            "  1) Install Python 3.12 via Homebrew, then new venv:\n"
            "       brew install python@3.12\n"
            "       /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv312\n"
            "       source .venv312/bin/activate\n"
            "       pip install -U pip && pip install datasets numpy huggingface_hub\n\n"
            "  2) Use Apple system Python 3.9 (often stable):\n"
            "       cd …/mini-rag-app && /usr/bin/python3 -m venv .venv_sys\n"
            "       source .venv_sys/bin/activate\n"
            "       pip install datasets numpy huggingface_hub\n\n"
            "Then run again from `src/` with PYTHONPATH=.\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    use_streaming = bool(args.streaming)
    mode = "streaming" if use_streaming else "download+memory-mapped (recommended)"
    print(
        f"Loading reciTAL/mlsum (tu) split={args.split!r} ({mode})...",
        flush=True,
    )
    if not use_streaming:
        print(
            "  (first run downloads data to the HF cache; ~249k train rows, be patient)\n"
            "  If you see HTTP 'range requests' errors, do NOT use --streaming.",
            flush=True,
        )

    ds = None
    last_err: BaseException | None = None
    # Default streaming=False: avoids fsspec HTTP range-request failures with IterableDataset.
    # Try parquet conversion branch first when not streaming (often no custom script).
    revision_order = ("refs/convert/parquet", None) if not use_streaming else (None,)
    for revision in revision_order:
        try:
            kw = dict(
                path="reciTAL/mlsum",
                name="tu",
                split=args.split,
                streaming=use_streaming,
                trust_remote_code=True,
            )
            if revision is not None:
                kw["revision"] = revision
            ds = load_dataset(**kw)
            if revision:
                print(f"  OK using revision={revision!r}", flush=True)
            break
        except BaseException as e:
            last_err = e
            ds = None
    if ds is None:
        print(
            "\nFailed to load MLSUM. If you see 'Dataset scripts are no longer supported':\n"
            "  pip install 'datasets>=2.16,<3.0'\n"
            "Legacy dataset repos need `datasets` 2.x, or load from the parquet conversion branch.\n",
            file=sys.stderr,
            flush=True,
        )
        raise last_err if last_err else RuntimeError("load_dataset failed")

    n_written = 0
    with out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            if args.max_rows is not None and idx >= args.max_rows:
                break
            text = row.get("text") or ""
            rec = {"id": str(idx), "text": str(text)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 50000 == 0:
                print(f"  wrote {n_written} rows...", flush=True)

    print(f"Wrote {n_written} articles to {out}", flush=True)
    root = _repo_root()
    try:
        hint = out.relative_to(root)
    except ValueError:
        hint = out
    print(
        "Use with: cd src && PYTHONPATH=. python -m experiments.runner "
        f"--corpus-mode mlsum --mlsum-manifest {hint} --source local ...",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
