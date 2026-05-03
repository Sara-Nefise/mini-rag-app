"""Download eneSadi/turkuaz-rag from Hugging Face into a project data folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    # src/experiments/download_dataset.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Turkuaz-RAG dataset files into a local directory (Hugging Face snapshot)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <project>/data/turkuaz-rag)",
    )
    args = parser.parse_args(argv)

    out = args.out if args.out is not None else _repo_root() / "data" / "turkuaz-rag"
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from e

    snapshot_download(
        repo_id="eneSadi/turkuaz-rag",
        repo_type="dataset",
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print(f"Saved under: {out}", flush=True)
    print("Load in benchmark with: --source local  (or set --data-dir to this path)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
