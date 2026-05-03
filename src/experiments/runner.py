"""CLI benchmark runner for Turkuaz-RAG retrieval experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from experiments.corpus import build_doc_pool
from experiments.mlsum_corpus import (
    build_mlsum_gold_map,
    load_mlsum_manifest,
    load_mlsum_manifest_subset,
)
from experiments.metrics import aggregate_means, mrr_first_relevant, recall_single_both
from experiments.retrieval_backend import (
    DenseEmbeddingIndex,
    embed_documents_batched,
    make_embedding_client_from_settings,
)
from experiments.systems import RetrievalSystems, default_system_names
from experiments.turkuaz_loader import load_huggingface, load_jsonl, load_local_directory
from experiments.types import TurkuazSample

KS = (2, 5, 10)


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return _repo_src().parent


def load_samples(args: argparse.Namespace) -> List[TurkuazSample]:
    if args.source == "huggingface":
        return load_huggingface(limit=args.limit)
    if args.source == "local":
        data_path = Path(args.data_dir)
        if not data_path.is_absolute():
            data_path = (_project_root() / data_path).resolve()
        return load_local_directory(data_path, limit=args.limit)
    path = Path(args.jsonl_path)
    return load_jsonl(path, limit=args.limit)


def parse_systems(arg: str) -> List[str]:
    if arg.strip().lower() == "all":
        return ["no_rag"] + default_system_names()
    return [s.strip() for s in arg.split(",") if s.strip()]


def evaluate_row(
    ranked_ids: Sequence[int],
    gold: tuple[int, int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"mrr": mrr_first_relevant(ranked_ids, gold)}
    for k in KS:
        s, b = recall_single_both(list(ranked_ids), gold, k)
        out[f"single@{k}"] = s
        out[f"both@{k}"] = b
    return out


def main(argv: Sequence[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(_repo_src() / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Turkuaz-RAG retrieval benchmark")
    parser.add_argument(
        "--source",
        choices=("huggingface", "local", "jsonl"),
        default="jsonl",
        help="local = read CSV/zip from --data-dir (download first via experiments.download_dataset)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/turkuaz-rag",
        help="When --source local: folder under project root (default: data/turkuaz-rag)",
    )
    parser.add_argument("--jsonl-path", type=str, default="experiments/fixtures/sample_eval.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--systems", type=str, default="all", help="Comma list or 'all'")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory under src/ (default: experiments/results)",
    )
    parser.add_argument(
        "--mock-embeddings",
        action="store_true",
        help="Deterministic vectors (no API calls); for wiring tests only.",
    )
    parser.add_argument(
        "--corpus-mode",
        choices=("closed", "mlsum"),
        default="closed",
        help="closed=Turkuaz contexts only; mlsum=full MLSUM manifest (Scenario 2).",
    )
    parser.add_argument(
        "--mlsum-manifest",
        type=str,
        default="data/mlsum-tu/manifest.jsonl",
        help="JSONL from build_mlsum_manifest (project-root-relative unless absolute).",
    )
    parser.add_argument(
        "--mlsum-max-docs",
        type=int,
        default=None,
        help="Load only first N manifest lines (debug); must cover all gold ids.",
    )
    parser.add_argument(
        "--mlsum-subset",
        action="store_true",
        help=(
            "Load only gold articles for the current samples + extra distractor docs "
            "(fast local/mock runs; not full 249k corpus)."
        ),
    )
    parser.add_argument(
        "--mlsum-extra-noise",
        type=int,
        default=5000,
        help="With --mlsum-subset: how many non-gold manifest rows to add as retrieval noise.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    src_root = _repo_src()
    out_root = src_root / args.output_dir
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args)
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        return 1

    systems_requested = parse_systems(args.systems)

    if args.corpus_mode == "closed":
        doc_texts, gold_map = build_doc_pool(samples)
    elif args.corpus_mode == "mlsum":
        man_path = Path(args.mlsum_manifest)
        if not man_path.is_absolute():
            man_path = (_project_root() / man_path).resolve()
        if args.mlsum_subset:
            print(
                f"MLSUM subset mode: gold ids for {len(samples)} samples + "
                f"{args.mlsum_extra_noise} noise docs (streaming manifest)...",
                flush=True,
            )
            doc_texts, id_to_idx = load_mlsum_manifest_subset(
                man_path,
                samples,
                extra_noise_docs=args.mlsum_extra_noise,
            )
        else:
            doc_texts, id_to_idx = load_mlsum_manifest(man_path, max_docs=args.mlsum_max_docs)
        gold_map = build_mlsum_gold_map(samples, id_to_idx)
    else:
        raise RuntimeError(f"Unknown corpus mode: {args.corpus_mode}")

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "corpus_mode": args.corpus_mode,
        "jsonl_path": str(Path(args.jsonl_path)) if args.source == "jsonl" else None,
        "data_dir": str((_project_root() / Path(args.data_dir)).resolve())
        if args.source == "local"
        else None,
        "mlsum_manifest": str(Path(args.mlsum_manifest))
        if args.corpus_mode == "mlsum"
        else None,
        "mlsum_max_docs": args.mlsum_max_docs,
        "mlsum_subset": args.mlsum_subset if args.corpus_mode == "mlsum" else False,
        "mlsum_extra_noise": args.mlsum_extra_noise if args.corpus_mode == "mlsum" else None,
        "limit": args.limit,
        "n_samples": len(samples),
        "n_unique_docs": len(doc_texts),
        "systems": systems_requested,
        "ks": list(KS),
        "mock_embeddings": args.mock_embeddings,
    }

    # Dense index + BM25 (skip embedding if only no_rag)
    need_embed = any(s != "no_rag" for s in systems_requested)
    engines: RetrievalSystems | None = None
    if need_embed:
        if args.mock_embeddings:
            from experiments.mock_embeddings import MockEmbeddingClient

            dim = int(os.environ.get("EMBEDDING_MODEL_SIZE", "1536"))
            emb = MockEmbeddingClient(dim)
        else:
            emb = make_embedding_client_from_settings()
        print(f"Embedding {len(doc_texts)} unique documents...", flush=True)
        mat = embed_documents_batched(emb, doc_texts)
        dense_index = DenseEmbeddingIndex(mat)
        from experiments.bm25 import BM25Index

        bm25 = BM25Index(doc_texts)
        engines = RetrievalSystems(emb, dense_index, mat, bm25, doc_texts)

    per_sample_path = run_dir / "per_sample.jsonl"
    rows_for_agg: Dict[str, List[dict]] = {s: [] for s in systems_requested}

    with per_sample_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            gold = gold_map[sample.sample_id]
            record: Dict[str, Any] = {
                "sample_id": sample.sample_id,
                "question_type": sample.question_type,
                "question": sample.question,
                "gold_doc_ids": list(gold),
                "systems": {},
            }
            for sys_name in systems_requested:
                if sys_name == "no_rag":
                    ranked: List[int] = []
                else:
                    assert engines is not None
                    ranked = engines.run(sys_name, sample.question)
                metrics_row = evaluate_row(ranked, gold)
                metrics_row["ranked_ids"] = ranked[:50]
                record["systems"][sys_name] = metrics_row
                rows_for_agg[sys_name].append(metrics_row)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {"run_id": run_id, "per_system": {}}
    for sys_name, rows in rows_for_agg.items():
        summary["per_system"][sys_name] = aggregate_means(rows, KS)

    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote results to {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
